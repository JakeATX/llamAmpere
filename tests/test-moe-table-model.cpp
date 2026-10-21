#include "../ggml/src/ggml-cuda/moe-table-model.h"
extern "C" {
#include "../vendor/hash/sha256/sha256.h"
}
#include <array>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
constexpr size_t layer_bytes = 40 + 2 * (512 + 512 * 512);
void put32(std::vector<uint8_t> & data, size_t at, uint32_t value) {
    for (int i = 0; i < 4; ++i) data[at + i] = uint8_t(value >> (i * 8));
}
void digest(std::vector<uint8_t> & data) {
    sha256_hash(data.data() + 32, data.data() + 64, data.size() - 64);
}
std::vector<uint8_t> fixture(unsigned nl, unsigned lead) {
    std::vector<uint8_t> data(64 + nl * layer_bytes);
    std::memcpy(data.data(), "MTPFTB02", 8);
    for (const auto field : {std::pair<size_t, unsigned>{8, 2}, {12, nl}, {16, 512},
                            {20, lead}, {24, 3}, {28, 512}}) put32(data, field.first, field.second);
    for (unsigned layer = 0; layer < nl; ++layer) {
        const size_t at = 64 + layer * layer_bytes;
        put32(data, at, 0x3f800000); // exact1.0f
        const bool active = layer >= lead && layer + 1 < nl;
        put32(data, at + 4, active ? 100 : 0);
        put32(data, at + 8, active ? 1 : 0);
        put32(data, at + 12, 512);
        for (unsigned k = 0; k < 3; ++k) put32(data, at + 16 + 8 * k, (k + 1) * 128);
        if (active) {
            const size_t offset = at + 40 + 2 * (512 + 3 * 512 + 9);
            data[offset] = 0x2e; data[offset + 1] = 0xfb; // -1234
        }
    }
    digest(data);
    return data;
}
void write(const std::string & path, const std::vector<uint8_t> & data) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char *>(data.data()), data.size());
    if (!output) throw std::runtime_error("fixture write failed");
}
void rejected(const std::string & path, std::vector<uint8_t> data, bool rehash = true) {
    if (rehash) digest(data);
    write(path, data);
    bool caught = false;
    try { (void) moe_tp::load_model(path); } catch (const std::runtime_error &) { caught = true; }
    if (!caught) throw std::runtime_error("malformed artifact accepted");
}
}
int main(int argc, char ** argv) {
    // Pass an exported artifact to validate Python/C++ interchange independently.
    if (argc == 3 && std::string(argv[1]) == "--read") {
        auto model = moe_tp::load_model(argv[2]);
        std::cout << "layers=" << model.layers.size() << " lead=" << model.lead << " method=" << model.method << '\n';
        return 0;
    }
    if (argc != 2) { std::cerr << "usage: test-moe-table-model temporary-file\n"; return 2; }
    const std::string path = argv[1];
    for (unsigned lead : {1u, 2u, 3u}) {
        auto data = fixture(5, lead);
        write(path, data);
        const auto model = moe_tp::load_model(path, 5);
        if (model.lead != lead || model.method != 3 || model.layers.size() != 5 ||
                model.layers[lead].weights[3 * 512 + 9] != -1234 ||
                model.layers[lead].projection_bytes != std::array<uint64_t, 3>{128, 256, 384} ||
                model.layers[lead - 1].active()) throw std::runtime_error("artifact roundtrip mismatch");
        bool wrong_model = false;
        try { (void) moe_tp::load_model(path, 4); } catch (const std::runtime_error &) { wrong_model = true; }
        if (!wrong_model) throw std::runtime_error("wrong model dimensions accepted");
    }
    const auto valid = fixture(5, 2);
    auto calibrated = valid;
    for (unsigned layer = 0; layer < 5; ++layer) {
        const size_t at = 64 + layer * layer_bytes + 8;
        put32(calibrated, at, layer >= 2 && layer < 4 ? 3 : 2);
    }
    digest(calibrated);
    write(path, calibrated);
    const auto thresholded = moe_tp::load_model(path, 5);
    if (!thresholded.layers[2].positive_only() || thresholded.layers[2].weights[3 * 512 + 9] != -1234) {
        throw std::runtime_error("positive-only signed model rejected");
    }
    auto bad = valid; bad.back() ^= 1; rejected(path, bad, false);
    bad = valid; bad.pop_back(); rejected(path, bad);
    bad = valid; put32(bad, 12, 128); rejected(path, bad);
    bad = valid; put32(bad, 20, 4); rejected(path, bad);
    bad = valid; put32(bad, 24, 1); rejected(path, bad);
    bad = valid; put32(bad, 64, 0x7f800000); rejected(path, bad); // infinite scale
    bad = valid; put32(bad, 64 + 4, 1); put32(bad, 64 + 8, 1); rejected(path, bad); // before lead
    bad = valid; put32(bad, 64 + 4 * layer_bytes + 4, 1);
    put32(bad, 64 + 4 * layer_bytes + 8, 1); rejected(path, bad); // cropped final target
    bad = valid; put32(bad, 64 + 16, 0); rejected(path, bad); // zero projection bytes
    bad = valid; bad[64 + 40] = 1; rejected(path, bad); // inactive vote
    bad = valid; bad[64 + 2 * layer_bytes + 40] = 0; bad[64 + 2 * layer_bytes + 41] = 0x80;
    rejected(path, bad); // -32768 violates score packing contract
    std::remove(path.c_str());
    std::cout << "transition artifact: signed votes, geometry, dimensions, flags, bounds and SHA256 PASS\n";
}
