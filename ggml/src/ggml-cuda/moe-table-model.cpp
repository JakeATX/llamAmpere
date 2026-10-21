#include "moe-table-model.h"
#if !defined(GGML_MOE_TABLE_NO_HASH) && __has_include("../../../vendor/hash/sha256/sha256.h")
#define GGML_MOE_TABLE_HAS_HASH 1
extern "C" {
#include "../../../vendor/hash/sha256/sha256.h"
}
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace moe_tp {
namespace {
constexpr size_t header_bytes = 64;
constexpr size_t experts = 512;
constexpr size_t layer_bytes = 40 + 2 * (experts + experts * experts);
void require(bool condition, const char * reason) {
    if (!condition) throw std::runtime_error(std::string("MoE transition model: ") + reason);
}
uint32_t u32(const uint8_t * p) {
    return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
}
int16_t i16(const uint8_t * p) {
    const unsigned bits = unsigned(p[0]) | unsigned(p[1]) << 8;
    return int16_t(bits < 32768 ? int(bits) : int(bits) - 65536);
}
uint64_t u64(const uint8_t * p) { return u32(p) | uint64_t(u32(p + 4)) << 32; }
}

Model load_model(const std::string & path, unsigned expected_layers) {
#if !defined(GGML_MOE_TABLE_HAS_HASH)
    (void) path; (void) expected_layers;
    throw std::runtime_error("MoE transition model requires the bundled SHA256 sources");
#else
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    require(bool(input), "cannot open file");
    const auto length = input.tellg();
    require(length >= std::streamoff(header_bytes + 2 * layer_bytes) &&
            length <= std::streamoff(header_bytes + 128 * layer_bytes), "unexpected file length");
    std::vector<uint8_t> data(size_t(length), uint8_t{});
    input.seekg(0);
    require(bool(input.read(reinterpret_cast<char *>(data.data()), data.size())), "incomplete file read");
    const uint8_t * h = data.data();
    require(std::memcmp(h, "MTPFTB02", 8) == 0 && u32(h + 8) == 2, "unsupported version");
    const unsigned layers = u32(h + 12);
    require(layers >= 2 && layers <= 128 && (!expected_layers || layers == expected_layers) &&
            u32(h + 16) == experts && u32(h + 28) == experts,
            "dimension mismatch");
    require(length == std::streamoff(header_bytes + layers * layer_bytes), "unexpected file length");
    Model result;
    result.lead = u32(h + 20);
    result.method = u32(h + 24);
    require(result.lead >= 1 && result.lead <= 3 && result.lead < layers, "unsupported lead");
    require(result.method >= 1 && result.method <= 3, "unsupported scoring method");
    std::copy(h + 32, h + 64, result.payload_sha256.begin());
    std::array<uint8_t, 32> actual;
    sha256_hash(actual.data(), h + header_bytes, data.size() - header_bytes);
    require(actual == result.payload_sha256, "payload SHA256 mismatch");
    result.layers.reserve(layers);
    for (unsigned target = 0; target < layers; ++target) {
        const uint8_t * p = h + header_bytes + target * layer_bytes;
        ModelLayer layer;
        const uint32_t scale_bits = u32(p);
        std::memcpy(&layer.scale, &scale_bits, sizeof(float));
        layer.training_rows = u32(p + 4);
        layer.flags = u32(p + 8);
        layer.top_n = u32(p + 12);
        require(std::isfinite(layer.scale) && layer.scale > 0 && !(layer.flags & ~3u), "invalid scale/flags");
        require(layer.top_n >= 1 && layer.top_n <= experts, "invalid row top-N");
        require(layer.active() == bool(layer.training_rows) && (result.method == 3 || layer.positive_only()),
                "inconsistent active/scoring flags");
        require(target >= result.lead || !layer.active(), "active layer before lead");
        // The final FFN may see cropped output rows, unlike the source pool.
        // Deployment never predicts it: replay and runtime must issue the same bytes.
        require(target + 1 < layers || !layer.active(), "active final target violates exclude_final_v1");
        for (int kind = 0; kind < 3; ++kind) {
            layer.projection_bytes[kind] = u64(p + 16 + 8 * kind);
            require(layer.projection_bytes[kind] && layer.projection_bytes[kind] <= (64u << 20),
                    "invalid projection bytes");
        }
        layer.bias.resize(experts);
        layer.weights.resize(experts * experts);
        size_t offset = 40;
        for (auto * values : {&layer.bias, &layer.weights}) for (auto & value : *values) {
            value = i16(p + offset);
            offset += 2;
            require(value != -32768, "coefficient exceeds exporter bounds");
            require(layer.active() || value == 0, "inactive layer contains votes");
        }
        result.layers.push_back(std::move(layer));
    }
    return result;
#endif
}
}
