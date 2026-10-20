#include "../ggml/src/ggml-moe-generation-profile.h"

#undef NDEBUG
#include <cassert>
#include <cstring>
#include <iterator>
#include <fstream>
#include <unistd.h>
#include <sys/stat.h>

static std::vector<char> read_file(const std::string & path) {
    std::ifstream file(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

int main() {
    char directory[] = "/tmp/moe-generation-profile-XXXXXX";
    assert(mkdtemp(directory));
    const std::string path = std::string(directory) + "/generation.profile";
    const uint64_t signature = UINT64_C(0xfedcba9876543210);
    const std::vector<ggml_moe_generation_profile_slice> slices = {{2, 2}, {0, 2}};
    const std::vector<unsigned long long> counts = {8, 0, UINT64_MAX, 3};
    assert(ggml_moe_generation_profile_write(path, signature, slices, counts, 7, 64));
    const std::string alias = path + ".alias";
    assert(symlink(path.c_str(), alias.c_str()) == 0);
    assert(ggml_moe_generation_profile_same_file(path.c_str(), alias.c_str()));
    assert(!ggml_moe_generation_profile_same_file(path.c_str(), nullptr));
    remove(alias.c_str());
    const auto bytes = read_file(path);
    assert(bytes.size() == 9 * sizeof(uint64_t));
    uint64_t header[5], saved[4];
    memcpy(header, bytes.data(), sizeof(header));
    memcpy(saved, bytes.data() + sizeof(header), sizeof(saved));
    assert(header[0] == UINT64_C(0x31464f5250454f4d));
    assert(header[1] == 2 && header[2] == 4 && header[3] == signature);
    assert(saved[0] == UINT64_MAX && saved[1] == 3 && saved[2] == 8 && saved[3] == 0);
    uint64_t checksum = UINT64_C(14695981039346656037);
    const auto * raw = reinterpret_cast<const unsigned char *>(saved);
    for (size_t i = 0; i < sizeof(saved); ++i) { checksum = (checksum ^ raw[i]) * UINT64_C(1099511628211); }
    assert(header[4] == checksum);
    const auto metadata_bytes = read_file(path + ".json");
    const std::string metadata(metadata_bytes.begin(), metadata_bytes.end());
    assert(metadata.find("\"completed_scopes\": 64") != std::string::npos);
    assert(metadata.find("\"tail_incomplete\": true") != std::string::npos);
    assert(metadata.find("not accepted tokens") != std::string::npos);
    assert(!ggml_moe_generation_profile_write(path, signature, {{3, 2}}, counts, 7, 64));
    assert(!ggml_moe_generation_profile_write(path, signature, {{0, 2}, {1, 2}}, counts, 7, 64));
    assert(!ggml_moe_generation_profile_write(path, signature, {{0, 2}}, counts, 7, 64));
    assert(!ggml_moe_generation_profile_write(path, signature, slices, {0, 0, 0, 0}, 7, 64));
    assert(read_file(path) == bytes);
    // A repeated cumulative snapshot replaces the file; it does not add counts.
    assert(ggml_moe_generation_profile_write(path, signature, slices, counts, 7, 128));
    assert(read_file(path) == bytes);
    assert(!ggml_moe_generation_profile_write(path + "/missing", signature, slices, counts, 7, 64));
    assert(read_file(path) == bytes);
    assert(access((path + ".tmp").c_str(), F_OK) != 0);
    remove((path + ".json").c_str());
    assert(mkdir((path + ".json").c_str(), 0700) == 0);
    assert(!ggml_moe_generation_profile_write(path, signature, slices, counts, 7, 192));
    assert(read_file(path) == bytes);
    assert(rmdir((path + ".json").c_str()) == 0);
    remove(path.c_str());
    assert(rmdir(directory) == 0);
    puts("generation profile format/mapping/identity/count/atomic-write tests passed");
}
