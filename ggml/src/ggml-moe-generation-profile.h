#pragma once

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <string>
#include <vector>
#if defined(__unix__)
#include <sys/stat.h>
#include <unistd.h>
#endif

static inline bool ggml_moe_generation_profile_same_file(const char * a, const char * b) {
    if (!a || !b || !*a || !*b) { return false; }
    if (strcmp(a, b) == 0) { return true; }
#if defined(__unix__)
    struct stat sa = {}, sb = {};
    return stat(a, &sa) == 0 && stat(b, &sb) == 0 && sa.st_dev == sb.st_dev && sa.st_ino == sb.st_ino;
#else
    return false;
#endif
}

// fork: Ordered slices preserve the identity of the validated startup profile.
struct ggml_moe_generation_profile_slice {
    size_t offset;
    size_t size;
};

static inline uint64_t ggml_moe_generation_profile_hash(uint64_t hash, const void * data, size_t size) {
    const auto * bytes = static_cast<const unsigned char *>(data);
    for (size_t i = 0; i < size; ++i) {
        hash = (hash ^ bytes[i]) * UINT64_C(1099511628211);
    }
    return hash;
}

static inline bool ggml_moe_generation_profile_write(const std::string & path, uint64_t signature,
        const std::vector<ggml_moe_generation_profile_slice> & slices,
        const std::vector<unsigned long long> & counts, uint64_t epoch, uint64_t scopes) {
    static_assert(sizeof(unsigned long long) == sizeof(uint64_t), "profile counters require 64 bits");
    if (path.empty() || slices.empty() || slices.size() > 4096 || counts.empty() || !epoch || !scopes) {
        return false;
    }
    size_t entries = 0;
    uint64_t checksum = UINT64_C(14695981039346656037);
    bool observed = false;
    for (const auto & slice : slices) {
        if (!slice.size || slice.size > 65536 || slice.offset > counts.size() || slice.size > counts.size() - slice.offset ||
                entries > counts.size() - slice.size) {
            return false;
        }
        for (const auto & previous : slices) {
            if (&previous == &slice) { break; }
            if (slice.offset < previous.offset + previous.size && previous.offset < slice.offset + slice.size) {
                return false;
            }
        }
        checksum = ggml_moe_generation_profile_hash(checksum, counts.data() + slice.offset, slice.size * sizeof(uint64_t));
        for (size_t e = 0; e < slice.size; ++e) { observed |= counts[slice.offset + e] != 0; }
        entries += slice.size;
    }
    if (entries != counts.size() || !observed) { return false; }
    std::string suffix = ".tmp";
#if defined(__unix__)
    suffix += "." + std::to_string((unsigned long long) getpid());
#endif
    suffix += "." + std::to_string(epoch) + "." + std::to_string(scopes);
    const std::string temporary = path + suffix;
    FILE * file = fopen(temporary.c_str(), "wb");
    if (!file) { return false; }
    const uint64_t header[] = {UINT64_C(0x31464f5250454f4d), (uint64_t) slices.size(), entries, signature, checksum};
    bool ok = fwrite(header, sizeof(header), 1, file) == 1;
    for (const auto & slice : slices) {
        if (ok) { ok = fwrite(counts.data() + slice.offset, sizeof(uint64_t), slice.size, file) == slice.size; }
    }
    ok = fclose(file) == 0 && ok;
    if (!ok || rename(temporary.c_str(), path.c_str()) != 0) {
        remove(temporary.c_str());
        return false;
    }
    const std::string metadata = path + ".json";
    const std::string metadata_temporary = metadata + suffix;
    file = fopen(metadata_temporary.c_str(), "wb");
    if (!file) { return false; }
    ok = fprintf(file, "{\n  \"kind\": \"device-route-operations\",\n  \"accounting\": \"target projection routes including speculative verification and replay; not accepted tokens\",\n  \"prompt_counts_merged\": false,\n  \"phase_filter\": \"device dispatch lifetime; use a fresh process with post-prompt activation for generation-only collection\",\n  \"epoch\": %llu,\n  \"completed_scopes\": %llu,\n  \"tail_incomplete\": true,\n  \"checksum\": \"0x%016llx\",\n  \"signature\": \"0x%016llx\",\n  \"entries\": %llu\n}\n",
            (unsigned long long) epoch, (unsigned long long) scopes, (unsigned long long) checksum,
            (unsigned long long) signature, (unsigned long long) entries) > 0;
    ok = fclose(file) == 0 && ok;
    if (!ok || rename(metadata_temporary.c_str(), metadata.c_str()) != 0) {
        remove(metadata_temporary.c_str());
        return false;
    }
    return true;
}
