#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#ifdef __CUDACC__
#define SJKVARN_LOW_HD __host__ __device__ __forceinline__
#else
#define SJKVARN_LOW_HD inline
#endif

namespace sj_kvarn_lowbits {

// A prefill block covers 64 rows and cannot cross a 128-row record.
SJKVARN_LOW_HD bool strip_master_expand_location(int sink, int end, int position, const int32_t * table,
        size_t table_records, size_t records, int & record) {
    record = -1;
    if (sink < 0 || sink%128 || end < sink || (end - sink)%128 || position < sink ||
        position >= end || position%64 || records > size_t(INT32_MAX)) return false;
    const int group = (position - sink)/128;
    if (table && size_t(group) >= table_records) return false;
    const int physical = table ? table[group] : group;
    if (physical < 0 || size_t(physical) >= records) return false;
    record = physical;
    return true;
}

// Returns 1 for a new head record, 0 for unused scratch, -1 for an invalid seal range.
SJKVARN_LOW_HD int strip_master_seal_location(int sink, int cap, int end, int old, int source_rows, int max_groups,
        const int32_t * table, size_t table_records, size_t master_records, int heads, int local,
        int & row, int & record) {
    row = record = -1;
    if (sink < 0 || cap <= 0 || cap%128 || old < sink || end < old ||
        (old - sink)%128 || (end - old)%128 || source_rows < sink || cap > source_rows - sink ||
        heads <= 0 || local < 0 || max_groups <= 0 || (end - old)/128 > max_groups ||
        master_records > size_t(INT32_MAX) || master_records%heads) return -1;
    const int group = local/heads;
    if (group >= (end - old)/128) return 0;
    const int logical = (old - sink)/128 + group;
    if (table && size_t(logical) >= table_records) return -1;
    const int page = table ? table[logical] : logical;
    if (page < 0 || size_t(page) >= master_records/heads) return -1;
    row = sink + int((int64_t(old) - sink + int64_t(group)*128)%cap);
    record = page*heads + local%heads;
    return 1;
}

// Compact token-axis strips keep one history byte per channel. No code is re-encoded.
SJKVARN_LOW_HD int stage_strip_bytes(int bits_k, int bits_v) {
    return 256*(2*bits_k + 1 + 2*bits_v + 1) + 96;
}

SJKVARN_LOW_HD int stage_source_byte(int bits, int strip, int byte) {
    const int channel_bytes = 2*bits + 1;
    const int channel = byte/channel_bytes, b = byte%channel_bytes;
    return b == 0 && strip == 0 ? -1 : channel*16*bits + strip*2*bits + b - 1;
}

// Fixed D256/G128 token-axis records. The host master has the full compact stage layout.
struct strip_master_layout {
    int bits_k, bits_v;
    int k_payload, v_payload, canonical_bytes;
    int k_strip, v_strip, strip_bytes, bytes;
};

SJKVARN_LOW_HD bool make_strip_master_layout(int bits_k, int bits_v, strip_master_layout & l) {
    l = {};
    if ((bits_k != 2 && bits_k != 3) || (bits_v != 2 && bits_v != 3)) return false;
    l.bits_k = bits_k; l.bits_v = bits_v;
    l.k_payload = 4096*bits_k; l.v_payload = 4096*bits_v;
    l.canonical_bytes = l.k_payload + l.v_payload + 2304;
    l.k_strip = 256*(2*bits_k + 1); l.v_strip = 256*(2*bits_v + 1);
    l.strip_bytes = stage_strip_bytes(bits_k, bits_v);
    l.bytes = 1536 + 8*l.strip_bytes;
    return true;
}

struct strip_master_expand_plan {
    int rows;
    size_t scratch_bytes, output_bytes, total_bytes;
};

// Budget covers canonical scratch and both f16 outputs. Each 64-row block has its own record copy.
SJKVARN_LOW_HD bool make_strip_master_expand_plan(int bits_k, int bits_v, int rows, int heads,
        size_t budget, strip_master_expand_plan & plan) {
    plan = {};
    strip_master_layout l;
    if (!make_strip_master_layout(bits_k, bits_v, l) || rows <= 0 || rows%64 || heads <= 0 || heads > 65535) return false;
    const size_t bytes_head = size_t(l.canonical_bytes) + 2*64*512;
    if (size_t(heads) > SIZE_MAX/bytes_head) return false;
    const size_t bytes_block = size_t(heads)*bytes_head;
    const size_t limit = budget/bytes_block;
    const size_t blocks = size_t(rows/64) < limit ? size_t(rows/64) : limit;
    if (!blocks) return false;
    plan.rows = int(blocks*64);
    plan.scratch_bytes = blocks*size_t(heads)*size_t(l.canonical_bytes);
    plan.output_bytes = blocks*size_t(heads)*64*512;
    plan.total_bytes = blocks*bytes_block;
    return true;
}

SJKVARN_LOW_HD bool strip_master_prefill_fallback_safe(bool master, int64_t queries) {
    return !master || queries <= 8;
}

SJKVARN_LOW_HD bool strip_master_prefill_workspace(int bits_k, int bits_v, int rows, int heads,
        size_t existing, size_t budget, size_t & offset, size_t & total) {
    offset = total = 0;
    strip_master_expand_plan p;
    if (existing > SIZE_MAX - 127 || !make_strip_master_expand_plan(bits_k, bits_v, rows, heads, SIZE_MAX, p) ||
        p.rows != rows) return false;
    const size_t aligned = (existing + 127)/128*128;
    if (aligned > budget || p.scratch_bytes > budget - aligned) return false;
    offset = aligned;
    total = aligned + p.scratch_bytes;
    return true;
}

// Destination master byte -> canonical source byte; -1 supplies zero history or rejects an invalid byte.
SJKVARN_LOW_HD int strip_master_source_byte(const strip_master_layout & l, int byte) {
    if (byte < 0 || byte >= l.bytes) return -1;
    const int meta = l.k_payload + l.v_payload;
    if (byte < 1024) return meta + byte;
    if (byte < 1536) return meta + 1280 + byte - 1024;
    const int strip = (byte - 1536)/l.strip_bytes, b = (byte - 1536)%l.strip_bytes;
    if (b < l.k_strip + l.v_strip) {
        const bool v = b >= l.k_strip;
        const int from = stage_source_byte(v ? l.bits_v : l.bits_k, strip, v ? b - l.k_strip : b);
        return from < 0 ? -1 : (v ? l.k_payload : 0) + from;
    }
    const int token_meta = b - l.k_strip - l.v_strip;
    const int base = token_meta < 32 ? 1024 : token_meta < 64 ? 1792 : 2048;
    return meta + base + strip*32 + token_meta%32;
}

// Canonical byte -> its unique non-history byte in the master. Saved state omits duplicate history.
SJKVARN_LOW_HD int strip_master_byte(const strip_master_layout & l, int byte) {
    if (byte < 0 || byte >= l.canonical_bytes) return -1;
    const int meta = l.k_payload + l.v_payload;
    if (byte < meta) {
        const bool v = byte >= l.k_payload;
        const int bits = v ? l.bits_v : l.bits_k, b = v ? byte - l.k_payload : byte;
        const int channel = b/(16*bits), row = b%(16*bits);
        return 1536 + (row/(2*bits))*l.strip_bytes + (v ? l.k_strip : 0) +
            channel*(2*bits + 1) + 1 + row%(2*bits);
    }
    const int b = byte - meta;
    if (b < 1024) return b;
    if (b >= 1280 && b < 1792) return 1024 + b - 1280;
    const int base = b < 1280 ? 1024 : b < 2048 ? 1792 : 2048;
    const int field = b < 1280 ? 0 : b < 2048 ? 32 : 64;
    return 1536 + ((b - base)/32)*l.strip_bytes + l.k_strip + l.v_strip + field + (b - base)%32;
}

SJKVARN_LOW_HD bool strip_master_range(const strip_master_layout & l, int offset, int size) {
    return l.bytes > 0 && offset >= 0 && size >= 0 && offset <= l.canonical_bytes && size <= l.canonical_bytes - offset;
}

SJKVARN_LOW_HD void strip_master_store(const strip_master_layout & l, uint8_t * master, int byte, uint8_t value) {
    const int to = strip_master_byte(l, byte);
    master[to] = value;
    if (byte < l.k_payload + l.v_payload) {
        const bool v = byte >= l.k_payload;
        const int bits = v ? l.bits_v : l.bits_k;
        const int row = (v ? byte - l.k_payload : byte)%(16*bits);
        // The last payload byte in a strip is also the next strip's history.
        if (row%(2*bits) == 2*bits - 1 && row < 14*bits) master[to + l.strip_bytes - 2*bits] = value;
    }
}

SJKVARN_LOW_HD void strip_master_zero_history(const strip_master_layout & l, uint8_t * master) {
    for (int channel = 0; channel < 256; ++channel) {
        master[1536 + channel*(2*l.bits_k + 1)] = 0;
        master[1536 + l.k_strip + channel*(2*l.bits_v + 1)] = 0;
    }
}

// One-record canonical intervals with separate data storage; the caller splits views and record boundaries.
SJKVARN_LOW_HD bool strip_master_get(const strip_master_layout & l, const uint8_t * master, uint8_t * data, int offset, int size) {
    if (!strip_master_range(l, offset, size) || (size && (!master || !data))) return false;
    for (int i = 0; i < size; ++i) data[i] = master[strip_master_byte(l, offset + i)];
    return true;
}

SJKVARN_LOW_HD bool strip_master_set(const strip_master_layout & l, uint8_t * master, const uint8_t * data, int offset, int size) {
    if (!strip_master_range(l, offset, size) || (size && (!master || !data))) return false;
    for (int i = 0; i < size; ++i) strip_master_store(l, master, offset + i, data[i]);
    if (size) strip_master_zero_history(l, master);
    return true;
}

SJKVARN_LOW_HD bool strip_master_fill(const strip_master_layout & l, uint8_t * master, uint8_t value, int offset, int size) {
    if (!strip_master_range(l, offset, size) || (size && !master)) return false;
    for (int i = 0; i < size; ++i) strip_master_store(l, master, offset + i, value);
    if (size) strip_master_zero_history(l, master);
    return true;
}

// Layout comes from make_strip_master_layout; storage must cover the reported physical bytes.
SJKVARN_LOW_HD bool strip_master_records_size(const strip_master_layout & l, size_t records, size_t & canonical, size_t & physical) {
    canonical = physical = 0;
    if (l.bytes <= 0 || l.canonical_bytes <= 0 || records > SIZE_MAX/size_t(l.bytes) ||
        records > SIZE_MAX/size_t(l.canonical_bytes)) return false;
    canonical = records*size_t(l.canonical_bytes);
    physical = records*size_t(l.bytes);
    return true;
}

SJKVARN_LOW_HD bool strip_master_records_range(const strip_master_layout & l, size_t records, size_t offset, size_t size) {
    size_t canonical, physical;
    return strip_master_records_size(l, records, canonical, physical) && offset <= canonical && size <= canonical - offset;
}

SJKVARN_LOW_HD bool strip_master_view_range(const strip_master_layout & l, size_t records, size_t base,
        size_t bytes, size_t offset, size_t size, size_t & absolute) {
    absolute = 0;
    if (!strip_master_records_range(l, records, base, bytes) || offset > bytes || size > bytes - offset) return false;
    absolute = base + offset;
    return true;
}

SJKVARN_LOW_HD bool strip_master_clock_layout(int bits_k, int bits_v, size_t records, size_t heads,
        size_t stride, size_t stage_stride, size_t capacity, strip_master_layout & l) {
    size_t canonical, physical;
    if (!make_strip_master_layout(bits_k, bits_v, l) || !records || !heads || heads > size_t(INT32_MAX) ||
        records > size_t(INT32_MAX)/heads || !strip_master_records_size(l, records*heads, canonical, physical)) return false;
    return stride == heads*size_t(l.canonical_bytes) && stage_stride == heads*size_t(l.bytes) && capacity >= physical;
}

// Packed head records with no gaps. The caller owns storage, stream joins and cache invalidation.
// Offsets are from the arena start, including a view's byte offset; data must be separate from master.
SJKVARN_LOW_HD bool strip_master_get_records(const strip_master_layout & l, const uint8_t * master, size_t records, uint8_t * data, size_t offset, size_t size) {
    if (!strip_master_records_range(l, records, offset, size) || (size && (!master || !data))) return false;
    while (size) {
        const size_t record = offset/size_t(l.canonical_bytes);
        const int in_record = int(offset%size_t(l.canonical_bytes));
        const size_t left = size_t(l.canonical_bytes - in_record);
        const int n = int(size < left ? size : left);
        strip_master_get(l, master + record*size_t(l.bytes), data, in_record, n);
        offset += size_t(n); size -= size_t(n); data += n;
    }
    return true;
}

SJKVARN_LOW_HD bool strip_master_set_records(const strip_master_layout & l, uint8_t * master, size_t records, const uint8_t * data, size_t offset, size_t size) {
    if (!strip_master_records_range(l, records, offset, size) || (size && (!master || !data))) return false;
    while (size) {
        const size_t record = offset/size_t(l.canonical_bytes);
        const int in_record = int(offset%size_t(l.canonical_bytes));
        const size_t left = size_t(l.canonical_bytes - in_record);
        const int n = int(size < left ? size : left);
        strip_master_set(l, master + record*size_t(l.bytes), data, in_record, n);
        offset += size_t(n); size -= size_t(n); data += n;
    }
    return true;
}

SJKVARN_LOW_HD bool strip_master_fill_records(const strip_master_layout & l, uint8_t * master, size_t records, uint8_t value, size_t offset, size_t size) {
    if (!strip_master_records_range(l, records, offset, size) || (size && !master)) return false;
    while (size) {
        const size_t record = offset/size_t(l.canonical_bytes);
        const int in_record = int(offset%size_t(l.canonical_bytes));
        const size_t left = size_t(l.canonical_bytes - in_record);
        const int n = int(size < left ? size : left);
        strip_master_fill(l, master + record*size_t(l.bytes), value, in_record, n);
        offset += size_t(n); size -= size_t(n);
    }
    return true;
}

struct strip_master_arena {
    int bits_k, bits_v;
    size_t base;
    size_t records;
    size_t capacity;
};

// Sorted, disjoint physical reservations; padding has no canonical clear bytes.
inline bool strip_master_fill_arenas(uint8_t * buffer, size_t bytes, const strip_master_arena * arenas,
        size_t count, uint8_t value, size_t offset, size_t size) {
    if (offset > bytes || size > bytes - offset || (size && !buffer) || (count && !arenas)) return false;
    size_t previous_end = 0;
    for (size_t i = 0; i < count; ++i) {
        const auto & a = arenas[i];
        strip_master_layout checked{};
        size_t canonical, physical;
        if (!make_strip_master_layout(a.bits_k, a.bits_v, checked) ||
            !a.records || !strip_master_records_size(checked, a.records, canonical, physical) ||
            canonical > physical || a.base < previous_end || a.base > bytes ||
            a.capacity > bytes - a.base || physical > a.capacity) return false;
        previous_end = a.base + a.capacity;
    }
    const size_t end = offset + size;
    size_t cursor = offset;
    for (size_t i = 0; i < count && cursor < end; ++i) {
        const auto & a = arenas[i];
        const size_t physical_end = a.base + a.capacity;
        if (physical_end <= cursor) continue;
        const size_t gap_end = end < a.base ? end : a.base;
        if (cursor < gap_end) {
            std::memset(buffer + cursor, value, gap_end - cursor);
            cursor = gap_end;
        }
        if (cursor == end) break;
        strip_master_layout checked{};
        if (!make_strip_master_layout(a.bits_k, a.bits_v, checked)) return false;
        const size_t canonical_end = a.base + a.records*size_t(checked.canonical_bytes);
        const size_t fill_end = end < canonical_end ? end : canonical_end;
        if (cursor < fill_end && !strip_master_fill_records(checked, buffer + a.base, a.records,
                value, cursor - a.base, fill_end - cursor)) return false;
        cursor = end < physical_end ? end : physical_end;
    }
    if (cursor < end) std::memset(buffer + cursor, value, end - cursor);
    return true;
}

template<int bits>
SJKVARN_LOW_HD uint32_t stage_window(const uint8_t * payload, int token, int channel) {
    const int bit = 2 + token*bits;
    const uint8_t * p = payload + channel*(2*bits + 1) + bit/8;
    uint32_t v = p[0];
    if (bit%8 + bits + 6 > 8) v |= uint32_t(p[1]) << 8;
    return (v >> (bit%8)) & ((1u << (bits + 6)) - 1);
}

// Read only bytes touched by a code, including the last code of a record.
template<int bits>
SJKVARN_LOW_HD uint32_t unpack(const uint8_t * payload, const int token, const int channel) {
    static_assert(bits == 2 || bits == 3, "Packed lower-bit code required");
    const unsigned bit = (token*256u + channel)*bits;
    const unsigned shift = bit & 7;
    const unsigned byte = bit >> 3;
    unsigned value = payload[byte];
    if (shift + bits > 8) {
        value |= unsigned(payload[byte + 1]) << 8;
    }
    return (value >> shift) & ((1u << bits) - 1);
}

// Reconstruct the existing MMA nibble word in registers without expanding the cache.
template<int bits, bool is_v>
SJKVARN_LOW_HD uint32_t fragment_word(const uint8_t * payload, const int word) {
    static_assert(bits >= 2 && bits <= 4, "Unsupported width");
    if constexpr (bits == 4) {
        return reinterpret_cast<const uint32_t *>(payload)[word];
    } else {
        const int lane = word & 31;
        const int tile = (word >> 5) & 15;
        const int strip = word >> 9;
        uint32_t result = 0;
        for (int nibble = 0; nibble < 8; ++nibble) {
            const int l = nibble & 3;
            const int e = nibble >> 2;
            const int row = (l & 1)*8 + (lane >> 2);
            const int col = 2*((l >> 1)*4 + (lane & 3)) + e;
            const int token = strip*16 + (is_v ? col : row);
            const int channel = tile*16 + (is_v ? row : col);
            result |= unpack<bits>(payload, token, channel) << (4*nibble);
        }
        return result;
    }
}

// Map fragment-ordered metadata slots to natural channel order for 2/3-bit records.
SJKVARN_LOW_HD int k_channel(const int slot) {
    const int lane_class = slot / 64;
    const int tile = (slot % 64) / 4;
    const int pair = (slot % 4) / 2;
    return tile*16 + 2*(lane_class + 4*pair) + (slot & 1);
}

SJKVARN_LOW_HD int v_channel(const int slot) {
    return ((slot % 32) / 2)*16 + slot/32 + (slot & 1)*8;
}

}

#undef SJKVARN_LOW_HD
