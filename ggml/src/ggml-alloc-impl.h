#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

class ggml_borrowed_arena {
public:
    struct span { char * data; size_t size; };

    explicit ggml_borrowed_arena(std::vector<span> spans, size_t alignment = 128) : spans(std::move(spans)), alignment(alignment) {}

    void * alloc(size_t size, size_t * actual_size) {
        if (alignment == 0 || size > SIZE_MAX - (alignment - 1)) {
            return nullptr;
        }
        size = (size + alignment - 1) / alignment * alignment;
        for (size_t next = chunk; next < spans.size(); ++next) {
            const size_t start = next == chunk ? offset : 0;
            if (start <= spans[next].size && size <= spans[next].size - start) {
                void * ptr = spans[next].data + start;
                stack.push_back({ptr, size, chunk, offset});
                chunk = next;
                offset = start + size;
                *actual_size = size;
                return ptr;
            }
        }
        return nullptr;
    }

    bool free(void * ptr, size_t size) {
        if (stack.empty() || stack.back().ptr != ptr || stack.back().size != size) {
            return false;
        }
        chunk = stack.back().chunk;
        offset = stack.back().offset;
        stack.pop_back();
        return true;
    }

    bool empty() const { return stack.empty(); }

private:
    struct allocation { void * ptr; size_t size; size_t chunk; size_t offset; };
    std::vector<span> spans;
    size_t alignment;
    size_t chunk = 0;
    size_t offset = 0;
    std::vector<allocation> stack;
};
