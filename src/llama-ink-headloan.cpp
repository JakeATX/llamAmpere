#include "llama-ink-headloan.h"

#include "llama-arch.h"
#include "llama-impl.h"

#include <cstdlib>
#include <cstring>

bool llama_ink_headloan_wanted(int arch) {
    static const bool on = [] { const char * e = getenv("LLAMA_INK_HEAD_LOAN"); return e && atoi(e) != 0; }();
    return on && (arch == LLM_ARCH_INKLING2 || arch == LLM_ARCH_INKLING2_LOOP);
}

llama_ink_headloan::llama_ink_headloan(ggml_tensor * head, ggml_backend_buffer_type_t dev_buft) :
        head(head), dev_buft(dev_buft) {
    if (!head || !head->buffer || !ggml_backend_buffer_is_host(head->buffer) || !dev_buft) {
        return;
    }
    host_buf   = head->buffer;
    host_data  = head->data;
    host_extra = head->extra;
    dev_bytes  = ggml_backend_buft_get_alloc_size(dev_buft, head);
    if (!upload()) {
        LLAMA_LOG_WARN("%s: [HEADLOAN] no VRAM for the head (%.0f MiB): it stays in host memory\n", __func__, dev_bytes/1048576.0);
        return;
    }
    LLAMA_LOG_WARN("%s: [HEADLOAN] head %s %.0f MiB in its own VRAM buffer; streamed prompts lend it (LLAMA_INK_HEAD_LOAN=0 = off)\n",
            __func__, head->name, dev_bytes/1048576.0);
}

llama_ink_headloan::~llama_ink_headloan() {
    if (dev_buf) {
        head->buffer = host_buf;
        head->data   = host_data;
        head->extra  = host_extra;
        ggml_backend_buffer_free(dev_buf);
        dev_buf = nullptr;
    }
}

bool llama_ink_headloan::upload() {
    GGML_ASSERT(!dev_buf);
    dev_buf = ggml_backend_buft_alloc_buffer(dev_buft, dev_bytes);
    if (!dev_buf) {
        return false;
    }
    ggml_backend_buffer_set_usage(dev_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    head->buffer = nullptr;
    head->data   = nullptr;
    head->extra  = nullptr;
    if (ggml_backend_tensor_alloc(dev_buf, head, ggml_backend_buffer_get_base(dev_buf)) != GGML_STATUS_SUCCESS) {
        head->buffer = host_buf;
        head->data   = host_data;
        head->extra  = host_extra;
        ggml_backend_buffer_free(dev_buf);
        dev_buf = nullptr;
        return false;
    }
    ggml_backend_tensor_set(head, host_data, 0, ggml_nbytes(head)); // synchronous: the host copy is read-only
    lent = false;
    return true;
}

bool llama_ink_headloan::lend() {
    if (lent || !dev_buf) {
        return lent;
    }
    head->buffer = host_buf;
    head->data   = host_data;
    head->extra  = host_extra;
    ggml_backend_buffer_free(dev_buf);
    dev_buf = nullptr;
    lent    = true;
    return true;
}

bool llama_ink_headloan::restore() {
    if (!lent) {
        return true;
    }
    if (!upload()) {
        if (!fail_logged) {
            LLAMA_LOG_WARN("%s: [HEADLOAN] cannot upload the head again (%.0f MiB): it runs from host memory until it fits\n",
                    __func__, dev_bytes/1048576.0);
            fail_logged = true;
        }
        return false;
    }
    fail_logged = false;
    return true;
}
