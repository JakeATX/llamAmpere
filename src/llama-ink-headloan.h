#pragma once

// [HEADLOAN] (#247) the output head lends its VRAM to streamed prompts (Inkling 2 on 8/10/12 GB cards).
//
// A prompt needs the head for its last row only. With LLAMA_INK_HEAD_LOAN=1 the loader keeps the head in host
// memory and the context uploads it into a VRAM buffer of its own before the MoE cache is sized (so the boot layout is
// the same as without the flag). While the prompt expert streamer is armed, the VRAM buffer is freed and the head points
// at the host copy: the prompt ladder can take that VRAM (469 MiB for the Inkling 2 Mini Q8_0 head) and the last row's
// logits are computed on the CPU (op offload is off on these cards). With mmap (default) the host copy is the mapped
// file (pageable), so an upload runs at pageable-copy speed. When the streamer is released, the head is uploaded
// again before the MoE cache is refilled. Same weights; the CPU head of the prompt's last row rounds differently from
// the GPU head (first reply token only). Off by default.

#include "ggml.h"
#include "ggml-backend.h"

struct llama_model;

// LLAMA_INK_HEAD_LOAN=1 on an Inkling 2 model: the loader places the head in host memory
bool llama_ink_headloan_wanted(int arch);

class llama_ink_headloan {
public:
    // head: the model's output tensor, in a host buffer; dev_buft: the GPU buffer type it runs from
    llama_ink_headloan(ggml_tensor * head, ggml_backend_buffer_type_t dev_buft);
    ~llama_ink_headloan(); // the head points back at its host copy

    bool ok() const { return dev_buf != nullptr || lent; }
    bool is_lent() const { return lent; }
    size_t bytes() const { return dev_bytes; }

    bool lend();    // free the VRAM copy; the head runs from host memory
    bool restore(); // upload the head into VRAM again (false: no VRAM, the head stays on the host; retry later)

private:
    bool upload();

    ggml_tensor *              head      = nullptr;
    ggml_backend_buffer_t      host_buf  = nullptr; // owned by the model
    void *                     host_data = nullptr;
    void *                     host_extra = nullptr;
    ggml_backend_buffer_type_t dev_buft  = nullptr;
    ggml_backend_buffer_t      dev_buf   = nullptr; // owned here
    size_t                     dev_bytes = 0;
    bool                       lent      = false;
    bool                       fail_logged = false;
};
