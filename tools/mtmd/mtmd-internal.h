#pragma once

#include "mtmd.h"

#include <string>
#include <vector>

// !!! Internal header, to be used by mtmd, the server and unit tests only !!!

#define MTMD_INTERNAL_HEADER

// The server requires pinned weights when the memory fit counted them as host memory.
MTMD_API mtmd_context * mtmd_init_from_file_impl(const char * mmproj_fname,
        const llama_model * text_model,
        const mtmd_context_params & ctx_params,
        bool host_weights_strict);

// bitmap is null for text parts
struct mtmd_internal_part {
    std::string text;
    const mtmd_bitmap * bitmap;
    // only used for text parts
    bool parse_special = false;
};

// [QWEN_VIDEO] merged parts are erased from `parts`, so one group always maps to one part
std::vector<std::vector<const mtmd_bitmap *>> mtmd_group_mergeable_bitmaps(std::vector<mtmd_internal_part> & parts, int n_merge);

struct mtmd_image_preprocessor;

// mtmd_get_video_config() for a given template, without a context
mtmd_video_config mtmd_video_config_init(mtmd_video_template tmpl, const mtmd_image_preprocessor * preproc,
                                         int32_t src_width, int32_t src_height,
                                         int32_t src_n_frames, float src_fps);
