#pragma once

#include "server-common.h"
#include "common.h"
#include "src/llama-context.h"
#include "src/llama-model.h"
#include "ggml/src/ggml-backend-moe-cache.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#if defined(__linux__)
#include <sys/stat.h>
#include <sys/sysmacros.h>
#endif

static size_t server_setup_host_experts(const llama_model & model) {
    size_t bytes = 0;
    for (const auto & entry : model.tensors_by_name) {
        const auto * t = entry.second;
        if (entry.first.find("_exps") == std::string::npos && entry.first.find("_chexps") == std::string::npos) {
            continue;
        }
        const auto * base = t->view_src ? t->view_src : t;
        if (base->buffer && ggml_backend_buffer_is_host(base->buffer)) {
            bytes += ggml_nbytes(t);
        }
    }
    return bytes;
}

static const char * server_setup_draft_device(const llama_context * ctx) {
    if (!ctx) { return "none"; }
    const auto & model = ctx->get_model();
    bool gpu = false;
    bool cpu = false;
    const bool mtp = ctx->get_cparams().ctx_type == LLAMA_CONTEXT_TYPE_MTP;
    for (uint32_t i = mtp ? model.hparams.n_layer() : 0; i < model.layers.size(); ++i) {
        const auto dev = model.dev_layer(i);
        const bool on_gpu = dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU;
        gpu |= on_gpu;
        cpu |= !on_gpu;
        const auto & layer = model.layers[i];
        for (const auto * t : {layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps}) {
            cpu |= t && t->buffer && ggml_backend_buffer_is_host(t->buffer);
        }
    }
    return gpu ? (cpu ? "mixed" : "GPU") : "CPU";
}

static std::string server_setup_kv(const llama_context * ctx) {
    if (!ctx) { return "none"; }
    auto name = [](ggml_type type) { return type == GGML_TYPE_COUNT ? "mixed/none" : ggml_type_name(type); };
    const std::string types = string_format("%s/%s", name(llama_get_kv_cache_type_k(ctx)), name(llama_get_kv_cache_type_v(ctx)));
    const auto * sj = ctx->get_sj_kvarn();
    if (!sj) { return types; }
    auto codec = [](ggml_type type) { return type == GGML_TYPE_I16 ? "trellis" : type == GGML_TYPE_TURBO4_0 ? "turbo4" : "scalar"; };
    return string_format("sjkvarn%u/sjkvarn%u:%s(staging:%s,sink:%s,edge:%u@%u/%u:%s)", sj->bits_k, sj->bits_v, codec(sj->body_type), types.c_str(), sj->sink_type == GGML_TYPE_COUNT ? "staging" : name(sj->sink_type), sj->edge_layers, sj->edge_bits_k, sj->edge_bits_v, codec(sj->edge_body_type));
}

static void server_setup_report(const common_params & params, llama_context * ctx, llama_context * draft) {
    const auto & model = ctx->get_model();
    const size_t host_experts = server_setup_host_experts(model);
    const bool guarded = model.hparams.n_expert > 0 && host_experts > 0;
    int warnings = 0;
    auto warn = [&](const std::string & text) {
        if (guarded) {
            ++warnings;
            SRV_WRN("[FNGUARD] WARNING %s\n", text.c_str());
        }
    };
    size_t cache_bytes = 0;
    const int cache_status = ggml_backend_sched_moe_cache_status(ctx->get_sched(), &cache_bytes);
    const bool cache_on = cache_status == 1 && cache_bytes > 0;
    if (cache_status < 0) {
        warn("expert cache status is unavailable from this backend");
    } else if (!cache_on) {
        warn("expert cache has no allocated GPU pool; use --moe-cache on and enable warmup (auto may allocate nothing)");
    }
    bool gpu_exists = false;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        gpu_exists |= ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU;
    }
    const char * draft_device = server_setup_draft_device(draft);
    if (gpu_exists && draft && std::strcmp(draft_device, "GPU") != 0) {
        warn("draft/MTP has CPU work while a GPU exists; remove CPU draft overrides and check draft GPU layers");
    }
    const int cores = common_cpu_get_num_physical_cores();
    if (llama_n_threads(ctx) != cores || llama_n_threads_batch(ctx) != cores) {
        warn(string_format("threads differ from %d physical cores; set -t %d -tb %d", cores, cores, cores));
    }
    if (draft && (llama_n_threads(draft) != cores || llama_n_threads_batch(draft) != cores)) {
        warn(string_format("draft threads differ from %d physical cores; set -td %d -tbd %d", cores, cores, cores));
    }
    uint32_t context = llama_n_ctx_seq(ctx);
    if (params.kv_unified_per_slot > 0) {
        context = std::min(context, (uint32_t) params.kv_unified_per_slot);
    }
    const uint32_t batch = llama_n_batch(ctx);
    const uint32_t ubatch = llama_n_ubatch(ctx);
    const uint32_t passes = (context + (uint64_t) batch - 1) / batch;
    if (batch < context) {
        warn(string_format("prompt batch %u is smaller than slot context %u: at least %u prompt passes; raise -b", batch, context, passes));
    }
    if (ubatch < 4096) {
        warn(string_format("prompt ubatch %u is below 4096; raise -ub", ubatch));
    }
    uint64_t total = 0, available = 0;
    if (!common_host_memory(total, available)) {
        warn("cannot check available host RAM");
    } else {
#if defined(__linux__)
        std::ifstream groups("/proc/self/cgroup");
        std::string group;
        while (std::getline(groups, group)) {
            if (group.rfind("0::", 0) != 0) { continue; }
            const auto dir = std::filesystem::path("/sys/fs/cgroup") / group.substr(3).substr(1);
            uint64_t limit = 0, current = 0;
            if (std::ifstream(dir / "memory.max") >> limit && std::ifstream(dir / "memory.current") >> current) {
                uint64_t file_bytes = 0, value = 0;
                std::string key;
                std::ifstream stats(dir / "memory.stat");
                while (stats >> key >> value) { if (key == "file") { file_bytes = value; } }
                const uint64_t used = current - std::min(current, file_bytes);
                available = std::min(available, limit > used ? limit - used : 0);
            }
        }
#endif
        const size_t draft_experts = draft && &draft->get_model() != &model ? server_setup_host_experts(draft->get_model()) : 0;
        if (host_experts + draft_experts > available) {
            warn(string_format("host experts need %.2f GiB, available RAM/job headroom is %.2f GiB; disk paging is likely", (host_experts + draft_experts) / 1073741824.0, available / 1073741824.0));
        }
    }
    std::set<std::string> files = {params.model.path};
    if (!params.speculative.draft.mparams.path.empty()) { files.insert(params.speculative.draft.mparams.path); }
#if defined(__linux__)
    for (const auto & source : model.b1_src) {
        std::error_code ec;
        const auto path = std::filesystem::read_symlink("/proc/self/fd/" + std::to_string(source.second.fd), ec);
        if (!ec) { files.insert(path.string()); }
    }
#endif
    for (const auto & file : files) {
        std::error_code ec;
        const auto space = std::filesystem::space(file, ec);
        if (ec || space.capacity == 0) {
            warn("cannot check model disk space: " + file);
        } else if (space.available < space.capacity / 100 || space.available < 15ull*1024*1024*1024) {
            warn("model disk has less than 1% or 15 GiB free: " + file);
        }
#if defined(__linux__)
        struct stat st;
        if (stat(file.c_str(), &st) != 0) { warn("cannot check model disk type: " + file); continue; }
        auto device = std::filesystem::canonical("/sys/dev/block/" + std::to_string(major(st.st_dev)) + ":" + std::to_string(minor(st.st_dev)), ec);
        while (!ec && !device.empty() && device != device.root_path()) {
            int rotational = -1;
            if (std::ifstream(device / "queue/rotational") >> rotational) {
                if (rotational != 0) { warn("model is on a slow rotational disk: " + file); }
                break;
            }
            device = device.parent_path();
        }
        if (ec) { warn("model disk speed is unknown (not a local block device): " + file); }
#endif
    }
    const int b1 = ctx->b1_status();
    SRV_INF("[FNGUARD] effective setup: guarded=%d cache=%s cache_MiB=%.2f cache_status=%d B1=%s passes_min=%u passes_basis=slot-context B1_groups=runtime threads=%d/%d draft_threads=%d/%d physical_cores=%d batch=%u ubatch=%u context=%u draft=%s KV=%s draft_KV=%s warnings=%d\n",
            guarded, cache_status < 0 ? "unknown" : cache_on ? "on" : "off", cache_bytes / 1048576.0, cache_status,
            b1 < 0 ? "pending" : b1 ? "on" : "off", passes, llama_n_threads(ctx), llama_n_threads_batch(ctx), draft ? llama_n_threads(draft) : 0, draft ? llama_n_threads_batch(draft) : 0, cores,
            batch, ubatch, context, draft_device, server_setup_kv(ctx).c_str(), server_setup_kv(draft).c_str(), warnings);
}
