#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "../ggml/src/ggml-impl.h"
#include "../ggml/src/ggml-backend-moe-cache.h"
#include "../ggml/src/ggml-moe-prefetch.h"

#ifdef GGML_CUDA
#include <cuda_runtime.h>
#endif
#include "../ggml/src/ggml-cuda/moe-cache.cuh"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#if defined(__linux__)
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace {

constexpr int64_t n_in      = 256;
constexpr int64_t n_out     = 128;
constexpr int64_t n_expert  = 64;
constexpr int64_t n_used    = 2;
constexpr int64_t n_tokens  = 1;
constexpr int64_t multi_n_used   = 6;
constexpr int64_t multi_n_tokens = 6;
constexpr int     max_steps = 160;

struct log_capture {
    std::mutex mutex;
    std::condition_variable cv;
    std::string text;

    void clear() {
        std::lock_guard<std::mutex> lock(mutex);
        text.clear();
    }

    std::string get() {
        std::lock_guard<std::mutex> lock(mutex);
        return text;
    }

    bool wait_for(
            const char * pattern,
            const std::atomic<bool> & stop,
            std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex);
        return cv.wait_for(lock, timeout, [&] {
            return text.find(pattern) != std::string::npos || stop.load();
        }) && text.find(pattern) != std::string::npos;
    }
};

static void log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    if (level == GGML_LOG_LEVEL_ERROR) {
        std::fputs(text, stderr);
        std::fflush(stderr);
    }
    log_capture & capture = *static_cast<log_capture *>(user_data);
    {
        std::lock_guard<std::mutex> lock(capture.mutex);
        capture.text += text;
    }
    capture.cv.notify_all();
}

static void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

struct invalidation_record {
    const void * base;
    size_t size;
};

static std::vector<invalidation_record> * invalidation_records = nullptr;

static void record_invalidation(const void * base, size_t size) {
    if (invalidation_records) {
        invalidation_records->push_back({base, size});
    }
}

static void * test_provider_session_create(
        void * const * backends, int n_backends,
        const struct ggml_moe_cache_config * config) {
    (void) backends; (void) n_backends; (void) config;
    return nullptr;
}

static void test_provider_session_destroy(void * session) { (void) session; }
static void test_provider_session_enter(void * session) { (void) session; }
static void test_provider_session_leave(void * session) { (void) session; }

static bool run_resume_lifecycle_cpu() {
    auto cpu = ggml_backend_cpu_init();
    if (!cpu) { return false; }
    static int created, destroyed, resumed, configured, last_mode, entered, left;
    ggml_moe_cache_api api = {};
    api.owner = ggml_backend_dev_backend_reg(ggml_backend_get_device(cpu));
    const auto previous_api = ggml_moe_cache_get(api.owner);
    api.session_create = [](void * const *, int, const ggml_moe_cache_config *) -> void * {
        ++created;
        return new int(created);
    };
    api.session_destroy = [](void * p) { ++destroyed; delete (int *) p; };
    api.session_enter = [](void * p) { entered = p ? *(int *) p : 0; };
    api.session_leave = [](void * p) { left = p ? *(int *) p : 0; };
    api.prompt_resume = [](void *) { ++resumed; };
    api.prefetch_configure = [](void *, int mode) { ++configured; last_mode = mode; };
    bool ok = true;
    for (bool hook : {true, false}) {
        created = destroyed = resumed = configured = 0;
        last_mode = -99;
        if (!hook) { api.prompt_resume = nullptr; }
        ggml_moe_cache_register(&api);
        auto sched = ggml_backend_sched_new(&cpu, nullptr, 1, 32, false, true);
        ok &= created == 1 && resumed == 0;
        ggml_backend_sched_moe_cache_resume(sched);
        ok &= resumed == 0;
        ggml_backend_sched_moe_cache_prefetch_configure(sched, 3);
        ok &= configured == 1 && last_mode == 3;
        for (int cycle = 0; cycle < 2; ++cycle) {
            ok &= ggml_backend_sched_moe_cache_suspend(sched);
            ok &= !ggml_backend_sched_moe_cache_suspend(sched);
            ggml_backend_sched_moe_cache_resume(sched);
            ggml_backend_sched_moe_cache_resume(sched);
            ok &= created == cycle + 2 && destroyed == cycle + 1 && resumed == (hook ? cycle + 1 : 0);
            ok &= configured == (cycle == 0 ? 2 : 4) && last_mode == (cycle == 0 ? 3 : 0);
            if (cycle == 0) {
                // A later explicit disable must also survive the next resume.
                ggml_backend_sched_moe_cache_prefetch_configure(sched, 0);
            }
        }
        ggml_backend_sched_free(sched);
        ok &= destroyed == created;
    }
    created = destroyed = entered = left = 0;
    ggml_moe_cache_register(&api);
    auto donor = ggml_backend_sched_new(&cpu, nullptr, 1, 32, false, true);
    auto borrower = ggml_backend_sched_new(&cpu, nullptr, 1, 32, false, true);
    ggml_backend_sched_set_moe_cache(borrower, GGML_MOE_CACHE_MODE_OFF, 0);
    ok &= created == 2 && destroyed == 1;
    ok &= ggml_backend_sched_set_moe_cache_donor(borrower, donor);
    ok &= !ggml_backend_sched_set_moe_cache_donor(borrower, borrower);
    ok &= !ggml_backend_sched_set_moe_cache_donor(donor, borrower);
    ggml_init_params ip = {16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true};
    auto ctx = ggml_init(ip);
    auto input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    auto output = ggml_scale(ctx, input, 2.0f);
    ggml_set_input(input);
    ggml_set_output(output);
    auto graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    auto compute = [&](int expected) {
        entered = left = -1;
        ggml_backend_sched_reset(borrower);
        if (!ggml_backend_sched_alloc_graph(borrower, graph)) { return false; }
        const float values[] = {1.0f, -2.0f, 3.0f, -4.0f};
        ggml_backend_tensor_set(input, values, 0, sizeof(values));
        if (ggml_backend_sched_graph_compute(borrower, graph) != GGML_STATUS_SUCCESS) { return false; }
        float actual[4] = {};
        ggml_backend_tensor_get(output, actual, 0, sizeof(actual));
        bool same = entered == expected && left == expected;
        for (int i = 0; i < 4; ++i) { same &= actual[i] == 2.0f * values[i]; }
        return same;
    };
    ok &= compute(1);
    ok &= ggml_backend_sched_moe_cache_suspend(donor);
    ok &= compute(0);
    ggml_backend_sched_moe_cache_resume(donor);
    ok &= compute(3);
    ok &= ggml_backend_sched_set_donor(borrower, nullptr);
    ggml_backend_sched_free(donor);
    ok &= compute(-1);
    ggml_backend_sched_free(borrower);
    ggml_free(ctx);
    ok &= destroyed == created;
    ggml_moe_cache_unregister(api.owner);
    if (previous_api.owner) { ggml_moe_cache_register(&previous_api); }
    ggml_backend_free(cpu);
    printf("cache-resume-lifecycle-cpu: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

static void * resident_test_host = nullptr;
static std::vector<float> resident_test_original;
static int resident_test_restores = 0;

static int resident_test_restore(const void * data, size_t size) {
    const uintptr_t begin = (uintptr_t)data;
    const uintptr_t base = (uintptr_t)resident_test_host;
    const size_t bytes = resident_test_original.size() * sizeof(float);
    if (size && size <= UINTPTR_MAX - begin && begin < base + bytes && base < begin + size) {
        memcpy(resident_test_host, resident_test_original.data(), bytes);
        ++resident_test_restores;
    }
    return 1;
}

static bool run_resident_scheduler_cpu(ggml_backend_t cpu, bool events = false, bool rotating = false) {
    struct event_state { int records = 0; };
    struct copy_slot { const void * data; ggml_backend_event_t event; };
    struct upload_state {
        ggml_backend_t cpu;
        ggml_backend_buffer_type_t buft;
        bool accept;
        int copies = 0;
        int waits = 0;
        bool ordered = true;
        int event_waits = 0;
        int event_reuses = 0;
        int event_syncs = 0;
        int event_records = 0;
        int event_created = 0;
        int event_freed = 0;
        bool rotating = false;
        ggml_backend_event_t waited_event = nullptr;
        std::vector<copy_slot> copy_slots{};
    } state{cpu, nullptr, true};
    state.rotating = rotating;
    // Separate buffer types force real scheduler copies without a GPU.
    auto buft = *ggml_backend_cpu_buffer_type();
    auto device = *ggml_backend_get_device(cpu);
    auto backend = *cpu;
    state.buft = &buft;
    buft.device = &device;
    buft.iface.is_host = +[](ggml_backend_buffer_type_t) { return false; };
    device.context = &state;
    device.iface.get_type = +[](ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_GPU; };
    device.iface.get_buffer_type = +[](ggml_backend_dev_t dev) { return ((upload_state *)dev->context)->buft; };
    device.iface.supports_buft = +[](ggml_backend_dev_t dev, ggml_backend_buffer_type_t type) {
        return type == ((upload_state *)dev->context)->buft;
    };
    backend.device = &device;
    backend.context = &state;
    backend.iface.graph_compute = +[](ggml_backend_t stream, ggml_cgraph * graph) {
        auto * original = ((upload_state *)stream->context)->cpu;
        return original->iface.graph_compute(original, graph);
    };
    backend.iface.synchronize = +[](ggml_backend_t stream) { ++((upload_state *)stream->context)->waits; };
    backend.iface.cpy_tensor_async = nullptr;
    if (events) {
        device.iface.event_new = +[](ggml_backend_dev_t dev) {
            ++((upload_state *)dev->context)->event_created;
            return new ggml_backend_event{dev, new event_state};
        };
        device.iface.event_free = +[](ggml_backend_dev_t dev, ggml_backend_event_t event) {
            ++((upload_state *)dev->context)->event_freed;
            delete (event_state *)event->context;
            delete event;
        };
        device.iface.event_synchronize = +[](ggml_backend_dev_t dev, ggml_backend_event_t) {
            ++((upload_state *)dev->context)->event_syncs;
        };
        backend.iface.event_wait = +[](ggml_backend_t stream, ggml_backend_event_t event) {
            auto & test = *(upload_state *)stream->context;
            ++test.waits;
            ++test.event_waits;
            test.waited_event = event;
            if (((event_state *)event->context)->records > 0) { ++test.event_reuses; }
        };
        backend.iface.event_record = +[](ggml_backend_t stream, ggml_backend_event_t event) {
            ++((upload_state *)stream->context)->event_records;
            ++((event_state *)event->context)->records;
        };
    }
    static int owner;
    ggml_moe_cache_api api{};
    api.owner = &owner;
    api.session_create = test_provider_session_create;
    api.session_destroy = test_provider_session_destroy;
    api.session_enter = test_provider_session_enter;
    api.session_leave = test_provider_session_leave;
    api.host_restore = resident_test_restore;
    api.resident_copy = +[](void * opaque, ggml_tensor * dst, size_t offset, const void * data, size_t size) {
        auto & test = *(upload_state *)((ggml_backend_t)opaque)->context;
        if (!test.accept || data != resident_test_host || size != resident_test_original.size() * sizeof(float)) { return 0; }
        test.ordered &= test.waits > 0;
        if (test.rotating) {
            test.ordered &= test.waited_event != nullptr;
            bool found = false;
            for (const auto & slot : test.copy_slots) {
                if (slot.data == dst->data) {
                    test.ordered &= slot.event == test.waited_event;
                    found = true;
                } else {
                    test.ordered &= slot.event != test.waited_event;
                }
            }
            if (!found) { test.copy_slots.push_back({dst->data, test.waited_event}); }
        }
        ++test.copies;
        memcpy((char *)dst->data + offset, resident_test_original.data(), size);
        return 1;
    };
    ggml_moe_cache_register(&api);
    ggml_moe_cache_resident_notify(true);
    const char * old_upload = getenv("GGML_MOE_RESIDENT_UPLOAD");
    const std::string saved_upload = old_upload ? old_upload : "";
    bool ok = true;
    for (int test = 0; test < 6; ++test) {
        bool case_ok = true;
        set_env("GGML_MOE_RESIDENT_UPLOAD", test == 0 ? "0" : test == 2 ? "invalid" : "1");
        state.accept = test != 3;
        ggml_backend_t backends[] = {&backend, cpu};
        auto sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, events, false);
        ggml_backend_sched_set_moe_cache(sched, GGML_MOE_CACHE_MODE_OFF, 0);
        const int copies = ggml_backend_sched_get_n_copies(sched);
        const int iterations = rotating ? 2 * copies + 1 : 2;
        auto * ctx = ggml_init({(16 + iterations) * ggml_tensor_overhead() + iterations * ggml_graph_overhead(), nullptr, true});
        if (!ctx) { ggml_backend_sched_free(sched); ok = false; break; }
        auto * weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16);
        auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
        if (!buffer) { ggml_free(ctx); ggml_backend_sched_free(sched); ok = false; break; }
        if (test != 4) { ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS); }
        if (test == 5) { ggml_set_input(weight); }
        auto * duplicate = ggml_dup(ctx, weight);
        auto * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, duplicate);
        ggml_backend_sched_set_tensor_backend(sched, duplicate, &backend);
        const bool allocated = ggml_backend_sched_alloc_graph(sched, graph);
        case_ok &= allocated;
        case_ok &= ggml_backend_sched_get_tensor_backend(sched, duplicate) == &backend;
        resident_test_host = weight->data;
        resident_test_original.assign(16, 1.25f);
        resident_test_restores = 0;
        state.copies = 0;
        state.ordered = true;
        state.copy_slots.clear();
        state.event_waits = state.event_reuses = state.event_syncs = state.event_records = 0;
        const bool direct = test == 1;
        for (int replay = 0; allocated && replay < iterations; ++replay) {
            if (rotating && replay) {
                ggml_backend_sched_reset(sched);
                duplicate = ggml_dup(ctx, weight);
                graph = ggml_new_graph(ctx);
                ggml_build_forward_expand(graph, duplicate);
                ggml_backend_sched_set_tensor_backend(sched, duplicate, &backend);
                if (!ggml_backend_sched_alloc_graph(sched, graph)) { case_ok = false; break; }
            }
            const int before_restore = resident_test_restores;
            memset(weight->data, 0, ggml_nbytes(weight));
            state.waits = 0;
            state.waited_event = nullptr;
            case_ok &= (rotating ? ggml_backend_sched_graph_compute_async(sched, graph) :
                    ggml_backend_sched_graph_compute(sched, graph)) == GGML_STATUS_SUCCESS;
            std::vector<float> output(16);
            ggml_backend_tensor_get(duplicate, output.data(), 0, ggml_nbytes(duplicate));
            case_ok &= output == resident_test_original;
            case_ok &= direct ? resident_test_restores == before_restore : resident_test_restores > before_restore;
            case_ok &= state.copies == (direct ? replay + 1 : 0);
            if (direct) {
                case_ok &= std::all_of((float *)weight->data, (float *)weight->data + 16, [](float v) { return v == 0.0f; });
            }
        }
        if (rotating && direct) {
            case_ok &= copies > 1 && state.copy_slots.size() == (size_t)copies;
            printf("permanent-scheduler-rotating-cpu-slots: %zu/%d iterations=%d\n", state.copy_slots.size(), copies, iterations);
        }
        if (events) {
            case_ok &= state.event_records >= iterations;
            case_ok &= test == 5 ? state.event_syncs >= iterations :
                    state.event_waits >= iterations && state.event_reuses >= (rotating ? iterations - copies : 1);
        }
        ggml_backend_sched_free(sched);
        case_ok &= state.event_created == state.event_freed && state.ordered;
        printf("permanent-scheduler%s-cpu-case-%d: %s\n", rotating ? "-rotating" : events ? "-events" : "", test, case_ok ? "OK" : "FAIL");
        ok &= case_ok;
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
    ok &= state.ordered;
    set_env("GGML_MOE_RESIDENT_UPLOAD", old_upload ? saved_upload.c_str() : nullptr);
    ggml_moe_cache_resident_notify(false);
    ggml_moe_cache_unregister(&owner);
    resident_test_host = nullptr;
    resident_test_original.clear();
    printf("permanent-scheduler%s-cpu: %s\n", rotating ? "-rotating" : events ? "-events" : "", ok ? "OK" : "FAIL");
    return ok;
}

static bool run_resident_host_barriers(ggml_backend_t cpu) {
    ggml_init_params params{16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true};
    auto * ctx = ggml_init(params);
    if (!ctx) { return false; }
    auto * weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16);
    auto * destination = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16);
    auto * result = ggml_scale(ctx, weight, 2.0f);
    auto * metadata = ggml_view_1d(ctx, weight, 16, 0);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    if (!buffer) { ggml_free(ctx); return false; }
    static int owner;
    ggml_moe_cache_api api{};
    api.owner = &owner;
    api.session_create = test_provider_session_create;
    api.session_destroy = test_provider_session_destroy;
    api.session_enter = test_provider_session_enter;
    api.session_leave = test_provider_session_leave;
    api.host_restore = resident_test_restore;
    ggml_moe_cache_register(&api);
    ggml_moe_cache_resident_notify(true);
    resident_test_host = weight->data;
    resident_test_original.assign(16, 1.25f);
    resident_test_restores = 0;
    auto discard = [&] { memset(weight->data, 0, ggml_nbytes(weight)); };
    bool ok = true;
    std::vector<float> output(16);
    discard();
    ggml_backend_tensor_get(weight, output.data(), 0, ggml_nbytes(weight));
    ok &= output == resident_test_original;
    discard();
    ggml_backend_tensor_set_async(cpu, destination, weight->data, 0, ggml_nbytes(weight));
    ggml_backend_synchronize(cpu);
    ggml_backend_tensor_get(destination, output.data(), 0, ggml_nbytes(destination));
    ok &= output == resident_test_original;
    discard();
    ggml_backend_tensor_copy(weight, destination);
    ggml_backend_tensor_get(destination, output.data(), 0, ggml_nbytes(destination));
    ok &= output == resident_test_original;
    discard();
    ggml_backend_tensor_get_2d_async(cpu, weight, output.data(), 0, 4 * sizeof(float), 4,
            4 * sizeof(float), 4 * sizeof(float));
    ggml_backend_synchronize(cpu);
    ok &= output == resident_test_original;
    discard();
    // Metadata nodes and unused leafs must not restore released source pages.
    ggml_cgraph metadata_graph = *graph;
    metadata_graph.n_nodes = 1;
    metadata_graph.nodes = &metadata;
    const int before_metadata = resident_test_restores;
    ok &= ggml_backend_graph_compute(cpu, &metadata_graph) == GGML_STATUS_SUCCESS;
    ok &= resident_test_restores == before_metadata;
    ggml_backend_cpu_set_n_threads(cpu, 2);
    ok &= ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS;
    ggml_backend_tensor_get(result, output.data(), 0, ggml_nbytes(result));
    ok &= output == std::vector<float>(16, 2.5f);
    ok &= resident_test_restores >= 5;

    const char * old_upload = getenv("GGML_MOE_RESIDENT_UPLOAD");
    const std::string saved_upload = old_upload ? old_upload : "";
    auto upload_backend = *cpu;
    upload_backend.iface.set_tensor_async = +[](ggml_backend_t, ggml_tensor * dst, const void * data, size_t offset, size_t size) {
        memcpy((char *)dst->data + offset, data, size);
    };
    int uploads = 0;
    api.resident_copy = +[](void * opaque, ggml_tensor * dst, size_t offset, const void * data, size_t size) {
        if (data != resident_test_host || size != resident_test_original.size() * sizeof(float)) { return 0; }
        ++*(int *)((ggml_backend_t)opaque)->context;
        memcpy((char *)dst->data + offset, resident_test_original.data(), size);
        return 1;
    };
    upload_backend.context = &uploads;
    ggml_moe_cache_register(&api);
    for (const char * mode : {"0", "1", "invalid"}) {
        set_env("GGML_MOE_RESIDENT_UPLOAD", mode);
        discard();
        const int before = resident_test_restores;
        ggml_backend_tensor_set_async(&upload_backend, destination, weight->data, 0, ggml_nbytes(weight));
        ggml_backend_tensor_get(destination, output.data(), 0, ggml_nbytes(destination));
        ok &= output == resident_test_original;
        ok &= resident_test_restores == before + (strcmp(mode, "1") == 0 ? 0 : 1);
        if (strcmp(mode, "1") == 0) {
            ok &= std::all_of((float *)weight->data, (float *)weight->data + 16, [](float v) { return v == 0.0f; });
        }
    }
    ok &= uploads == 1;
    // Refused partial uploads must still restore the host source before reading.
    set_env("GGML_MOE_RESIDENT_UPLOAD", "1");
    discard();
    const int before_refusal = resident_test_restores;
    ggml_backend_tensor_set_async(&upload_backend, destination, (const char *)weight->data + sizeof(float), 0, 15 * sizeof(float));
    ok &= resident_test_restores == before_refusal + 1 && uploads == 1;
    ggml_backend_tensor_get(destination, output.data(), 0, ggml_nbytes(destination));
    ok &= output == resident_test_original;
    discard();
    const int before_copy = resident_test_restores;
    ggml_backend_tensor_copy_async(cpu, &upload_backend, weight, destination);
    ggml_backend_tensor_get(destination, output.data(), 0, ggml_nbytes(destination));
    ok &= output == resident_test_original && uploads == 2 && resident_test_restores == before_copy;
    set_env("GGML_MOE_RESIDENT_UPLOAD", old_upload ? saved_upload.c_str() : nullptr);
    ggml_moe_cache_resident_notify(false);
    ggml_moe_cache_unregister(&owner);
    resident_test_host = nullptr;
    resident_test_original.clear();
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    printf("permanent-host-read-barriers: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

#if defined(GGML_CUDA) && defined(__linux__)
static uint64_t resident_test_hash(uint64_t hash, const void * data, size_t size) {
    const auto * bytes = (const uint8_t *)data;
    for (size_t i = 0; i < size; ++i) { hash = (hash ^ bytes[i]) * UINT64_C(1099511628211); }
    return hash;
}

static bool run_permanent_original_copy(ggml_backend_t gpu, ggml_backend_t cpu) {
    auto api = ggml_moe_cache_get(ggml_backend_dev_backend_reg(ggml_backend_get_device(gpu)));
    if (!api.resident_create || !api.resident_dropped || !api.resident_spans || !api.resident_destroy) { return false; }
    ggml_init_params params{8 * ggml_tensor_overhead(), nullptr, true};
    auto * ctx = ggml_init(params);
    auto * a = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_0, n_in, n_out, n_expert);
    auto * b = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_0, n_in, n_out, n_expert);
    auto * host_buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    if (!host_buffer) { ggml_free(ctx); return false; }
    std::vector<uint8_t> original(ggml_nbytes(a));
    for (size_t i = 0; i < original.size(); ++i) { original[i] = (uint8_t)(i * 13 + i / 997); }
    memcpy(a->data, original.data(), original.size());
    memcpy(b->data, original.data(), original.size());
    ggml_moe_cache_tensor_desc descs[] = {
        {"blk.0.ffn_up_exps.weight", a->data, a->nb[2], n_in, n_out, n_expert, GGML_TYPE_Q4_0},
        {"blk.1.ffn_up_exps.weight", b->data, b->nb[2], n_in, n_out, n_expert, GGML_TYPE_Q4_0},
    };
    uint64_t signature = UINT64_C(14695981039346656037);
    for (const auto & d : descs) {
        signature = resident_test_hash(signature, d.name, strlen(d.name) + 1);
        const uint64_t shape[] = {d.expert_size, n_in, n_out, n_expert, GGML_TYPE_Q4_0};
        signature = resident_test_hash(signature, shape, sizeof(shape));
        for (size_t j = 0; j < 64; ++j) {
            signature = resident_test_hash(signature, (const char *)d.data + ((original.size() - 64) / 63) * j, 64);
        }
    }
    std::vector<uint64_t> counts(2 * n_expert, 1);
    const uint64_t checksum = resident_test_hash(UINT64_C(14695981039346656037), counts.data(), counts.size() * sizeof(uint64_t));
    const uint64_t header[] = {UINT64_C(0x31464f5250454f4d), 2, 2 * n_expert, signature, checksum};
    char path[] = "/tmp/llama-permanent-profile-XXXXXX";
    int fd = mkstemp(path);
    FILE * file = fd >= 0 ? fdopen(fd, "wb") : nullptr;
    bool ok = file && fwrite(header, sizeof(header), 1, file) == 1 &&
        fwrite(counts.data(), sizeof(uint64_t), counts.size(), file) == counts.size();
    if (file) { ok &= fclose(file) == 0; }
    const char * old_path = getenv("GGML_MOE_CACHE_PROFILE_LOAD");
    const std::string saved = old_path ? old_path : "";
    set_env("GGML_MOE_CACHE_PROFILE_LOAD", path);
    void * resident = ok ? api.resident_create(ggml_backend_get_device(gpu), descs, 2, 1u << 20) : nullptr;
    ok &= resident != nullptr;
    if (resident) {
        const size_t count = api.resident_spans(resident, nullptr, 0);
        std::vector<ggml_moe_resident_span> spans(count);
        api.resident_spans(resident, spans.data(), spans.size());
        auto * output = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, a->nb[2]);
        auto * device_buffer = ggml_backend_alloc_buffer(gpu, a->nb[2]);
        ggml_backend_tensor_alloc(device_buffer, output, ggml_backend_buffer_get_base(device_buffer));
        int restores = 0;
        api.resident_dropped(resident, +[](void * data) { ++*(int *)data; return 1; }, &restores);
        for (const auto & span : spans) { memset(const_cast<void *>(span.host), 0, span.size); }
        std::vector<uint8_t> got(a->nb[2]);
        for (const auto & span : spans) {
            ok &= ggml_moe_cache_resident_copy(gpu, output, 0, span.host, span.size);
            ggml_backend_synchronize(gpu);
            ggml_backend_tensor_get(output, got.data(), 0, span.size);
            const size_t offset = (uintptr_t)span.host - (uintptr_t)a->data;
            ok &= memcmp(got.data(), original.data() + offset, span.size) == 0;
        }
        ok &= restores == 0;
        ggml_moe_cache_host_restore(a->data, ggml_nbytes(a));
        ok &= restores == 1 && memcmp(a->data, original.data(), original.size()) == 0;
        api.resident_destroy(resident);
        ggml_backend_buffer_free(device_buffer);
    }
    set_env("GGML_MOE_CACHE_PROFILE_LOAD", old_path ? saved.c_str() : nullptr);
    unlink(path);
    ggml_backend_buffer_free(host_buffer);
    ggml_free(ctx);
    printf("permanent-original-D2D-and-host-restore: %s\n", ok ? "OK" : "FAIL");
    return ok;
}
#endif

// The tests exercise the selected device's provider directly, so a machine
// with several providers (CUDA + Vulkan, say) always tests a consistent
// pair of session creator and session callbacks. Populated in main().
static ggml_moe_cache_api g_test_api;

static ggml_moe_cache_api & active_api() {
    return g_test_api;
}

struct reset_buffer_context {
    uint8_t data[64];
    bool reset = false;
};

static void * reset_buffer_get_base(ggml_backend_buffer_t buffer) {
    return ((reset_buffer_context *)buffer->context)->data;
}

static void reset_buffer_reset(ggml_backend_buffer_t buffer) {
    ((reset_buffer_context *)buffer->context)->reset = true;
}

static bool run_invalidation_hook_coverage(ggml_backend_t cpu) {
    const ggml_init_params params = {
        4 * ggml_tensor_overhead(),
        nullptr,
        true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "cache-invalidation-hooks: failed to create context\n");
        return false;
    }

    ggml_tensor * src = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
    ggml_tensor * dst = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    if (!buffer) {
        fprintf(stderr, "cache-invalidation-hooks: failed to allocate tensors\n");
        ggml_free(ctx);
        return false;
    }
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // Register a test provider whose invalidate hook records buffer mutations.
    // The provider registry is the supported way to install a hook: the buffer
    // mutation path iterates the registered providers.
    static int dummy_owner = 0;
    ggml_moe_cache_api test_api = {};
    test_api.owner = &dummy_owner;
    test_api.session_create = test_provider_session_create;
    test_api.session_destroy = test_provider_session_destroy;
    test_api.session_enter = test_provider_session_enter;
    test_api.session_leave = test_provider_session_leave;
    test_api.invalidate = record_invalidation;
    ggml_moe_cache_register(&test_api);

    std::vector<float> data(64, 0.25f);
    std::vector<invalidation_record> records;
    invalidation_records = &records;

    bool ok = true;
    auto check = [&](const char * operation,
                     std::initializer_list<invalidation_record> expected) {
        bool matches = records.size() == expected.size();
        size_t index = 0;
        for (const invalidation_record & item : expected) {
            if (index >= records.size() || records[index].base != item.base ||
                records[index].size != item.size) {
                matches = false;
            }
            index++;
        }
        if (!matches) {
            fprintf(stderr,
                    "cache-invalidation-hooks: %s produced %zu records, expected %zu\n",
                    operation, records.size(), expected.size());
        }
        records.clear();
        ok &= matches;
    };

    char * dst_base = (char *)dst->data;
    void * buffer_base = ggml_backend_buffer_get_base(buffer);
    const size_t buffer_size = ggml_backend_buffer_get_size(buffer);

    ggml_backend_tensor_set(dst, data.data(), 4, 16);
    check("tensor set", {{dst_base + 4, 16}});

    ggml_backend_tensor_set_async(cpu, dst, data.data(), 8, 20);
    check("async tensor set", {{dst_base + 8, 20}});

    ggml_backend_tensor_set_2d(
            dst, data.data(), 0, 8, 3, 16, 8);
    check("2D tensor set", {
            {dst_base, 8}, {dst_base + 16, 8}, {dst_base + 32, 8}});

    ggml_backend_tensor_set_2d_async(
            cpu, dst, data.data(), 4, 8, 3, 16, 8);
    check("async 2D tensor set", {
            {dst_base + 4, 8}, {dst_base + 20, 8}, {dst_base + 36, 8}});

    ggml_backend_tensor_memset(dst, 0, 12, 24);
    check("tensor memset", {{dst_base + 12, 24}});

    if (!ggml_backend_buffer_copy_tensor(src, dst)) {
        fprintf(stderr, "cache-invalidation-hooks: direct tensor copy failed\n");
        ok = false;
    }
    check("direct tensor copy", {{dst_base, ggml_nbytes(dst)}});

    ggml_backend_tensor_copy(src, dst);
    check("tensor copy", {{dst_base, ggml_nbytes(dst)}});

    ggml_backend_tensor_copy_async(cpu, cpu, src, dst);
    check("async tensor copy", {{dst_base, ggml_nbytes(dst)}});

    ggml_backend_buffer_clear(buffer, 0);
    check("buffer clear", {{buffer_base, buffer_size}});

    ggml_backend_buffer_set_usage(
            buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    check("weight usage removal", {{buffer_base, buffer_size}});

    ggml_backend_tensor_set(dst, data.data(), 0, 16);
    check("non-weight tensor set", {});

    ggml_backend_buffer_set_usage(
            buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    check("weight usage restoration", {});

    ggml_backend_buffer_reset(buffer);
    check("buffer reset without callback", {});

    reset_buffer_context reset_context;
    ggml_backend_buffer_i reset_iface = {};
    reset_iface.get_base = reset_buffer_get_base;
    reset_iface.reset = reset_buffer_reset;
    ggml_backend_buffer_t reset_buffer = ggml_backend_buffer_init(
            ggml_backend_cpu_buffer_type(), reset_iface,
            &reset_context, sizeof(reset_context.data));
    ggml_backend_buffer_set_usage(
            reset_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_reset(reset_buffer);
    check("buffer reset", {
            {reset_context.data, sizeof(reset_context.data)}});
    if (!reset_context.reset) {
        fprintf(stderr, "cache-invalidation-hooks: reset callback was not called\n");
        ok = false;
    }
    ggml_backend_buffer_free(reset_buffer);
    check("reset buffer free", {
            {reset_context.data, sizeof(reset_context.data)}});

    ggml_backend_buffer_free(buffer);
    buffer = nullptr;
    check("buffer free", {{buffer_base, buffer_size}});

    ggml_moe_cache_unregister(&dummy_owner);
    invalidation_records = nullptr;
    ggml_free(ctx);
    printf("cache-invalidation-hooks: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

static bool has_positive_field(const std::string & text, const char * field) {
    size_t position = 0;
    while ((position = text.find(field, position)) != std::string::npos) {
        position += strlen(field);
        char * end = nullptr;
        const long long value = strtoll(text.c_str() + position, &end, 10);
        if (end != text.c_str() + position && value > 0) {
            return true;
        }
    }
    return false;
}

static long long max_field_value(const std::string & text, const char * field) {
    long long result = -1;
    size_t position = 0;
    while ((position = text.find(field, position)) != std::string::npos) {
        position += strlen(field);
        char * end = nullptr;
        const long long value = strtoll(text.c_str() + position, &end, 10);
        if (end != text.c_str() + position) {
            result = std::max(result, value);
        }
    }
    return result;
}

static size_t count_field_at_least(
        const std::string & text, const char * field, long long minimum) {
    size_t result = 0;
    size_t position = 0;
    while ((position = text.find(field, position)) != std::string::npos) {
        position += strlen(field);
        char * end = nullptr;
        const long long value = strtoll(text.c_str() + position, &end, 10);
        if (end != text.c_str() + position && value >= minimum) {
            result++;
        }
    }
    return result;
}

static size_t count_occurrences(const std::string & text, const char * pattern) {
    size_t result = 0;
    size_t position = 0;
    while ((position = text.find(pattern, position)) != std::string::npos) {
        result++;
        position += strlen(pattern);
    }
    return result;
}

// Any GPU/IGPU device whose backend registered a moe-cache provider. The
// test exercises the selected device's provider (see g_test_api below).
static ggml_backend_dev_t find_provider_device() {
    ggml_backend_load_all();
    for (size_t index = 0; index < ggml_backend_dev_count(); index++) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
        if ((ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_GPU ||
             ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_IGPU) &&
            ggml_moe_cache_get(reg).owner != nullptr) {
            return device;
        }
    }
    return nullptr;
}

static ggml_backend_dev_t find_other_provider_device(
        ggml_backend_dev_t excluded) {
    for (size_t index = 0; index < ggml_backend_dev_count(); index++) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
        if (device != excluded &&
            (ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_GPU ||
             ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_IGPU) &&
            ggml_moe_cache_get(reg).owner != nullptr) {
            return device;
        }
    }
    return nullptr;
}

static long provider_physical_device(ggml_backend_dev_t device) {
    const char * description = ggml_backend_dev_description(device);
    const char * marker = description ? strstr(description, "(physical device ") : nullptr;
    if (!marker) {
        return -1;
    }
    marker += strlen("(physical device ");
    char * end = nullptr;
    const long physical = strtol(marker, &end, 10);
    return end != marker ? physical : -1;
}

static ggml_backend_t init_cpu_backend() {
    for (size_t index = 0; index < ggml_backend_dev_count(); index++) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        if (ggml_backend_dev_type(device) !=
                GGML_BACKEND_DEVICE_TYPE_CPU) {
            continue;
        }
        ggml_backend_t backend =
            ggml_backend_dev_init(device, nullptr);
        if (!backend) {
            continue;
        }
        ggml_backend_reg_t reg =
            ggml_backend_dev_backend_reg(device);
        auto set_n_threads =
            (ggml_backend_set_n_threads_t)
                ggml_backend_reg_get_proc_address(
                    reg, "ggml_backend_set_n_threads");
        if (set_n_threads) {
            set_n_threads(backend, 4);
        }
        return backend;
    }
    return nullptr;
}

static bool compare_output(
        const std::vector<float> & reference,
        const std::vector<float> & actual,
        double max_nmse) {
    double squared_error = 0.0;
    double squared_reference = 0.0;
    for (size_t index = 0; index < reference.size(); index++) {
        if (!std::isfinite(actual[index])) {
            return false;
        }
        const double difference = (double) actual[index] - reference[index];
        squared_error += difference * difference;
        squared_reference += (double) reference[index] * reference[index];
    }
    return squared_error / std::max(squared_reference, 1e-12) <= max_nmse;
}

static void configure_cache(
        const char * fail_stage,
        const char * max_batch = "1",
        const char * dedicated_mmv = "1") {
    set_env("GGML_CUDA_MOE_CACHE", "1");
    set_env("GGML_CUDA_MOE_CACHE_MODE", "on");
    set_env("GGML_CUDA_MOE_CACHE_BUDGET_MB", "4");
    set_env("GGML_CUDA_MOE_CACHE_RESERVE_MB", "0");
    set_env("GGML_CUDA_MOE_CACHE_MIN_EXPERT_KB", "1");
    set_env("GGML_CUDA_MOE_CACHE_MAX_BATCH", max_batch);
    set_env("GGML_CUDA_MOE_CACHE_INSERTS", "4");
    set_env("GGML_CUDA_MOE_CACHE_ADMIT_AFTER", "1");
    set_env("GGML_CUDA_MOE_CACHE_THROTTLE", "1");
    set_env("GGML_CUDA_MOE_CACHE_QUEUE", "16");
    set_env("GGML_CUDA_MOE_CACHE_STATS", "1");
    set_env("GGML_CUDA_MOE_CACHE_NDEV", "1");
    set_env("GGML_CUDA_MOE_CACHE_MIN_CC", "0");
    set_env("GGML_CUDA_MOE_CACHE_SERIAL_FILL", nullptr);
    set_env("GGML_CUDA_MOE_CACHE_DEDICATED_MMV", dedicated_mmv);
    set_env("GGML_CUDA_MOE_CACHE_OVERLAP_CPU_ROWS", "0");
    set_env("GGML_CUDA_MOE_CACHE_FAIL", fail_stage);
}

static bool run_capability_queries(
        ggml_backend_dev_t provider_device, ggml_backend_t cpu) {
    const ggml_moe_cache_api api =
        ggml_moe_cache_get(ggml_backend_dev_backend_reg(provider_device));
    if (!api.query_config || !api.query_device || !api.query_shape) {
        fprintf(stderr, "cache-capabilities: query API is incomplete\n");
        return false;
    }

    configure_cache(nullptr);
    ggml_moe_cache_config config = {};
    bool ok = api.query_config(1, 0, &config) == 1;
    ok &= config.min_devices == 1; // automatic mode now accepts a single device
    ok &= config.budget_bytes == 4u * 1024 * 1024;
    ok &= config.reserve_bytes == 0;
    ok &= config.minimum_slab_bytes == 1024u * 1024 * 1024;
    ok &= config.min_expert_bytes == 1024;
    ok &= config.min_expert_explicit == 1;
    ok &= config.overlap_cpu_rows == 0;

    ggml_moe_cache_device_caps device = {};
    ok &= api.query_device(provider_device, &config, &device) == 1;
    ok &= device.logical_device >= 0;
    ok &= device.physical_device >= 0;
    ok &= device.compute_capability >= config.min_compute_capability;
    ok &= device.min_expert_bytes == 1024;
    ok &= api.query_device(cpu->device, &config, &device) == 0;

    const size_t expert_size = ggml_row_size(GGML_TYPE_Q4_0, n_in) * n_out;
    ggml_moe_cache_shape_caps shape = {};
    ok &= api.query_shape(
            GGML_TYPE_Q4_0, n_in, n_out, 64, expert_size, &shape) == 1;
    ok &= shape.pool_bytes == expert_size * 64;
    ok &= shape.minimum_bytes == shape.scratch_bytes + shape.pool_bytes;
    ok &= api.query_shape(
            GGML_TYPE_F32, n_in, n_out, 64, expert_size, &shape) == 0;
    ok &= api.query_shape(
            GGML_TYPE_Q4_0, n_in, n_out, 64, expert_size - 1, &shape) == 0;
    ok &= api.query_shape(
            GGML_TYPE_Q4_0, n_in, n_out, 1, expert_size, &shape) == 1;
    ok &= shape.pool_bytes == expert_size * 64;
    ok &= api.query_shape(
            GGML_TYPE_Q4_0, n_in, n_out, 0, expert_size, &shape) == 0;

    set_env("GGML_CUDA_MOE_CACHE", "0");
    set_env("GGML_CUDA_MOE_CACHE_MODE", "off");
    ok &= api.query_config(-1, 0, &config) == 0;

    set_env("GGML_CUDA_MOE_CACHE", "1");
    set_env("GGML_CUDA_MOE_CACHE_MODE", "auto");
    set_env("GGML_CUDA_MOE_CACHE_BUDGET_MB", nullptr);
    set_env("GGML_CUDA_MOE_CACHE_MIN_EXPERT_KB", nullptr);
    set_env("GGML_CUDA_MOE_CACHE_MAX_BATCH", nullptr);
    set_env("GGML_CUDA_MOE_CACHE_MIN_CC", nullptr);
    set_env("GGML_CUDA_MOE_CACHE_OVERLAP_CPU_ROWS", nullptr);
    ok &= api.query_config(1, 0, &config) == 1;
    ok &= config.min_devices == 1; // automatic mode now accepts a single device
    ok &= config.minimum_slab_bytes == 1024u * 1024 * 1024;
    ok &= config.min_compute_capability == 800;
    ok &= config.min_expert_bytes == 512u * 1024;
    ok &= config.min_expert_explicit == 0;
    ok &= config.max_batch == 8;
    ok &= config.overlap_cpu_rows == -1;
    ok &= api.query_device(provider_device, &config, &device) == 1;
    ok &= device.min_expert_bytes == 512u * 1024;

    ok &= api.query_config(0, 0, &config) == 1;
    ok &= config.min_devices == 1;
    ok &= config.minimum_slab_bytes == 0;
    ok &= config.min_compute_capability == 700;
    ok &= config.min_expert_bytes == 1024u * 1024;
    ok &= config.min_expert_explicit == 0;
    ok &= config.max_batch == 8;
    ok &= config.overlap_cpu_rows == -1;
    ok &= api.query_device(provider_device, &config, &device) == 1;
    ok &= device.min_expert_bytes == (device.compute_capability >= 800
            ? 512u * 1024 : 1024u * 1024);
    configure_cache(nullptr);

    printf("cache-capabilities: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

struct test_graph {
    ggml_context * ctx = nullptr;
    ggml_tensor * out = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
};

static test_graph make_graph(
        ggml_backend_t cpu,
        ggml_tensor * weights,
        ggml_tensor * activations,
        ggml_tensor * ids) {
    ggml_init_params params = {
        8 * ggml_tensor_overhead() + ggml_graph_overhead(),
        nullptr,
        true,
    };
    test_graph result;
    result.ctx = ggml_init(params);
    if (!result.ctx) {
        return result;
    }
    result.out = ggml_mul_mat_id(result.ctx, weights, activations, ids);
    ggml_set_name(result.out, "moe_cache_test_out");
    result.graph = ggml_new_graph(result.ctx);
    ggml_build_forward_expand(result.graph, result.out);
    result.buffer = ggml_backend_alloc_ctx_tensors(result.ctx, cpu);
    return result;
}

static test_graph make_fused_graph(
        ggml_backend_t cpu,
        ggml_tensor * up_weights,
        ggml_tensor * gate_weights,
        ggml_tensor * activations,
        ggml_tensor * ids,
        ggml_tensor * down_weights = nullptr) {
    ggml_init_params params = {
        12 * ggml_tensor_overhead() + ggml_graph_overhead(),
        nullptr,
        true,
    };
    test_graph result;
    result.ctx = ggml_init(params);
    if (!result.ctx) {
        return result;
    }
    ggml_tensor * up = ggml_mul_mat_id(
            result.ctx, up_weights, activations, ids);
    ggml_tensor * gate = ggml_mul_mat_id(
            result.ctx, gate_weights, activations, ids);
    result.out = ggml_swiglu_split(result.ctx, gate, up);
    ggml_set_name(up, "moe_cache_fused_up");
    ggml_set_name(gate, "moe_cache_fused_gate");
    ggml_set_name(result.out, "moe_cache_fused_out");
    if (down_weights) {
        result.out = ggml_mul_mat_id(result.ctx, down_weights, result.out, ids);
    }
    result.graph = ggml_new_graph(result.ctx);
    ggml_build_forward_expand(result.graph, result.out);
    result.buffer = ggml_backend_alloc_ctx_tensors(result.ctx, cpu);
    return result;
}

static test_graph make_clamped_fused_graph(
        ggml_backend_t cpu,
        ggml_tensor * up_weights,
        ggml_tensor * gate_weights,
        ggml_tensor * activations,
        ggml_tensor * ids) {
    ggml_init_params params = {
        16 * ggml_tensor_overhead() + ggml_graph_overhead(),
        nullptr,
        true,
    };
    test_graph result;
    result.ctx = ggml_init(params);
    if (!result.ctx) {
        return result;
    }
    ggml_tensor * up = ggml_mul_mat_id(
            result.ctx, up_weights, activations, ids);
    ggml_tensor * gate = ggml_mul_mat_id(
            result.ctx, gate_weights, activations, ids);
    ggml_tensor * up_clamped =
        ggml_clamp(result.ctx, up, -0.25f, 0.25f);
    ggml_tensor * gate_clamped = ggml_clamp(
            result.ctx, gate,
            -std::numeric_limits<float>::infinity(), 0.20f);
    result.out = ggml_swiglu_split(
            result.ctx, gate_clamped, up_clamped);
    ggml_set_name(up, "moe_cache_clamped_up");
    ggml_set_name(gate, "moe_cache_clamped_gate");
    ggml_set_name(up_clamped, "moe_cache_clamped_up_limit");
    ggml_set_name(gate_clamped, "moe_cache_clamped_gate_limit");
    ggml_set_name(result.out, "moe_cache_clamped_out");
    result.graph = ggml_new_graph(result.ctx);
    ggml_build_forward_expand(result.graph, result.out);
    result.buffer = ggml_backend_alloc_ctx_tensors(result.ctx, cpu);
    return result;
}

static void free_graph(test_graph & graph) {
    if (graph.buffer) {
        ggml_backend_buffer_free(graph.buffer);
    }
    if (graph.ctx) {
        ggml_free(graph.ctx);
    }
    graph = {};
}

static bool run_scenario(
        const char * name,
        const char * fail_stage,
        ggml_backend_t gpu,
        ggml_backend_t cpu,
        test_graph & graph,
        const std::vector<float> & reference,
        log_capture & capture,
        const char * max_batch = "1",
        const char * required_field = nullptr,
        const char * dedicated_mmv = "1") {
    configure_cache(fail_stage, max_batch, dedicated_mmv);
    capture.clear();

    ggml_backend_t backends[] = { gpu, cpu };
    ggml_backend_sched_t scheduler = ggml_backend_sched_new(
            backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, false);
    if (!scheduler) {
        fprintf(stderr, "%s: failed to create scheduler\n", name);
        return false;
    }
    ggml_backend_sched_set_tensor_backend(scheduler, graph.out, cpu);
    if (!ggml_backend_sched_alloc_graph(scheduler, graph.graph) ||
        ggml_backend_sched_get_tensor_backend(scheduler, graph.out) != cpu) {
        fprintf(stderr, "%s: MUL_MAT_ID was not assigned to CPU\n", name);
        ggml_backend_sched_free(scheduler);
        return false;
    }

    bool output_ok = true;
    bool live_hit = false;
    std::vector<float> actual(reference.size());
    for (int step = 0; step < max_steps; step++) {
        const enum ggml_status status =
            ggml_backend_sched_graph_compute(scheduler, graph.graph);
        if (status != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "%s: graph compute failed at step %d: %s\n",
                    name, step, ggml_status_to_string(status));
            output_ok = false;
            break;
        }
        ggml_backend_tensor_get(
                graph.out, actual.data(), 0, actual.size() * sizeof(float));
        if (!compare_output(reference, actual, 5e-4)) {
            fprintf(stderr, "%s: output mismatch at step %d\n", name, step);
            output_ok = false;
            break;
        }
        if (!fail_stage && has_positive_field(capture.get(), "hits=")) {
            live_hit = true;
        }
        if (step >= 64) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    ggml_backend_sched_free(scheduler);
    const std::string log = capture.get();
    bool stage_ok = false;
    if (!fail_stage) {
        stage_ok = live_hit &&
            log.find("[moe-cache] enabled:") != std::string::npos &&
            max_field_value(log, "dispatch-fail=") == 0 &&
            max_field_value(log, "collect-fail=") == 0 &&
            (!required_field || has_positive_field(log, required_field));
    } else if (strcmp(fail_stage, "dispatch") == 0) {
        stage_ok = has_positive_field(log, "dispatch-fail=");
    } else if (strcmp(fail_stage, "collect") == 0) {
        stage_ok = has_positive_field(log, "collect-fail=");
    } else if (strcmp(fail_stage, "insert") == 0) {
        stage_ok = has_positive_field(log, "fill-fail=");
    } else if (strcmp(fail_stage, "slab") == 0) {
        stage_ok = log.find("allocation failed") != std::string::npos;
    }
    if (required_field) {
        stage_ok &= has_positive_field(log, required_field);
    }
    if (!stage_ok) {
        fprintf(stderr, "%s: cache stage was not observed\n%s", name, log.c_str());
    }
    printf("%s: %s\n", name, output_ok && stage_ok ? "OK" : "FAIL");
    return output_ok && stage_ok;
}

static bool run_multi_token_scenario(
        ggml_backend_t gpu,
        ggml_backend_t cpu,
        ggml_tensor * weights,
        ggml_tensor * gate_weights,
        log_capture & capture,
        bool fused_enabled,
        bool dedup_enabled) {
    const ggml_init_params params = {
        4 * ggml_tensor_overhead(),
        nullptr,
        true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "cache-multi-token: failed to create tensor context\n");
        return false;
    }

    ggml_tensor * ids = ggml_new_tensor_2d(
            ctx, GGML_TYPE_I32, multi_n_used, multi_n_tokens);
    ggml_tensor * activations = ggml_new_tensor_3d(
            ctx, GGML_TYPE_F32, n_in, 1, multi_n_tokens);
    ggml_set_name(ids, "moe_cache_multi_token_ids");
    ggml_set_name(activations, "moe_cache_multi_token_activations");

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    if (!buffer) {
        fprintf(stderr, "cache-multi-token: failed to allocate CPU tensors\n");
        ggml_free(ctx);
        return false;
    }

    std::vector<int32_t> ids_data(multi_n_used * multi_n_tokens);
    for (int64_t token = 0; token < multi_n_tokens; token++) {
        for (int64_t id = 0; id < multi_n_used; id++) {
            ids_data[token * multi_n_used + id] =
                (int32_t) ((token * 7 + id * 3) % n_expert);
        }
    }
    ggml_backend_tensor_set(
            ids, ids_data.data(), 0,
            ids_data.size() * sizeof(ids_data[0]));

    std::vector<float> activation_data(ggml_nelements(activations));
    for (size_t index = 0; index < activation_data.size(); index++) {
        activation_data[index] =
            0.31f * std::sin((float) index * 0.053f) -
            0.12f * std::cos((float) index * 0.089f);
    }
    ggml_backend_tensor_set(
            activations, activation_data.data(), 0,
            activation_data.size() * sizeof(float));

    test_graph graph = make_graph(cpu, weights, activations, ids);
    test_graph fused_graph = make_fused_graph(
            cpu, weights, gate_weights, activations, ids);
    test_graph clamped_fused_graph = make_clamped_fused_graph(
            cpu, weights, gate_weights, activations, ids);
    bool ok = graph.ctx && graph.buffer;
    if (fused_enabled) {
        ok = ok && fused_graph.ctx && fused_graph.buffer &&
            clamped_fused_graph.ctx && clamped_fused_graph.buffer;
    }
    if (!ok) {
        fprintf(stderr, "cache-multi-token: failed to create graphs\n");
    } else {
        set_env("GGML_CUDA_MOE_CACHE", "0");
        auto make_reference = [&](const char * name, test_graph & test,
                                  std::vector<float> & reference) {
            if (ggml_backend_graph_compute(cpu, test.graph) !=
                    GGML_STATUS_SUCCESS) {
                fprintf(stderr, "%s: CPU reference compute failed\n", name);
                return false;
            }
            reference.resize(ggml_nelements(test.out));
            ggml_backend_tensor_get(
                    test.out, reference.data(), 0,
                    reference.size() * sizeof(float));
            return true;
        };
        std::vector<float> reference;
        std::vector<float> fused_reference;
        std::vector<float> clamped_fused_reference;
        ok = make_reference("cache-multi-token", graph, reference) &&
            make_reference(
                    "cache-fused-multi-token", fused_graph,
                    fused_reference) &&
            make_reference(
                    "cache-fused-clamped-multi-token",
                    clamped_fused_graph, clamped_fused_reference);
        ok = ok && run_scenario(
                "cache-multi-token", nullptr, gpu, cpu,
                graph, reference, capture, nullptr,
                dedup_enabled ? "act-dedup=" : nullptr);
        if (!fused_enabled) {
            printf("cache-fused-multi-token: SKIP (provider-specific)\n");
            printf("cache-fused-clamped-multi-token: SKIP (provider-specific)\n");
        } else {
            ok = ok && run_scenario(
                    "cache-fused-multi-token", nullptr, gpu, cpu,
                    fused_graph, fused_reference, capture,
                    nullptr, "fusion-nodes=");
            ok = ok && run_scenario(
                    "cache-fused-clamped-multi-token", nullptr, gpu, cpu,
                    clamped_fused_graph, clamped_fused_reference, capture,
                    nullptr, "fusion-nodes=");
        }
    }

    free_graph(clamped_fused_graph);
    free_graph(fused_graph);
    free_graph(graph);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

struct mxfp4_fixture {
    ggml_context * ctx = nullptr;
    ggml_tensor * weights = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    test_graph graph;
    std::vector<float> reference;
};

static void free_mxfp4_fixture(mxfp4_fixture & fixture) {
    free_graph(fixture.graph);
    if (fixture.buffer) {
        ggml_backend_buffer_free(fixture.buffer);
    }
    if (fixture.ctx) {
        ggml_free(fixture.ctx);
    }
    fixture = {};
}

static bool init_mxfp4_fixture(
        const char * name,
        const char * weight_name,
        int64_t mxfp4_n_in,
        int64_t mxfp4_n_out,
        ggml_backend_t cpu,
        mxfp4_fixture & fixture) {
    const ggml_init_params params = {
        8 * ggml_tensor_overhead(),
        nullptr,
        true,
    };
    fixture.ctx = ggml_init(params);
    if (!fixture.ctx) {
        fprintf(stderr, "%s: failed to create tensor context\n", name);
        return false;
    }

    fixture.weights = ggml_new_tensor_3d(
            fixture.ctx, GGML_TYPE_MXFP4,
            mxfp4_n_in, mxfp4_n_out, n_expert);
    ggml_tensor * ids = ggml_new_tensor_2d(
            fixture.ctx, GGML_TYPE_I32, n_used, n_tokens);
    ggml_tensor * activations = ggml_new_tensor_3d(
            fixture.ctx, GGML_TYPE_F32, mxfp4_n_in, 1, n_tokens);
    ggml_set_name(fixture.weights, weight_name);
    ggml_set_name(ids, "moe_cache_mxfp4_ids");
    ggml_set_name(activations, "moe_cache_mxfp4_activations");

    fixture.buffer = ggml_backend_alloc_ctx_tensors(fixture.ctx, cpu);
    if (!fixture.buffer) {
        fprintf(stderr, "%s: failed to allocate CPU tensors\n", name);
        free_mxfp4_fixture(fixture);
        return false;
    }
    ggml_backend_buffer_set_usage(
            fixture.buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    std::vector<float> weights_f32(ggml_nelements(fixture.weights));
    for (size_t index = 0; index < weights_f32.size(); index++) {
        weights_f32[index] =
            0.18f * std::sin((float) (index % 991) * 0.023f) +
            0.07f * std::cos((float) (index % 421) * 0.037f);
    }
    std::vector<uint8_t> weights_mxfp4(ggml_nbytes(fixture.weights));
    const size_t quantized = ggml_quantize_chunk(
            GGML_TYPE_MXFP4, weights_f32.data(), weights_mxfp4.data(),
            0, mxfp4_n_out * n_expert, mxfp4_n_in, nullptr);
    if (quantized != weights_mxfp4.size()) {
        fprintf(stderr, "%s: unexpected quantized size: %zu != %zu\n",
                name, quantized, weights_mxfp4.size());
        free_mxfp4_fixture(fixture);
        return false;
    }
    ggml_backend_tensor_set(
            fixture.weights, weights_mxfp4.data(), 0, weights_mxfp4.size());

    const int32_t ids_data[n_used] = { 0, 1 };
    ggml_backend_tensor_set(ids, ids_data, 0, sizeof(ids_data));
    std::vector<float> activation_data(ggml_nelements(activations));
    for (size_t index = 0; index < activation_data.size(); index++) {
        activation_data[index] =
            0.45f * std::sin((float) index * 0.061f) -
            0.16f * std::cos((float) index * 0.097f);
    }
    ggml_backend_tensor_set(
            activations, activation_data.data(), 0,
            activation_data.size() * sizeof(float));

    fixture.graph = make_graph(
            cpu, fixture.weights, activations, ids);
    if (!fixture.graph.ctx || !fixture.graph.buffer) {
        fprintf(stderr, "%s: failed to create graph\n", name);
        free_mxfp4_fixture(fixture);
        return false;
    }

    set_env("GGML_CUDA_MOE_CACHE", "0");
    if (ggml_backend_graph_compute(cpu, fixture.graph.graph) !=
            GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: CPU reference compute failed\n", name);
        free_mxfp4_fixture(fixture);
        return false;
    }
    fixture.reference.resize(ggml_nelements(fixture.graph.out));
    ggml_backend_tensor_get(
            fixture.graph.out, fixture.reference.data(), 0,
            fixture.reference.size() * sizeof(float));
    return true;
}

static bool run_mxfp4_shared_pool(
        ggml_backend_t gpu,
        ggml_backend_t cpu,
        log_capture & capture) {
    mxfp4_fixture up;
    mxfp4_fixture down;
    bool initialized = init_mxfp4_fixture(
            "cache-mxfp4-up", "blk.6.ffn_up_exps.weight",
            n_in, n_out, cpu, up);
    initialized = initialized && init_mxfp4_fixture(
            "cache-mxfp4-down", "blk.6.ffn_down_exps.weight",
            n_out, n_in, cpu, down);
    if (!initialized) {
        free_mxfp4_fixture(up);
        free_mxfp4_fixture(down);
        return false;
    }
    if (ggml_nbytes(up.weights) != ggml_nbytes(down.weights)) {
        fprintf(stderr, "cache-mxfp4-shared-pool: expert sizes differ\n");
        free_mxfp4_fixture(up);
        free_mxfp4_fixture(down);
        return false;
    }

    configure_cache(nullptr);
    capture.clear();
    ggml_backend_t backends[] = { gpu, cpu };
    ggml_backend_sched_t scheduler = ggml_backend_sched_new(
            backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, false);
    if (!scheduler) {
        fprintf(stderr, "cache-mxfp4-shared-pool: failed to create scheduler\n");
        free_mxfp4_fixture(up);
        free_mxfp4_fixture(down);
        return false;
    }

    auto run_orientation = [&](const char * name, mxfp4_fixture & fixture,
                               long long hits_before) {
        ggml_backend_sched_reset(scheduler);
        ggml_backend_sched_set_tensor_backend(
                scheduler, fixture.graph.out, cpu);
        if (!ggml_backend_sched_alloc_graph(
                    scheduler, fixture.graph.graph) ||
            ggml_backend_sched_get_tensor_backend(
                    scheduler, fixture.graph.out) != cpu) {
            fprintf(stderr, "%s: MUL_MAT_ID was not assigned to CPU\n", name);
            return false;
        }

        std::vector<float> actual(fixture.reference.size());
        for (int step = 0; step < max_steps; step++) {
            const enum ggml_status status =
                ggml_backend_sched_graph_compute(
                        scheduler, fixture.graph.graph);
            if (status != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "%s: graph compute failed at step %d: %s\n",
                        name, step, ggml_status_to_string(status));
                return false;
            }
            ggml_backend_tensor_get(
                    fixture.graph.out, actual.data(), 0,
                    actual.size() * sizeof(float));
            if (!compare_output(fixture.reference, actual, 5e-4)) {
                fprintf(stderr, "%s: output mismatch at step %d\n", name, step);
                return false;
            }
            const std::string live_log = capture.get();
            if (max_field_value(live_log, "hits=") > hits_before) {
                return max_field_value(live_log, "dispatch-fail=") == 0 &&
                       max_field_value(live_log, "collect-fail=") == 0;
            }
            if (step >= 64) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        fprintf(stderr, "%s: cache hit was not observed\n", name);
        return false;
    };

    const bool up_ok = run_orientation("cache-mxfp4-up", up, 0);
    const long long up_hits = max_field_value(capture.get(), "hits=");
    const bool down_ok = up_ok &&
        run_orientation("cache-mxfp4-down", down, up_hits);
    ggml_backend_sched_free(scheduler);

    const std::string log = capture.get();
    const bool one_pool = count_occurrences(log, " pool[") == 1;
    const bool no_failures =
        max_field_value(log, "dispatch-fail=") == 0 &&
        max_field_value(log, "collect-fail=") == 0;
    if (!one_pool) {
        fprintf(stderr,
                "cache-mxfp4-shared-pool: expected one shared pool\n%s",
                log.c_str());
    }
    const bool ok = up_ok && down_ok && one_pool && no_failures;
    printf("cache-mxfp4-shared-pool: %s\n", ok ? "OK" : "FAIL");
    free_mxfp4_fixture(up);
    free_mxfp4_fixture(down);
    return ok;
}

static bool run_invalidation_scenario(
        ggml_backend_t gpu,
        ggml_backend_t cpu,
        test_graph & graph,
        ggml_tensor * weights,
        const std::vector<uint8_t> & replacement_row,
        const std::vector<float> & old_reference,
        log_capture & capture) {
    configure_cache(nullptr);
    capture.clear();

    ggml_backend_t backends[] = { gpu, cpu };
    ggml_backend_sched_t scheduler = ggml_backend_sched_new(
            backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, false);
    if (!scheduler) {
        fprintf(stderr, "cache-invalidate: failed to create scheduler\n");
        return false;
    }
    ggml_backend_sched_set_tensor_backend(scheduler, graph.out, cpu);
    if (!ggml_backend_sched_alloc_graph(scheduler, graph.graph) ||
        ggml_backend_sched_get_tensor_backend(scheduler, graph.out) != cpu) {
        fprintf(stderr, "cache-invalidate: MUL_MAT_ID was not assigned to CPU\n");
        ggml_backend_sched_free(scheduler);
        return false;
    }

    std::vector<float> actual(old_reference.size());
    long long hits_before = -1;
    bool output_ok = true;
    for (int step = 0; step < max_steps; step++) {
        const enum ggml_status status =
            ggml_backend_sched_graph_compute(scheduler, graph.graph);
        if (status != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "cache-invalidate: warmup failed at step %d: %s\n",
                    step, ggml_status_to_string(status));
            output_ok = false;
            break;
        }
        ggml_backend_tensor_get(
                graph.out, actual.data(), 0, actual.size() * sizeof(float));
        if (!compare_output(old_reference, actual, 5e-4)) {
            fprintf(stderr, "cache-invalidate: warmup mismatch at step %d\n", step);
            output_ok = false;
            break;
        }
        hits_before = max_field_value(capture.get(), "hits=");
        if (hits_before > 0) {
            break;
        }
        if (step >= 64) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    if (output_ok && hits_before <= 0) {
        fprintf(stderr, "cache-invalidate: no cache hit before mutation\n");
        output_ok = false;
    }

    std::vector<float> new_reference(old_reference.size());
    if (output_ok) {
        ggml_backend_tensor_set(
                weights, replacement_row.data(), 0, replacement_row.size());
        if (ggml_backend_graph_compute(cpu, graph.graph) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "cache-invalidate: CPU reference compute failed\n");
            output_ok = false;
        } else {
            ggml_backend_tensor_get(
                    graph.out, new_reference.data(), 0,
                    new_reference.size() * sizeof(float));
            float max_change = 0.0f;
            for (size_t index = 0; index < new_reference.size(); index++) {
                max_change = std::max(
                        max_change, std::abs(new_reference[index] - old_reference[index]));
            }
            if (max_change < 0.01f) {
                fprintf(stderr, "cache-invalidate: mutation did not change the reference\n");
                output_ok = false;
            }
        }
    }

    capture.clear();
    bool repopulated = false;
    if (output_ok) {
        for (int step = 0; step < max_steps; step++) {
            const enum ggml_status status =
                ggml_backend_sched_graph_compute(scheduler, graph.graph);
            if (status != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "cache-invalidate: compute failed at step %d: %s\n",
                        step, ggml_status_to_string(status));
                output_ok = false;
                break;
            }
            ggml_backend_tensor_get(
                    graph.out, actual.data(), 0, actual.size() * sizeof(float));
            if (!compare_output(new_reference, actual, 5e-4)) {
                fprintf(stderr, "cache-invalidate: stale output at step %d\n", step);
                output_ok = false;
                break;
            }
            if (max_field_value(capture.get(), "hits=") > hits_before) {
                repopulated = true;
                break;
            }
            if (step >= 64) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }

    ggml_backend_sched_free(scheduler);
    if (output_ok && !repopulated) {
        fprintf(stderr, "cache-invalidate: mutated expert was not repopulated\n%s",
                capture.get().c_str());
        output_ok = false;
    }
    printf("cache-invalidate: %s\n", output_ok ? "OK" : "FAIL");
    return output_ok;
}

constexpr int64_t stress_n_out  = 65;
constexpr int64_t stress_n_used = 64;

struct stress_fixture {
    ggml_context * ctx = nullptr;
    ggml_tensor * weights = nullptr;
    ggml_tensor * ids = nullptr;
    ggml_tensor * activations = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    test_graph graph;
    std::vector<float> reference;
};

static void free_stress_fixture(stress_fixture & fixture) {
    free_graph(fixture.graph);
    if (fixture.buffer) {
        ggml_backend_buffer_free(fixture.buffer);
    }
    if (fixture.ctx) {
        ggml_free(fixture.ctx);
    }
    fixture = {};
}

static bool init_stress_fixture(stress_fixture & fixture, ggml_backend_t cpu) {
    const ggml_init_params params = {
        8 * ggml_tensor_overhead(),
        nullptr,
        true,
    };
    fixture.ctx = ggml_init(params);
    if (!fixture.ctx) {
        return false;
    }

    fixture.weights = ggml_new_tensor_3d(
            fixture.ctx, GGML_TYPE_Q4_0, n_in, stress_n_out, n_expert);
    fixture.ids = ggml_new_tensor_2d(
            fixture.ctx, GGML_TYPE_I32, stress_n_used, n_tokens);
    fixture.activations = ggml_new_tensor_3d(
            fixture.ctx, GGML_TYPE_F32, n_in, stress_n_used, n_tokens);
    ggml_set_name(fixture.weights, "blk.1.ffn_up_exps.weight");
    ggml_set_name(fixture.ids, "moe_cache_stress_ids");
    ggml_set_name(fixture.activations, "moe_cache_stress_activations");

    fixture.buffer = ggml_backend_alloc_ctx_tensors(fixture.ctx, cpu);
    if (!fixture.buffer) {
        free_stress_fixture(fixture);
        return false;
    }
    ggml_backend_buffer_set_usage(
            fixture.buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    std::vector<float> weights_f32(ggml_nelements(fixture.weights));
    for (size_t index = 0; index < weights_f32.size(); index++) {
        weights_f32[index] =
            0.13f * std::sin((float) (index % 983) * 0.019f) -
            0.04f * std::cos((float) (index % 419) * 0.029f);
    }
    std::vector<uint8_t> weights_q4(ggml_nbytes(fixture.weights));
    const size_t quantized = ggml_quantize_chunk(
            GGML_TYPE_Q4_0, weights_f32.data(), weights_q4.data(),
            0, stress_n_out * n_expert, n_in, nullptr);
    if (quantized != weights_q4.size()) {
        fprintf(stderr, "stress: unexpected quantized size\n");
        free_stress_fixture(fixture);
        return false;
    }
    ggml_backend_tensor_set(
            fixture.weights, weights_q4.data(), 0, weights_q4.size());

    std::vector<int32_t> ids_data(stress_n_used);
    for (int32_t index = 0; index < stress_n_used; index++) {
        ids_data[index] = index;
    }
    ggml_backend_tensor_set(
            fixture.ids, ids_data.data(), 0,
            ids_data.size() * sizeof(ids_data[0]));

    std::vector<float> activation_data(
            ggml_nelements(fixture.activations));
    for (size_t index = 0; index < activation_data.size(); index++) {
        activation_data[index] =
            0.35f * std::sin((float) index * 0.067f) +
            0.17f * std::cos((float) index * 0.103f);
    }
    ggml_backend_tensor_set(
            fixture.activations, activation_data.data(), 0,
            activation_data.size() * sizeof(float));

    fixture.graph = make_graph(
            cpu, fixture.weights, fixture.activations, fixture.ids);
    if (!fixture.graph.ctx || !fixture.graph.buffer) {
        fprintf(stderr, "stress: failed to create graph\n");
        free_stress_fixture(fixture);
        return false;
    }

    set_env("GGML_CUDA_MOE_CACHE", "0");
    if (ggml_backend_graph_compute(cpu, fixture.graph.graph) !=
            GGML_STATUS_SUCCESS) {
        fprintf(stderr, "stress: CPU reference compute failed\n");
        free_stress_fixture(fixture);
        return false;
    }
    fixture.reference.resize(ggml_nelements(fixture.graph.out));
    ggml_backend_tensor_get(
            fixture.graph.out, fixture.reference.data(), 0,
            fixture.reference.size() * sizeof(float));
    return true;
}

static ggml_backend_sched_t make_scheduler(
        const char * name,
        ggml_backend_t gpu,
        ggml_backend_t cpu,
        test_graph & graph) {
    ggml_backend_t backends[] = { gpu, cpu };
    ggml_backend_sched_t scheduler = ggml_backend_sched_new(
            backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, false);
    if (!scheduler) {
        fprintf(stderr, "%s: failed to create scheduler\n", name);
        return nullptr;
    }
    ggml_backend_sched_set_tensor_backend(scheduler, graph.out, cpu);
    if (!ggml_backend_sched_alloc_graph(scheduler, graph.graph) ||
        ggml_backend_sched_get_tensor_backend(scheduler, graph.out) != cpu) {
        fprintf(stderr, "%s: MUL_MAT_ID was not assigned to CPU\n", name);
        ggml_backend_sched_free(scheduler);
        return nullptr;
    }
    return scheduler;
}

static bool compute_matches(
        const char * name,
        ggml_backend_sched_t scheduler,
        test_graph & graph,
        const std::vector<float> & reference,
        int step) {
    const enum ggml_status status =
        ggml_backend_sched_graph_compute(scheduler, graph.graph);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: graph compute failed at step %d: %s\n",
                name, step, ggml_status_to_string(status));
        return false;
    }
    std::vector<float> actual(reference.size());
    ggml_backend_tensor_get(
            graph.out, actual.data(), 0, actual.size() * sizeof(float));
    if (!compare_output(reference, actual, 5e-4)) {
        fprintf(stderr, "%s: output mismatch at step %d\n", name, step);
        return false;
    }
    return true;
}

static bool run_precensus_invalidation(
        ggml_backend_t gpu,
        ggml_backend_t cpu,
        test_graph & graph,
        ggml_tensor * weights,
        const uint8_t * unchanged_expert,
        size_t expert_size,
        const std::vector<float> & reference,
        log_capture & capture) {
    configure_cache(nullptr);
    capture.clear();

    ggml_backend_sched_t scheduler = make_scheduler(
            "cache-precensus-invalidate", gpu, cpu, graph);
    if (!scheduler) {
        return false;
    }

    bool output_ok = compute_matches(
            "cache-precensus-invalidate", scheduler, graph, reference, 0);
    if (output_ok) {
        ggml_backend_tensor_set(
                weights, unchanged_expert,
                (n_expert - 1) * expert_size, expert_size);
    }
    for (int step = 1; step < 80 && output_ok; step++) {
        output_ok = compute_matches(
                "cache-precensus-invalidate", scheduler, graph,
                reference, step);
        if (step >= 64) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    ggml_backend_sched_free(scheduler);
    const std::string log = capture.get();
    const bool census_ok =
        log.find("slots=64 ") != std::string::npos;
    if (!census_ok) {
        fprintf(stderr,
                "cache-precensus-invalidate: tensor census was not stable\n%s",
                log.c_str());
    }
    printf("cache-precensus-invalidate: %s\n",
            output_ok && census_ok ? "OK" : "FAIL");
    return output_ok && census_ok;
}

static bool run_concurrent_sessions(
        ggml_backend_dev_t provider_device,
        ggml_backend_t gpu,
        ggml_backend_t cpu,
        stress_fixture & fixture,
        log_capture & capture) {
    configure_cache(nullptr);
    set_env("GGML_CUDA_MOE_CACHE_STATS", "0");
    capture.clear();

    ggml_backend_t gpu_second =
        ggml_backend_dev_init(provider_device, nullptr);
    ggml_backend_t cpu_second = init_cpu_backend();
    if (!gpu_second || !cpu_second) {
        fprintf(stderr, "cache-concurrent: failed to create second backends\n");
        if (gpu_second) {
            ggml_backend_free(gpu_second);
        }
        if (cpu_second) {
            ggml_backend_free(cpu_second);
        }
        return false;
    }
    test_graph second_graph = make_graph(
            cpu_second, fixture.weights, fixture.activations, fixture.ids);
    if (!second_graph.ctx || !second_graph.buffer) {
        fprintf(stderr, "cache-concurrent: failed to create second graph\n");
        free_graph(second_graph);
        ggml_backend_free(gpu_second);
        ggml_backend_free(cpu_second);
        return false;
    }

    ggml_backend_sched_t first = make_scheduler(
            "cache-concurrent-1", gpu, cpu, fixture.graph);
    ggml_backend_sched_t second = make_scheduler(
            "cache-concurrent-2", gpu_second, cpu_second, second_graph);
    if (!first || !second) {
        if (first) {
            ggml_backend_sched_free(first);
        }
        if (second) {
            ggml_backend_sched_free(second);
        }
        free_graph(second_graph);
        ggml_backend_free(gpu_second);
        ggml_backend_free(cpu_second);
        return false;
    }

    constexpr int concurrent_steps = 112;
    std::atomic<int> ready{0};
    std::atomic<bool> start{false};
    std::atomic<bool> output_ok{true};
    auto run = [&](const char * name, ggml_backend_sched_t scheduler,
                   test_graph & graph) {
        ready.fetch_add(1);
        while (!start.load()) {
            std::this_thread::yield();
        }
        for (int step = 0; step < concurrent_steps && output_ok.load(); step++) {
            if (!compute_matches(
                    name, scheduler, graph, fixture.reference, step)) {
                output_ok.store(false);
                break;
            }
            if (step >= 64) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    };

    std::thread first_thread(
            run, "cache-concurrent-1", first, std::ref(fixture.graph));
    std::thread second_thread(
            run, "cache-concurrent-2", second, std::ref(second_graph));
    while (ready.load() != 2) {
        std::this_thread::yield();
    }
    start.store(true);
    first_thread.join();
    second_thread.join();

    ggml_backend_sched_free(first);
    ggml_backend_sched_free(second);
    const std::string log = capture.get();
    const bool cache_ok =
        count_occurrences(log, " pool[") >= 2 &&
        count_field_at_least(log, "hits=", stress_n_used) >= 2 &&
        count_field_at_least(log, "used=", stress_n_used) >= 2;
    if (!cache_ok) {
        fprintf(stderr, "cache-concurrent: full cache use was not observed\n%s",
                log.c_str());
    }

    free_graph(second_graph);
    ggml_backend_free(gpu_second);
    ggml_backend_free(cpu_second);
    printf("cache-concurrent: %s\n",
            output_ok.load() && cache_ok ? "OK" : "FAIL");
    return output_ok.load() && cache_ok;
}

static bool run_fused_concurrent_sessions(
        ggml_backend_dev_t provider_device,
        ggml_backend_t gpu,
        ggml_backend_t cpu,
        ggml_tensor * up_weights,
        ggml_tensor * gate_weights,
        ggml_tensor * activations,
        ggml_tensor * ids,
        test_graph & first_graph,
        const std::vector<float> & reference,
        log_capture & capture) {
    configure_cache(nullptr);
    set_env("GGML_CUDA_MOE_CACHE_BUDGET_MB", "16");
    set_env("GGML_CUDA_MOE_CACHE_STATS", "0");
    capture.clear();

    ggml_backend_t gpu_second =
        ggml_backend_dev_init(provider_device, nullptr);
    ggml_backend_t cpu_second = init_cpu_backend();
    if (!gpu_second || !cpu_second) {
        fprintf(stderr, "cache-fused-concurrent: failed to create second backends\n");
        if (gpu_second) {
            ggml_backend_free(gpu_second);
        }
        if (cpu_second) {
            ggml_backend_free(cpu_second);
        }
        return false;
    }

    test_graph second_graph = make_fused_graph(
            cpu_second, up_weights, gate_weights, activations, ids);
    if (!second_graph.ctx || !second_graph.buffer) {
        fprintf(stderr, "cache-fused-concurrent: failed to create second graph\n");
        free_graph(second_graph);
        ggml_backend_free(gpu_second);
        ggml_backend_free(cpu_second);
        return false;
    }

    ggml_backend_sched_t first = make_scheduler(
            "cache-fused-concurrent-1", gpu, cpu, first_graph);
    ggml_backend_sched_t second = make_scheduler(
            "cache-fused-concurrent-2", gpu_second, cpu_second,
            second_graph);
    if (!first || !second) {
        if (first) {
            ggml_backend_sched_free(first);
        }
        if (second) {
            ggml_backend_sched_free(second);
        }
        free_graph(second_graph);
        ggml_backend_free(gpu_second);
        ggml_backend_free(cpu_second);
        return false;
    }

    std::atomic<int> ready{0};
    std::atomic<bool> start{false};
    std::atomic<bool> output_ok{true};
    auto run = [&](const char * name, ggml_backend_sched_t scheduler,
                   test_graph & graph) {
        ready.fetch_add(1);
        while (!start.load()) {
            std::this_thread::yield();
        }
        for (int step = 0; step < max_steps && output_ok.load(); step++) {
            if (!compute_matches(name, scheduler, graph, reference, step)) {
                output_ok.store(false);
                break;
            }
            if (step >= 64) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    };

    std::thread first_thread(
            run, "cache-fused-concurrent-1", first,
            std::ref(first_graph));
    std::thread second_thread(
            run, "cache-fused-concurrent-2", second,
            std::ref(second_graph));
    while (ready.load() != 2) {
        std::this_thread::yield();
    }
    start.store(true);
    first_thread.join();
    second_thread.join();

    ggml_backend_sched_free(first);
    ggml_backend_sched_free(second);
    const std::string log = capture.get();
    const bool cache_ok =
        count_field_at_least(log, "fusion-nodes=", 1) >= 2 &&
        max_field_value(log, "dispatch-fail=") == 0 &&
        max_field_value(log, "collect-fail=") == 0;
    if (!cache_ok) {
        fprintf(stderr,
                "cache-fused-concurrent: fusion was not observed in both sessions\n%s",
                log.c_str());
    }

    free_graph(second_graph);
    ggml_backend_free(gpu_second);
    ggml_backend_free(cpu_second);
    configure_cache(nullptr);
    printf("cache-fused-concurrent: %s\n",
            output_ok.load() && cache_ok ? "OK" : "FAIL");
    return output_ok.load() && cache_ok;
}

static bool run_fused_repeated_lifecycle(
        ggml_backend_t gpu,
        ggml_backend_t cpu,
        test_graph & graph,
        const std::vector<float> & reference,
        log_capture & capture) {
    configure_cache(nullptr);
    constexpr int cycles = 8;
    bool ok = true;
    for (int cycle = 0; cycle < cycles && ok; cycle++) {
        capture.clear();
        ggml_backend_sched_t scheduler = make_scheduler(
                "cache-fused-lifecycle", gpu, cpu, graph);
        if (!scheduler) {
            ok = false;
            break;
        }

        bool fused = false;
        for (int step = 0; step < max_steps && ok && !fused; step++) {
            ok = compute_matches(
                    "cache-fused-lifecycle", scheduler, graph,
                    reference, step);
            fused = has_positive_field(
                    capture.get(), "fusion-nodes=");
            if (step >= 64 && !fused) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        ok &= fused;
        ggml_backend_sched_free(scheduler);
    }

    printf("cache-fused-lifecycle: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

static bool run_repeated_lifecycle(
        ggml_backend_t gpu,
        ggml_backend_t cpu,
        stress_fixture & fixture,
        log_capture & capture) {
    configure_cache(nullptr);
    capture.clear();

    constexpr int cycles = 8;
    constexpr int census_steps = 65;
    bool output_ok = true;
    for (int cycle = 0; cycle < cycles && output_ok; cycle++) {
        ggml_backend_sched_t scheduler = make_scheduler(
                "cache-lifecycle", gpu, cpu, fixture.graph);
        if (!scheduler) {
            output_ok = false;
            break;
        }
        for (int step = 0; step < census_steps; step++) {
            const enum ggml_status status =
                ggml_backend_sched_graph_compute(scheduler, fixture.graph.graph);
            if (status != GGML_STATUS_SUCCESS) {
                fprintf(stderr,
                        "cache-lifecycle: compute failed in cycle %d step %d: %s\n",
                        cycle, step, ggml_status_to_string(status));
                output_ok = false;
                break;
            }
        }
        if (output_ok) {
            std::vector<float> actual(fixture.reference.size());
            ggml_backend_tensor_get(
                    fixture.graph.out, actual.data(), 0,
                    actual.size() * sizeof(float));
            output_ok = compare_output(
                    fixture.reference, actual, 5e-4);
            if (!output_ok) {
                fprintf(stderr,
                        "cache-lifecycle: output mismatch in cycle %d\n", cycle);
            }
        }
        ggml_backend_sched_free(scheduler);
    }

    const std::string log = capture.get();
    const bool cache_ok =
        count_occurrences(log, " pool[") >= cycles &&
        has_positive_field(log, "enqueued=");
    if (!cache_ok) {
        fprintf(stderr, "cache-lifecycle: fill startup was not observed\n%s",
                log.c_str());
    }
    printf("cache-lifecycle: %s\n",
            output_ok && cache_ok ? "OK" : "FAIL");
    return output_ok && cache_ok;
}

// B1's opt-in suspend (LLAMA_B1_MOE_CACHE_SUSPEND=1): the scheduler frees its session and pool, computes without
// a cache, and resume makes a new session that fills again; output stays correct throughout.
static bool run_suspend_resume(
        ggml_backend_t gpu,
        ggml_backend_t cpu,
        stress_fixture & fixture,
        log_capture & capture,
        bool return_reserve = false) {
    configure_cache(nullptr);
    if (return_reserve) {
        set_env("GGML_CUDA_MOE_CACHE_RESERVE_MB", "3072");
        set_env("GGML_CUDA_MOE_CACHE_RETURN_RESERVE_MB", "2560");
    }
    capture.clear();
    ggml_backend_sched_t scheduler = make_scheduler(
            "cache-suspend", gpu, cpu, fixture.graph);
    if (!scheduler) {
        printf("cache-suspend: FAIL\n");
        return false;
    }
    constexpr int steps = 65;
    bool ok = true;
    int  step = 0;
    for (int i = 0; i < steps && ok; i++) {
        ok = compute_matches("cache-suspend", scheduler, fixture.graph, fixture.reference, step++);
    }
    if (return_reserve) {
        (void) ggml_backend_sched_moe_cache_restore(scheduler);
        ok &= capture.get().find("prompt-return reserve bytes:") == std::string::npos;
    }
    const size_t pools_before = count_occurrences(capture.get(), " pool[");
    const bool s1 = ok && ggml_backend_sched_moe_cache_suspend(scheduler);
    const bool s2 = ggml_backend_sched_moe_cache_suspend(scheduler); // already suspended: no-op
    if (!s1 || s2) {
        fprintf(stderr, "cache-suspend: suspend returned %d then %d (want 1 then 0)\n", (int) s1, (int) s2);
        ok = false;
    }
    for (int i = 0; i < 4 && ok; i++) {
        ok = compute_matches("cache-suspend", scheduler, fixture.graph, fixture.reference, step++);
    }
    ggml_backend_sched_moe_cache_resume(scheduler);
    ggml_backend_sched_moe_cache_resume(scheduler); // not suspended: no-op
    if (return_reserve) {
        (void) ggml_backend_sched_moe_cache_restore(scheduler);
        const std::string log = capture.get();
        ok &= count_occurrences(log, "prompt-return reserve bytes:") == 1 &&
            log.find("decode=2684354560") != std::string::npos;
    }
    for (int i = 0; i < steps && ok; i++) {
        ok = compute_matches("cache-suspend", scheduler, fixture.graph, fixture.reference, step++);
    }
    const size_t pools_after = count_occurrences(capture.get(), " pool[");
    if (ok && pools_after <= pools_before) {
        fprintf(stderr, "cache-suspend: no new pool after resume (%zu -> %zu)\n%s",
                pools_before, pools_after, capture.get().c_str());
        ok = false;
    }
    // a second cycle on the made-again session
    if (ok && !ggml_backend_sched_moe_cache_suspend(scheduler)) {
        fprintf(stderr, "cache-suspend: second suspend found no session\n");
        ok = false;
    }
    ggml_backend_sched_moe_cache_resume(scheduler);
    if (return_reserve) {
        (void) ggml_backend_sched_moe_cache_restore(scheduler);
        ok &= count_occurrences(capture.get(), "prompt-return reserve bytes:") == 2;
        set_env("GGML_CUDA_MOE_CACHE_RETURN_RESERVE_MB", nullptr);
    }
    ggml_backend_sched_free(scheduler);
    printf("cache-suspend: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

static bool run_fill_invalidation(
        ggml_backend_t gpu,
        ggml_backend_t cpu,
        stress_fixture & fixture,
        log_capture & capture) {
    configure_cache(nullptr);
    capture.clear();

    ggml_backend_sched_t scheduler = make_scheduler(
            "cache-fill-invalidate", gpu, cpu, fixture.graph);
    if (!scheduler) {
        return false;
    }

    std::vector<float> replacement_f32(n_in * stress_n_out);
    for (size_t index = 0; index < replacement_f32.size(); index++) {
        replacement_f32[index] =
            1.1f + 0.2f * std::sin((float) index * 0.043f);
    }
    std::vector<uint8_t> replacement_q4(
            ggml_row_size(GGML_TYPE_Q4_0, n_in) * stress_n_out);
    const size_t replacement_size = ggml_quantize_chunk(
            GGML_TYPE_Q4_0, replacement_f32.data(), replacement_q4.data(),
            0, stress_n_out, n_in, nullptr);
    if (replacement_size != replacement_q4.size()) {
        fprintf(stderr,
                "cache-fill-invalidate: unexpected replacement size\n");
        ggml_backend_sched_free(scheduler);
        return false;
    }

    std::atomic<bool> stop{false};
    std::atomic<bool> mutation_started{false};
    std::atomic<bool> mutation_done{false};
    std::thread mutator([&] {
        if (capture.wait_for(
                " pool[", stop, std::chrono::seconds(5))) {
            mutation_started.store(true);
            ggml_backend_tensor_set(
                    fixture.weights, replacement_q4.data(), 0,
                    replacement_q4.size());
            mutation_done.store(true);
        }
    });

    bool output_ok = true;
    for (int step = 0; step < max_steps && !mutation_done.load(); step++) {
        const enum ggml_status status =
            ggml_backend_sched_graph_compute(scheduler, fixture.graph.graph);
        if (status != GGML_STATUS_SUCCESS) {
            fprintf(stderr,
                    "cache-fill-invalidate: warmup failed at step %d: %s\n",
                    step, ggml_status_to_string(status));
            output_ok = false;
            break;
        }
        if (mutation_started.load()) {
            const auto deadline =
                std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (!mutation_done.load() &&
                   std::chrono::steady_clock::now() < deadline) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            break;
        }
    }
    stop.store(true);
    capture.cv.notify_all();
    mutator.join();
    if (!mutation_started.load() || !mutation_done.load()) {
        fprintf(stderr,
                "cache-fill-invalidate: concurrent mutation did not complete\n");
        output_ok = false;
    }

    std::vector<float> new_reference(fixture.reference.size());
    if (output_ok &&
        ggml_backend_graph_compute(cpu, fixture.graph.graph) !=
            GGML_STATUS_SUCCESS) {
        fprintf(stderr,
                "cache-fill-invalidate: CPU reference compute failed\n");
        output_ok = false;
    }
    if (output_ok) {
        ggml_backend_tensor_get(
                fixture.graph.out, new_reference.data(), 0,
                new_reference.size() * sizeof(float));
        float max_change = 0.0f;
        for (size_t index = 0; index < new_reference.size(); index++) {
            max_change = std::max(
                    max_change,
                    std::abs(new_reference[index] - fixture.reference[index]));
        }
        if (max_change < 0.01f) {
            fprintf(stderr,
                    "cache-fill-invalidate: mutation did not change output\n");
            output_ok = false;
        }
    }

    capture.clear();
    bool repopulated = false;
    if (output_ok) {
        for (int step = 0; step < max_steps; step++) {
            if (!compute_matches(
                    "cache-fill-invalidate", scheduler, fixture.graph,
                    new_reference, step)) {
                output_ok = false;
                break;
            }
            if (has_positive_field(capture.get(), "hits=")) {
                repopulated = true;
                break;
            }
            if (step >= 64) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }
    ggml_backend_sched_free(scheduler);
    if (output_ok && !repopulated) {
        fprintf(stderr,
                "cache-fill-invalidate: cache was not repopulated\n%s",
                capture.get().c_str());
        output_ok = false;
    }
    printf("cache-fill-invalidate: %s\n", output_ok ? "OK" : "FAIL");
    return output_ok;
}

static void * create_direct_session(
        ggml_backend_t gpu, ggml_backend_t cpu) {
    const ggml_moe_cache_api api =
        ggml_moe_cache_get(ggml_backend_dev_backend_reg(ggml_backend_get_device(gpu)));
    if (!api.session_create) {
        return nullptr;
    }
    void * backends[] = { gpu, cpu };
    return api.session_create(backends, 2, nullptr);
}

static bool run_explicit_session_config(
        ggml_backend_t gpu, ggml_backend_t cpu) {
    configure_cache(nullptr);
    const ggml_moe_cache_api api =
        ggml_moe_cache_get(ggml_backend_dev_backend_reg(ggml_backend_get_device(gpu)));
    ggml_moe_cache_config config = {};
    if (!api.query_config ||
        !api.query_config(0, 4, &config)) {
        fprintf(stderr, "cache-explicit-config: failed to query configuration\n");
        return false;
    }

    set_env("GGML_CUDA_MOE_CACHE", "0");
    set_env("GGML_CUDA_MOE_CACHE_MODE", "off");
    void * backends[] = { gpu, cpu };
    void * session = api.session_create(backends, 2, &config);
    const bool ok = session != nullptr;
    if (session) {
        api.session_destroy(session);
    }
    bool invalid_rejected = true;
    for (int overlap_cpu_rows : { -2, 9 }) {
        config.overlap_cpu_rows = overlap_cpu_rows;
        void * invalid = api.session_create(backends, 2, &config);
        invalid_rejected &= invalid == nullptr;
        if (invalid) {
            api.session_destroy(invalid);
        }
    }
    config.overlap_cpu_rows = 0;
    for (int min_expert_explicit : { -1, 2 }) {
        config.min_expert_explicit = min_expert_explicit;
        void * invalid = api.session_create(backends, 2, &config);
        invalid_rejected &= invalid == nullptr;
        if (invalid) {
            api.session_destroy(invalid);
        }
    }
    configure_cache(nullptr);
    printf("cache-explicit-config: %s\n", ok && invalid_rejected ? "OK" : "FAIL");
    return ok && invalid_rejected;
}

static bool direct_begin_ready(
        const char * name, const void * base, size_t expert_size,
        int64_t direct_n_in, int64_t direct_n_out,
        int direct_type, int64_t direct_n_expert) {
    const ggml_moe_cache_api api = ggml_moe_cache_active();
    if (!api.begin || !api.end) {
        return false;
    }
    void * node = api.begin(
            name, base, expert_size, direct_n_in, direct_n_out,
            direct_type, direct_n_expert, 1, 1);
    if (!node) {
        return false;
    }
    api.end(node);
    return true;
}

static bool wait_for_direct_pool(
        const char * name, const void * base, size_t expert_size,
        int64_t direct_n_in, int64_t direct_n_out,
        int direct_type, int64_t direct_n_expert) {
    for (int step = 0; step < 80; step++) {
        if (direct_begin_ready(
                name, base, expert_size, direct_n_in, direct_n_out,
                direct_type, direct_n_expert)) {
            return true;
        }
    }
    return false;
}

static bool run_policy_diagnostics(
        ggml_backend_t gpu, ggml_backend_t cpu,
        ggml_tensor * weights, log_capture & capture) {
    configure_cache(nullptr);
    capture.clear();

    void * session = create_direct_session(gpu, cpu);
    if (!session) {
        fprintf(stderr, "cache-policy-diagnostics: failed to create session\n");
        return false;
    }

    const size_t expert_size = ggml_nbytes(weights) / weights->ne[2];
    const ggml_moe_cache_api api =
        ggml_moe_cache_get(ggml_backend_dev_backend_reg(ggml_backend_get_device(gpu)));
    api.session_enter(session);
    void * first = api.begin(
            weights->name, weights->data, expert_size,
            weights->ne[0], weights->ne[1], weights->type,
            weights->ne[2], 1, 1);
    const bool census_waited = first == nullptr;
    if (first) {
        api.end(first);
    }
    void * second = api.begin(
            weights->name, weights->data, expert_size,
            weights->ne[0], weights->ne[1], weights->type,
            weights->ne[2], 1, 1);
    const bool census_completed = second != nullptr;
    if (second) {
        api.end(second);
    }
    for (int64_t n_tokens : { 2, 3 }) {
        void * oversize = api.begin(
                weights->name, weights->data, expert_size,
                weights->ne[0], weights->ne[1], weights->type,
                weights->ne[2], n_tokens, n_tokens);
        if (oversize) {
            api.end(oversize);
        }
    }
    void * excess_rows = api.begin(
            weights->name, weights->data, expert_size,
            weights->ne[0], weights->ne[1], weights->type,
            weights->ne[2], 1, 65);
    if (excess_rows) {
        api.end(excess_rows);
    }
    api.session_leave(session);
    api.session_destroy(session);

    const std::string log = capture.get();
    const bool configured =
        log.find("configured: mode=on devices=1 budget=4 MiB cap") != std::string::npos &&
        log.find("max-batch=1") != std::string::npos;
    const bool capacity =
        log.find("capacity: cap=4 MiB granted=4 MiB") != std::string::npos &&
        log.find("free=") != std::string::npos &&
        log.find("reserve=0 MiB") != std::string::npos;
    const bool bypass = count_occurrences(log, "above max-batch=1") == 1;
    const bool row_bypass = count_occurrences(log, "above row limit=64") == 1;
    const bool pool = count_occurrences(log, " pool[") == 1;
    const bool ok = census_waited && census_completed && configured &&
        capacity && bypass && row_bypass && pool;
    if (!ok) {
        fprintf(stderr, "cache-policy-diagnostics: expected diagnostics were not observed\n%s", log.c_str());
    }
    printf("cache-policy-diagnostics: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

static int direct_plan_one(
        const char * name, const void * base, size_t expert_size,
        int64_t direct_n_in, int64_t direct_n_out,
        int direct_type, int64_t direct_n_expert, int32_t expert) {
    const ggml_moe_cache_api api = ggml_moe_cache_active();
    if (!api.begin || !api.plan || !api.end) {
        return -1;
    }
    void * node = api.begin(
            name, base, expert_size, direct_n_in, direct_n_out,
            direct_type, direct_n_expert, 1, 1);
    if (!node) {
        return -1;
    }
    int32_t slot = -1;
    const int hits = api.plan(node, &expert, 1, &slot);
    api.end(node);
    return hits;
}

static int direct_plan_many(
        const char * name, const void * base, size_t expert_size,
        int64_t direct_n_in, int64_t direct_n_out,
        int direct_type, int64_t direct_n_expert,
        const int32_t * experts, int n_experts,
        int64_t direct_n_tokens = 1) {
    const ggml_moe_cache_api api = ggml_moe_cache_active();
    if (!api.begin || !api.plan || !api.end) {
        return -1;
    }
    void * node = api.begin(
            name, base, expert_size, direct_n_in, direct_n_out,
            direct_type, direct_n_expert, direct_n_tokens, n_experts);
    if (!node) {
        return -1;
    }
    std::vector<int32_t> slots(n_experts, -1);
    const int hits = api.plan(
            node, experts, n_experts, slots.data());
    api.end(node);
    return hits;
}

static bool wait_for_direct_resident(
        ggml_tensor * weights, int32_t expert) {
    const size_t expert_size = ggml_nbytes(weights) / weights->ne[2];
    (void) direct_plan_one(
            weights->name, weights->data, expert_size,
            weights->ne[0], weights->ne[1], weights->type,
            weights->ne[2], expert);
    for (int attempt = 0; attempt < 200; attempt++) {
        if (direct_plan_one(
                weights->name, weights->data, expert_size,
                weights->ne[0], weights->ne[1], weights->type,
                weights->ne[2], expert) == 1) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
}

static bool run_fused_partial_invalidation(
        ggml_backend_t gpu,
        ggml_backend_t cpu,
        ggml_tensor * up_weights,
        ggml_tensor * gate_weights,
        ggml_tensor * activations,
        const uint8_t * up_row,
        const uint8_t * gate_row,
        const std::vector<float> & reference,
        log_capture & capture) {
    const ggml_moe_cache_api api =
        ggml_moe_cache_get(ggml_backend_dev_backend_reg(ggml_backend_get_device(gpu)));
    if (!api.fused_begin || !api.collect || !api.end) {
        fprintf(stderr, "cache-fused-partial: incomplete cache API\n");
        return false;
    }

    configure_cache(nullptr);
    capture.clear();
    void * session = create_direct_session(gpu, cpu);
    if (!session) {
        fprintf(stderr, "cache-fused-partial: failed to create session\n");
        return false;
    }

    const size_t expert_size =
        ggml_nbytes(up_weights) / up_weights->ne[2];
    const ggml_moe_cache_tensor_desc up = {
        up_weights->name, up_weights->data, expert_size,
        up_weights->ne[0], up_weights->ne[1], up_weights->ne[2],
        (int32_t)up_weights->type,
    };
    const ggml_moe_cache_tensor_desc gate = {
        gate_weights->name, gate_weights->data, expert_size,
        gate_weights->ne[0], gate_weights->ne[1], gate_weights->ne[2],
        (int32_t)gate_weights->type,
    };
    const float * act_rows[2] = {
        (const float *)activations->data,
        (const float *)activations->data,
    };

    api.session_enter(session);
    bool pool_ready = false;
    for (int step = 0; step < 80 && !pool_ready; step++) {
        const bool up_ready = direct_begin_ready(
                up_weights->name, up_weights->data, expert_size,
                up_weights->ne[0], up_weights->ne[1], up_weights->type,
                up_weights->ne[2]);
        const bool gate_ready = direct_begin_ready(
                gate_weights->name, gate_weights->data, expert_size,
                gate_weights->ne[0], gate_weights->ne[1], gate_weights->type,
                gate_weights->ne[2]);
        pool_ready = up_ready && gate_ready;
    }

    bool partial_ok = pool_ready &&
        wait_for_direct_resident(up_weights, 0) &&
        wait_for_direct_resident(gate_weights, 0);

    auto execute = [&](const int32_t * experts, int n_ids,
                       uint64_t expected_mask,
                       const int * reference_rows) {
        uint64_t hit_mask = 0;
        void * node = api.fused_begin(
                &up, &gate, GGML_GLU_OP_SWIGLU,
                -std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::infinity(),
                -std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::infinity(),
                experts, n_ids, 1, act_rows, &hit_mask);
        if (!node || hit_mask != expected_mask) {
            if (node) {
                api.end(node);
            }
            return false;
        }

        int n_hits = 0;
        float output[2 * n_out];
        float * rows[2] = { nullptr, nullptr };
        for (int row = 0; row < n_ids; row++) {
            if (hit_mask & (UINT64_C(1) << row)) {
                rows[n_hits] = output + n_hits * n_out;
                n_hits++;
            }
        }
        const bool collected =
            active_api().collect(node, n_hits, rows, n_out) == 1;
        active_api().end(node);
        if (!collected) {
            return false;
        }

        for (int hit = 0; hit < n_hits; hit++) {
            const int reference_row = reference_rows[hit];
            std::vector<float> expected(
                    reference.begin() + reference_row * n_out,
                    reference.begin() + (reference_row + 1) * n_out);
            std::vector<float> actual(
                    output + hit * n_out,
                    output + (hit + 1) * n_out);
            if (!compare_output(expected, actual, 5e-4)) {
                return false;
            }
        }
        return true;
    };

    const int32_t partial_ids[2] = { 0, 1 };
    const int partial_reference[1] = { 0 };
    partial_ok &= execute(
            partial_ids, 2, UINT64_C(1), partial_reference);

    const int32_t duplicate_ids[2] = { 0, 0 };
    const int duplicate_reference[2] = { 0, 0 };
    partial_ok &= execute(
            duplicate_ids, 2, UINT64_C(3), duplicate_reference);

    auto invalidated = [&](ggml_tensor * weights,
                           const uint8_t * unchanged_row) {
        ggml_backend_tensor_set(
                weights, unchanged_row, 0, expert_size);
        const int32_t expert = 0;
        uint64_t hit_mask = 0;
        void * node = active_api().fused_begin(
                &up, &gate, GGML_GLU_OP_SWIGLU,
                -std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::infinity(),
                -std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::infinity(),
                &expert, 1, 1, act_rows, &hit_mask);
        if (node) {
            active_api().end(node);
            return false;
        }
        if (hit_mask != 0 || !wait_for_direct_resident(weights, expert)) {
            return false;
        }
        const int reference_row[1] = { 0 };
        return execute(&expert, 1, UINT64_C(1), reference_row);
    };

    const bool up_invalidation_ok = partial_ok &&
        invalidated(up_weights, up_row);
    const bool gate_invalidation_ok = up_invalidation_ok &&
        invalidated(gate_weights, gate_row);
    active_api().session_leave(session);
    active_api().session_destroy(session);

    const bool stats_ok = has_positive_field(
            capture.get(), "fusion-nodes=");
    printf("cache-fused-partial-duplicate: %s\n",
            partial_ok ? "OK" : "FAIL");
    printf("cache-fused-up-invalidate: %s\n",
            up_invalidation_ok ? "OK" : "FAIL");
    printf("cache-fused-gate-invalidate: %s\n",
            gate_invalidation_ok && stats_ok ? "OK" : "FAIL");
    return gate_invalidation_ok && stats_ok;
}

static bool run_cpu_overlap_policy(
        ggml_backend_t gpu, ggml_backend_t cpu,
        ggml_tensor * weights, log_capture & capture) {
    configure_cache(nullptr);
    set_env("GGML_CUDA_MOE_CACHE_OVERLAP_CPU_ROWS", "1");
    capture.clear();

    void * session = create_direct_session(gpu, cpu);
    if (!session) {
        fprintf(stderr, "cache-cpu-overlap: failed to create session\n");
        configure_cache(nullptr);
        return false;
    }

    const size_t expert_size = ggml_nbytes(weights) / weights->ne[2];
    active_api().session_enter(session);
    bool ok = wait_for_direct_pool(
            weights->name, weights->data, expert_size,
            weights->ne[0], weights->ne[1], weights->type, weights->ne[2]);
    const int32_t experts[] = { 0, 1 };
    for (int index = 0; index < 2 && ok; index++) {
        (void) direct_plan_one(
                weights->name, weights->data, expert_size,
                weights->ne[0], weights->ne[1], weights->type,
                weights->ne[2], experts[index]);
        bool resident = false;
        for (int attempt = 0; attempt < 100; attempt++) {
            if (direct_plan_one(
                    weights->name, weights->data, expert_size,
                    weights->ne[0], weights->ne[1], weights->type,
                    weights->ne[2], experts[index]) == 1) {
                resident = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        ok &= resident;
    }
    if (ok) {
        ok = direct_plan_many(
                weights->name, weights->data, expert_size,
                weights->ne[0], weights->ne[1], weights->type,
                weights->ne[2], experts, 2) == 1;
    }
    active_api().session_leave(session);
    active_api().session_destroy(session);

    ok &= has_positive_field(capture.get(), "cpu-overlap=");
    configure_cache(nullptr);
    printf("cache-cpu-overlap: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

static bool run_adaptive_cpu_overlap_policy(
        ggml_backend_t gpu, ggml_backend_t cpu,
        ggml_tensor * weights, log_capture & capture) {
    configure_cache(nullptr, "8");
    set_env("GGML_CUDA_MOE_CACHE_OVERLAP_CPU_ROWS", nullptr);
    capture.clear();

    void * session = create_direct_session(gpu, cpu);
    if (!session) {
        fprintf(stderr, "cache-cpu-overlap-auto: failed to create session\n");
        configure_cache(nullptr);
        return false;
    }

    const size_t expert_size = ggml_nbytes(weights) / weights->ne[2];
    active_api().session_enter(session);
    bool ok = wait_for_direct_pool(
            weights->name, weights->data, expert_size,
            weights->ne[0], weights->ne[1], weights->type, weights->ne[2]);
    for (int32_t expert = 0; expert < 8 && ok; expert++) {
        (void) direct_plan_one(
                weights->name, weights->data, expert_size,
                weights->ne[0], weights->ne[1], weights->type,
                weights->ne[2], expert);
        bool resident = false;
        for (int attempt = 0; attempt < 100; attempt++) {
            if (direct_plan_one(
                    weights->name, weights->data, expert_size,
                    weights->ne[0], weights->ne[1], weights->type,
                    weights->ne[2], expert) == 1) {
                resident = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        ok &= resident;
    }

    const int32_t single_token_experts[] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    if (ok) {
        ok = direct_plan_many(
                weights->name, weights->data, expert_size,
                weights->ne[0], weights->ne[1], weights->type,
                weights->ne[2], single_token_experts, 8) == 6;
    }

    int32_t multi_token_experts[multi_n_used * multi_n_tokens];
    for (int token = 0; token < multi_n_tokens; token++) {
        for (int id = 0; id < multi_n_used; id++) {
            multi_token_experts[token * multi_n_used + id] = id;
        }
    }
    if (ok) {
        ok = direct_plan_many(
                weights->name, weights->data, expert_size,
                weights->ne[0], weights->ne[1], weights->type,
                weights->ne[2], multi_token_experts,
                multi_n_used * multi_n_tokens, multi_n_tokens) == 30;
    }
    active_api().session_leave(session);
    active_api().session_destroy(session);

    ok &= capture.get().find("cpu-overlap=auto") != std::string::npos;
    ok &= has_positive_field(capture.get(), "cpu-overlap=");
    configure_cache(nullptr);
    printf("cache-cpu-overlap-auto: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

// Drives the adaptive cpu-overlap calibration in moe-cache.cu
// (moe_cache_calibration_on_enter/on_leave) through a full convergence:
// enough single-token decode scopes to exhaust the unmeasured global
// warm-up, then the whole round-robin candidate sweep, then confirms it
// locks in and stops adjusting. Each scope is one session_enter/plan/leave
// cycle over a single already-resident expert -- moe_cache_plan() updates
// the calibration scope stats unconditionally per planned node, so neither
// multiple experts nor a hit are required, only n_tokens == 1 per scope.
static bool run_adaptive_cpu_overlap_calibration(
        ggml_backend_t gpu, ggml_backend_t cpu,
        ggml_tensor * weights, log_capture & capture) {
    configure_cache(nullptr, "8");
    set_env("GGML_CUDA_MOE_CACHE_OVERLAP_CPU_ROWS", nullptr);
    capture.clear();

    void * session = create_direct_session(gpu, cpu);
    if (!session) {
        fprintf(stderr, "cache-cpu-overlap-calibration: failed to create session\n");
        configure_cache(nullptr);
        return false;
    }

    const size_t expert_size = ggml_nbytes(weights) / weights->ne[2];
    active_api().session_enter(session);
    bool ok = wait_for_direct_pool(
            weights->name, weights->data, expert_size,
            weights->ne[0], weights->ne[1], weights->type, weights->ne[2]);
    for (int attempt = 0; attempt < 100 && ok; attempt++) {
        if (direct_plan_one(
                weights->name, weights->data, expert_size,
                weights->ne[0], weights->ne[1], weights->type,
                weights->ne[2], /* expert */ 0) == 1) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    active_api().session_leave(session);

    // moe-cache.cu: moe_cache_calib_global_warmup_scopes (64) +
    // moe_cache_calib_rounds (3) x moe_cache_calib_candidate_count (6) x
    // (moe_cache_calib_round_warmup_scopes (1) + _measured_scopes (4)) = 154.
    // Run comfortably past that so calibration has definitely completed,
    // then a few more to confirm it stays locked in afterward.
    constexpr int total_scopes = 170;
    constexpr int expected_round_lines = 18; // 3 rounds x 6 candidates
    for (int step = 0; step < total_scopes && ok; step++) {
        active_api().session_enter(session);
        direct_plan_one(
                weights->name, weights->data, expert_size,
                weights->ne[0], weights->ne[1], weights->type,
                weights->ne[2], /* expert */ 0);
        active_api().session_leave(session);
    }
    active_api().session_destroy(session);

    const std::string log = capture.get();
    ok &= log.find("cpu-overlap calibration: warm-up complete") != std::string::npos;
    ok &= count_occurrences(log, "cpu-overlap calibration: round=") == expected_round_lines;
    ok &= count_occurrences(log, "cpu-overlap calibration complete:") == 1;

    configure_cache(nullptr);
    printf("cache-cpu-overlap-calibration: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

static bool run_scope_isolation(
        ggml_backend_t gpu, ggml_backend_t cpu, ggml_tensor * weights) {
    if (!active_api().session_enter || !active_api().session_leave ||
        !active_api().begin || !active_api().end ||
        !active_api().invalidate || !active_api().session_destroy) {
        fprintf(stderr, "cache-scope: incomplete cache API\n");
        return false;
    }

    configure_cache(nullptr);
    void * outer = create_direct_session(gpu, cpu);
    if (!outer) {
        fprintf(stderr, "cache-scope: failed to create outer session\n");
        return false;
    }

    const size_t expert_size = ggml_nbytes(weights) / weights->ne[2];
    active_api().session_enter(outer);
    const bool warmed = wait_for_direct_pool(
            weights->name, weights->data, expert_size,
            weights->ne[0], weights->ne[1], weights->type, weights->ne[2]);
    active_api().session_leave(outer);

    set_env("GGML_CUDA_MOE_CACHE_RESERVE_MB", "1048576");
    void * dormant = create_direct_session(gpu, cpu);
    if (!dormant) {
        fprintf(stderr, "cache-scope: failed to create dormant session\n");
        active_api().session_destroy(outer);
        return false;
    }
    active_api().session_enter(dormant);
    const bool dormant_begin = direct_begin_ready(
            weights->name, weights->data, expert_size,
            weights->ne[0], weights->ne[1], weights->type, weights->ne[2]);
    active_api().session_leave(dormant);

    active_api().session_enter(outer);
    const bool outer_before = direct_begin_ready(
            weights->name, weights->data, expert_size,
            weights->ne[0], weights->ne[1], weights->type, weights->ne[2]);

    active_api().session_enter(nullptr);
    const bool null_leaked = direct_begin_ready(
            weights->name, weights->data, expert_size,
            weights->ne[0], weights->ne[1], weights->type, weights->ne[2]);
    active_api().session_leave(nullptr);
    const bool outer_after_null = direct_begin_ready(
            weights->name, weights->data, expert_size,
            weights->ne[0], weights->ne[1], weights->type, weights->ne[2]);

    active_api().session_enter(dormant);
    const bool dormant_leaked = direct_begin_ready(
            weights->name, weights->data, expert_size,
            weights->ne[0], weights->ne[1], weights->type, weights->ne[2]);
    active_api().session_leave(dormant);
    const bool outer_after_dormant = direct_begin_ready(
            weights->name, weights->data, expert_size,
            weights->ne[0], weights->ne[1], weights->type, weights->ne[2]);
    active_api().session_leave(outer);

    active_api().session_destroy(dormant);
    active_api().session_destroy(outer);
    const bool ok = warmed && !dormant_begin && outer_before &&
        !null_leaked && outer_after_null &&
        !dormant_leaked && outer_after_dormant;
    printf("cache-scope: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

static bool run_shape_liveness(
        ggml_backend_t gpu, ggml_backend_t cpu) {
    configure_cache(nullptr);
    set_env("GGML_CUDA_MOE_CACHE_BUDGET_MB", "16");
    void * session = create_direct_session(gpu, cpu);
    if (!session) {
        fprintf(stderr, "cache-shape-liveness: failed to create session\n");
        return false;
    }

    constexpr int64_t shape_a_in = 256;
    constexpr int64_t shape_b_in = 512;
    constexpr int64_t shape_out = 128;
    constexpr int64_t shape_experts = 64;
    const size_t shape_a_expert =
        ggml_row_size(GGML_TYPE_Q4_0, shape_a_in) * shape_out;
    const size_t shape_b_expert =
        ggml_row_size(GGML_TYPE_Q4_0, shape_b_in) * shape_out;
    std::vector<uint8_t> shape_a(shape_a_expert * shape_experts);
    std::vector<uint8_t> shape_b(shape_b_expert * shape_experts);

    active_api().session_enter(session);
    (void)direct_begin_ready(
            "blk.2.ffn_up_exps.weight", shape_a.data(), shape_a_expert,
            shape_a_in, shape_out, GGML_TYPE_Q4_0, shape_experts);
    active_api().invalidate(shape_a.data(), shape_a.size());
    const bool shape_b_ready = wait_for_direct_pool(
            "blk.3.ffn_up_exps.weight", shape_b.data(), shape_b_expert,
            shape_b_in, shape_out, GGML_TYPE_Q4_0, shape_experts);
    const bool shape_a_ready = wait_for_direct_pool(
            "blk.2.ffn_up_exps.weight", shape_a.data(), shape_a_expert,
            shape_a_in, shape_out, GGML_TYPE_Q4_0, shape_experts);
    active_api().session_leave(session);
    active_api().session_destroy(session);

    const bool ok = shape_b_ready && shape_a_ready;
    printf("cache-shape-liveness: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

static bool run_exact_shape_inventory(
        ggml_backend_t gpu, ggml_backend_t cpu,
        log_capture & capture) {
    configure_cache(nullptr);
    set_env("GGML_CUDA_MOE_CACHE_BUDGET_MB", "16");
    set_env("GGML_CUDA_MOE_CACHE_ADMIT_AFTER", nullptr);
    set_env("GGML_CUDA_MOE_CACHE_THROTTLE", "8");
    set_env("GGML_CUDA_MOE_CACHE_STATS", "0");
    capture.clear();

    void * session = create_direct_session(gpu, cpu);
    if (!session) {
        fprintf(stderr, "cache-shape-inventory: failed to create session\n");
        return false;
    }

    constexpr int64_t direct_n_in = 256;
    constexpr int64_t direct_n_out = 128;
    constexpr int64_t first_n_expert = 16;
    constexpr int64_t second_n_expert = 49;
    const size_t expert_size =
        ggml_row_size(GGML_TYPE_Q4_0, direct_n_in) * direct_n_out;
    std::vector<uint8_t> first(expert_size * first_n_expert);
    std::vector<uint8_t> second(expert_size * second_n_expert);
    std::vector<uint8_t> late(expert_size);

    active_api().session_enter(session);
    (void)direct_begin_ready(
            "blk.7.ffn_up_exps.weight", first.data(), expert_size,
            direct_n_in, direct_n_out, GGML_TYPE_Q4_0, first_n_expert);
    (void)direct_begin_ready(
            "blk.8.ffn_up_exps.weight", second.data(), expert_size,
            direct_n_in, direct_n_out, GGML_TYPE_Q4_0, second_n_expert);
    bool ready = false;
    for (int step = 0; step < 40 && !ready; step++) {
        ready |= direct_begin_ready(
                "blk.7.ffn_up_exps.weight", first.data(), expert_size,
                direct_n_in, direct_n_out, GGML_TYPE_Q4_0, first_n_expert);
        ready |= direct_begin_ready(
                "blk.8.ffn_up_exps.weight", second.data(), expert_size,
                direct_n_in, direct_n_out, GGML_TYPE_Q4_0, second_n_expert);
    }
    const bool complete_first_miss = direct_plan_one(
            "blk.7.ffn_up_exps.weight", first.data(), expert_size,
            direct_n_in, direct_n_out, GGML_TYPE_Q4_0, first_n_expert, 0) == 0;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    const bool complete_second_hit = direct_plan_one(
            "blk.7.ffn_up_exps.weight", first.data(), expert_size,
            direct_n_in, direct_n_out, GGML_TYPE_Q4_0, first_n_expert, 0) == 1;
    const bool late_first_miss = direct_plan_one(
            "blk.9.ffn_up_exps.weight", late.data(), expert_size,
            direct_n_in, direct_n_out, GGML_TYPE_Q4_0, 1, 0) == 0;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    const bool late_second_miss = direct_plan_one(
            "blk.9.ffn_up_exps.weight", late.data(), expert_size,
            direct_n_in, direct_n_out, GGML_TYPE_Q4_0, 1, 0) == 0;
    bool late_hit = false;
    for (int attempt = 0; attempt < 100 && !late_hit; attempt++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        late_hit = direct_plan_one(
                "blk.9.ffn_up_exps.weight", late.data(), expert_size,
                direct_n_in, direct_n_out, GGML_TYPE_Q4_0, 1, 0) == 1;
    }
    active_api().session_leave(session);
    active_api().session_destroy(session);

    const std::string log = capture.get();
    const bool ok = ready &&
        max_field_value(log, "slots=") == first_n_expert + second_n_expert &&
        log.find("entries=65 coverage=complete") != std::string::npos &&
        log.find("admit=1-complete/2-partial/8-replace") != std::string::npos &&
        complete_first_miss && complete_second_hit &&
        late_first_miss && late_second_miss && late_hit &&
        log.find("fills=serial") != std::string::npos;
    if (!ok) {
        fprintf(stderr, "cache-shape-inventory: unexpected pool inventory\n%s",
                log.c_str());
    }
    printf("cache-shape-inventory: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

static bool run_complete_pool_allocation(
        ggml_backend_t gpu, ggml_backend_t cpu) {
    configure_cache(nullptr);
    set_env("GGML_CUDA_MOE_CACHE_BUDGET_MB", "4");
    void * session = create_direct_session(gpu, cpu);
    if (!session) {
        fprintf(stderr, "cache-complete-pools: failed to create session\n");
        return false;
    }

    constexpr int64_t direct_n_in = 512;
    constexpr int64_t common_n_out = 64;
    constexpr int64_t rare_n_out = 128;
    constexpr int64_t direct_n_expert = 64;
    const size_t common_expert =
        ggml_row_size(GGML_TYPE_Q4_0, direct_n_in) * common_n_out;
    const size_t rare_expert =
        ggml_row_size(GGML_TYPE_Q4_0, direct_n_in) * rare_n_out;
    std::vector<std::vector<uint8_t>> common_tensors(
            8, std::vector<uint8_t>(common_expert * direct_n_expert));
    std::vector<uint8_t> rare_tensor(rare_expert * direct_n_expert);

    active_api().session_enter(session);
    for (const auto & tensor : common_tensors) {
        (void)direct_begin_ready(
                "blk.9.ffn_up_exps.weight", tensor.data(), common_expert,
                direct_n_in, common_n_out, GGML_TYPE_Q4_0, direct_n_expert);
    }
    (void)direct_begin_ready(
            "blk.9.ffn_down_exps.weight", rare_tensor.data(), rare_expert,
            direct_n_in, rare_n_out, GGML_TYPE_Q4_0, direct_n_expert);

    const bool common_ready = wait_for_direct_pool(
            "blk.9.ffn_up_exps.weight", common_tensors[0].data(), common_expert,
            direct_n_in, common_n_out, GGML_TYPE_Q4_0, direct_n_expert);
    const bool rare_ready = direct_begin_ready(
            "blk.9.ffn_down_exps.weight", rare_tensor.data(), rare_expert,
            direct_n_in, rare_n_out, GGML_TYPE_Q4_0, direct_n_expert);
    active_api().session_leave(session);
    active_api().session_destroy(session);

    const bool ok = common_ready && rare_ready;
    printf("cache-complete-pools: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

static bool run_shared_budget(
        ggml_backend_t gpu, ggml_backend_t cpu,
        log_capture & capture) {
    configure_cache(nullptr);
    size_t free_bytes = 0;
    size_t total_bytes = 0;
#ifdef GGML_CUDA
    // The cache's budget path queries cudaMemGetInfo, not the backend device
    // memory API. On UMA devices (Grace Blackwell, Jetson/Orin) the backend
    // reports /proc/meminfo while cudaMemGetInfo reports the CUDA view, and
    // the two disagree. Match the cache's query so the reserve leaves the
    // expected remainder on every device class.
    {
        int prev_device = 0;
        cudaError_t err = cudaGetDevice(&prev_device);
        if (err != cudaSuccess) {
            fprintf(stderr, "cache-shared-budget: cudaGetDevice failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        // Restore the device afterwards: run_route_override runs next and
        // exercises routing across two devices, so the current device must
        // not leak out of this test. Fall back to the current device rather
        // than device 0 when the physical device is not reported.
        const long physical = provider_physical_device(gpu->device);
        const int target = physical >= 0 ? (int) physical : prev_device;
        if (target != prev_device) {
            err = cudaSetDevice(target);
            if (err != cudaSuccess) {
                fprintf(stderr, "cache-shared-budget: cudaSetDevice failed: %s\n",
                        cudaGetErrorString(err));
                return false;
            }
        }
        err = cudaMemGetInfo(&free_bytes, &total_bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, "cache-shared-budget: cudaMemGetInfo failed: %s\n",
                    cudaGetErrorString(err));
            cudaSetDevice(prev_device);
            return false;
        }
        if (target != prev_device) {
            cudaSetDevice(prev_device);
        }
    }
#else
    ggml_backend_dev_memory(gpu->device, &free_bytes, &total_bytes);
#endif
    const size_t free_mib = free_bytes >> 20;
    if (free_mib < 64) {
        printf("cache-shared-budget: SKIP (insufficient free VRAM)\n");
        return true;
    }

    const std::string reserve = std::to_string(free_mib - 24);
    set_env("GGML_CUDA_MOE_CACHE_BUDGET_MB", nullptr);
    set_env("GGML_CUDA_MOE_CACHE_RESERVE_MB", reserve.c_str());
    capture.clear();

    void * first = create_direct_session(gpu, cpu);
    void * second = create_direct_session(gpu, cpu);
    if (!first || !second) {
        fprintf(stderr, "cache-shared-budget: failed to create sessions\n");
        if (first) {
            active_api().session_destroy(first);
        }
        if (second) {
            active_api().session_destroy(second);
        }
        configure_cache(nullptr);
        return false;
    }

    constexpr int64_t direct_n_in = 256;
    constexpr int64_t direct_n_out = 128;
    constexpr int64_t direct_n_expert = 64;
    const size_t expert_size =
        ggml_row_size(GGML_TYPE_Q4_0, direct_n_in) * direct_n_out;
    std::vector<uint8_t> weights(expert_size * direct_n_expert);

    active_api().session_enter(first);
    const bool first_ready = wait_for_direct_pool(
            "blk.6.ffn_up_exps.weight", weights.data(), expert_size,
            direct_n_in, direct_n_out, GGML_TYPE_Q4_0, direct_n_expert);
    active_api().session_leave(first);

    active_api().session_enter(second);
    const bool second_ready = wait_for_direct_pool(
            "blk.6.ffn_up_exps.weight", weights.data(), expert_size,
            direct_n_in, direct_n_out, GGML_TYPE_Q4_0, direct_n_expert);
    active_api().session_leave(second);

    const std::string shared_log = capture.get();
    bool return_reserve_ok = true;
    if (free_mib >= 1600 && active_api().restore) {
        const std::string decode_reserve = std::to_string(free_mib - 48);
        set_env("GGML_CUDA_MOE_CACHE_RETURN_RESERVE_MB", decode_reserve.c_str());
        capture.clear();
        (void) active_api().restore(first);
        const std::string startup_return = capture.get();
        return_reserve_ok &= startup_return.find("prompt-return reserve bytes:") == std::string::npos;
        return_reserve_ok &= active_api().trim && active_api().trim(first, 2 * expert_size) > 0;
        capture.clear();
        (void) active_api().restore(first);
        const std::string first_return = capture.get();
        // The second session's larger live reservation must still win.
        return_reserve_ok &= first_return.find("prompt-return reserve bytes:") != std::string::npos &&
            first_return.find("reserve=" + std::to_string((free_mib - 24) << 20)) != std::string::npos;
        return_reserve_ok &= active_api().trim && active_api().trim(second, 2 * expert_size) > 0;
        capture.clear();
        (void) active_api().restore(second);
        const std::string second_return = capture.get();
        return_reserve_ok &= second_return.find("prompt-return reserve bytes:") != std::string::npos &&
            second_return.find("reserve=" + std::to_string((free_mib - 48) << 20)) != std::string::npos;
        if (!return_reserve_ok) {
            fprintf(stderr, "cache-shared-budget: prompt-return reserve failed\n%s%s", first_return.c_str(), second_return.c_str());
        }
        set_env("GGML_CUDA_MOE_CACHE_RETURN_RESERVE_MB", nullptr);
    } else {
        printf("cache-prompt-return-reserve: SKIP (needs 1600 MiB free and restore API)\n");
    }
    active_api().session_destroy(second);
    active_api().session_destroy(first);

    const bool divided = count_field_at_least(shared_log, "granted=", 8) == 2 &&
        max_field_value(shared_log, "granted=") <= 16;

    capture.clear();
    void * replacement = create_direct_session(gpu, cpu);
    bool replacement_ready = false;
    if (replacement) {
        active_api().session_enter(replacement);
        replacement_ready = wait_for_direct_pool(
                "blk.6.ffn_up_exps.weight", weights.data(), expert_size,
                direct_n_in, direct_n_out, GGML_TYPE_Q4_0, direct_n_expert);
        active_api().session_leave(replacement);
        active_api().session_destroy(replacement);
    }
    const std::string replacement_log = capture.get();
    const bool released = max_field_value(replacement_log, "granted=") >= 20;

    capture.clear();
    const std::string high_reserve = std::to_string(free_mib - 8);
    set_env("GGML_CUDA_MOE_CACHE_RESERVE_MB", high_reserve.c_str());
    void * high = create_direct_session(gpu, cpu);
    set_env("GGML_CUDA_MOE_CACHE_RESERVE_MB", reserve.c_str());
    void * low = create_direct_session(gpu, cpu);
    if (high) {
        active_api().session_destroy(high);
    }
    void * after_high = create_direct_session(gpu, cpu);

    bool low_ready = false;
    if (low) {
        active_api().session_enter(low);
        low_ready = wait_for_direct_pool(
                "blk.6.ffn_up_exps.weight", weights.data(), expert_size,
                direct_n_in, direct_n_out, GGML_TYPE_Q4_0, direct_n_expert);
        active_api().session_leave(low);
    }
    bool after_high_ready = false;
    if (after_high) {
        active_api().session_enter(after_high);
        after_high_ready = wait_for_direct_pool(
                "blk.6.ffn_up_exps.weight", weights.data(), expert_size,
                direct_n_in, direct_n_out, GGML_TYPE_Q4_0, direct_n_expert);
        active_api().session_leave(after_high);
    }
    if (after_high) {
        active_api().session_destroy(after_high);
    }
    if (low) {
        active_api().session_destroy(low);
    }
    const std::string reserve_log = capture.get();
    const bool reserve_lowered = high && low && after_high &&
        low_ready && after_high_ready &&
        count_field_at_least(reserve_log, "granted=", 8) == 2;
    configure_cache(nullptr);

    const bool ok = first_ready && second_ready && divided && return_reserve_ok &&
        replacement_ready && released && reserve_lowered;
    if (!ok) {
        fprintf(stderr, "cache-shared-budget: unexpected claim behavior\n%s%s%s",
                shared_log.c_str(), replacement_log.c_str(), reserve_log.c_str());
    }
    printf("cache-shared-budget: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

static bool run_route_override(
        ggml_backend_dev_t first_device,
        ggml_backend_t gpu, ggml_backend_t cpu,
        log_capture & capture) {
    ggml_backend_dev_t second_device =
        find_other_provider_device(first_device);
    if (!second_device) {
        printf("cache-route-override: SKIP (one CUDA device)\n");
        return true;
    }
    const long first_physical = provider_physical_device(first_device);
    if (first_physical >= 0 &&
        first_physical == provider_physical_device(second_device)) {
        printf("cache-route-override: SKIP (one physical CUDA device)\n");
        return true;
    }
    ggml_backend_t second_gpu =
        ggml_backend_dev_init(second_device, nullptr);
    if (!second_gpu) {
        fprintf(stderr,
                "cache-route-override: failed to initialize second device\n");
        return false;
    }

    configure_cache(nullptr);
    set_env("GGML_CUDA_MOE_CACHE_BUDGET_MB", "8");
    set_env("GGML_CUDA_MOE_CACHE_NDEV", "2");
    capture.clear();
    void * backends[] = { gpu, second_gpu, cpu };
    void * session = active_api().session_create(backends, 3, nullptr);
    if (!session) {
        fprintf(stderr,
                "cache-route-override: failed to create session\n");
        ggml_backend_free(second_gpu);
        return false;
    }

    constexpr int64_t shape_a_in = 1024;
    constexpr int64_t shape_a_out = 160;
    constexpr int64_t shape_b_in = 512;
    constexpr int64_t shape_b_out = 384;
    constexpr int64_t shape_experts = 64;
    const size_t shape_a_expert =
        ggml_row_size(GGML_TYPE_Q4_0, shape_a_in) * shape_a_out;
    const size_t shape_b_expert =
        ggml_row_size(GGML_TYPE_Q4_0, shape_b_in) * shape_b_out;
    std::vector<uint8_t> shape_a(shape_a_expert * shape_experts);
    std::vector<uint8_t> shape_b(shape_b_expert * shape_experts);

    active_api().session_enter(session);
    const bool shape_a_ready = wait_for_direct_pool(
            "blk.4.ffn_up_exps.weight", shape_a.data(), shape_a_expert,
            shape_a_in, shape_a_out, GGML_TYPE_Q4_0, shape_experts);
    const bool shape_b_ready = wait_for_direct_pool(
            "blk.4.ffn_down_exps.weight", shape_b.data(), shape_b_expert,
            shape_b_in, shape_b_out, GGML_TYPE_Q4_0, shape_experts);
    active_api().session_leave(session);

    ggml_moe_cache_config config = {};
    ggml_moe_cache_device_caps first_caps = {};
    ggml_moe_cache_device_caps second_caps = {};
    const bool queried = active_api().query_config(0, 8, &config) &&
        active_api().query_device(first_device, &config, &first_caps) &&
        active_api().query_device(second_device, &config, &second_caps);
    const bool expect_parallel = queried && first_caps.compute_capability >= 800 &&
        second_caps.compute_capability >= 800;
    active_api().session_destroy(session);
    const bool fill_policy = capture.get().find(
            expect_parallel ? "fills=parallel" : "fills=serial") != std::string::npos;

    ggml_moe_cache_config automatic = {};
    capture.clear();
    const bool automatic_queried =
        active_api().query_config(1, 4, &automatic);
    void * dormant = automatic_queried
        ? active_api().session_create(backends, 3, &automatic) : nullptr;
    const bool dormant_created = dormant != nullptr;
    bool dormant_begin = false;
    if (dormant) {
        active_api().session_enter(dormant);
        dormant_begin = direct_begin_ready(
                "blk.4.ffn_up_exps.weight", shape_a.data(), shape_a_expert,
                shape_a_in, shape_a_out, GGML_TYPE_Q4_0, shape_experts);
        active_api().session_leave(dormant);
        active_api().session_destroy(dormant);
    }
    const bool slab_floor = dormant_created && !dormant_begin &&
        capture.get().find("automatic slab floor") != std::string::npos;
    ggml_backend_free(second_gpu);

    const bool ok = shape_a_ready && shape_b_ready && queried && fill_policy &&
        automatic_queried && slab_floor;
    printf("cache-route-override: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

static bool run_single_gpu_auto(
        ggml_backend_dev_t provider_device, ggml_backend_t gpu, ggml_backend_t cpu,
        log_capture & capture) {
    const ggml_moe_cache_api api =
        ggml_moe_cache_get(ggml_backend_dev_backend_reg(provider_device));
    if (!api.query_config || !api.session_create || !api.session_destroy) {
        printf("cache-single-gpu-auto: FAIL (incomplete cache API)\n");
        return false;
    }

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    ggml_backend_dev_memory(provider_device, &free_bytes, &total_bytes);
    if (free_bytes < 1024u * 1024 * 1024 + 64u * 1024 * 1024) {
        printf("cache-single-gpu-auto: SKIP (device free memory below the auto slab floor)\n");
        return true;
    }

    configure_cache(nullptr);
    set_env("GGML_CUDA_MOE_CACHE_MODE", "auto");
    set_env("GGML_CUDA_MOE_CACHE_BUDGET_MB", "2048");
    set_env("GGML_CUDA_MOE_CACHE_RESERVE_MB", "0");
    capture.clear();

    ggml_moe_cache_config config = {};
    if (!api.query_config(1, 2048, &config)) {
        printf("cache-single-gpu-auto: FAIL (query_config)\n");
        return false;
    }
    bool ok = config.min_devices == 1 &&
        config.minimum_slab_bytes == 1024u * 1024 * 1024;

    void * backends[] = { gpu, cpu };
    void * session = api.session_create(backends, 2, &config);
    ok &= session != nullptr;
    if (session) {
        constexpr int64_t shape_in = 1024;
        constexpr int64_t shape_out = 160;
        constexpr int64_t shape_experts = 64;
        const size_t expert_size =
            ggml_row_size(GGML_TYPE_Q4_0, shape_in) * shape_out;
        std::vector<uint8_t> data(expert_size * shape_experts);
        active_api().session_enter(session);
        const bool ready = wait_for_direct_pool(
                "blk.4.ffn_up_exps.weight", data.data(), expert_size,
                shape_in, shape_out, GGML_TYPE_Q4_0, shape_experts);
        const bool no_dormant_log =
            capture.get().find("automatic slab floor") == std::string::npos;
        active_api().session_leave(session);
        api.session_destroy(session);
        ok &= ready && no_dormant_log;
        if (!ready) {
            fprintf(stderr, "cache-single-gpu-auto: auto session did not engage\n%s\n",
                    capture.get().c_str());
        }
    }
    configure_cache(nullptr);
    printf("cache-single-gpu-auto: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

static bool run_admission_policy_once(
        ggml_backend_t gpu, ggml_backend_t cpu,
        int new_expert_misses, log_capture & capture) {
    configure_cache(nullptr);
    set_env("GGML_CUDA_MOE_CACHE_BUDGET_MB", "8");
    set_env("GGML_CUDA_MOE_CACHE_ADMIT_AFTER", "2");
    set_env("GGML_CUDA_MOE_CACHE_THROTTLE", "8");
    set_env("GGML_CUDA_MOE_CACHE_STATS", "0");
    capture.clear();

    void * session = create_direct_session(gpu, cpu);
    if (!session) {
        return false;
    }

    constexpr int64_t policy_n_in = 1024;
    constexpr int64_t policy_n_out = 206;
    constexpr int64_t policy_n_expert = 65;
    const size_t expert_size =
        ggml_row_size(GGML_TYPE_Q4_0, policy_n_in) * policy_n_out;
    std::vector<uint8_t> weights(expert_size * policy_n_expert);
    const char * name = "blk.5.ffn_up_exps.weight";

    active_api().session_enter(session);
    bool ok = wait_for_direct_pool(
            name, weights.data(), expert_size,
            policy_n_in, policy_n_out,
            GGML_TYPE_Q4_0, policy_n_expert);
    for (int32_t expert = 0; expert < 64 && ok; expert++) {
        if (direct_plan_one(
                name, weights.data(), expert_size,
                policy_n_in, policy_n_out,
                GGML_TYPE_Q4_0, policy_n_expert, expert) != 0) {
            ok = false;
            break;
        }
        if (expert == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        if (direct_plan_one(
                name, weights.data(), expert_size,
                policy_n_in, policy_n_out,
                GGML_TYPE_Q4_0, policy_n_expert, expert) != 0) {
            ok = false;
            break;
        }

        bool hit = false;
        for (int attempt = 0; attempt < 100; attempt++) {
            if (direct_plan_one(
                    name, weights.data(), expert_size,
                    policy_n_in, policy_n_out,
                    GGML_TYPE_Q4_0, policy_n_expert, expert) == 1) {
                hit = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        ok &= hit;
    }

    for (int miss = 0; miss < new_expert_misses && ok; miss++) {
        ok &= direct_plan_one(
                name, weights.data(), expert_size,
                policy_n_in, policy_n_out,
                GGML_TYPE_Q4_0, policy_n_expert, 64) == 0;
    }
    if (new_expert_misses == 8 && ok) {
        bool hit = false;
        for (int attempt = 0; attempt < 100; attempt++) {
            if (direct_plan_one(
                    name, weights.data(), expert_size,
                    policy_n_in, policy_n_out,
                    GGML_TYPE_Q4_0, policy_n_expert, 64) == 1) {
                hit = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        ok &= hit;
    }

    active_api().session_leave(session);
    active_api().session_destroy(session);
    const std::string log = capture.get();
    const long long expected_enqueued =
        new_expert_misses == 8 ? 65 : 64;
    const long long expected_evictions =
        new_expert_misses == 8 ? 1 : 0;
    return ok &&
        max_field_value(log, "slots=") == 64 &&
        max_field_value(log, "enqueued=") == expected_enqueued &&
        max_field_value(log, "evictions=") == expected_evictions;
}

static bool run_admission_policy(
        ggml_backend_t gpu, ggml_backend_t cpu,
        log_capture & capture) {
    const bool seven = run_admission_policy_once(
            gpu, cpu, 7, capture);
    const bool eight = run_admission_policy_once(
            gpu, cpu, 8, capture);
    const bool ok = seven && eight;
    printf("cache-admission-policy: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

static bool run_prompt_warm(
        ggml_backend_t gpu, ggml_backend_t cpu, ggml_tensor * weights,
        test_graph & graph, const std::vector<float> & reference, log_capture & capture) {
    configure_cache(nullptr);
    set_env("GGML_CUDA_MOE_CACHE_OVERLAP_CPU_ROWS", nullptr);
    capture.clear();
    const auto & api = active_api();
    void * session = create_direct_session(gpu, cpu);
    if (!session || !api.warm) {
        if (session) {
            api.session_destroy(session);
        }
        fprintf(stderr, "cache-prompt-warm: missing session or warm callback\n");
        return false;
    }
    const ggml_moe_cache_tensor_desc descriptor = {
        weights->name, weights->data, weights->nb[2], weights->ne[0], weights->ne[1], weights->ne[2], (int32_t) weights->type,
    };
    std::vector<uint64_t> usage(weights->ne[2], 0);
    usage[0] = 100;
    usage[1] = 50;
    const uint64_t * counts[] = {usage.data()};
    bool ok = true;
#if defined(__linux__)
    const char * previous_save = getenv("GGML_MOE_CACHE_PROFILE_SAVE");
    const bool had_save = previous_save != nullptr;
    const std::string saved_path = previous_save ? previous_save : "";
    std::string invalid_profile;
    if (const char * directory = getenv("GGML_MOE_CACHE_TEST_TMPDIR")) {
        const std::string path = std::string(directory) + "/moe-invalid-profile-XXXXXX";
        std::vector<char> filename(path.begin(), path.end());
        filename.push_back('\0');
        const int fd = mkstemp(filename.data());
        ok &= fd >= 0;
        if (fd >= 0) {
            close(fd);
            unlink(filename.data());
            invalid_profile = filename.data();
            set_env("GGML_MOE_CACHE_PROFILE_SAVE", invalid_profile.c_str());
        }
    }
#endif
    for (int field = 0; field < 5; ++field) {
        auto invalid = descriptor;
        switch (field) {
            case 0: invalid.name = nullptr; break;
            case 1: invalid.data = nullptr; break;
            case 2: invalid.n_in = 0; break;
            case 3: invalid.type = GGML_TYPE_F32; break;
            case 4: invalid.expert_size++; break;
        }
        ok &= api.warm(session, &invalid, counts, 1) == 0;
    }
#if defined(__linux__)
    if (!invalid_profile.empty()) {
        // Reject shapes before sampling weights or writing a profile.
        ok &= capture.get().find("[moe-cache] profile ") == std::string::npos;
        FILE * file = fopen(invalid_profile.c_str(), "rb");
        ok &= file == nullptr;
        if (file) {
            fclose(file);
        }
        remove(invalid_profile.c_str());
        remove((invalid_profile + ".tmp").c_str());
        set_env("GGML_MOE_CACHE_PROFILE_SAVE", had_save ? saved_path.c_str() : nullptr);
    }
#endif
    const uint64_t * missing_counts[] = {nullptr};
    ok &= api.warm(session, &descriptor, missing_counts, 1) == 0;
    ok &= api.warm(session, &descriptor, counts, 1) == 2;
    const std::string warm_log = capture.get();
    ok &= warm_log.find("hits=") == std::string::npos &&
        warm_log.find("cpu-overlap calibration:") == std::string::npos;
    api.session_enter(session);
    ok &= direct_plan_one(descriptor.name, descriptor.data, descriptor.expert_size,
            descriptor.n_in, descriptor.n_out, descriptor.type, descriptor.n_expert, 0) == 1;
    ok &= direct_plan_one(descriptor.name, descriptor.data, descriptor.expert_size,
            descriptor.n_in, descriptor.n_out, descriptor.type, descriptor.n_expert, 1) == 1;
    ok &= ggml_backend_graph_compute(cpu, graph.graph) == GGML_STATUS_SUCCESS;
    std::vector<float> actual(reference.size());
    ggml_backend_tensor_get(graph.out, actual.data(), 0, actual.size() * sizeof(float));
    ok &= compare_output(reference, actual, 5e-4);
    api.session_leave(session);
    api.session_destroy(session);
    ok &= max_field_value(capture.get(), "dispatch-fail=") == 0 &&
        max_field_value(capture.get(), "collect-fail=") == 0;
    printf("cache-prompt-warm: %s\n", ok ? "OK" : "FAIL");
    if (!ok) {
        fprintf(stderr, "%s", capture.get().c_str());
    }
    return ok;
}

static bool run_prompt_warm_ranking(ggml_backend_t gpu, ggml_backend_t cpu, log_capture & capture) {
    configure_cache(nullptr);
    set_env("GGML_CUDA_MOE_CACHE_BUDGET_MB", "8");
    set_env("GGML_CUDA_MOE_CACHE_THROTTLE", "8");
    capture.clear();
    const auto & api = active_api();
    void * session = create_direct_session(gpu, cpu);
    if (!session || !api.warm) {
        if (session) {
            api.session_destroy(session);
        }
        return false;
    }
    constexpr int64_t shape_in = 1024;
    constexpr int64_t shape_out = 206;
    constexpr int64_t experts = 65;
    const size_t expert_bytes = ggml_row_size(GGML_TYPE_Q4_0, shape_in) * shape_out;
    std::vector<uint8_t> data(expert_bytes * experts);
    std::vector<uint64_t> usage(experts);
    for (size_t expert = 0; expert < usage.size(); ++expert) {
        usage[expert] = expert + 1;
    }
    const ggml_moe_cache_tensor_desc descriptor = {
        "blk.5.ffn_up_exps.weight", data.data(), expert_bytes, shape_in, shape_out, experts, GGML_TYPE_Q4_0,
    };
    const uint64_t * counts[] = {usage.data()};
    bool ok = api.warm(session, &descriptor, counts, 1) == 64;
    ok &= api.warm(session, &descriptor, counts, 1) == 0;
    api.session_enter(session);
    // The hottest entries are at the end of the input, and must survive the excess cold entry.
    ok &= direct_plan_one(descriptor.name, data.data(), expert_bytes,
            shape_in, shape_out, GGML_TYPE_Q4_0, experts, 64) == 1;
    ok &= direct_plan_one(descriptor.name, data.data(), expert_bytes,
            shape_in, shape_out, GGML_TYPE_Q4_0, experts, 1) == 1;
    ok &= direct_plan_one(descriptor.name, data.data(), expert_bytes,
            shape_in, shape_out, GGML_TYPE_Q4_0, experts, 0) == 0;
    api.session_leave(session);
    api.session_destroy(session);
    const std::string log = capture.get();
    ok &= max_field_value(log, "slots=") == 64 && max_field_value(log, "evictions=") == 0;
    printf("cache-prompt-warm-ranking: %s\n", ok ? "OK" : "FAIL");
    if (!ok) {
        fprintf(stderr, "%s", log.c_str());
    }
    return ok;
}

static bool run_prompt_warm_decode_heat(ggml_backend_t gpu, ggml_backend_t cpu, log_capture & capture) {
    configure_cache(nullptr);
    set_env("GGML_CUDA_MOE_CACHE_BUDGET_MB", "8");
    const char * previous_hot = getenv("GGML_CUDA_MOE_CACHE_HOT_USES");
    const bool had_hot = previous_hot != nullptr;
    const std::string saved_hot = previous_hot ? previous_hot : "";
    set_env("GGML_CUDA_MOE_CACHE_HOT_USES", "4");
    capture.clear();
    const auto & api = active_api();
    void * session = create_direct_session(gpu, cpu);
    if (!session || !api.warm) {
        if (session) {
            api.session_destroy(session);
        }
        set_env("GGML_CUDA_MOE_CACHE_HOT_USES", had_hot ? saved_hot.c_str() : nullptr);
        return false;
    }
    constexpr int64_t shape_in = 1024, shape_out = 206, experts = 66;
    const size_t expert_bytes = ggml_row_size(GGML_TYPE_Q4_0, shape_in) * shape_out;
    std::vector<uint8_t> data(expert_bytes * experts);
    std::vector<uint64_t> usage(experts, 1000);
    usage[0] = usage[65] = 0;
    const ggml_moe_cache_tensor_desc descriptor = {
        "blk.5.ffn_up_exps.weight", data.data(), expert_bytes, shape_in, shape_out, experts, GGML_TYPE_Q4_0,
    };
    const uint64_t * counts[] = {usage.data()};
    bool ok = api.warm(session, &descriptor, counts, 1) == 64;
    api.session_enter(session);
    auto plan = [&](int32_t expert) {
        return direct_plan_one(descriptor.name, data.data(), expert_bytes,
                shape_in, shape_out, GGML_TYPE_Q4_0, experts, expert);
    };
    auto admit = [&](int32_t expert) {
        if (plan(expert) != 0) {
            return false;
        }
        for (int attempt = 0; attempt < 200; ++attempt) {
            if (plan(expert) == 1) {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return false;
    };
    ok &= admit(0);
    ok &= admit(65);
    // A first decode hit must survive the next admission despite large prompt counts.
    ok &= plan(0) == 1;
    // Actual decode hits can then mature the new expert beyond the hot threshold.
    for (int hit = 0; hit < 4; ++hit) {
        ok &= plan(0) == 1;
    }
    api.session_leave(session);
    api.session_destroy(session);
    const std::string log = capture.get();
    ok &= max_field_value(log, "slots=") == 64 &&
        max_field_value(log, "evictions=") == 2 && max_field_value(log, "heat=") == 0;
    set_env("GGML_CUDA_MOE_CACHE_HOT_USES", had_hot ? saved_hot.c_str() : nullptr);
    printf("cache-prompt-warm-decode-heat: %s\n", ok ? "OK" : "FAIL");
    if (!ok) {
        fprintf(stderr, "%s", log.c_str());
    }
    return ok;
}

static bool run_profile_persistence(ggml_backend_t gpu, ggml_backend_t cpu, log_capture & capture) {
#if defined(__linux__)
    const char * directory = getenv("GGML_MOE_CACHE_TEST_TMPDIR");
    if (!directory) {
        printf("cache-profile-persistence: SKIP (set GGML_MOE_CACHE_TEST_TMPDIR)\n");
        return true;
    }
    const auto & api = active_api();
    if (!api.warm || !api.trim) {
        return false;
    }
    configure_cache(nullptr);
    set_env("GGML_CUDA_MOE_CACHE_BUDGET_MB", "8");
    set_env("GGML_CUDA_MOE_CACHE_THROTTLE", "8");
    const std::string pattern = std::string(directory) + "/moe-profile-XXXXXX";
    std::vector<char> filename(pattern.begin(), pattern.end());
    filename.push_back('\0');
    const int fd = mkstemp(filename.data());
    if (fd < 0) {
        return false;
    }
    close(fd);
    const std::string path = filename.data();
    const char * previous_save = getenv("GGML_MOE_CACHE_PROFILE_SAVE");
    const bool had_save = previous_save != nullptr;
    const std::string saved_path = previous_save ? previous_save : "";
    const char * previous_load = getenv("GGML_MOE_CACHE_PROFILE_LOAD");
    const bool had_load = previous_load != nullptr;
    const std::string loaded_path = previous_load ? previous_load : "";
    const char * previous_blend = getenv("GGML_MOE_CACHE_PROFILE_BLEND_PCT");
    const bool had_blend = previous_blend != nullptr;
    const std::string saved_blend = previous_blend ? previous_blend : "";
    const char * previous_global = getenv("GGML_MOE_CACHE_PROFILE_GLOBAL");
    const bool had_global = previous_global != nullptr;
    const std::string saved_global = previous_global ? previous_global : "";
    set_env("GGML_MOE_CACHE_PROFILE_GLOBAL", nullptr);
    set_env("GGML_MOE_CACHE_PROFILE_BLEND_PCT", nullptr);
    set_env("GGML_MOE_CACHE_PROFILE_LOAD", nullptr);
    set_env("GGML_MOE_CACHE_PROFILE_SAVE", path.c_str());
    constexpr int64_t shape_in = 1024, shape_out = 206, experts = 65;
    const size_t expert_bytes = ggml_row_size(GGML_TYPE_Q4_0, shape_in) * shape_out;
    std::vector<uint8_t> data(expert_bytes * experts);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = (uint8_t)(i % 251);
    }
    std::vector<uint64_t> usage(experts);
    for (size_t e = 0; e < usage.size(); ++e) {
        usage[e] = e + 1;
    }
    const uint64_t * counts[] = {usage.data()};
    const ggml_moe_cache_tensor_desc descriptor = {
        "blk.5.ffn_up_exps.weight", data.data(), expert_bytes, shape_in, shape_out, experts, GGML_TYPE_Q4_0,
    };
    auto read_file = [&]() {
        std::vector<uint8_t> bytes;
        FILE * file = fopen(path.c_str(), "rb");
        if (file) {
            if (fseek(file, 0, SEEK_END) == 0) {
                const long size = ftell(file);
                if (size > 0 && size <= 1048576 && fseek(file, 0, SEEK_SET) == 0) {
                    bytes.resize((size_t)size);
                    if (fread(bytes.data(), 1, bytes.size(), file) != bytes.size()) {
                        bytes.clear();
                    }
                }
            }
            fclose(file);
        }
        return bytes;
    };
    auto write_file = [&](const std::vector<uint8_t> & bytes) {
        FILE * file = fopen(path.c_str(), "wb");
        if (!file) {
            return false;
        }
        const bool written = fwrite(bytes.data(), 1, bytes.size(), file) == bytes.size();
        return fclose(file) == 0 && written;
    };
    bool ok = true;
    std::vector<uint8_t> first;
    for (int repetition = 0; repetition < 2; ++repetition) {
        void * session = create_direct_session(gpu, cpu);
        if (!session) {
            ok = false;
            break;
        }
        ok &= api.warm(session, &descriptor, counts, 1) == 64;
        api.session_destroy(session);
        const auto bytes = read_file();
        ok &= !bytes.empty();
        if (repetition == 0) {
            first = bytes;
        } else {
            // The second save must reproduce the complete file, not just its counts.
            ok &= first == bytes;
        }
    }
    const auto original = read_file();
    ok &= !original.empty();
    set_env("GGML_MOE_CACHE_PROFILE_SAVE", nullptr);
    set_env("GGML_MOE_CACHE_PROFILE_LOAD", path.c_str());
    // A fresh allocation proves that profile identity does not depend on host addresses.
    std::vector<uint8_t> relocated = data;
    auto loaded = descriptor;
    loaded.data = relocated.data();
    void * reader = create_direct_session(gpu, cpu);
    if (reader) {
        capture.clear();
        ok &= api.warm(reader, &loaded, nullptr, 1) == 64;
        ok &= capture.get().find("profile loaded:") != std::string::npos;
        std::vector<uint64_t> zeros(experts, 0);
        const uint64_t * zero_counts[] = {zeros.data()};
        set_env("GGML_MOE_CACHE_PROFILE_SAVE", path.c_str());
        ok &= api.warm(reader, &loaded, zero_counts, 1) == 0;
        ok &= original == read_file();
        set_env("GGML_MOE_CACHE_PROFILE_SAVE", nullptr);
        api.session_enter(reader);
        ok &= direct_plan_one(loaded.name, loaded.data, expert_bytes,
                shape_in, shape_out, GGML_TYPE_Q4_0, experts, 64) == 1;
        ok &= direct_plan_one(loaded.name, loaded.data, expert_bytes,
                shape_in, shape_out, GGML_TYPE_Q4_0, experts, 1) == 1;
        ok &= direct_plan_one(loaded.name, loaded.data, expert_bytes,
                shape_in, shape_out, GGML_TYPE_Q4_0, experts, 0) == 0;
        api.session_leave(reader);
        api.session_destroy(reader);
        ok &= max_field_value(capture.get(), "slots=") == 64;
    } else {
        ok = false;
    }
    // A resumed session receives prompt counts before an explicit profile load.
    reader = create_direct_session(gpu, cpu);
    if (reader) {
        capture.clear();
        std::vector<uint64_t> zeros(experts, 0);
        const uint64_t * zero_counts[] = {zeros.data()};
        ok &= api.warm(reader, &loaded, zero_counts, 1) == 0;
        ok &= capture.get().find("profile loaded:") != std::string::npos;
        api.session_enter(reader);
        ok &= direct_plan_one(loaded.name, loaded.data, expert_bytes,
                shape_in, shape_out, GGML_TYPE_Q4_0, experts, 64) == 1;
        ok &= direct_plan_one(loaded.name, loaded.data, expert_bytes,
                shape_in, shape_out, GGML_TYPE_Q4_0, experts, 0) == 0;
        api.session_leave(reader);
        api.session_destroy(reader);
    } else { ok = false; }
    // Prompt and profile rank opposite ends of a full pool.
    std::vector<uint64_t> prompt_usage(experts);
    for (size_t e = 0; e < prompt_usage.size(); ++e) {
        // Different profile/prompt corpus sizes must not change the mixture.
        prompt_usage[e] = 1000 * (experts - e);
    }
    const uint64_t * prompt_counts[] = {prompt_usage.data()};
    for (const char * blend : {"", "0", "75", "100"}) {
        set_env("GGML_MOE_CACHE_PROFILE_BLEND_PCT", blend[0] ? blend : nullptr);
        reader = create_direct_session(gpu, cpu);
        bool blend_ok = reader != nullptr;
        if (reader) {
            capture.clear();
            const bool profile_wins = strcmp(blend, "75") == 0 || strcmp(blend, "100") == 0;
            blend_ok &= api.warm(reader, &loaded, prompt_counts, 1) == (profile_wins ? 0 : 1);
            const std::string warm_log = capture.get();
            blend_ok &= warm_log.find("profile loaded:") != std::string::npos;
            blend_ok &= warm_log.find("decode lookup counters unchanged") != std::string::npos;
            blend_ok &= warm_log.find("hits=") == std::string::npos;
            blend_ok &= warm_log.find("cpu-overlap calibration:") == std::string::npos;
            api.session_enter(reader);
            blend_ok &= direct_plan_one(loaded.name, loaded.data, expert_bytes,
                    shape_in, shape_out, GGML_TYPE_Q4_0, experts, profile_wins ? 64 : 0) == 1;
            api.session_leave(reader);
            if (strcmp(blend, "75") == 0) {
                blend_ok &= api.trim(reader, 2 * expert_bytes) == 2 * expert_bytes;
                api.session_enter(reader);
                blend_ok &= direct_plan_one(loaded.name, loaded.data, expert_bytes,
                        shape_in, shape_out, GGML_TYPE_Q4_0, experts, 64) == 1;
                blend_ok &= direct_plan_one(loaded.name, loaded.data, expert_bytes,
                        shape_in, shape_out, GGML_TYPE_Q4_0, experts, 1) == 0;
                blend_ok &= direct_plan_one(loaded.name, loaded.data, expert_bytes,
                        shape_in, shape_out, GGML_TYPE_Q4_0, experts, 2) == 0;
                api.session_leave(reader);
            }
            api.session_enter(reader);
            blend_ok &= direct_plan_one(loaded.name, loaded.data, expert_bytes,
                    shape_in, shape_out, GGML_TYPE_Q4_0, experts, profile_wins ? 0 : 64) == 0;
            api.session_leave(reader);
            api.session_destroy(reader);
            blend_ok &= original == read_file();
        }
        printf("cache-profile-prompt-blend pct=%s: %s\n", blend[0] ? blend : "unset", blend_ok ? "OK" : "FAIL");
        if (!blend_ok) {
            fprintf(stderr, "%s", capture.get().c_str());
        }
        ok &= blend_ok;
        set_env("GGML_MOE_CACHE_PROFILE_BLEND_PCT", nullptr);
    }
    // The same profile must load against an interleaved expert arena.
    std::vector<uint8_t> interleaved(expert_bytes * experts * 3, 0xa5);
    for (int64_t e = 0; e < experts; ++e) {
        memcpy(interleaved.data() + (size_t)e * expert_bytes * 3, data.data() + (size_t)e * expert_bytes, expert_bytes);
    }
    auto arena_descriptor = loaded;
    arena_descriptor.data = interleaved.data();
    arena_descriptor.expert_size = expert_bytes * 3;
    for (bool global : {false, true}) {
        set_env("GGML_MOE_CACHE_PROFILE_GLOBAL", global ? "1" : nullptr);
        void * arena_reader = create_direct_session(gpu, cpu);
        bool arena_ok = arena_reader != nullptr;
        if (arena_reader) {
            capture.clear();
            arena_ok &= api.warm(arena_reader, &arena_descriptor, nullptr, 1) == 64;
            arena_ok &= capture.get().find("profile loaded:") != std::string::npos;
            if (global) {
                arena_ok &= capture.get().find("global profile plan:") != std::string::npos;
                arena_ok &= api.warm(arena_reader, &arena_descriptor, prompt_counts, 1) == 0;
                arena_ok &= capture.get().find("global profile warm:") != std::string::npos;
                api.session_enter(arena_reader);
                arena_ok &= direct_plan_one(arena_descriptor.name, arena_descriptor.data,
                        arena_descriptor.expert_size, shape_in, shape_out, GGML_TYPE_Q4_0, experts, 64) == 1;
                arena_ok &= direct_plan_one(arena_descriptor.name, arena_descriptor.data,
                        arena_descriptor.expert_size, shape_in, shape_out, GGML_TYPE_Q4_0, experts, 0) == 0;
                api.session_leave(arena_reader);
            }
            api.session_destroy(arena_reader);
        }
        printf("cache-profile-arena global=%d: %s\n", (int)global, arena_ok ? "OK" : "FAIL");
        ok &= arena_ok;
    }
    set_env("GGML_MOE_CACHE_PROFILE_GLOBAL", nullptr);
    auto rejected = [&](const ggml_moe_cache_tensor_desc & candidate) {
        void * session = create_direct_session(gpu, cpu);
        if (!session) {
            return false;
        }
        capture.clear();
        const bool failed = api.warm(session, &candidate, nullptr, 1) == 0 &&
            capture.get().find("profile rejected:") != std::string::npos;
        api.session_destroy(session);
        return failed;
    };
    auto wrong_shape = loaded;
    wrong_shape.n_in /= 2;
    wrong_shape.n_out *= 2;
    ok &= rejected(wrong_shape);
    auto wrong_type = loaded;
    wrong_type.type = GGML_TYPE_IQ4_NL;
    ok &= rejected(wrong_type);
    relocated[0] ^= 1;
    ok &= rejected(loaded);
    relocated[0] ^= 1;
    if (!original.empty()) {
        auto corrupt = original;
        corrupt.back() ^= 1;
        ok &= write_file(corrupt) && rejected(loaded);
        auto truncated = original;
        truncated.pop_back();
        ok &= write_file(truncated) && rejected(loaded);
        auto extended = original;
        extended.push_back(0);
        ok &= write_file(extended) && rejected(loaded);
        ok &= write_file(original);
    }
    // Unobserved first-layer experts precede a hot later layer at startup.
    ggml_moe_cache_tensor_desc layers[] = {descriptor, descriptor};
    layers[0].name = "blk.0.ffn_up_exps.weight";
    layers[0].n_expert = 64;
    layers[1].data = data.data() + 64 * expert_bytes;
    layers[1].n_expert = 1;
    std::vector<uint64_t> early_counts(64, 0);
    const uint64_t later_count = 100;
    const uint64_t * layer_counts[] = {early_counts.data(), &later_count};
    set_env("GGML_MOE_CACHE_PROFILE_LOAD", nullptr);
    set_env("GGML_MOE_CACHE_PROFILE_SAVE", path.c_str());
    void * writer = create_direct_session(gpu, cpu);
    if (writer) {
        ok &= api.warm(writer, layers, layer_counts, 2) == 1;
        api.session_destroy(writer);
    } else { ok = false; }
    set_env("GGML_MOE_CACHE_PROFILE_SAVE", nullptr);
    set_env("GGML_MOE_CACHE_PROFILE_LOAD", path.c_str());
    reader = create_direct_session(gpu, cpu);
    if (reader) {
        ok &= api.warm(reader, layers, nullptr, 2) == 64;
        api.session_enter(reader);
        for (int32_t e : {0, 63}) {
            ok &= direct_plan_one(layers[0].name, layers[0].data, expert_bytes,
                    shape_in, shape_out, GGML_TYPE_Q4_0, 64, e) == 1;
        }
        ok &= direct_plan_one(layers[1].name, layers[1].data, expert_bytes,
                shape_in, shape_out, GGML_TYPE_Q4_0, 1, 0) == 0;
        api.session_leave(reader);
        api.session_destroy(reader);
    } else { ok = false; }
    remove(path.c_str());
    remove((path + ".tmp").c_str());
    set_env("GGML_MOE_CACHE_PROFILE_SAVE", had_save ? saved_path.c_str() : nullptr);
    set_env("GGML_MOE_CACHE_PROFILE_LOAD", had_load ? loaded_path.c_str() : nullptr);
    set_env("GGML_MOE_CACHE_PROFILE_BLEND_PCT", had_blend ? saved_blend.c_str() : nullptr);
    set_env("GGML_MOE_CACHE_PROFILE_GLOBAL", had_global ? saved_global.c_str() : nullptr);
    printf("cache-profile-persistence: %s\n", ok ? "OK" : "FAIL");
    return ok;
#else
    GGML_UNUSED(gpu);
    GGML_UNUSED(cpu);
    GGML_UNUSED(capture);
    printf("cache-profile-persistence: SKIP (requires Linux temporary files)\n");
    return true;
#endif
}

static bool run_prompt_cache_lending(ggml_backend_t gpu, ggml_backend_t cpu, log_capture & capture) {
    configure_cache(nullptr);
    set_env("GGML_CUDA_MOE_CACHE_BUDGET_MB", "8");
    capture.clear();
    const auto & api = active_api();
    void * session = create_direct_session(gpu, cpu);
    if (!session || !api.warm || !api.trim || !api.restore) {
        if (session) {
            api.session_destroy(session);
        }
        return false;
    }
    constexpr int64_t shape_in = 1024, shape_out = 206, experts = 65;
    const size_t expert_bytes = ggml_row_size(GGML_TYPE_Q4_0, shape_in) * shape_out;
    std::vector<uint8_t> data(expert_bytes * experts);
    std::vector<uint64_t> usage(experts);
    for (size_t expert = 0; expert < usage.size(); ++expert) {
        usage[expert] = expert + 1;
    }
    const ggml_moe_cache_tensor_desc descriptor = {
        "blk.5.ffn_up_exps.weight", data.data(), expert_bytes, shape_in, shape_out, experts, GGML_TYPE_Q4_0,
    };
    const uint64_t * counts[] = {usage.data()};
    bool ok = api.warm(session, &descriptor, counts, 1) == 64;
    ok &= api.trim(session, 2 * expert_bytes) == 2 * expert_bytes;
    api.session_enter(session);
    ok &= direct_plan_one(descriptor.name, data.data(), expert_bytes,
            shape_in, shape_out, GGML_TYPE_Q4_0, experts, 64) == 1;
    ok &= direct_plan_one(descriptor.name, data.data(), expert_bytes,
            shape_in, shape_out, GGML_TYPE_Q4_0, experts, 1) == 0;
    api.session_leave(session);
    ok &= api.restore(session) >= 2 * expert_bytes;
    ok &= api.warm(session, &descriptor, counts, 1) >= 2;
    // A new prompt's hottest expert must replace a cold resident without a saved profile.
    usage[0] = 1000;
    ok &= api.warm(session, &descriptor, counts, 1) == 1;
    api.session_enter(session);
    ok &= direct_plan_one(descriptor.name, data.data(), expert_bytes,
            shape_in, shape_out, GGML_TYPE_Q4_0, experts, 0) == 1;
    api.session_leave(session);
    api.session_destroy(session);
    printf("cache-prompt-lending: %s\n", ok ? "OK" : "FAIL");
    if (!ok) {
        fprintf(stderr, "%s", capture.get().c_str());
    }
    return ok;
}

// A profile must influence swap admission even when its predicted expert has
// never appeared in decode. Placement and swap ordering must preserve the math.
static bool run_profile_swap_prior(ggml_backend_t gpu, ggml_backend_t cpu, log_capture & capture) {
#if defined(__linux__) && defined(GGML_CUDA)
    const char * directory = getenv("GGML_MOE_CACHE_TEST_TMPDIR");
    if (!directory || !active_api().warm || !active_api().device_prepare) {
        fprintf(stderr, "cache-profile-swap-prior: missing temporary directory or device API\n");
        return false;
    }
    struct saved_environment {
        struct entry { const char * name; bool present; std::string value; };
        std::vector<entry> entries;
        saved_environment() {
            for (const char * name : {
                    "GGML_CUDA_MOE_CACHE", "GGML_CUDA_MOE_CACHE_MODE", "GGML_CUDA_MOE_CACHE_BUDGET_MB",
                    "GGML_CUDA_MOE_CACHE_RESERVE_MB", "GGML_CUDA_MOE_CACHE_MIN_EXPERT_KB",
                    "GGML_CUDA_MOE_CACHE_MAX_BATCH", "GGML_CUDA_MOE_CACHE_INSERTS",
                    "GGML_CUDA_MOE_CACHE_ADMIT_AFTER", "GGML_CUDA_MOE_CACHE_THROTTLE",
                    "GGML_CUDA_MOE_CACHE_QUEUE", "GGML_CUDA_MOE_CACHE_STATS", "GGML_CUDA_MOE_CACHE_NDEV",
                    "GGML_CUDA_MOE_CACHE_MIN_CC", "GGML_CUDA_MOE_CACHE_SERIAL_FILL",
                    "GGML_CUDA_MOE_CACHE_DEDICATED_MMV", "GGML_CUDA_MOE_CACHE_OVERLAP_CPU_ROWS",
                    "GGML_CUDA_MOE_CACHE_FAIL", "GGML_MOE_CACHE_DEVICE", "GGML_MOE_CACHE_DEVICE_STATS",
                    "GGML_MOE_CACHE_DEVICE_CAPTURE", "GGML_MOE_CACHE_PROFILE_SAVE",
                    "GGML_MOE_CACHE_PROFILE_LOAD", "GGML_MOE_CACHE_PROFILE_BLEND_PCT",
                    "GGML_MOE_CACHE_PROFILE_GLOBAL", "GGML_MOE_CACHE_PROFILE_PRIOR_PCT",
                    "GGML_MOE_CACHE_PROFILE_PRIOR_DECAY", "GGML_MOE_CACHE_SWAP",
                    "GGML_MOE_CACHE_SWAP_INTERVAL", "GGML_MOE_CACHE_SWAP_DECAY_PCT",
                    "GGML_MOE_CACHE_SWAP_MARGIN", "GGML_MOE_CACHE_TRACE"}) {
                const char * value = getenv(name);
                entries.push_back({name, value != nullptr, value ? value : ""});
            }
        }
        ~saved_environment() {
            for (const auto & item : entries) {
                set_env(item.name, item.present ? item.value.c_str() : nullptr);
            }
        }
    } saved;
    configure_cache(nullptr, "8");
    set_env("GGML_CUDA_MOE_CACHE_BUDGET_MB", "8");
    set_env("GGML_MOE_CACHE_PROFILE_LOAD", nullptr);
    set_env("GGML_MOE_CACHE_PROFILE_BLEND_PCT", "0");
    set_env("GGML_MOE_CACHE_PROFILE_GLOBAL", nullptr);
    set_env("GGML_MOE_CACHE_DEVICE", nullptr);
    set_env("GGML_MOE_CACHE_DEVICE_STATS", nullptr);
    set_env("GGML_MOE_CACHE_SWAP", nullptr);
    set_env("GGML_MOE_CACHE_TRACE", nullptr);
    const std::string pattern = std::string(directory) + "/moe-swap-profile-XXXXXX";
    std::vector<char> filename(pattern.begin(), pattern.end());
    filename.push_back('\0');
    const int fd = mkstemp(filename.data());
    if (fd < 0) { return false; }
    close(fd);
    set_env("GGML_MOE_CACHE_PROFILE_SAVE", filename.data());

    constexpr int64_t width = 1024, rows = 206, experts = 65;
    ggml_init_params params = {8 * ggml_tensor_overhead(), nullptr, true};
    ggml_context * ctx = ggml_init(params);
    ggml_context * inputs = ggml_init(params);
    auto * weights = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_0, width, rows, experts);
    ggml_set_name(weights, "blk.7.ffn_up_exps.weight");
    auto * acts = ggml_new_tensor_3d(inputs, GGML_TYPE_F32, width, 1, 1);
    auto * ids = ggml_new_tensor_2d(inputs, GGML_TYPE_I32, 2, 1);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    auto input_buffer = ggml_backend_alloc_ctx_tensors(inputs, cpu);
    bool ok = buffer && input_buffer;
    if (ok) {
        ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        std::vector<float> values(width * rows * experts);
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = 0.2f * std::sin((float)(i % 10007) * 0.013f);
        }
        std::vector<uint8_t> quantized(ggml_nbytes(weights));
        ok &= ggml_quantize_chunk(GGML_TYPE_Q4_0, values.data(), quantized.data(),
                0, rows * experts, width, nullptr) == quantized.size();
        ggml_backend_tensor_set(weights, quantized.data(), 0, quantized.size());
        values.resize(width);
        ggml_backend_tensor_set(acts, values.data(), 0, width * sizeof(float));
    }
    // Swap victims need exact mapped fallback sources, including initially hot experts.
    const bool registered = ok && cudaHostRegister(weights->data, ggml_nbytes(weights),
            cudaHostRegisterPortable | cudaHostRegisterMapped) == cudaSuccess;
    ok &= registered;
    if (!registered) { (void)cudaGetLastError(); }
    const ggml_moe_cache_tensor_desc descriptor = {weights->name, weights->data,
        weights->nb[2], width, rows, experts, GGML_TYPE_Q4_0};
    std::vector<uint64_t> profile(experts, 0), prompt(experts, 1000);
    profile[64] = 1000000;
    prompt[64] = 0;
    const uint64_t * profile_counts[] = {profile.data()};
    const uint64_t * prompt_counts[] = {prompt.data()};
    void * writer = ok ? create_direct_session(gpu, cpu) : nullptr;
    ok &= writer != nullptr;
    if (writer) {
        ok &= active_api().warm(writer, &descriptor, profile_counts, 1) == 1;
        active_api().session_destroy(writer);
    }
    set_env("GGML_MOE_CACHE_PROFILE_SAVE", nullptr);
    set_env("GGML_MOE_CACHE_PROFILE_LOAD", filename.data());
    set_env("GGML_MOE_CACHE_DEVICE", "1");
    set_env("GGML_MOE_CACHE_DEVICE_STATS", "1");
    set_env("GGML_MOE_CACHE_DEVICE_CAPTURE", nullptr);
    set_env("GGML_MOE_CACHE_SWAP", "1");
    set_env("GGML_MOE_CACHE_SWAP_INTERVAL", "1");
    set_env("GGML_MOE_CACHE_SWAP_DECAY_PCT", "95");
    set_env("GGML_MOE_CACHE_SWAP_MARGIN", "2");
    std::vector<std::vector<float>> control_outputs(2);
    for (int mode = 0; mode < 4 && ok; ++mode) {
        const int prior_pct = mode == 0 ? 0 : mode == 3 ? 200 : 100;
        set_env("GGML_MOE_CACHE_PROFILE_PRIOR_PCT", mode == 0 ? "0" : mode == 3 ? "200" : "100");
        set_env("GGML_MOE_CACHE_PROFILE_PRIOR_DECAY", mode == 2 ? "1" : nullptr);
        // A nondecaying empirical score needs finite first-snapshot scaling.
        // 200% makes its initial two routes exceed the admission margin.
        set_env("GGML_MOE_CACHE_SWAP_DECAY_PCT", mode == 3 ? "100" : "95");
        capture.clear();
        test_graph reference = make_graph(cpu, weights, acts, ids);
        test_graph candidate = make_graph(gpu, weights, acts, ids);
        ggml_backend_t backends[] = {gpu, cpu};
        auto scheduler = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, false);
        bool case_ok = scheduler && reference.buffer && candidate.buffer;
        if (case_ok) {
            case_ok &= ggml_backend_sched_moe_cache_warm(scheduler, &descriptor, prompt_counts, 1) >= 0;
            case_ok &= ggml_backend_sched_moe_cache_device_prepare(scheduler, &descriptor, 1) == 1;
            case_ok &= ggml_backend_sched_alloc_graph(scheduler, candidate.graph);
            for (int phase = 0; phase < 2 && case_ok; ++phase) {
                const int32_t route[] = {phase == 0 ? 0 : 64, phase == 0 ? 0 : 64};
                ggml_backend_tensor_set(ids, route, 0, sizeof(route));
                case_ok &= ggml_backend_graph_compute(cpu, reference.graph) == GGML_STATUS_SUCCESS;
                std::vector<float> expected(ggml_nelements(reference.out)), actual(expected.size()), first;
                ggml_backend_tensor_get(reference.out, expected.data(), 0, expected.size() * sizeof(float));
                for (int replay = 0; replay < (phase == 0 ? 16 : 1) && case_ok; ++replay) {
                    case_ok &= ggml_backend_sched_graph_compute(scheduler, candidate.graph) == GGML_STATUS_SUCCESS;
                    ggml_backend_tensor_get(candidate.out, actual.data(), 0, actual.size() * sizeof(float));
                    case_ok &= compare_output(expected, actual, 5e-4);
                    if (replay == 0) { first = actual; }
                    else { case_ok &= actual == first; }
                    // Drain private swap/readback streams deterministically before the next decision.
                    case_ok &= cudaDeviceSynchronize() == cudaSuccess;
                }
                if (mode == 0) { control_outputs[phase] = actual; }
                else { case_ok &= actual == control_outputs[phase]; }
            }
        }
        if (scheduler) { ggml_backend_sched_free(scheduler); }
        const std::string log = capture.get();
        case_ok &= log.find("newly_uncovered_bytes=0") != std::string::npos;
        // Final teardown counters include the sole probe of the never-routed
        // profile expert. Default cannot promote it before that first probe.
        case_ok &= max_field_value(log, "dispatches=") == 17;
        case_ok &= max_field_value(log, "misses=") == (mode == 0 ? 2 : 0);
        case_ok &= max_field_value(log, "hits=") == (mode == 0 ? 32 : 34);
        case_ok &= mode == 0 ? max_field_value(log, " swaps=") == 0 : max_field_value(log, " swaps=") >= 1;
        case_ok &= mode == 0 ? log.find("swap profile prior:") == std::string::npos :
            log.find(mode == 2 ? "mode=decaying-seed" : "mode=persistent") != std::string::npos;
        printf("cache-profile-swap-prior pct=%d mode=%s counter_decay=%d: %s\n", prior_pct,
                mode == 2 ? "decaying-seed" : "persistent", mode == 3 ? 100 : 95, case_ok ? "OK" : "FAIL");
        if (!case_ok) { fprintf(stderr, "%s", log.c_str()); }
        ok &= case_ok;
        free_graph(candidate);
        free_graph(reference);
    }
    if (registered) { ok &= cudaHostUnregister(weights->data) == cudaSuccess; }
    ggml_backend_buffer_free(input_buffer);
    ggml_backend_buffer_free(buffer);
    ggml_free(inputs);
    ggml_free(ctx);
    unlink(filename.data());
    return ok;
#else
    (void) gpu; (void) cpu; (void) capture;
    printf("cache-profile-swap-prior: SKIP (requires CUDA and temporary-file support)\n");
    return true;
#endif
}

static bool run_swap_whole(ggml_backend_t gpu, ggml_backend_t cpu, log_capture & capture, bool incompatible = false) {
#if defined(__linux__) && defined(GGML_CUDA)
    const char * directory = getenv("GGML_MOE_CACHE_TEST_TMPDIR");
    if (!directory || !active_api().warm || !active_api().device_prepare) {
        fprintf(stderr, "cache-swap-whole: missing temporary directory or device API (tmp=%d warm=%d prepare=%d)\n",
                directory != nullptr, active_api().warm != nullptr, active_api().device_prepare != nullptr);
        return false;
    }
    struct saved_env {
        struct entry { const char * name; bool present; std::string value; };
        std::vector<entry> entries;
        saved_env() {
            for (const char * name : {"GGML_CUDA_MOE_CACHE", "GGML_CUDA_MOE_CACHE_MODE",
                    "GGML_CUDA_MOE_CACHE_BUDGET_MB", "GGML_CUDA_MOE_CACHE_RESERVE_MB",
                    "GGML_CUDA_MOE_CACHE_MIN_EXPERT_KB", "GGML_CUDA_MOE_CACHE_MAX_BATCH",
                    "GGML_CUDA_MOE_CACHE_INSERTS", "GGML_CUDA_MOE_CACHE_ADMIT_AFTER",
                    "GGML_CUDA_MOE_CACHE_THROTTLE", "GGML_CUDA_MOE_CACHE_QUEUE",
                    "GGML_CUDA_MOE_CACHE_STATS", "GGML_CUDA_MOE_CACHE_NDEV", "GGML_CUDA_MOE_CACHE_MIN_CC",
                    "GGML_CUDA_MOE_CACHE_SERIAL_FILL", "GGML_CUDA_MOE_CACHE_DEDICATED_MMV",
                    "GGML_CUDA_MOE_CACHE_OVERLAP_CPU_ROWS", "GGML_CUDA_MOE_CACHE_FAIL",
                    "GGML_MOE_CACHE_DEVICE", "GGML_MOE_CACHE_DEVICE_STATS", "GGML_MOE_CACHE_DEVICE_CAPTURE",
                    "GGML_MOE_CACHE_PROFILE_SAVE", "GGML_MOE_CACHE_PROFILE_LOAD", "GGML_MOE_CACHE_PROFILE_GLOBAL",
                    "GGML_MOE_CACHE_PROFILE_PRIOR_PCT", "GGML_MOE_CACHE_SWAP", "GGML_MOE_CACHE_SWAP_WHOLE",
                    "GGML_MOE_CACHE_SWAP_LAYER", "GGML_MOE_CACHE_SWAP_MAX", "GGML_MOE_CACHE_SWAP_MB",
                    "GGML_MOE_CACHE_SWAP_INTERVAL", "GGML_MOE_CACHE_SWAP_DECAY_PCT",
                    "GGML_MOE_CACHE_SWAP_DECAY_AFTER", "GGML_MOE_CACHE_SWAP_HYST_PCT",
                    "GGML_MOE_CACHE_SWAP_MARGIN_MILLI", "GGML_MOE_CACHE_TRACE"}) {
                const char * value = getenv(name);
                entries.push_back({name, value != nullptr, value ? value : ""});
            }
        }
        ~saved_env() { for (const auto & e : entries) { set_env(e.name, e.present ? e.value.c_str() : nullptr); } }
    } saved;
    configure_cache(nullptr, "48");
    set_env("GGML_CUDA_MOE_CACHE_BUDGET_MB", "48");
    for (const char * name : {"GGML_MOE_CACHE_PROFILE_LOAD", "GGML_MOE_CACHE_PROFILE_GLOBAL",
            "GGML_MOE_CACHE_PROFILE_PRIOR_PCT", "GGML_MOE_CACHE_DEVICE", "GGML_MOE_CACHE_SWAP",
            "GGML_MOE_CACHE_DEVICE_CAPTURE", "GGML_MOE_CACHE_TRACE"}) { set_env(name, nullptr); }
    std::string pattern = std::string(directory) + "/moe-whole-XXXXXX";
    std::vector<char> filename(pattern.begin(), pattern.end());
    filename.push_back('\0');
    const int fd = mkstemp(filename.data());
    if (fd < 0) { std::perror("cache-swap-whole: create profile"); return false; }
    close(fd);
    constexpr int tensors = 6;
    constexpr int64_t width = 1024, rows = 1024, experts = 33;
    ggml_init_params params{64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true};
    auto * ctx = ggml_init(params);
    ggml_tensor * weights[tensors], * acts[tensors], * ids[tensors];
    ggml_moe_cache_tensor_desc desc[tensors];
    std::vector<uint64_t> counts[tensors];
    const uint64_t * count_ptrs[tensors];
    const char * kinds[] = {"up", "gate", "down"};
    for (int t = 0; t < tensors; ++t) {
        const int64_t n_in = incompatible && t >= 3 ? width / 2 : width;
        const int64_t n_out = incompatible && t >= 3 ? rows * 2 : rows;
        weights[t] = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_0, n_in, n_out, experts);
        ggml_format_name(weights[t], "blk.%d.ffn_%s_exps.weight", t / 3, kinds[t % 3]);
        acts[t] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_in, 1, 1);
        ids[t] = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 1);
        counts[t].resize(experts);
        for (int e = 0; e < experts; ++e) { counts[t][e] = t < 3 ? 1000 - e : 1; }
        count_ptrs[t] = counts[t].data();
    }
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    bool ok = buffer != nullptr;
    const char * setup_stage = "CPU tensor allocation";
    if (buffer) { ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS); }
    bool registered = false;
    setup_stage = ok ? "weight quantization" : setup_stage;
    for (int t = 0; t < tensors && ok; ++t) {
        const int64_t n_in = weights[t]->ne[0], n_out = weights[t]->ne[1];
        std::vector<float> values(n_in * n_out * experts);
        for (size_t i = 0; i < values.size(); ++i) { values[i] = 0.1f * std::sin((float) (i % 997) * 0.017f + t); }
        std::vector<uint8_t> quantized(ggml_nbytes(weights[t]));
        ok &= ggml_quantize_chunk(GGML_TYPE_Q4_0, values.data(), quantized.data(), 0, n_out * experts, n_in, nullptr) == quantized.size();
        ggml_backend_tensor_set(weights[t], quantized.data(), 0, quantized.size());
        values.resize(n_in);
        ggml_backend_tensor_set(acts[t], values.data(), 0, n_in * sizeof(float));
        const int32_t routes[] = {t < 3 ? 0 : 32, t < 3 ? 0 : 32};
        ggml_backend_tensor_set(ids[t], routes, 0, sizeof(routes));
        desc[t] = {weights[t]->name, weights[t]->data, weights[t]->nb[2], n_in, n_out, experts, GGML_TYPE_Q4_0};
    }
    if (ok) {
        setup_stage = "CUDA host registration";
        const cudaError_t error = cudaHostRegister(ggml_backend_buffer_get_base(buffer), ggml_backend_buffer_get_size(buffer),
                cudaHostRegisterPortable | cudaHostRegisterMapped);
        registered = error == cudaSuccess;
        ok &= registered;
        if (!registered) {
            fprintf(stderr, "cache-swap-whole: host registration failed: %s\n", cudaGetErrorString(error));
            (void) cudaGetLastError();
        }
    }
    set_env("GGML_MOE_CACHE_PROFILE_SAVE", filename.data());
    if (ok) { setup_stage = "profile session creation"; }
    auto writer = ok ? create_direct_session(gpu, cpu) : nullptr;
    ok &= writer != nullptr;
    if (writer) {
        setup_stage = "profile warm";
        const int queued = active_api().warm(writer, desc, count_ptrs, tensors);
        ok &= queued > 0;
        printf("cache-swap-whole profile writer incompatible=%d queued=%d\n", incompatible, queued);
        active_api().session_destroy(writer);
    }
    set_env("GGML_MOE_CACHE_PROFILE_SAVE", nullptr);
    set_env("GGML_MOE_CACHE_PROFILE_LOAD", filename.data());
    set_env("GGML_MOE_CACHE_PROFILE_GLOBAL", "1");
    set_env("GGML_MOE_CACHE_DEVICE", "1");
    set_env("GGML_MOE_CACHE_DEVICE_STATS", "1");
    set_env("GGML_MOE_CACHE_SWAP_INTERVAL", "1");
    set_env("GGML_MOE_CACHE_SWAP_DECAY_PCT", "70");
    set_env("GGML_MOE_CACHE_SWAP_DECAY_AFTER", "1");
    set_env("GGML_MOE_CACHE_SWAP_HYST_PCT", "0");
    set_env("GGML_MOE_CACHE_SWAP_MARGIN_MILLI", "1500");
    set_env("GGML_MOE_CACHE_SWAP_LAYER", "0");
    std::vector<std::vector<float>> control(tensors), expected(tensors);
    if (ok) { setup_stage = "CPU reference"; }
    for (int t = 0; t < tensors && ok; ++t) {
        auto reference = make_graph(cpu, weights[t], acts[t], ids[t]);
        ok &= reference.buffer && ggml_backend_graph_compute(cpu, reference.graph) == GGML_STATUS_SUCCESS;
        if (ok) {
            expected[t].resize(ggml_nelements(reference.out));
            ggml_backend_tensor_get(reference.out, expected[t].data(), 0, expected[t].size() * sizeof(float));
        }
        free_graph(reference);
    }
    if (!ok) {
        fprintf(stderr, "cache-swap-whole setup incompatible=%d failed at %s\n%s", incompatible, setup_stage, capture.get().c_str());
    }
    for (int mode = 0; mode < (incompatible ? 2 : 6) && ok; ++mode) {
        set_env("GGML_MOE_CACHE_SWAP", mode == 0 ? "0" : "1");
        set_env("GGML_MOE_CACHE_SWAP_WHOLE", "1");
        set_env("GGML_MOE_CACHE_SWAP_MAX", mode == 2 ? "2" : "3");
        set_env("GGML_MOE_CACHE_SWAP_MB", mode == 3 ? "1" : "2");
        set_env("GGML_MOE_CACHE_SWAP_LAYER", mode == 4 ? "1" : "0");
        desc[5].name = mode == 5 ? "blk.1.ffn_unknown_exps.weight" : weights[5]->name;
        capture.clear();
        auto * graph_ctx = ggml_init(params);
        auto * graph = ggml_new_graph(graph_ctx);
        ggml_tensor * outputs[tensors];
        for (int t = 0; t < tensors; ++t) {
            outputs[t] = ggml_mul_mat_id(graph_ctx, weights[t], acts[t], ids[t]);
            ggml_build_forward_expand(graph, outputs[t]);
        }
        ggml_backend_t backends[] = {gpu, cpu};
        auto scheduler = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, false);
        bool case_ok = scheduler != nullptr;
        if (case_ok) {
            case_ok &= ggml_backend_sched_moe_cache_warm(scheduler, desc, count_ptrs, tensors) >= 0;
            case_ok &= ggml_backend_sched_moe_cache_device_prepare(scheduler, desc, tensors) == 1;
            case_ok &= ggml_backend_sched_alloc_graph(scheduler, graph);
            for (int round = 0; round < 12 && case_ok; ++round) {
                case_ok &= ggml_backend_sched_graph_compute(scheduler, graph) == GGML_STATUS_SUCCESS;
                for (int t = 0; t < tensors; ++t) {
                    std::vector<float> actual(ggml_nelements(outputs[t]));
                    ggml_backend_tensor_get(outputs[t], actual.data(), 0, actual.size() * sizeof(float));
                    case_ok &= compare_output(expected[t], actual, 5e-4);
                    if (mode == 0 && round == 0) { control[t] = actual; }
                    else { case_ok &= actual == control[t]; }
                }
                case_ok &= cudaDeviceSynchronize() == cudaSuccess;
            }
        }
        if (scheduler) { ggml_backend_sched_free(scheduler); }
        const std::string log = capture.get();
        const long long swaps = max_field_value(log, " swaps=");
        if (mode == 0) { case_ok &= swaps <= 0; }
        else if (mode != 5) { case_ok &= mode == 1 && !incompatible ? swaps == 3 : swaps == 0; }
        if (mode != 0 && mode != 5) {
            case_ok &= log.find("whole swap initial:") != std::string::npos;
            case_ok &= log.find("partial=0") != std::string::npos;
            case_ok &= log.find("whole swap refused:") == std::string::npos;
        }
        if (mode == 5) { case_ok &= log.find("whole swap refused:") != std::string::npos; }
        if (mode == 1 && !incompatible) {
            case_ok &= max_field_value(log, "evictions=") >= 3;
            case_ok &= max_field_value(log, "hits=") > max_field_value(log, "misses=");
        }
        printf("cache-swap-whole incompatible=%d mode=%d: %s\n", incompatible, mode, case_ok ? "OK" : "FAIL");
        if (!case_ok) { fprintf(stderr, "%s", log.c_str()); }
        ok &= case_ok;
        ggml_free(graph_ctx);
    }
    if (registered) { ok &= cudaHostUnregister(ggml_backend_buffer_get_base(buffer)) == cudaSuccess; }
    if (buffer) { ggml_backend_buffer_free(buffer); }
    ggml_free(ctx);
    unlink(filename.data());
    return ok;
#else
    (void) gpu; (void) cpu; (void) capture; (void) incompatible;
    return false;
#endif
}

static bool run_prefetch_router(ggml_backend_t gpu, ggml_backend_t cpu) {
    bool ok = true;
    for (int tokens : {1, 4}) {
        for (int activation = 0; activation < 3; ++activation) {
            std::vector<int32_t> reference;
            for (auto backend : {cpu, gpu}) {
                ggml_init_params params = {64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true};
                auto * ctx = ggml_init(params);
                auto * router = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 512);
                auto * hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, tokens);
                auto * bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 512);
                auto * scores = ggml_mul_mat(ctx, router, hidden);
                if (activation == 0) {
                    scores = ggml_sigmoid(ctx, scores);
                } else if (activation == 1) {
                    scores = ggml_soft_max(ctx, scores);
                } else {
                    scores = ggml_sqrt(ctx, ggml_softplus(ctx, scores));
                }
                scores = ggml_add(ctx, scores, bias);
                if (tokens > 1) {
                    scores = ggml_cont(ctx, ggml_transpose(ctx, scores));
                    scores = ggml_pool_2d(ctx, scores, GGML_OP_POOL_MAX, tokens, 1, tokens, 1, 0, 0);
                    scores = ggml_reshape_1d(ctx, scores, 512);
                }
                auto * ids = ggml_cont(ctx, ggml_argsort_top_k(ctx, scores, 32));
                auto * graph = ggml_new_graph(ctx);
                ggml_build_forward_expand(graph, ids);
                auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
                if (!buffer) {
                    ggml_free(ctx);
                    return false;
                }
                std::vector<float> weights(256 * 512);
                for (int expert = 0; expert < 512; ++expert) {
                    std::fill(weights.begin() + expert * 256, weights.begin() + (expert + 1) * 256,
                              (expert - 256) / 256.0f);
                }
                std::vector<float> input(256 * tokens, 1.0f / 256.0f);
                if (tokens > 1) {
                    std::fill(input.end() - 256, input.end(), -2.0f / 256.0f);
                }
                std::vector<float> correction(512, 0.0f);
                ggml_backend_tensor_set(router, weights.data(), 0, weights.size() * sizeof(float));
                ggml_backend_tensor_set(hidden, input.data(), 0, input.size() * sizeof(float));
                ggml_backend_tensor_set(bias, correction.data(), 0, correction.size() * sizeof(float));
                ok &= ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
                std::vector<int32_t> actual(32);
                ggml_backend_tensor_get(ids, actual.data(), 0, actual.size() * sizeof(int32_t));
                for (int rank = 0; rank < 32; ++rank) {
                    ok &= actual[rank] == (tokens == 1 ? 511 - rank : rank);
                }
                if (backend == cpu) {
                    reference = actual;
                } else {
                    ok &= actual == reference;
                }
                ggml_backend_buffer_free(buffer);
                ggml_free(ctx);
            }
        }
    }
    printf("cache-prefetch-router: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

// Exercise actual copies, selected publication, retirement, and same-graph reuse.
static bool run_prefetch_staging(ggml_backend_t gpu, ggml_backend_t cpu,
        ggml_tensor * weights, log_capture & capture, bool device_mode, ggml_tensor * gate_weights = nullptr) {
    configure_cache(nullptr, "8");
    set_env("GGML_CUDA_MOE_CACHE_BUDGET_MB", "16");
    set_env("GGML_MOE_CACHE_DEVICE", device_mode ? "1" : nullptr);
    set_env("GGML_MOE_CACHE_DEVICE_CAPTURE", device_mode ? "1" : nullptr);
    set_env("GGML_MOE_CACHE_DEVICE_FUSED", gate_weights ? "1" : nullptr);
    set_env("GGML_MOE_CACHE_DEVICE_STATS", device_mode ? "1" : nullptr);
    capture.clear();
    const auto api = active_api();
    if (!api.prefetch || !api.prefetch_observe || !api.prefetch_configure) {
        return false;
    }
    void * session = create_direct_session(gpu, cpu);
    if (!session) {
        return false;
    }
    api.prefetch_configure(session, 1);
    ggml_moe_cache_tensor_desc descriptors[2] = {};
    const int banks = gate_weights ? 2 : 1;
    for (int i = 0; i < banks; ++i) {
        auto * w = i ? gate_weights : weights;
        descriptors[i] = {i ? "blk.1.ffn_gate_exps.weight" : "blk.1.ffn_up_exps.weight", w->data,
            w->nb[2], w->ne[0], w->ne[1], w->ne[2], (int32_t) w->type};
    }
    std::vector<uint64_t> usage(weights->ne[2], 0);
    for (int i = 0; i < 8; ++i) {
        usage[i] = 100;
    }
    const uint64_t * counts[] = {usage.data(), usage.data()};
    bool ok = api.warm(session, descriptors, counts, banks) == 8 * banks;
    if (device_mode) {
        ok &= api.device_prepare(session, descriptors, banks) == 1;
    }
    ggml_init_params params = {4 * ggml_tensor_overhead(), nullptr, true};
    ggml_context * host = ggml_init(params);
    ggml_context * card = ggml_init(params);
    ggml_tensor * host_acts = ggml_new_tensor_3d(host, GGML_TYPE_F32, weights->ne[0], 1, 1);
    ggml_tensor * host_ids = ggml_new_tensor_2d(host, GGML_TYPE_I32, 2, 1);
    ggml_tensor * card_acts = ggml_new_tensor_3d(card, GGML_TYPE_F32, weights->ne[0], 1, 1);
    ggml_tensor * card_ids = ggml_new_tensor_2d(card, GGML_TYPE_I32, 2, 1);
    auto host_buffer = ggml_backend_alloc_ctx_tensors(host, cpu);
    auto card_buffer = ggml_backend_alloc_ctx_tensors(card, gpu);
    std::vector<float> values(weights->ne[0], 0.125f);
    const int32_t route[] = {63, 63};
    ggml_backend_tensor_set(host_acts, values.data(), 0, values.size() * sizeof(float));
    ggml_backend_tensor_set(card_acts, values.data(), 0, values.size() * sizeof(float));
    ggml_backend_tensor_set(host_ids, route, 0, sizeof(route));
    ggml_backend_tensor_set(card_ids, route, 0, sizeof(route));
    test_graph reference = gate_weights ? make_fused_graph(cpu, weights, gate_weights, host_acts, host_ids) :
            make_graph(cpu, weights, host_acts, host_ids);
    test_graph candidate = gate_weights ? make_fused_graph(gpu, weights, gate_weights, card_acts, card_ids) :
            make_graph(device_mode ? gpu : cpu, weights,
                    device_mode ? card_acts : host_acts, device_mode ? card_ids : host_ids);
    auto fusion_count = (int64_t (*)(ggml_backend_t, const char *)) ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(ggml_backend_get_device(gpu)), "ggml_backend_cuda_fusion_count");
    const int64_t fused_before = gate_weights && fusion_count ? fusion_count(gpu, "mul_mat_glu") : 0;
    ok &= ggml_backend_graph_compute(cpu, reference.graph) == GGML_STATUS_SUCCESS;
    std::vector<float> expected(ggml_nelements(reference.out)), actual(expected.size()), first;
    ggml_backend_tensor_get(reference.out, expected.data(), 0, expected.size() * sizeof(float));
    auto compute = [&]() {
        ok &= ggml_backend_graph_compute(device_mode ? gpu : cpu, candidate.graph) == GGML_STATUS_SUCCESS;
        ggml_backend_tensor_get(candidate.out, actual.data(), 0, actual.size() * sizeof(float));
        ok &= compare_output(expected, actual, 5e-4);
        if (first.empty()) {
            first = actual;
        } else if (device_mode) {
            ok &= first == actual;
        }
    };
    for (int round = 0; round < 4 && ok; ++round) {
        api.session_enter(session);
        api.prefetch_observe(session, 0, nullptr, 0);
        // Establish a generous real interval; completion is polled, not assumed.
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        api.prefetch_observe(session, 1, nullptr, 0);
        api.prefetch_observe(session, 0, nullptr, 0);
        const int32_t ranked[] = {0, 63}; // Resident expert must not be copied.
        api.prefetch(session, descriptors, banks, ranked, 2, 0, 1, 0);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        bool completed = true;
        for (int i = 0; i < banks; ++i) {
            const auto & desc = descriptors[i];
            bool bank_completed = false;
            do {
                // Planning can promote a queued prediction to demand. Finish this
                // bank before promoting another, which would preempt its copy.
                bank_completed = direct_plan_one(desc.name, desc.data, desc.expert_size,
                        desc.n_in, desc.n_out, desc.type, desc.n_expert, 63) == 1;
                if (!bank_completed) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            } while (!bank_completed && std::chrono::steady_clock::now() < deadline);
            completed &= bank_completed;
        }
        ok &= completed;
        api.prefetch_observe(session, 1, nullptr, 0);
        api.prefetch_observe(session, 1, route, 2);
        compute();
        api.prefetch_observe(session, 2, nullptr, 0);
        if (!device_mode) {
            api.session_enter(nullptr);
        }
        compute(); // Device graph remains exact after overlay retirement.
        if (!device_mode) {
            api.session_leave(nullptr);
        }
        api.session_leave(session);
    }
    if (gate_weights) {
        ok &= fusion_count && fusion_count(gpu, "mul_mat_glu") > fused_before;
    }
    api.session_destroy(session);
    const std::string log = capture.get();
    ok &= count_occurrences(log, "[moe-prefetch] ") == 4 * banks;
    ok &= count_occurrences(log, "outcome=used") == 4 * banks;
    const size_t logical_bytes = ggml_row_size(weights->type, weights->ne[0]) * weights->ne[1];
    const std::string copy_bytes = " bytes=" + std::to_string(logical_bytes) + " issue_us=";
    ok &= count_occurrences(log, copy_bytes.c_str()) == 4 * banks;
    for (size_t pos = log.find("[moe-prefetch]"); pos != std::string::npos;
            pos = log.find("[moe-prefetch]", pos + 1)) {
        const size_t stamp = log.find("issue_us=", pos);
        long long issue = 0, start = 0, done = 0;
        ok &= stamp != std::string::npos && sscanf(log.c_str() + stamp,
                "issue_us=%lld start_us=%lld done_us=%lld", &issue, &start, &done) == 3 &&
                issue > 0 && start >= issue && done >= start;
    }
    if (device_mode) {
        ok &= max_field_value(log, "replays=") > 0;
    }
    ok &= log.find("expert=0 ") == std::string::npos;
    ok &= max_field_value(log, "fill-fail=") == 0;
    free_graph(candidate);
    free_graph(reference);
    ggml_backend_buffer_free(card_buffer);
    ggml_backend_buffer_free(host_buffer);
    ggml_free(card);
    ggml_free(host);
    set_env("GGML_MOE_CACHE_DEVICE", nullptr);
    set_env("GGML_MOE_CACHE_DEVICE_CAPTURE", nullptr);
    set_env("GGML_MOE_CACHE_DEVICE_FUSED", nullptr);
    set_env("GGML_MOE_CACHE_DEVICE_STATS", nullptr);
    printf("cache-prefetch-staging-%s: %s\n", gate_weights ? "fused" : device_mode ? "device" : "legacy", ok ? "OK" : "FAIL");
    if (!ok) {
        fprintf(stderr, "%s", log.c_str());
    }
    return ok;
}

static bool run_device_tables_case(ggml_backend_t gpu, ggml_backend_t cpu,
        ggml_tensor * weights, log_capture & capture, float * neighbor_output = nullptr) {
    if (!active_api().device_prepare || !active_api().device_supports) {
        return false;
    }
    bool ok = true;
    for (int route_mode : {0, 1, 2}) {
    for (int tokens : {1, 4}) {
        for (int channels : {1, 2}) {
            configure_cache(nullptr, "8");
            set_env("GGML_MOE_CACHE_DEVICE", "1");
            set_env("GGML_MOE_CACHE_DEVICE_STATS", "1");
            capture.clear();
            ggml_init_params params = {8 * ggml_tensor_overhead(), nullptr, true};
            ggml_context * inputs = ggml_init(params);
            ggml_tensor * acts = ggml_new_tensor_3d(inputs, GGML_TYPE_F32, weights->ne[0], channels, tokens);
            ggml_tensor * ids = ggml_new_tensor_2d(inputs, GGML_TYPE_I32, 2, tokens);
            ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(inputs, cpu);
            std::vector<float> values(ggml_nelements(acts));
            for (size_t i = 0; i < values.size(); ++i) {
                values[i] = 0.2f * std::sin((float)i * 0.013f);
            }
            // One warm expert, cold experts, and duplicate IDs in later tokens.
            std::vector<int32_t> routing(2 * tokens);
            for (int t = 0; t < tokens; ++t) {
                routing[2*t] = route_mode == 0 ? 0 : route_mode == 1 ? 1 : t % 3;
                routing[2*t + 1] = route_mode == 0 ? 0 : route_mode == 1 ? 2 : t == 2 ? routing[2*t] : (t + 1) % 3;
            }
            ggml_backend_tensor_set(acts, values.data(), 0, values.size() * sizeof(float));
            ggml_backend_tensor_set(ids, routing.data(), 0, routing.size() * sizeof(int32_t));
            test_graph reference = make_graph(cpu, weights, acts, ids);
            ok &= ggml_backend_graph_compute(cpu, reference.graph) == GGML_STATUS_SUCCESS;
            std::vector<float> expected(ggml_nelements(reference.out));
            ggml_backend_tensor_get(reference.out, expected.data(), 0, expected.size() * sizeof(float));
            const bool unsupported = weights->type == GGML_TYPE_IQ1_M;
            test_graph candidate = make_graph(unsupported ? cpu : gpu, weights, acts, ids);
            ggml_backend_t backends[] = {gpu, cpu};
            ggml_backend_sched_t scheduler = ggml_backend_sched_new(backends, nullptr, 2,
                    GGML_DEFAULT_GRAPH_SIZE, false, false);
            const ggml_moe_cache_tensor_desc descriptor = {weights->name, weights->data,
                weights->nb[2], weights->ne[0], weights->ne[1], weights->ne[2], (int32_t) weights->type};
            std::vector<uint64_t> usage(weights->ne[2], 0);
            usage[0] = 100;
            const uint64_t * counts[] = {usage.data()};
            ok &= ggml_backend_sched_moe_cache_warm(scheduler, &descriptor, counts, 1) == 1;
            const bool prepared = ggml_backend_sched_moe_cache_device_prepare(scheduler, &descriptor, 1) == 1;
            ok &= prepared != unsupported;
            if (unsupported) {
                // A failed opt-in must leave CPU scheduling and the legacy
                // 23-type cache path usable without a device pointer table.
                ok &= ggml_backend_sched_alloc_graph(scheduler, candidate.graph);
                ok &= ggml_backend_sched_get_tensor_backend(scheduler, candidate.out) == cpu;
                ok &= ggml_backend_sched_graph_compute(scheduler, candidate.graph) == GGML_STATUS_SUCCESS;
                std::vector<float> fallback(expected.size());
                ggml_backend_tensor_get(candidate.out, fallback.data(), 0, fallback.size() * sizeof(float));
                ok &= compare_output(expected, fallback, 5e-4);
                ok &= capture.get().find("IQ1_M mapped dispatch is not supported") != std::string::npos;
            } else if (prepared) {
                ok &= ggml_backend_sched_alloc_graph(scheduler, candidate.graph);
                ok &= ggml_backend_sched_get_tensor_backend(scheduler, candidate.out) == gpu;
                ok &= ggml_backend_sched_get_buffer_size(scheduler, gpu) < ggml_nbytes(weights);
                std::vector<float> first(expected.size()), second(expected.size());
                auto read_output = [&](std::vector<float> & output) {
                    void * destination = neighbor_output ? (void *)neighbor_output : output.data();
                    ggml_backend_tensor_get(candidate.out, destination, 0, output.size() * sizeof(float));
                    if (neighbor_output) {
                        memcpy(output.data(), neighbor_output, output.size() * sizeof(float));
                    }
                };
                ok &= ggml_backend_sched_graph_compute(scheduler, candidate.graph) == GGML_STATUS_SUCCESS;
                read_output(first);
                ok &= compare_output(expected, first, 5e-4);
                ok &= ggml_backend_sched_graph_compute(scheduler, candidate.graph) == GGML_STATUS_SUCCESS;
                read_output(second);
                ok &= first == second;
                // Unrelated activation writes must not invalidate expert tables.
                ggml_backend_tensor_set(acts, values.data(), 0, values.size() * sizeof(float));
                ok &= ggml_backend_sched_graph_compute(scheduler, candidate.graph) == GGML_STATUS_SUCCESS;
                read_output(second);
                ok &= first == second;
                active_api().invalidate(weights->data, ggml_nbytes(weights));
                ok &= ggml_backend_sched_graph_compute(scheduler, candidate.graph) == GGML_STATUS_FAILED;
            }
            ggml_backend_sched_free(scheduler);
            if (prepared) {
                const std::string log = capture.get();
                ok &= max_field_value(log, "dispatches=") == 3;
                if (neighbor_output) {
                    ok &= log.find("mapped registration boundary copy:") != std::string::npos;
                }
                if (route_mode == 0) {
                    ok &= max_field_value(log, "hits=") == 6 * tokens;
                    ok &= max_field_value(log, "misses=") == 0;
                } else if (route_mode == 1) {
                    ok &= max_field_value(log, "hits=") == 0;
                    ok &= max_field_value(log, "misses=") == 6 * tokens;
                } else {
                    ok &= max_field_value(log, "hits=") > 0;
                    ok &= max_field_value(log, "misses=") > 0;
                }
            }
            ggml_backend_buffer_free(candidate.buffer);
            ggml_free(candidate.ctx);
            ggml_backend_buffer_free(reference.buffer);
            ggml_free(reference.ctx);
            ggml_backend_buffer_free(buffer);
            ggml_free(inputs);
            if (!ok) {
                fprintf(stderr, "cache-device-tables failed type=%s tokens=%d channels=%d\n%s", ggml_type_name(weights->type), tokens, channels, capture.get().c_str());
            }
        }
    }
    }
    set_env("GGML_MOE_CACHE_DEVICE", nullptr);
    set_env("GGML_MOE_CACHE_DEVICE_STATS", nullptr);
    printf("cache-device-tables type=%s mode=%s: %s\n", ggml_type_name(weights->type),
            weights->type == GGML_TYPE_IQ1_M ? "rejected-with-CPU-fallback" : "mapped",
            ok ? "OK" : "FAIL");
    return ok;
}

// A full warm cache must admit a repeatedly routed miss without invalidating
// the graph. Then route its evicted victim to verify the mapped fallback.
static bool run_device_host_budget() {
    bool ok = ggml_cuda_moe_cache_host_file_credit(1000, 1000, 100, 50, 50, 100) == 700;
    ok &= ggml_cuda_moe_cache_host_file_credit(1000, 1000, 100, 0, 0, 0) == 900;
    ok &= ggml_cuda_moe_cache_host_file_credit(1000, 1000, 0, 100, 0, 0) == 900;
    ok &= ggml_cuda_moe_cache_host_file_credit(1000, 1000, 0, 0, 100, 0) == 900;
    ok &= ggml_cuda_moe_cache_host_file_credit(1000, 1000, 0, 0, 0, 100) == 900;
    ok &= ggml_cuda_moe_cache_host_file_credit(1000, 800, 900, 500, 500, 0) == 0;
    ok &= ggml_cuda_moe_cache_host_file_credit(1000, 100, 0, 0, 0, 40) == 60;
    ok &= ggml_cuda_moe_cache_host_file_credit(1000, 100, 0, 0, 0, 200) == 0;
    ok &= ggml_cuda_moe_cache_host_file_credit(UINT64_MAX, UINT64_MAX, UINT64_MAX, 1, 1, 1) == 0;
    struct range { uintptr_t begin, end; };
    const range experts[] = {{0, 40}, {32, 72}, {70, 90}, {90, 153}};
    std::vector<range> chunks;
    constexpr size_t cap = 64;
    for (const auto & expert : experts) {
        if (!chunks.empty() && ggml_cuda_moe_cache_host_merge(chunks.back().begin, chunks.back().end,
                    expert.begin, expert.end, cap)) {
            chunks.back().end = std::max(chunks.back().end, expert.end);
        } else {
            chunks.push_back(expert);
        }
    }
    for (const auto & chunk : chunks) {
        ok &= chunk.end - chunk.begin <= cap;
    }
    for (const auto & expert : experts) {
        ok &= std::any_of(chunks.begin(), chunks.end(), [&](const range & chunk) {
            return chunk.begin <= expert.begin && chunk.end >= expert.end;
        });
    }
    ok &= !ggml_cuda_moe_cache_host_merge(0, 40, 32, 72, cap);
    ok &= ggml_cuda_moe_cache_host_merge(0, 40, 32, 60, cap);
    printf("cache-device-host-budget: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

// Reading this process's pagemap reveals present bits without touching the
// advised mapping. PFN access is unnecessary. The mapping remains readable.
static int present_source_pages(const void * base, size_t bytes) {
    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    const int fd = open("/proc/self/pagemap", O_RDONLY);
    if (fd < 0 || !page || (uintptr_t) base % page || bytes % page) {
        if (fd >= 0) { close(fd); }
        return -1;
    }
    int present = 0;
    for (size_t offset = 0; offset < bytes; offset += page) {
        uint64_t entry = 0;
        const off_t position = (off_t) (((uintptr_t) base + offset) / page * sizeof(entry));
        if (pread(fd, &entry, sizeof(entry), position) != sizeof(entry)) {
            close(fd);
            return -1;
        }
        present += (entry >> 63) & 1;
    }
    close(fd);
    return present;
}

static bool run_device_adaptation(ggml_backend_t gpu, ggml_backend_t cpu, log_capture & capture, bool with_capture = false, bool with_source_reuse = false, bool with_tuning = false) {
    configure_cache(nullptr, "8");
    // Include the bounded runtime workspace as well as the 64 expert slots.
    set_env("GGML_CUDA_MOE_CACHE_BUDGET_MB", "8");
    set_env("GGML_MOE_CACHE_DEVICE", "1");
    set_env("GGML_MOE_CACHE_DEVICE_STATS", "1");
    set_env("GGML_MOE_CACHE_DEVICE_CAPTURE", with_capture ? "1" : nullptr);
    set_env("GGML_MOE_CACHE_DEVICE_ADAPT_INTERVAL", with_tuning ? "2" : nullptr);
    set_env("GGML_MOE_CACHE_DEVICE_ADAPT_MB", with_tuning ? "1" : nullptr);
    capture.clear();
    constexpr int64_t width = 1024, rows = 206;
    const int64_t experts = with_tuning ? 80 : 65;
    const int route_count = with_tuning ? 16 : 2;
    ggml_init_params params = {8 * ggml_tensor_overhead(), nullptr, true};
    ggml_context * ctx = ggml_init(params);
    ggml_context * inputs = ggml_init(params);
    ggml_tensor * weights = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_0, width, rows, experts);
    ggml_set_name(weights, "blk.7.ffn_up_exps.weight");
    ggml_tensor * acts = ggml_new_tensor_3d(inputs, GGML_TYPE_F32, width, 1, 1);
    ggml_tensor * ids = ggml_new_tensor_2d(inputs, GGML_TYPE_I32, route_count, 1);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    // Registered weight pages must not overlap scheduler input copies.
    ggml_backend_buffer_t input_buffer = ggml_backend_alloc_ctx_tensors(inputs, cpu);
    std::vector<float> values(width * rows * experts);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = 0.2f * std::sin((float)(i % 10007) * 0.013f);
    }
    std::vector<uint8_t> quantized(ggml_nbytes(weights));
    bool ok = ggml_quantize_chunk(GGML_TYPE_Q4_0, values.data(), quantized.data(),
            0, rows * experts, width, nullptr) == quantized.size();
    auto require = [&](bool passed, int line) {
        if (!passed) {
            fprintf(stderr, "device adaptation fixture assertion failed: line=%d capture=%d source_reuse=%d\n",
                    line, (int) with_capture, (int) with_source_reuse);
        }
        ok &= passed;
    };
    ggml_backend_tensor_set(weights, quantized.data(), 0, quantized.size());
    void * readonly_mapping = MAP_FAILED;
    if (with_source_reuse) {
        const char * directory = getenv("GGML_MOE_CACHE_TEST_TMPDIR");
        require((directory != nullptr), __LINE__);
        std::string path = std::string(directory ? directory : "/mnt/models/fnfast") + "/moe-source-reuse-XXXXXX";
        std::vector<char> filename(path.begin(), path.end());
        filename.push_back('\0');
        int fd = mkstemp(filename.data());
        size_t written = 0;
        while (fd >= 0 && written < quantized.size()) {
            const ssize_t n = write(fd, quantized.data() + written, quantized.size() - written);
            if (n <= 0) { break; }
            written += (size_t)n;
        }
        if (fd >= 0) { close(fd); }
        fd = written == quantized.size() ? open(filename.data(), O_RDONLY) : -1;
        readonly_mapping = fd >= 0 ? mmap(nullptr, quantized.size(), PROT_READ, MAP_SHARED, fd, 0) : MAP_FAILED;
        if (fd >= 0) { close(fd); }
        unlink(filename.data());
        require((readonly_mapping != MAP_FAILED), __LINE__);
        if (readonly_mapping != MAP_FAILED) {
            ggml_backend_buffer_free(buffer);
            weights->buffer = nullptr;
            weights->data = nullptr;
            buffer = ggml_backend_cpu_buffer_from_ptr(readonly_mapping, quantized.size());
            require((ggml_backend_tensor_alloc(buffer, weights, readonly_mapping) == GGML_STATUS_SUCCESS), __LINE__);
            ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }
    }
    values.resize(width);
    ggml_backend_tensor_set(acts, values.data(), 0, width * sizeof(float));
    std::vector<int32_t> initial_ids(route_count, 64);
    if (with_tuning) {
        for (int i = 0; i < route_count; ++i) { initial_ids[i] += i; }
    }
    ggml_backend_tensor_set(ids, initial_ids.data(), 0, initial_ids.size() * sizeof(int32_t));
    test_graph reference = make_graph(cpu, weights, acts, ids);
    test_graph candidate = make_graph(gpu, weights, acts, ids);
    ggml_backend_t backends[] = {gpu, cpu};
    ggml_backend_sched_t scheduler = ggml_backend_sched_new(backends, nullptr, 2,
            GGML_DEFAULT_GRAPH_SIZE, false, false);
    const ggml_moe_cache_tensor_desc descriptor = {weights->name, weights->data,
        weights->nb[2], width, rows, experts, GGML_TYPE_Q4_0};
    std::vector<uint64_t> usage(experts, 1000);
    for (int64_t expert = 64; expert < experts; ++expert) { usage[expert] = 0; }
    const uint64_t * counts[] = {usage.data()};
    require((ggml_backend_sched_moe_cache_warm(scheduler, &descriptor, counts, 1) == 64), __LINE__);
    const bool prepared = ggml_backend_sched_moe_cache_device_prepare(scheduler, &descriptor, 1) == 1;
    require((prepared), __LINE__);
    if (prepared) {
        require((ggml_backend_sched_alloc_graph(scheduler, candidate.graph)), __LINE__);
        for (int expert : {64, 0}) {
            std::vector<int32_t> route(route_count, expert);
            if (with_tuning) {
                for (int i = 0; i < route_count; ++i) { route[i] += i; }
            }
            ggml_backend_tensor_set(ids, route.data(), 0, route.size() * sizeof(int32_t));
            require((ggml_backend_graph_compute(cpu, reference.graph) == GGML_STATUS_SUCCESS), __LINE__);
            std::vector<float> expected(ggml_nelements(reference.out));
            ggml_backend_tensor_get(reference.out, expected.data(), 0, expected.size() * sizeof(float));
            std::vector<float> actual(expected.size()), first;
            void * source_probe = nullptr;
            size_t source_probe_bytes = 0;
            if (with_source_reuse && expert == 64) {
                const size_t page = (size_t) sysconf(_SC_PAGESIZE);
                const uintptr_t begin = (uintptr_t) weights->data + 64 * descriptor.expert_size;
                const uintptr_t aligned = (begin + page - 1) & ~(page - 1);
                const uintptr_t end = (begin + descriptor.expert_size) & ~(page - 1);
                source_probe = (void *) aligned;
                source_probe_bytes = end > aligned ? end - aligned : 0;
                require((source_probe_bytes > 0), __LINE__);
                require((present_source_pages(source_probe, source_probe_bytes) > 0), __LINE__);
                // The cold expert's exact staging copy already exists. Drop
                // only original full-page PTEs; preserve bytes/read permissions.
                require((madvise(source_probe, source_probe_bytes, MADV_DONTNEED) == 0), __LINE__);
                require((present_source_pages(source_probe, source_probe_bytes) == 0), __LINE__);
            }
            // Same graph, enough replays to fill a cold expert and subsequently
            // count it as resident. The second phase must evict another slot.
            for (int replay = 0; replay < 128; ++replay) {
                require((ggml_backend_sched_graph_compute(scheduler, candidate.graph) == GGML_STATUS_SUCCESS), __LINE__);
                ggml_backend_tensor_get(candidate.out, actual.data(), 0, actual.size() * sizeof(float));
                require((compare_output(expected, actual, 5e-4)), __LINE__);
                if (replay == 0) {
                    first = actual;
                } else {
                    require((actual == first), __LINE__);
                }
            }
            if (source_probe) {
                const int touched = present_source_pages(source_probe, source_probe_bytes);
                require((touched == 0), __LINE__);
                fprintf(stderr, "cache-device-source-reuse original_full_pages=%zu present_after_promotion=%d; mapping stays readable\n",
                        source_probe_bytes / (size_t) sysconf(_SC_PAGESIZE), touched);
            }
        }
    }
    if (prepared && with_source_reuse) {
        auto replay_exact = [&]() {
            // The scheduler substitutes its own input-copy tensors into a graph.
            // After reset, rebuild these original edges before allocating again;
            // otherwise updates to ids would leave the old device copy unchanged.
            candidate.out->src[0] = weights;
            candidate.out->src[1] = acts;
            candidate.out->src[2] = ids;
            const int32_t route[] = {64, 64};
            ggml_backend_tensor_set(ids, route, 0, sizeof(route));
            bool matched = ggml_backend_graph_compute(cpu, reference.graph) == GGML_STATUS_SUCCESS;
            std::vector<float> expected(ggml_nelements(reference.out)), actual(expected.size());
            ggml_backend_tensor_get(reference.out, expected.data(), 0, expected.size() * sizeof(float));
            matched &= ggml_backend_sched_graph_compute(scheduler, candidate.graph) == GGML_STATUS_SUCCESS;
            ggml_backend_tensor_get(candidate.out, actual.data(), 0, actual.size() * sizeof(float));
            return matched && compare_output(expected, actual, 5e-4);
        };
        // Reprepare the same cold set: keep exact readonly copies and make no
        // new copy, while invalidating the old captured graph generation.
        ggml_backend_sched_reset(scheduler);
        size_t checkpoint = capture.get().size();
        require((ggml_backend_sched_moe_cache_warm(scheduler, &descriptor, counts, 1) >= 0), __LINE__);
        require((ggml_backend_sched_moe_cache_device_prepare(scheduler, &descriptor, 1) == 1), __LINE__);
        std::string reuse = capture.get().substr(checkpoint);
        require((reuse.find("source mappings reuse: retained_ranges=0 ") == std::string::npos), __LINE__);
        require((reuse.find("source mappings setup: new_copy_bytes=0 new_registered_bytes=0") != std::string::npos), __LINE__);
        require((replay_exact()), __LINE__);
        // Borrow two slots. Keep mappings for the old cold expert and allocate
        // only newly uncovered misses; never preserve unused source ranges.
        ggml_backend_sched_reset(scheduler);
        checkpoint = capture.get().size();
        require((ggml_backend_sched_moe_cache_trim(scheduler, 2 * descriptor.expert_size) == 2 * descriptor.expert_size), __LINE__);
        require((ggml_backend_sched_moe_cache_device_prepare(scheduler, &descriptor, 1) == 1), __LINE__);
        const std::string borrowed = capture.get().substr(checkpoint);
        require((max_field_value(borrowed, "newly_uncovered_bytes=") > 0), __LINE__);
        require((max_field_value(borrowed, "newly_uncovered_bytes=") <= 2 * descriptor.expert_size + 8192), __LINE__);
        require((replay_exact()), __LINE__);
        ggml_backend_sched_reset(scheduler);
        checkpoint = capture.get().size();
        require((ggml_backend_sched_moe_cache_restore(scheduler) >= 2 * descriptor.expert_size), __LINE__);
        require((ggml_backend_sched_moe_cache_warm(scheduler, &descriptor, counts, 1) >= 0), __LINE__);
        require((ggml_backend_sched_moe_cache_device_prepare(scheduler, &descriptor, 1) == 1), __LINE__);
        const std::string returned = capture.get().substr(checkpoint);
        require((max_field_value(returned, "released_bytes=") > 0), __LINE__);
        require((returned.find("source mappings setup: new_copy_bytes=0 new_registered_bytes=0") != std::string::npos), __LINE__);
        require((replay_exact()), __LINE__);
        require((memcmp(weights->data, quantized.data(), quantized.size()) == 0), __LINE__);
        // A changed descriptor must discard old source provenance even if
        // the base address is reused. Restore the original descriptor before
        // computing again; no incorrectly shaped table is ever executed.
        ggml_backend_sched_reset(scheduler);
        (void)ggml_backend_sched_moe_cache_restore(scheduler);
        checkpoint = capture.get().size();
        auto changed = descriptor;
        changed.n_out--;
        require((ggml_backend_sched_moe_cache_device_prepare(scheduler, &changed, 1) == 1), __LINE__);
        require((capture.get().substr(checkpoint).find("retained source mapping identity changed;") != std::string::npos), __LINE__);
        ggml_backend_sched_reset(scheduler);
        (void)ggml_backend_sched_moe_cache_restore(scheduler);
        require((ggml_backend_sched_moe_cache_warm(scheduler, &descriptor, counts, 1) >= 0), __LINE__);
        require((ggml_backend_sched_moe_cache_device_prepare(scheduler, &descriptor, 1) == 1), __LINE__);
        require((replay_exact()), __LINE__);
        // Source mutation/teardown invalidation also applies between table
        // lifetimes, so a retained copy cannot outlive its original bank.
        ggml_backend_sched_reset(scheduler);
        (void)ggml_backend_sched_moe_cache_restore(scheduler);
        checkpoint = capture.get().size();
        active_api().invalidate(weights->data, ggml_nbytes(weights));
        require((capture.get().substr(checkpoint).find("retained source mapping invalidated:") != std::string::npos), __LINE__);
        require((ggml_backend_sched_moe_cache_warm(scheduler, &descriptor, counts, 1) >= 0), __LINE__);
        require((ggml_backend_sched_moe_cache_device_prepare(scheduler, &descriptor, 1) == 1), __LINE__);
        require((capture.get().substr(checkpoint).find("source mappings reuse: retained_ranges=0 ") != std::string::npos), __LINE__);
        require((replay_exact()), __LINE__);
    }
    if (prepared && with_tuning) {
        // Invalid/out-of-range values retain defaults; valid endpoints only
        // configure a cap and do not allocate that many bytes in this fixture.
        struct policy_case { const char * interval; const char * mb; uint64_t scopes; size_t bytes; };
        const policy_case policies[] = {
            {"0", "1025", 64, 64u << 20}, {"1025", "0", 64, 64u << 20},
            {"-1", "1junk", 64, 64u << 20}, {"1", "1024", 1, 1024u << 20},
            {"1024", "1", 1024, 1u << 20},
        };
        for (const auto & policy : policies) {
            ggml_backend_sched_reset(scheduler);
            const size_t checkpoint = capture.get().size();
            set_env("GGML_MOE_CACHE_DEVICE_ADAPT_INTERVAL", policy.interval);
            set_env("GGML_MOE_CACHE_DEVICE_ADAPT_MB", policy.mb);
            require((ggml_backend_sched_moe_cache_warm(scheduler, &descriptor, counts, 1) >= 0), __LINE__);
            require((ggml_backend_sched_moe_cache_device_prepare(scheduler, &descriptor, 1) == 1), __LINE__);
            const std::string policy_log = capture.get().substr(checkpoint);
            require((policy_log.find("device adaptation policy: interval_scopes=" + std::to_string(policy.scopes) +
                    " cap_bytes=" + std::to_string(policy.bytes) + ";") != std::string::npos), __LINE__);
        }
    }
    ggml_backend_sched_free(scheduler);
    const std::string log = capture.get();
    require((log.find(with_tuning ? "slots=64 entries=80" : "slots=64 entries=65") != std::string::npos), __LINE__);
    const size_t arena_log = log.find("device table arena bytes:");
    const size_t pool_log = log.find("pool[0]:");
    require((arena_log != std::string::npos && pool_log != std::string::npos && arena_log < pool_log), __LINE__);
    const char * shared_mtp = getenv("GGML_MOE_CACHE_MTP_SHARED");
    if (shared_mtp && strcmp(shared_mtp, "1") == 0 && arena_log != std::string::npos) {
        const std::string arena_line = log.substr(arena_log, log.find('\n', arena_log) - arena_log);
        require((max_field_value(arena_line, "entries=") >= 4 * (uint64_t) descriptor.n_expert), __LINE__);
    }
    require((log.find("failure=none policy=materialized-before-expert-pools") != std::string::npos), __LINE__);
    require((log.find("policy=actual-allocator-padding") != std::string::npos), __LINE__);
    if (with_tuning) {
        for (int scope : {2, 4, 130, 132}) {
            require((log.find("device adaptation: scopes=" + std::to_string(scope) +
                    " counter_bytes=640 fills=8 copied_bytes=949248 cap_bytes=1048576") != std::string::npos), __LINE__);
        }
        require((max_field_value(log, "hits=") == 4000), __LINE__);
        require((max_field_value(log, "misses=") == 96), __LINE__);
        // Every refill line must obey its configured one-MiB boundary cap.
        size_t pos = 0;
        while ((pos = log.find("device adaptation: scopes=", pos)) != std::string::npos) {
            const size_t end = log.find('\n', pos);
            const std::string line = log.substr(pos, end - pos);
            require((max_field_value(line, " copied_bytes=") <= 1048576), __LINE__);
            pos = end == std::string::npos ? log.size() : end + 1;
        }
    } else {
        require((log.find("fills=1 copied_bytes=118656") != std::string::npos), __LINE__);
        require((max_field_value(log, "hits=") == 256), __LINE__);
        require((max_field_value(log, "misses=") == 256), __LINE__);
    }
    // Device-only routes do not emit legacy host-node eviction statistics.
    // Two refills in a full pool, each retaining a new victim fallback, prove
    // both replacements without depending on those unrelated counters.
    require((count_field_at_least(log, " copied_bytes=", 118656) == (with_tuning ? 4 : 2)), __LINE__);
    require((max_field_value(log, "fallback_staging_bytes=") == (with_tuning ? 32 : 2) * 118656), __LINE__);
    require((max_field_value(log, "original_source_bytes=") == 0), __LINE__);
    require((max_field_value(log, "pinned_source_bytes=") == (with_tuning ? 8 : 1) * 118656), __LINE__);
    if (with_capture) {
        require((max_field_value(log, "captures=") >= 1), __LINE__);
        require((max_field_value(log, "replays=") > 64), __LINE__);
    }
    ggml_backend_buffer_free(candidate.buffer);
    ggml_free(candidate.ctx);
    ggml_backend_buffer_free(reference.buffer);
    ggml_free(reference.ctx);
    ggml_backend_buffer_free(input_buffer);
    ggml_free(inputs);
    ggml_backend_buffer_free(buffer);
    if (readonly_mapping != MAP_FAILED) { munmap(readonly_mapping, quantized.size()); }
    ggml_free(ctx);
    set_env("GGML_MOE_CACHE_DEVICE", nullptr);
    set_env("GGML_MOE_CACHE_DEVICE_STATS", nullptr);
    set_env("GGML_MOE_CACHE_DEVICE_CAPTURE", nullptr);
    set_env("GGML_MOE_CACHE_DEVICE_ADAPT_INTERVAL", nullptr);
    set_env("GGML_MOE_CACHE_DEVICE_ADAPT_MB", nullptr);
    printf("%s: %s\n", with_tuning ? "cache-device-adaptation-tuning" : with_source_reuse ? "cache-device-source-reuse" : with_capture ? "cache-device-capture" : "cache-device-adaptation", ok ? "OK" : "FAIL");
    if (!ok) {
        fprintf(stderr, "%s", log.c_str());
    }
    return ok;
}

#if defined(GGML_CUDA) && defined(__linux__)
static bool run_permanent_device_table(ggml_backend_t gpu, ggml_backend_t cpu, log_capture & capture) {
    configure_cache(nullptr, "8");
    set_env("GGML_CUDA_MOE_CACHE_BUDGET_MB", "8");
    set_env("GGML_MOE_CACHE_DEVICE", "1");
    set_env("GGML_MOE_CACHE_DEVICE_CAPTURE", "1");
    set_env("GGML_MOE_CACHE_DEVICE_STATS", "1");
    set_env("GGML_MOE_CACHE_DEVICE_ADAPT_INTERVAL", "2");
    set_env("GGML_MOE_CACHE_DEVICE_ADAPT_MB", "1");
    capture.clear();
    constexpr int64_t width = 1024, rows = 206, experts = 80, routes = 16;
    ggml_init_params params{8 * ggml_tensor_overhead(), nullptr, true};
    auto * ctx = ggml_init(params);
    auto * inputs = ggml_init(params);
    auto * weights = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_0, width, rows, experts);
    // An unnumbered fixture bank has no output-only last layer to exclude.
    ggml_set_name(weights, "permanent.ffn_up_exps.weight");
    auto * acts = ggml_new_tensor_3d(inputs, GGML_TYPE_F32, width, 1, 1);
    auto * ids = ggml_new_tensor_2d(inputs, GGML_TYPE_I32, routes, 1);
    const size_t page = (size_t)sysconf(_SC_PAGESIZE);
    const size_t mapping_bytes = (ggml_nbytes(weights) + page - 1) / page * page;
    void * mapping = mmap(nullptr, mapping_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mapping == MAP_FAILED) { ggml_free(inputs); ggml_free(ctx); return false; }
    auto * buffer = ggml_backend_cpu_buffer_from_ptr(mapping, mapping_bytes);
    const bool allocated = buffer && ggml_backend_tensor_alloc(buffer, weights, mapping) == GGML_STATUS_SUCCESS;
    auto * input_buffer = ggml_backend_alloc_ctx_tensors(inputs, cpu);
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    bool ok = allocated && input_buffer;
    auto require = [&](bool value, int line) {
        if (!value) { fprintf(stderr, "permanent table assertion failed: line=%d\n", line); }
        ok &= value;
    };
    std::vector<float> values(width * rows * experts);
    for (size_t i = 0; i < values.size(); ++i) { values[i] = 0.2f * std::sin((float)(i % 10007) * 0.013f); }
    std::vector<uint8_t> quantized(ggml_nbytes(weights));
    require(ggml_quantize_chunk(GGML_TYPE_Q4_0, values.data(), quantized.data(), 0, rows * experts, width, nullptr) == quantized.size(), __LINE__);
    ggml_backend_tensor_set(weights, quantized.data(), 0, quantized.size());
    ggml_backend_tensor_set(acts, values.data(), 0, width * sizeof(float));
    auto reference = make_graph(cpu, weights, acts, ids);
    auto candidate = make_graph(gpu, weights, acts, ids);
    std::vector<std::vector<float>> expected(2, std::vector<float>(rows * routes));
    for (int phase = 0; phase < 2; ++phase) {
        std::vector<int32_t> route(routes);
        for (int i = 0; i < routes; ++i) { route[i] = (phase ? 0 : 64) + i; }
        ggml_backend_tensor_set(ids, route.data(), 0, route.size() * sizeof(int32_t));
        require(ggml_backend_graph_compute(cpu, reference.graph) == GGML_STATUS_SUCCESS, __LINE__);
        ggml_backend_tensor_get(reference.out, expected[phase].data(), 0, expected[phase].size() * sizeof(float));
    }
    const ggml_moe_cache_tensor_desc descriptor{weights->name, weights->data, weights->nb[2], width, rows, experts, GGML_TYPE_Q4_0};
    uint64_t signature = UINT64_C(14695981039346656037);
    signature = resident_test_hash(signature, descriptor.name, strlen(descriptor.name) + 1);
    const uint64_t shape[] = {descriptor.expert_size, width, rows, experts, GGML_TYPE_Q4_0};
    signature = resident_test_hash(signature, shape, sizeof(shape));
    for (size_t j = 0; j < 64; ++j) {
        signature = resident_test_hash(signature, quantized.data() + ((quantized.size() - 64) / 63) * j, 64);
    }
    std::vector<uint64_t> usage(experts, 1);
    for (int i = 0; i < 9; ++i) { usage[i] = 10000; }
    const uint64_t checksum = resident_test_hash(UINT64_C(14695981039346656037), usage.data(), usage.size() * sizeof(uint64_t));
    const uint64_t header[] = {UINT64_C(0x31464f5250454f4d), 1, experts, signature, checksum};
    char path[] = "/tmp/llama-permanent-table-XXXXXX";
    int fd = mkstemp(path);
    FILE * file = fd >= 0 ? fdopen(fd, "wb") : nullptr;
    require(file && fwrite(header, sizeof(header), 1, file) == 1 && fwrite(usage.data(), sizeof(uint64_t), usage.size(), file) == usage.size(), __LINE__);
    if (file) { require(fclose(file) == 0, __LINE__); }
    const char * old_path = getenv("GGML_MOE_CACHE_PROFILE_LOAD");
    const std::string saved_path = old_path ? old_path : "";
    set_env("GGML_MOE_CACHE_PROFILE_LOAD", path);
    auto & api = active_api();
    void * resident = ok ? api.resident_create(ggml_backend_get_device(gpu), &descriptor, 1, 9 * descriptor.expert_size) : nullptr;
    require(resident != nullptr, __LINE__);
    // The permanent prefix ends partway through a cold registered page. A direct
    // D2H restore to its final expert would cross mixed registration boundaries.
    const size_t hole_bytes = 9 * descriptor.expert_size / page * page;
    auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(gpu));
    auto pin = (bool (*)(void *, size_t))ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_host_pin");
    auto unpin = (void (*)(void *))ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_host_unpin");
    const bool registered = pin && unpin && pin((char *)mapping + hole_bytes, mapping_bytes - hole_bytes);
    require(registered, __LINE__);
    struct restore_state { void * hole; size_t bytes; int count; bool registered; bool (*pin)(void *, size_t); } state{mapping, hole_bytes, 0, false, pin};
    ggml_backend_t backends[] = {gpu, cpu};
    auto scheduler = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, false);
    std::vector<uint64_t> warm(experts, 1000);
    for (int i = 64; i < experts; ++i) { warm[i] = 0; }
    const uint64_t * counts[] = {warm.data()};
    if (resident && registered) {
        const size_t count = api.resident_spans(resident, nullptr, 0);
        require(count == 9, __LINE__);
        std::vector<ggml_moe_resident_span> spans(count);
        api.resident_spans(resident, spans.data(), count);
        api.resident_dropped(resident, +[](void * p) {
            auto & restored = *(restore_state *)p;
            restored.registered = restored.pin(restored.hole, restored.bytes);
            ++restored.count;
            return restored.registered ? 1 : 0;
        }, &state);
        for (const auto & span : spans) { memset(const_cast<void *>(span.host), 0, span.size); }
        require(madvise(mapping, hole_bytes, MADV_DONTNEED) == 0, __LINE__);
        std::vector<unsigned char> pages(hole_bytes / page);
        require(mincore(mapping, hole_bytes, pages.data()) == 0, __LINE__);
        for (unsigned char present : pages) { require(!(present & 1), __LINE__); }
        auto prepare = [&] {
            candidate.out->src[0] = weights;
            candidate.out->src[1] = acts;
            candidate.out->src[2] = ids;
            require(ggml_backend_sched_moe_cache_warm(scheduler, &descriptor, counts, 1) >= 0, __LINE__);
            require(ggml_backend_sched_moe_cache_device_prepare(scheduler, &descriptor, 1) == 1, __LINE__);
            require(ggml_backend_sched_alloc_graph(scheduler, candidate.graph), __LINE__);
            require(ggml_backend_sched_get_tensor_backend(scheduler, candidate.out) == gpu, __LINE__);
        };
        auto replay = [&](int phase, int iterations, int expected_restores = 0) {
            std::vector<int32_t> route(routes);
            for (int i = 0; i < routes; ++i) { route[i] = (phase ? 0 : 64) + i; }
            ggml_backend_tensor_set(ids, route.data(), 0, route.size() * sizeof(int32_t));
            std::vector<float> actual(expected[phase].size());
            for (int i = 0; i < iterations; ++i) {
                require(ggml_backend_sched_graph_compute(scheduler, candidate.graph) == GGML_STATUS_SUCCESS, __LINE__);
                ggml_backend_tensor_get(candidate.out, actual.data(), 0, actual.size() * sizeof(float));
                require(compare_output(expected[phase], actual, 5e-4), __LINE__);
            }
            require(state.count == expected_restores, __LINE__);
        };
        prepare();
        replay(0, 128);
        replay(1, 128);
        require(max_field_value(capture.get(), " copied_bytes=") > 0, __LINE__);
        ggml_backend_sched_reset(scheduler);
        require(ggml_backend_sched_moe_cache_trim(scheduler, 2 * descriptor.expert_size) > 0, __LINE__);
        prepare(); replay(1, 4);
        ggml_backend_sched_reset(scheduler);
        require(ggml_backend_sched_moe_cache_restore(scheduler) > 0, __LINE__);
        prepare(); replay(1, 4);
        ggml_backend_sched_reset(scheduler);
        require(ggml_backend_sched_moe_cache_suspend(scheduler), __LINE__);
        ggml_backend_sched_moe_cache_resume(scheduler);
        prepare(); replay(1, 4);
        require(ggml_moe_cache_resident_lookup(weights->data, descriptor.expert_size, ggml_backend_get_device(gpu)) != nullptr, __LINE__);
        require(state.count == 0, __LINE__);
        require(max_field_value(capture.get(), "replays=") > 64, __LINE__);
        // A generic fallback copies the whole object, crossing the separately
        // registered cold range and restored hole. Expert-sized D2D checks alone
        // do not cover this multi-turn path.
        auto * upload_ctx = ggml_init(params);
        auto * uploaded = ggml_dup_tensor(upload_ctx, weights);
        auto * upload_buffer = ggml_backend_alloc_ctx_tensors(upload_ctx, gpu);
        require(upload_buffer != nullptr, __LINE__);
        if (upload_buffer) {
            auto raw_upload = [&](const char * label) {
                const cudaError_t direct = cudaMemcpyAsync(uploaded->data, weights->data, quantized.size(),
                        cudaMemcpyHostToDevice, cudaStreamPerThread);
                printf("permanent-%s-whole-object-raw-H2D: %s\n", label, cudaGetErrorString(direct));
                require(direct == cudaSuccess || direct == cudaErrorInvalidValue, __LINE__);
                if (direct == cudaSuccess) { require(cudaStreamSynchronize(cudaStreamPerThread) == cudaSuccess, __LINE__); }
                else { (void)cudaGetLastError(); }
            };
            const char * old_upload = getenv("GGML_MOE_RESIDENT_UPLOAD");
            const std::string saved_upload = old_upload ? old_upload : "";
            set_env("GGML_MOE_RESIDENT_UPLOAD", "1");
            std::vector<uint8_t> copied(quantized.size());
            ggml_backend_tensor_set_async(gpu, uploaded, weights->data, 0, quantized.size());
            ggml_backend_synchronize(gpu);
            ggml_backend_tensor_get(uploaded, copied.data(), 0, copied.size());
            require(copied == quantized && state.count == 0, __LINE__);
            require(mincore(mapping, hole_bytes, pages.data()) == 0, __LINE__);
            for (unsigned char present : pages) { require(!(present & 1), __LINE__); }
            ggml_backend_tensor_set_async(gpu, uploaded, (const char *)weights->data + 37, 37, quantized.size() - 74);
            ggml_backend_synchronize(gpu);
            ggml_backend_tensor_get(uploaded, copied.data(), 0, copied.size());
            require(copied == quantized && state.count == 0, __LINE__);
            ggml_backend_tensor_copy_async(cpu, gpu, weights, uploaded);
            ggml_backend_synchronize(gpu);
            ggml_backend_tensor_get(uploaded, copied.data(), 0, copied.size());
            require(copied == quantized && state.count == 0, __LINE__);
            auto * copy_ctx = ggml_init({8 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true});
            auto * duplicate = ggml_dup(copy_ctx, weights);
            auto * copy_graph = ggml_new_graph(copy_ctx);
            ggml_build_forward_expand(copy_graph, duplicate);
            auto copy_sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, false);
            ggml_backend_sched_set_tensor_backend(copy_sched, duplicate, gpu);
            require(ggml_backend_sched_alloc_graph(copy_sched, copy_graph), __LINE__);
            require(ggml_backend_sched_get_tensor_backend(copy_sched, duplicate) == gpu, __LINE__);
            require(ggml_backend_sched_graph_compute(copy_sched, copy_graph) == GGML_STATUS_SUCCESS, __LINE__);
            ggml_backend_tensor_get(duplicate, copied.data(), 0, copied.size());
            require(copied == quantized && state.count == 0, __LINE__);
            ggml_backend_sched_free(copy_sched);
            ggml_free(copy_ctx);
            require(mincore(mapping, hole_bytes, pages.data()) == 0, __LINE__);
            for (unsigned char present : pages) { require(!(present & 1), __LINE__); }
            // Keep the original fallback coverage independent of the option.
            set_env("GGML_MOE_RESIDENT_UPLOAD", "0");
            raw_upload("sparse");
            // This call must restore the original host bytes before enqueueing
            // the full-object H2D copy, just as the multi-turn scheduler does.
            ggml_backend_tensor_set_async(gpu, uploaded, weights->data, 0, quantized.size());
            ggml_backend_synchronize(gpu);
            require(state.count == 1 && memcmp(weights->data, quantized.data(), quantized.size()) == 0, __LINE__);
            ggml_backend_tensor_get(uploaded, copied.data(), 0, copied.size());
            require(copied == quantized, __LINE__);
            raw_upload("restored");
            // Non-page-aligned offsets must split at registration boundaries,
            // not at arbitrary fixed chunk lengths.
            ggml_backend_tensor_set_async(gpu, uploaded, (const char *)weights->data + 37, 37, quantized.size() - 74);
            ggml_backend_synchronize(gpu);
            ggml_backend_tensor_get(uploaded, copied.data(), 0, copied.size());
            require(copied == quantized && state.count == 1, __LINE__);
            ggml_backend_tensor_set(uploaded, weights->data, 0, quantized.size());
            ggml_backend_tensor_get(uploaded, copied.data(), 0, copied.size());
            require(copied == quantized, __LINE__);
            // Both 2D entry points must preserve each row's source/destination
            // strides, including a row crossing the restored/cold boundary.
            const size_t source = hole_bytes - page + 37;
            const size_t width_bytes = 2 * page + 11, height = 3;
            const size_t source_stride = width_bytes + 127, target_stride = width_bytes + 97;
            std::vector<uint8_t> expected_upload = quantized;
            for (size_t row = 0; row < height; ++row) {
                memcpy(expected_upload.data() + 13 + row * target_stride,
                        quantized.data() + source + row * source_stride, width_bytes);
            }
            ggml_backend_tensor_set_2d_async(gpu, uploaded, (const char *)weights->data + source,
                    13, width_bytes, height, target_stride, source_stride);
            ggml_backend_synchronize(gpu);
            ggml_backend_tensor_get(uploaded, copied.data(), 0, copied.size());
            require(copied == expected_upload, __LINE__);
            set_env("GGML_MOE_RESIDENT_UPLOAD", old_upload ? saved_upload.c_str() : nullptr);
            for (size_t row = 0; row < height; ++row) {
                memcpy(expected_upload.data() + 19 + row * target_stride,
                        quantized.data() + source + row * source_stride, width_bytes);
            }
            ggml_backend_tensor_set_2d(uploaded, (const char *)weights->data + source,
                    19, width_bytes, height, target_stride, source_stride);
            ggml_backend_tensor_get(uploaded, copied.data(), 0, copied.size());
            require(copied == expected_upload, __LINE__);
        }
        ggml_backend_buffer_free(upload_buffer);
        ggml_free(upload_ctx);
        // Generic host restoration must leave already published cold aliases
        // and permanent addresses usable by the existing graph as well.
        replay(0, 4, 1);
        replay(1, 4, 1);
    }
    ggml_backend_sched_free(scheduler);
    if (resident) { api.resident_destroy(resident); }
    if (registered) { unpin((char *)mapping + hole_bytes); }
    if (state.registered) { unpin(mapping); }
    set_env("GGML_MOE_CACHE_PROFILE_LOAD", old_path ? saved_path.c_str() : nullptr);
    unlink(path);
    free_graph(candidate); free_graph(reference);
    ggml_backend_buffer_free(input_buffer); ggml_free(inputs);
    ggml_backend_buffer_free(buffer); ggml_free(ctx);
    require(munmap(mapping, mapping_bytes) == 0, __LINE__);
    for (const char * key : {"GGML_MOE_CACHE_DEVICE", "GGML_MOE_CACHE_DEVICE_CAPTURE", "GGML_MOE_CACHE_DEVICE_STATS", "GGML_MOE_CACHE_DEVICE_ADAPT_INTERVAL", "GGML_MOE_CACHE_DEVICE_ADAPT_MB"}) { set_env(key, nullptr); }
    printf("permanent-device-table-adapt-trim-suspend: %s\n", ok ? "OK" : "FAIL");
    if (!ok) { fprintf(stderr, "%s", capture.get().c_str()); }
    return ok;
}
#endif

static bool run_device_adaptation_starvation(ggml_backend_t gpu, ggml_backend_t cpu, log_capture & capture) {
    struct saved_env {
        const char * name;
        std::string value;
        bool present;
    };
    std::vector<saved_env> saved;
    for (const char * name : {"GGML_CUDA_MOE_CACHE_BUDGET_MB", "GGML_MOE_CACHE_DEVICE",
            "GGML_MOE_CACHE_DEVICE_STATS", "GGML_MOE_CACHE_DEVICE_CAPTURE",
            "GGML_MOE_CACHE_DEVICE_ADAPT_INTERVAL", "GGML_MOE_CACHE_DEVICE_ADAPT_MB",
            "GGML_MOE_CACHE_PROFILE_LOAD", "GGML_MOE_CACHE_PROFILE_SAVE"}) {
        const char * value = getenv(name);
        saved.push_back({name, value ? value : "", value != nullptr});
    }
    configure_cache(nullptr, "8");
    set_env("GGML_CUDA_MOE_CACHE_BUDGET_MB", "16");
    set_env("GGML_MOE_CACHE_DEVICE", "1");
    set_env("GGML_MOE_CACHE_DEVICE_STATS", "1");
    set_env("GGML_MOE_CACHE_DEVICE_CAPTURE", nullptr);
    set_env("GGML_MOE_CACHE_DEVICE_ADAPT_INTERVAL", "1");
    set_env("GGML_MOE_CACHE_DEVICE_ADAPT_MB", "1");
    set_env("GGML_MOE_CACHE_PROFILE_LOAD", nullptr);
    set_env("GGML_MOE_CACHE_PROFILE_SAVE", nullptr);
    capture.clear();
    const auto & api = active_api();
    constexpr int64_t width = 1024, banks = 2;
    const int64_t rows[] = {206, 104}, experts[] = {512, 256}, routes[] = {64, 1};
    ggml_init_params params = {8 * ggml_tensor_overhead(), nullptr, true};
    auto * weights_ctx = ggml_init(params);
    auto * inputs_ctx = ggml_init(params);
    ggml_tensor * weights[banks], * acts[banks], * ids[banks];
    ggml_moe_cache_tensor_desc descriptors[banks];
    std::vector<uint64_t> usage[banks];
    const uint64_t * counts[banks];
    for (int bank = 0; bank < banks; ++bank) {
        weights[bank] = ggml_new_tensor_3d(weights_ctx, GGML_TYPE_Q4_0, width, rows[bank], experts[bank]);
        ggml_set_name(weights[bank], bank ? "blk.1.ffn_up_exps.weight" : "blk.0.ffn_up_exps.weight");
        acts[bank] = ggml_new_tensor_3d(inputs_ctx, GGML_TYPE_F32, width, 1, 1);
        ids[bank] = ggml_new_tensor_2d(inputs_ctx, GGML_TYPE_I32, routes[bank], 1);
    }
    auto weights_buffer = ggml_backend_alloc_ctx_tensors(weights_ctx, cpu);
    auto inputs_buffer = ggml_backend_alloc_ctx_tensors(inputs_ctx, gpu);
    ggml_backend_buffer_set_usage(weights_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    bool ok = true;
    auto require = [&](bool passed, int line) {
        if (!passed) {
            fprintf(stderr, "device adaptation starvation assertion failed: line=%d\n", line);
        }
        ok &= passed;
    };
    test_graph graphs[banks];
    for (int bank = 0; bank < banks; ++bank) {
        std::vector<float> values(width * rows[bank]);
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = 0.2f * std::sin((float)(i % 997) * 0.017f);
        }
        std::vector<uint8_t> quantized(weights[bank]->nb[2]);
        require(ggml_quantize_chunk(GGML_TYPE_Q4_0, values.data(), quantized.data(),
                0, rows[bank], width, nullptr) == quantized.size(), __LINE__);
        for (int expert = 0; expert < experts[bank]; ++expert) {
            ggml_backend_tensor_set(weights[bank], quantized.data(), expert * quantized.size(), quantized.size());
        }
        values.resize(width);
        ggml_backend_tensor_set(acts[bank], values.data(), 0, values.size() * sizeof(float));
        descriptors[bank] = {weights[bank]->name, weights[bank]->data, weights[bank]->nb[2],
                width, rows[bank], experts[bank], GGML_TYPE_Q4_0};
        usage[bank].assign(experts[bank], 1000);
        counts[bank] = usage[bank].data();
        graphs[bank] = make_graph(gpu, weights[bank], acts[bank], ids[bank]);
    }
    auto * reference_ctx = ggml_init(params);
    auto * reference_acts = ggml_new_tensor_3d(reference_ctx, GGML_TYPE_F32, width, 1, 1);
    auto * reference_ids = ggml_new_tensor_2d(reference_ctx, GGML_TYPE_I32, 1, 1);
    auto reference_buffer = ggml_backend_alloc_ctx_tensors(reference_ctx, cpu);
    std::vector<float> activation(width);
    ggml_backend_tensor_get(acts[1], activation.data(), 0, activation.size() * sizeof(float));
    ggml_backend_tensor_set(reference_acts, activation.data(), 0, activation.size() * sizeof(float));
    const int32_t target = 255;
    ggml_backend_tensor_set(reference_ids, &target, 0, sizeof(target));
    test_graph reference = make_graph(cpu, weights[1], reference_acts, reference_ids);
    require(ggml_backend_graph_compute(cpu, reference.graph) == GGML_STATUS_SUCCESS, __LINE__);
    std::vector<float> expected(rows[1]);
    ggml_backend_tensor_get(reference.out, expected.data(), 0, expected.size() * sizeof(float));
    void * session = create_direct_session(gpu, cpu);
    require(session != nullptr, __LINE__);
    if (session) {
        require(api.warm(session, descriptors, counts, banks) >= 128, __LINE__);
        const bool prepared = api.device_prepare(session, descriptors, banks) == 1;
        require(prepared, __LINE__);
        if (prepared) {
            require(api.device_supports(session, gpu, graphs[0].out) == 1, __LINE__);
            require(api.device_supports(session, gpu, graphs[1].out) == 1, __LINE__);
            api.session_enter(session);
            // Both pools are full. Every large-pool resident and miss gets two uses,
            // so at least 371 higher-count misses cannot replace any resident.
            for (int repeat = 0; repeat < 2; ++repeat) {
                for (int first = 0; first < experts[0]; first += routes[0]) {
                    int32_t route[64];
                    for (int i = 0; i < routes[0]; ++i) { route[i] = first + i; }
                    ggml_backend_tensor_set(ids[0], route, 0, sizeof(route));
                    require(ggml_backend_graph_compute(gpu, graphs[0].graph) == GGML_STATUS_SUCCESS, __LINE__);
                    ggml_backend_synchronize(gpu);
                }
            }
            ggml_backend_tensor_set(ids[1], &target, 0, sizeof(target));
            require(ggml_backend_graph_compute(gpu, graphs[1].graph) == GGML_STATUS_SUCCESS, __LINE__);
            api.session_leave(session);
            const std::string first_log = capture.get();
            require(first_log.find("fills=1 copied_bytes=" + std::to_string(descriptors[1].expert_size)) !=
                    std::string::npos, __LINE__);
            api.print_stats(session);
            const std::string before = capture.get();
            const long long hits = max_field_value(before, "hits=");
            const long long misses = max_field_value(before, "misses=");
            const size_t checkpoint = before.size();
            std::vector<float> initial(rows[1]), actual(rows[1]);
            ggml_backend_tensor_get(graphs[1].out, initial.data(), 0, initial.size() * sizeof(float));
            require(compare_output(expected, initial, 5e-4), __LINE__);
            api.session_enter(session);
            require(ggml_backend_graph_compute(gpu, graphs[1].graph) == GGML_STATUS_SUCCESS, __LINE__);
            api.session_leave(session);
            ggml_backend_tensor_get(graphs[1].out, actual.data(), 0, actual.size() * sizeof(float));
            require(actual == initial, __LINE__);
            api.print_stats(session);
            const std::string after = capture.get().substr(checkpoint);
            require(max_field_value(after, "hits=") == hits + 1, __LINE__);
            require(max_field_value(after, "misses=") == misses, __LINE__);
            require(max_field_value(after, "dispatches=") == 18, __LINE__);
        }
        api.session_destroy(session);
    }
    free_graph(reference);
    ggml_backend_buffer_free(reference_buffer);
    ggml_free(reference_ctx);
    for (auto & graph : graphs) { free_graph(graph); }
    ggml_backend_buffer_free(inputs_buffer);
    ggml_backend_buffer_free(weights_buffer);
    ggml_free(inputs_ctx);
    ggml_free(weights_ctx);
    for (const auto & value : saved) {
        set_env(value.name, value.present ? value.value.c_str() : nullptr);
    }
    printf("cache-device-adaptation-starvation: %s\n", ok ? "OK" : "FAIL");
    if (!ok) { fprintf(stderr, "%s", capture.get().c_str()); }
    return ok;
}

// Keep all tensor addresses and a nonzero graph uid unchanged across cache
// session switches and release/reprepare. Only the cache generation can detect
// that a captured executable contains stale tables or routing counters.
static bool run_device_capture_lifetime(ggml_backend_t gpu, ggml_backend_t cpu,
        ggml_tensor * weights, log_capture & capture) {
    configure_cache(nullptr, "8");
    set_env("GGML_MOE_CACHE_DEVICE", "1");
    set_env("GGML_MOE_CACHE_DEVICE_STATS", "1");
    set_env("GGML_MOE_CACHE_DEVICE_CAPTURE", "1");
    capture.clear();
    ggml_init_params params = {4 * ggml_tensor_overhead(), nullptr, true};
    ggml_context * host = ggml_init(params);
    ggml_context * card = ggml_init(params);
    ggml_tensor * host_acts = ggml_new_tensor_3d(host, GGML_TYPE_F32, weights->ne[0], 1, 1);
    ggml_tensor * host_ids = ggml_new_tensor_2d(host, GGML_TYPE_I32, 2, 1);
    ggml_tensor * card_acts = ggml_new_tensor_3d(card, GGML_TYPE_F32, weights->ne[0], 1, 1);
    ggml_tensor * card_ids = ggml_new_tensor_2d(card, GGML_TYPE_I32, 2, 1);
    ggml_backend_buffer_t host_buffer = ggml_backend_alloc_ctx_tensors(host, cpu);
    ggml_backend_buffer_t card_buffer = ggml_backend_alloc_ctx_tensors(card, gpu);
    std::vector<float> values(weights->ne[0]);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = 0.2f * std::cos((float)i * 0.017f);
    }
    const int32_t ids[] = {0, 0};
    ggml_backend_tensor_set(host_acts, values.data(), 0, values.size() * sizeof(float));
    ggml_backend_tensor_set(card_acts, values.data(), 0, values.size() * sizeof(float));
    ggml_backend_tensor_set(host_ids, ids, 0, sizeof(ids));
    ggml_backend_tensor_set(card_ids, ids, 0, sizeof(ids));
    test_graph reference = make_graph(cpu, weights, host_acts, host_ids);
    test_graph candidate = make_graph(gpu, weights, card_acts, card_ids);
    candidate.graph->uid = UINT64_C(0x464e464153544447);
    bool ok = ggml_backend_graph_compute(cpu, reference.graph) == GGML_STATUS_SUCCESS;
    std::vector<float> expected(ggml_nelements(reference.out));
    ggml_backend_tensor_get(reference.out, expected.data(), 0, expected.size() * sizeof(float));
    const auto api = active_api();
    void * sessions[] = {create_direct_session(gpu, cpu), create_direct_session(gpu, cpu)};
    const ggml_moe_cache_tensor_desc descriptor = {weights->name, weights->data,
        weights->nb[2], weights->ne[0], weights->ne[1], weights->ne[2], (int32_t) weights->type};
    std::vector<uint64_t> usage(weights->ne[2], 0);
    usage[0] = 100;
    const uint64_t * counts[] = {usage.data()};
    bool prepared[] = {false, false};
    for (int i = 0; i < 2; ++i) {
        ok &= sessions[i] != nullptr;
        if (sessions[i]) {
            ok &= api.warm(sessions[i], &descriptor, counts, 1) == 1;
            prepared[i] = api.device_prepare(sessions[i], &descriptor, 1) == 1;
            ok &= prepared[i];
        }
    }
    auto replay = [&](int session_index) {
        if (!prepared[session_index]) {
            return;
        }
        std::vector<float> first, actual(expected.size());
        for (int n = 0; n < 128; ++n) {
            api.session_enter(sessions[session_index]);
            ok &= ggml_backend_graph_compute(gpu, candidate.graph) == GGML_STATUS_SUCCESS;
            api.session_leave(sessions[session_index]);
            ggml_backend_tensor_get(candidate.out, actual.data(), 0, actual.size() * sizeof(float));
            ok &= compare_output(expected, actual, 5e-4);
            if (n == 0) {
                first = actual;
            } else {
                ok &= first == actual;
            }
        }
    };
    replay(0);
    replay(1);
    replay(0);
    if (prepared[0]) {
        // Warm releases old captured pointers even if every expert remains in
        // the same slot and all graph uid/tensor properties are unchanged.
        ok &= api.warm(sessions[0], &descriptor, counts, 1) == 0;
        prepared[0] = api.device_prepare(sessions[0], &descriptor, 1) == 1;
        ok &= prepared[0];
        replay(0);
    }
    if (sessions[0]) {
        api.session_destroy(sessions[0]);
    }
    replay(1); // A's captured pointers are now freed; B must recapture safely.
    if (sessions[1]) {
        api.session_destroy(sessions[1]);
    }
    const std::string log = capture.get();
    // Each of A(initial), A(reprepared), B records actual graph launches.
    ok &= count_field_at_least(log, "captures=", 1) == 3;
    ok &= count_field_at_least(log, "captures=", 2) == 2;
    ok &= count_field_at_least(log, "replays=", 65) == 3;
    ggml_backend_buffer_free(candidate.buffer);
    ggml_free(candidate.ctx);
    ggml_backend_buffer_free(reference.buffer);
    ggml_free(reference.ctx);
    ggml_backend_buffer_free(card_buffer);
    ggml_backend_buffer_free(host_buffer);
    ggml_free(card);
    ggml_free(host);
    set_env("GGML_MOE_CACHE_DEVICE", nullptr);
    set_env("GGML_MOE_CACHE_DEVICE_STATS", nullptr);
    set_env("GGML_MOE_CACHE_DEVICE_CAPTURE", nullptr);
    printf("cache-device-capture-lifetime: %s\n", ok ? "OK" : "FAIL");
    if (!ok) {
        fprintf(stderr, "%s", log.c_str());
    }
    return ok;
}

// The normal router returns a top-k view with expert-wide row strides.
static bool run_device_argsort_view(ggml_backend_t gpu, ggml_backend_t cpu,
        ggml_tensor * weights, log_capture & capture) {
    bool ok = true;
    for (int tokens : {3, 4, 5}) {
        for (int channels : {1, 10}) {
            configure_cache(nullptr, "8");
            set_env("GGML_MOE_CACHE_DEVICE", "1");
            set_env("GGML_MOE_CACHE_DEVICE_STATS", "1");
            set_env("GGML_MOE_CACHE_DEVICE_CAPTURE", "1");
            capture.clear();
            ggml_init_params params = {16 * ggml_tensor_overhead(), nullptr, true};
            ggml_context * contexts[] = {ggml_init(params), ggml_init(params)};
            ggml_tensor * acts[2], * scores[2], * ids[2];
            ggml_backend_buffer_t buffers[2];
            ggml_backend_t backends[] = {cpu, gpu};
            std::vector<float> values(weights->ne[0] * channels * tokens);
            for (size_t i = 0; i < values.size(); ++i) values[i] = 0.2f * std::sin((float)i * 0.013f);
            std::vector<float> probabilities(512 * tokens);
            for (int t = 0; t < tokens; ++t) {
                for (int e = 0; e < 512; ++e) {
                    probabilities[t * 512 + e] = e < 64 ? 1000.0f - (float)((e + 64 - 3*t) % 64) : -(float)e;
                }
            }
            for (int i = 0; i < 2; ++i) {
                acts[i] = ggml_new_tensor_3d(contexts[i], GGML_TYPE_F32, weights->ne[0], channels, tokens);
                scores[i] = ggml_new_tensor_2d(contexts[i], GGML_TYPE_F32, 512, tokens);
                ids[i] = ggml_argsort_top_k(contexts[i], scores[i], 10);
                ok &= !ggml_is_contiguous(ids[i]) && ids[i]->nb[1] == 512*sizeof(int32_t);
                buffers[i] = ggml_backend_alloc_ctx_tensors(contexts[i], backends[i]);
                ggml_backend_tensor_set(acts[i], values.data(), 0, values.size()*sizeof(float));
                ggml_backend_tensor_set(scores[i], probabilities.data(), 0, probabilities.size()*sizeof(float));
            }
            test_graph reference = make_graph(cpu, weights, acts[0], ids[0]);
            test_graph candidate = make_graph(gpu, weights, acts[1], ids[1]);
            ok &= ggml_backend_graph_compute(cpu, reference.graph) == GGML_STATUS_SUCCESS;
            std::vector<float> expected(ggml_nelements(reference.out)), actual(expected.size()), first;
            ggml_backend_tensor_get(reference.out, expected.data(), 0, expected.size()*sizeof(float));
            const auto api = active_api();
            void * session = create_direct_session(gpu, cpu);
            const ggml_moe_cache_tensor_desc descriptor = {weights->name, weights->data, weights->nb[2],
                weights->ne[0], weights->ne[1], weights->ne[2], (int32_t)weights->type};
            std::vector<uint64_t> usage(weights->ne[2], 0); usage[0] = 100;
            const uint64_t * counts[] = {usage.data()};
            bool prepared = session && api.warm(session, &descriptor, counts, 1) == 1 &&
                    api.device_prepare(session, &descriptor, 1) == 1;
            ok &= prepared;
            if (prepared) {
                for (int run = 0; run < 12; ++run) {
                    api.session_enter(session);
                    ok &= api.device_supports(session, gpu, candidate.out) == 1;
                    ok &= ggml_backend_graph_compute(gpu, candidate.graph) == GGML_STATUS_SUCCESS;
                    api.session_leave(session);
                    ggml_backend_tensor_get(candidate.out, actual.data(), 0, actual.size()*sizeof(float));
                    ok &= compare_output(expected, actual, 5e-4);
                    if (run == 0) first = actual; else ok &= first == actual;
                }
            }
            if (session) api.session_destroy(session);
            const std::string log = capture.get();
            ok &= max_field_value(log, "dispatches=") == 12;
            ok &= max_field_value(log, "hits=") == 12;
            ok &= max_field_value(log, "misses=") == (uint64_t)(12 * (10*tokens - 1));
            ok &= max_field_value(log, "replays=") >= 8;
            if (!ok) fprintf(stderr, "argsort view tokens=%d channels=%d\n%s", tokens, channels, log.c_str());
            ggml_backend_buffer_free(candidate.buffer); ggml_free(candidate.ctx);
            ggml_backend_buffer_free(reference.buffer); ggml_free(reference.ctx);
            for (int i = 0; i < 2; ++i) { ggml_backend_buffer_free(buffers[i]); ggml_free(contexts[i]); }
        }
    }
    set_env("GGML_MOE_CACHE_DEVICE", nullptr);
    set_env("GGML_MOE_CACHE_DEVICE_STATS", nullptr);
    set_env("GGML_MOE_CACHE_DEVICE_CAPTURE", nullptr);
    printf("cache-device-argsort-view: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

// fork: exercise independent hot/cold gate and up tables through actual CUDA graph replay.
static bool run_device_fused_tables(ggml_backend_t gpu, ggml_backend_t cpu, log_capture & capture) {
    auto fusion_count = (int64_t (*)(ggml_backend_t, const char *)) ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(ggml_backend_get_device(gpu)), "ggml_backend_cuda_fusion_count");
    if (!fusion_count) {
        return false;
    }
    bool ok = true;
    std::vector<float> importance(n_in, 1.0f);
    for (const auto type : ggml_moe_cache_types) {
        if (type == GGML_TYPE_IQ1_M) {
            continue;
        }
        ggml_init_params weight_params = {2 * ggml_tensor_overhead(), nullptr, true};
        ggml_context * weights_ctx = ggml_init(weight_params);
        ggml_tensor * weights[] = {
            ggml_new_tensor_3d(weights_ctx, type, n_in, n_out, n_expert),
            ggml_new_tensor_3d(weights_ctx, type, n_in, n_out, n_expert),
        };
        ggml_set_name(weights[0], "blk.7.ffn_up_exps.weight");
        ggml_set_name(weights[1], "blk.7.ffn_gate_exps.weight");
        ggml_backend_buffer_t weights_buffer = ggml_backend_alloc_ctx_tensors(weights_ctx, cpu);
        ggml_backend_buffer_set_usage(weights_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        std::vector<float> values(n_in * n_out * n_expert);
        for (int bank = 0; bank < 2; ++bank) {
            for (size_t i = 0; i < values.size(); ++i) {
                values[i] = 0.15f * std::sin((float)(i % 997) * (0.017f + bank * 0.013f));
            }
            std::vector<uint8_t> quantized(ggml_nbytes(weights[bank]));
            const float * imatrix = ggml_quantize_requires_imatrix(type) ? importance.data() : nullptr;
            const size_t bytes = ggml_quantize_chunk(type, values.data(), quantized.data(),
                    0, n_out * n_expert, n_in, imatrix);
            ok &= bytes == quantized.size();
            ggml_backend_tensor_set(weights[bank], quantized.data(), 0, quantized.size());
        }
        for (int tokens : {1, 4}) {
            for (int channels : {1, 2}) {
                configure_cache(nullptr, "8");
                set_env("GGML_MOE_CACHE_DEVICE", "1");
                set_env("GGML_MOE_CACHE_DEVICE_STATS", "1");
                set_env("GGML_MOE_CACHE_DEVICE_CAPTURE", "1");
                set_env("GGML_MOE_CACHE_DEVICE_FUSED", "1");
                capture.clear();
                ggml_init_params params = {12 * ggml_tensor_overhead(), nullptr, true};
                ggml_context * input_ctx[] = {ggml_init(params), ggml_init(params)};
                ggml_backend_t backends[] = {cpu, gpu};
                ggml_tensor * acts[2];
                ggml_tensor * ids[2];
                ggml_backend_buffer_t buffers[2];
                std::vector<float> activations(n_in * channels * tokens);
                for (size_t i = 0; i < activations.size(); ++i) {
                    activations[i] = 0.2f * std::cos((float)i * 0.019f);
                }
                std::vector<int32_t> routing(512 * tokens, 63);
                for (int t = 0; t < tokens; ++t) {
                    routing[512*t] = t % 3;
                    routing[512*t + 1] = (t + 1) % 3;
                }
                for (int i = 0; i < 2; ++i) {
                    acts[i] = ggml_new_tensor_3d(input_ctx[i], GGML_TYPE_F32, n_in, channels, tokens);
                    ggml_tensor * storage = ggml_new_tensor_2d(input_ctx[i], GGML_TYPE_I32, 512, tokens);
                    ids[i] = ggml_view_2d(input_ctx[i], storage, 2, tokens, storage->nb[1], 0);
                    buffers[i] = ggml_backend_alloc_ctx_tensors(input_ctx[i], backends[i]);
                    ggml_backend_tensor_set(acts[i], activations.data(), 0, activations.size() * sizeof(float));
                    ggml_backend_tensor_set(storage, routing.data(), 0, routing.size() * sizeof(int32_t));
                }
                test_graph reference = make_fused_graph(cpu, weights[0], weights[1], acts[0], ids[0]);
                test_graph candidate = make_fused_graph(gpu, weights[0], weights[1], acts[1], ids[1]);
                ok &= ggml_backend_graph_compute(cpu, reference.graph) == GGML_STATUS_SUCCESS;
                std::vector<float> expected(ggml_nelements(reference.out)), actual(expected.size()), first;
                ggml_backend_tensor_get(reference.out, expected.data(), 0, expected.size() * sizeof(float));
                const auto api = active_api();
                void * session = create_direct_session(gpu, cpu);
                ggml_moe_cache_tensor_desc descriptors[2];
                std::vector<uint64_t> usage[2] = {std::vector<uint64_t>(n_expert, 0), std::vector<uint64_t>(n_expert, 0)};
                const uint64_t * counts[] = {usage[0].data(), usage[1].data()};
                for (int i = 0; i < 2; ++i) {
                    auto * w = weights[i];
                    descriptors[i] = {w->name, w->data, w->nb[2], n_in, n_out, n_expert, (int32_t)type};
                    usage[i][i] = 100;
                }
                bool prepared = session && api.warm(session, descriptors, counts, 2) == 2 &&
                        api.device_prepare(session, descriptors, 2) == 1;
                ok &= prepared;
                const int64_t before = fusion_count(gpu, "mul_mat_glu");
                if (prepared) {
                    for (int run = 0; run < 80; ++run) {
                        api.session_enter(session);
                        ok &= ggml_backend_graph_compute(gpu, candidate.graph) == GGML_STATUS_SUCCESS;
                        api.session_leave(session);
                        ggml_backend_tensor_get(candidate.out, actual.data(), 0, actual.size() * sizeof(float));
                        ok &= compare_output(expected, actual, 5e-4);
                        if (run == 0) {
                            first = actual;
                        } else {
                            ok &= first == actual;
                        }
                    }
                    ok &= fusion_count(gpu, "mul_mat_glu") > before;
                }
                if (session) {
                    api.session_destroy(session);
                }
                const std::string log = capture.get();
                ok &= max_field_value(log, "dispatches=") == 160;
                ok &= max_field_value(log, "replays=") >= 8;
                if (!ok) {
                    fprintf(stderr, "fused device type=%s tokens=%d channels=%d\n%s", ggml_type_name(type), tokens, channels, log.c_str());
                }
                free_graph(candidate);
                free_graph(reference);
                for (int i = 0; i < 2; ++i) {
                    ggml_backend_buffer_free(buffers[i]);
                    ggml_free(input_ctx[i]);
                }
            }
        }
        printf("cache-device-fused type=%s: %s\n", ggml_type_name(type), ok ? "OK" : "FAIL");
        ggml_backend_buffer_free(weights_buffer);
        ggml_free(weights_ctx);
    }
    for (const char * name : {"GGML_MOE_CACHE_DEVICE", "GGML_MOE_CACHE_DEVICE_STATS", "GGML_MOE_CACHE_DEVICE_CAPTURE", "GGML_MOE_CACHE_DEVICE_FUSED"}) {
        set_env(name, nullptr);
    }
    return ok;
}

// fork: retain exact expert images across stable-table updates and source invalidation.
static bool run_device_missjoin_retain_lifecycle(ggml_backend_t gpu, ggml_backend_t cpu, log_capture & capture) {
    const auto api = active_api();
    if (!api.warm || !api.device_prepare || !api.device_supports || !api.print_stats || !api.invalidate) {
        return false;
    }
    configure_cache(nullptr, "8");
    set_env("GGML_CUDA_MOE_CACHE_BUDGET_MB", "8");
    set_env("GGML_MOE_CACHE_DEVICE", "1");
    set_env("GGML_MOE_CACHE_DEVICE_CAPTURE", "1");
    set_env("GGML_MOE_CACHE_DEVICE_STATS_EVERY", "0");
    set_env("GGML_MOE_CACHE_DEVICE_FUSED", nullptr);
    set_env("GGML_MOE_CACHE_DEVICE_ADAPT_MB", "1");
    set_env("GGML_MOE_CACHE_SWAP", nullptr);
    set_env("GGML_MOE_CACHE_PROFILE_LOAD", nullptr);
    set_env("GGML_MOE_CACHE_PROFILE_SAVE", nullptr);
    set_env("GGML_MOE_PREFETCH_ASYNC", nullptr);
    set_env("GGML_MOE_MISSJOIN", "stage");
    set_env("GGML_MOE_MISSJOIN_SLOTS", "1");
    set_env("GGML_MOE_MISSJOIN_RETAIN", "1");
    set_env("GGML_MOE_MISSJOIN_MASK", nullptr);
    set_env("GGML_MOE_MISSJOIN_OVERLAP", nullptr);
    set_env("GGML_MOE_MMV_GROUPED", nullptr);
    if (!getenv("GGML_MOE_CACHE_DEVICE_PUBLISH_ASYNC")) {
        set_env("GGML_MOE_CACHE_DEVICE_PUBLISH_ASYNC", "0");
    }
    capture.clear();
    bool ok = true;
    auto require = [&](bool passed, int line) {
        if (!passed) {
            fprintf(stderr, "retained lifecycle assertion failed: line=%d\n", line);
        }
        ok &= passed;
    };
    constexpr int64_t width = 1024, rows = 206, experts = 65;
    ggml_init_params params = {8 * ggml_tensor_overhead(), nullptr, true};
    auto * weights_ctx = ggml_init(params);
    ggml_tensor * weights[] = {
        ggml_new_tensor_3d(weights_ctx, GGML_TYPE_Q4_0, width, rows, experts),
        ggml_new_tensor_3d(weights_ctx, GGML_TYPE_Q4_0, width, rows, 1),
    };
    ggml_set_name(weights[0], "blk.7.ffn_up_exps.weight");
    ggml_set_name(weights[1], "blk.8.ffn_up_exps.weight");
    auto weights_buffer = ggml_backend_alloc_ctx_tensors(weights_ctx, cpu);
    ggml_backend_buffer_set_usage(weights_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<uint8_t> quantized[2];
    for (int bank = 0; bank < 2; ++bank) {
        auto * w = weights[bank];
        std::vector<float> values(ggml_nelements(w));
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = 0.2f * std::sin((float)(i % 10007) * (0.013f + bank * 0.007f));
        }
        quantized[bank].resize(ggml_nbytes(w));
        require(ggml_quantize_chunk(w->type, values.data(), quantized[bank].data(),
                0, w->ne[1] * w->ne[2], width, nullptr) == quantized[bank].size(), __LINE__);
        ggml_backend_tensor_set(w, quantized[bank].data(), 0, quantized[bank].size());
    }
    ggml_context * input_ctx[] = {ggml_init(params), ggml_init(params)};
    ggml_backend_t backends[] = {cpu, gpu};
    ggml_tensor * acts[2], * ids[2][3];
    ggml_backend_buffer_t input_buffers[2];
    std::vector<float> activations(width);
    for (size_t i = 0; i < activations.size(); ++i) {
        activations[i] = 0.2f * std::cos((float)i * 0.019f);
    }
    std::vector<int32_t> heat(63);
    for (int i = 0; i < 63; ++i) { heat[i] = i + 1; }
    const int32_t target[] = {64, 64}, challenger[] = {0, 0};
    for (int backend = 0; backend < 2; ++backend) {
        acts[backend] = ggml_new_tensor_3d(input_ctx[backend], GGML_TYPE_F32, width, 1, 1);
        ids[backend][0] = ggml_new_tensor_2d(input_ctx[backend], GGML_TYPE_I32, 2, 1);
        ids[backend][1] = ggml_new_tensor_2d(input_ctx[backend], GGML_TYPE_I32, 63, 1);
        ids[backend][2] = ggml_new_tensor_2d(input_ctx[backend], GGML_TYPE_I32, 2, 1);
        input_buffers[backend] = ggml_backend_alloc_ctx_tensors(input_ctx[backend], backends[backend]);
        ggml_backend_tensor_set(acts[backend], activations.data(), 0, activations.size() * sizeof(float));
        ggml_backend_tensor_set(ids[backend][0], target, 0, sizeof(target));
        ggml_backend_tensor_set(ids[backend][1], heat.data(), 0, heat.size() * sizeof(int32_t));
        ggml_backend_tensor_set(ids[backend][2], challenger, 0, sizeof(challenger));
    }
    test_graph reference[3], candidate[3];
    std::vector<float> expected[3], control[3];
    for (int graph = 0; graph < 3; ++graph) {
        auto * w = weights[graph == 2 ? 1 : 0];
        reference[graph] = make_graph(cpu, w, acts[0], ids[0][graph]);
        candidate[graph] = make_graph(gpu, w, acts[1], ids[1][graph]);
        candidate[graph].graph->uid = UINT64_C(0x52455441494e0000) + graph;
        expected[graph].resize(ggml_nelements(reference[graph].out));
        require(ggml_backend_graph_compute(cpu, reference[graph].graph) == GGML_STATUS_SUCCESS, __LINE__);
        ggml_backend_tensor_get(reference[graph].out, expected[graph].data(), 0, expected[graph].size() * sizeof(float));
    }
    ggml_moe_cache_tensor_desc descriptors[2];
    std::vector<uint64_t> usage[] = {std::vector<uint64_t>(experts, 0), std::vector<uint64_t>(1, 0)};
    for (int expert = 0; expert < 64; ++expert) { usage[0][expert] = 1000 - expert; }
    const uint64_t * counts[] = {usage[0].data(), usage[1].data()};
    for (int bank = 0; bank < 2; ++bank) {
        auto * w = weights[bank];
        descriptors[bank] = {w->name, w->data, w->nb[2], width, rows, w->ne[2], (int32_t)w->type};
    }
    void * session = create_direct_session(gpu, cpu);
    require(session != nullptr, __LINE__);
    auto prepare = [&](bool staged, int queued) {
        set_env("GGML_MOE_CACHE_DEVICE_STATS", staged ? "1" : nullptr);
        set_env("GGML_MOE_CACHE_DEVICE_ADAPT_INTERVAL", staged ? "1" : "1024");
        const bool warmed = session && api.warm(session, descriptors, counts, 2) == queued;
        require(warmed, __LINE__);
        const bool prepared = warmed && api.device_prepare(session, descriptors, 2) == 1;
        require(prepared, __LINE__);
        if (prepared) {
            for (const auto & graph : candidate) {
                require(api.device_supports(session, gpu, graph.out) == 1, __LINE__);
            }
        }
        return prepared;
    };
    auto run = [&](std::initializer_list<int> graphs, bool save_control) {
        api.session_enter(session);
        for (int graph : graphs) {
            require(ggml_backend_graph_compute(gpu, candidate[graph].graph) == GGML_STATUS_SUCCESS, __LINE__);
            std::vector<float> actual(expected[graph].size());
            ggml_backend_tensor_get(candidate[graph].out, actual.data(), 0, actual.size() * sizeof(float));
            require(compare_output(expected[graph], actual, 5e-4), __LINE__);
            if (save_control) {
                control[graph] = actual;
            } else {
                require(control[graph].size() == actual.size() &&
                        memcmp(control[graph].data(), actual.data(), actual.size() * sizeof(float)) == 0, __LINE__);
            }
        }
        api.session_leave(session);
    };
    auto snapshot = [&](unsigned long long hits, unsigned long long misses, unsigned long long dispatches) {
        const size_t checkpoint = capture.get().size();
        api.print_stats(session);
        const std::string log = capture.get().substr(checkpoint);
        const size_t pos = log.find("[moe-cache] phase snapshot: ");
        unsigned long long id = 0, epoch = 0, got_hits = 0, got_misses = 0, got_dispatches = 0;
        int device = -1;
        const bool parsed = pos != std::string::npos && sscanf(log.c_str() + pos,
                "[moe-cache] phase snapshot: session=%llu epoch=%llu device=%d hits=%llu misses=%llu dispatches=%llu kind=device",
                &id, &epoch, &device, &got_hits, &got_misses, &got_dispatches) == 6;
        require(parsed && id > 0 && epoch > 0 && device >= 0 && got_hits == hits && got_misses == misses &&
                got_dispatches == dispatches, __LINE__);
        return epoch;
    };
    auto retained = [&](const std::string & log, unsigned long long dispatches, unsigned long long hits,
            unsigned long long copies) {
        unsigned long long got[7] = {};
        const size_t pos = log.rfind("[moe-cache] missjoin retain final: ");
        const bool parsed = pos != std::string::npos && sscanf(log.c_str() + pos,
                "[moe-cache] missjoin retain final: dispatches=%llu hit_experts=%llu hit_routes=%llu copies=%llu copy_bytes=%llu evictions=%llu fallback_misses=%llu",
                &got[0], &got[1], &got[2], &got[3], &got[4], &got[5], &got[6]) == 7;
        require(parsed && got[0] == dispatches && got[1] == hits && got[2] == 2 * hits && got[3] == copies &&
                got[4] == copies * descriptors[0].expert_size && got[5] == 0 && got[6] == 0, __LINE__);
        require(max_field_value(log, "captures=") > 0 && max_field_value(log, "replays=") > 0, __LINE__);
    };
    // The unmasked device-table control uses the identical MMV without staging.
    if (prepare(false, 64)) {
        run({0, 1, 2}, true);
        if (prepare(true, 0)) {
            const size_t lifecycle_checkpoint = capture.get().size();
            const auto epoch = snapshot(0, 0, 0);
            // The full pool's oldest zero-use slot is A0; retain A64 before promoting it.
            run({0, 0}, false);
            require(snapshot(0, 4, 2) == epoch, __LINE__);
            // A64 gets two hits, A1..63 get three each, and B0 gets four misses.
            // B has separate retained tags, so its admission cannot replace A64's retained bytes.
            run({0, 1, 1, 1, 2, 2}, false);
            require(snapshot(191, 8, 8) == epoch, __LINE__);
            // B0 displaced A64. Both cold A64 dispatches must reuse its original retained image.
            run({0, 0}, false);
            require(snapshot(191, 12, 10) == epoch, __LINE__);
            const size_t checkpoint = capture.get().size();
            api.invalidate(weights[0]->data, ggml_nbytes(weights[0]));
            api.invalidate(weights[1]->data, ggml_nbytes(weights[1]));
            const std::string released = capture.get().substr(checkpoint);
            retained(released, 10, 4, 2);
            require(released.find("retained source mapping invalidated:") != std::string::npos, __LINE__);
            const std::string lifecycle_log = capture.get().substr(lifecycle_checkpoint);
            for (int scope : {1, 2, 3}) {
                require(lifecycle_log.find("device adaptation: scopes=" + std::to_string(scope) +
                        " counter_bytes=528 fills=1 copied_bytes=118656") != std::string::npos, __LINE__);
            }
            const int publish_async = atoi(getenv("GGML_MOE_CACHE_DEVICE_PUBLISH_ASYNC"));
            require(lifecycle_log.find("; stable tables publish_async=" + std::to_string(publish_async)) !=
                    std::string::npos, __LINE__);

            // Reuse the source address and graph uid, but change the formerly retained expert.
            std::vector<float> replacement(width * rows);
            for (size_t i = 0; i < replacement.size(); ++i) {
                replacement[i] = 0.3f * std::cos((float)i * 0.037f);
            }
            const size_t offset = 64 * descriptors[0].expert_size;
            require(ggml_quantize_chunk(weights[0]->type, replacement.data(), quantized[0].data() + offset,
                    0, rows, width, nullptr) == descriptors[0].expert_size, __LINE__);
            ggml_backend_tensor_set(weights[0], quantized[0].data() + offset, offset, descriptors[0].expert_size);
            require(ggml_backend_graph_compute(cpu, reference[0].graph) == GGML_STATUS_SUCCESS, __LINE__);
            ggml_backend_tensor_get(reference[0].out, expected[0].data(), 0, expected[0].size() * sizeof(float));
            const auto old_control = control[0];
            if (prepare(false, 64)) {
                run({0}, true);
                require(memcmp(old_control.data(), control[0].data(), control[0].size() * sizeof(float)) != 0, __LINE__);
                if (prepare(true, 0)) {
                    const auto next_epoch = snapshot(0, 0, 0);
                    require(next_epoch != epoch, __LINE__);
                    // A fresh generation must copy changed bytes once, then hit on replay before promotion.
                    run({0, 0, 0}, false);
                    require(snapshot(0, 6, 3) == next_epoch, __LINE__);
                    const size_t next_checkpoint = capture.get().size();
                    api.invalidate(weights[0]->data, ggml_nbytes(weights[0]));
                    retained(capture.get().substr(next_checkpoint), 3, 2, 1);
                }
            }
        }
    }
    if (session) { api.session_destroy(session); }
    if (!ok) { fprintf(stderr, "%s", capture.get().c_str()); }
    for (int graph = 0; graph < 3; ++graph) {
        free_graph(candidate[graph]);
        free_graph(reference[graph]);
    }
    for (int backend = 0; backend < 2; ++backend) {
        ggml_backend_buffer_free(input_buffers[backend]);
        ggml_free(input_ctx[backend]);
    }
    ggml_backend_buffer_free(weights_buffer);
    ggml_free(weights_ctx);
    printf("cache-device-missjoin-retain-lifecycle: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

// fork: compare staging against the same fused CUDA kernels with no resident mask.
static bool run_device_fused_missjoin(ggml_backend_t gpu, ggml_backend_t cpu, log_capture & capture,
        bool overlap, bool mask_only, bool retain = false, bool fused = true, bool grouped = false,
        bool large_copy = false) {
    auto fusion_count = (int64_t (*)(ggml_backend_t, const char *)) ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(ggml_backend_get_device(gpu)), "ggml_backend_cuda_fusion_count");
    const auto api = active_api();
    if (!fusion_count || !api.warm || !api.device_prepare) {
        return false;
    }
    configure_cache(nullptr, "8");
    if (large_copy || grouped) {
        // Each weight bank needs at least 64 resident slots, even when only one expert is warm.
        set_env("GGML_CUDA_MOE_CACHE_BUDGET_MB", large_copy ? "256" : "64");
    }
    set_env("GGML_MOE_CACHE_DEVICE", "1");
    set_env("GGML_MOE_CACHE_DEVICE_CAPTURE", "1");
    set_env("GGML_MOE_CACHE_DEVICE_FUSED", fused ? "1" : nullptr);
    set_env("GGML_MOE_CACHE_DEVICE_ADAPT_INTERVAL", "1024");
    set_env("GGML_MOE_CACHE_SWAP", nullptr);
    set_env("GGML_MOE_PREFETCH_ASYNC", nullptr);
    set_env("GGML_MOE_MISSJOIN", "stage");
    set_env("GGML_MOE_MISSJOIN_OVERLAP", overlap ? "1" : nullptr);
    if (retain && !getenv("GGML_MOE_MISSJOIN_RETAIN")) {
        set_env("GGML_MOE_MISSJOIN_RETAIN", "2");
    }
    if (!retain) { set_env("GGML_MOE_MISSJOIN_RETAIN", nullptr); }
    if (!getenv("GGML_MOE_MISSJOIN_SLOTS")) {
        set_env("GGML_MOE_MISSJOIN_SLOTS", "2");
    }
    const int slots = std::max(1, std::min(atoi(getenv("GGML_MOE_MISSJOIN_SLOTS")), 256));
    const int banks = overlap ? 3 : 2;
    const int used = 8, phases = 4, runs = 12;
    const int32_t patterns[phases][used] = {
        {0, 1, 2, 3, 2, 3, 0, 1},
        {3, 2, 1, 0, 1, 0, 3, 2},
        {7, 7, 6, 6, 5, 5, 4, 4},
        {0, 0, 0, 0, 0, 0, 0, 0},
    };
    std::vector<unsigned long long> totals(6, 0);
    unsigned long long copied_bytes = 0, bypassed_experts = 0, bypassed_bytes = 0;
    const char * reuse_env = getenv("GGML_MOE_MISSJOIN_REUSE_ONLY");
    const bool reuse_only = !retain && reuse_env && strcmp(reuse_env, "1") == 0;
    // At 65 rows, IQ3_S needs 40 quant blocks per row to keep expert images 16-byte aligned.
    const int64_t input_width = large_copy ? 2560 : grouped ? 10240 : n_in;
    const int64_t output_rows = large_copy ? 640 : grouped ? 65 : n_out;
    std::vector<float> importance(input_width, 1.0f);
    bool ok = true;
    bool saw_retained_eviction = false;
    for (const auto type : grouped ? std::vector<ggml_type>{GGML_TYPE_Q4_0, GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL, GGML_TYPE_Q2_0} :
            std::vector<ggml_type>{GGML_TYPE_Q4_0, GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL}) {
        ggml_init_params weight_params = {3 * ggml_tensor_overhead(), nullptr, true};
        auto * weights_ctx = ggml_init(weight_params);
        ggml_tensor * weights[3] = {
            ggml_new_tensor_3d(weights_ctx, type, input_width, output_rows, n_expert),
            ggml_new_tensor_3d(weights_ctx, type, input_width, output_rows, n_expert),
            overlap ? ggml_new_tensor_3d(weights_ctx, GGML_TYPE_Q4_0, output_rows, input_width, n_expert) : nullptr,
        };
        ggml_set_name(weights[0], "blk.7.ffn_up_exps.weight");
        ggml_set_name(weights[1], "blk.7.ffn_gate_exps.weight");
        if (overlap) {
            ggml_set_name(weights[2], "blk.7.ffn_down_exps.weight");
        }
        auto weights_buffer = ggml_backend_alloc_ctx_tensors(weights_ctx, cpu);
        ggml_backend_buffer_set_usage(weights_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (int bank = 0; bank < banks; ++bank) {
            auto * w = weights[bank];
            std::vector<float> values(ggml_nelements(w));
            for (size_t i = 0; i < values.size(); ++i) {
                values[i] = 0.15f * std::sin((float)(i % 997) * (0.017f + bank * 0.013f));
            }
            std::vector<uint8_t> quantized(ggml_nbytes(w));
            const float * imatrix = ggml_quantize_requires_imatrix(w->type) ? importance.data() : nullptr;
            ok &= ggml_quantize_chunk(w->type, values.data(), quantized.data(),
                    0, w->ne[1] * w->ne[2], w->ne[0], imatrix) == quantized.size();
            if (grouped && bank == 0) {
                // All tested formats keep d first; one selected expert has a NaN in the last partial row block.
                const uint16_t nan_scale = 0x7e01;
                memcpy(quantized.data() + (output_rows - 1)*ggml_row_size(type, input_width), &nan_scale, sizeof(nan_scale));
            }
            ggml_backend_tensor_set(w, quantized.data(), 0, quantized.size());
        }
        // Ampere's mapped IQ3_S dispatch stops at six rows; the other
        // copy formats also cover eight-row windows and 63 cold experts.
        for (int tokens : large_copy ? std::vector<int>{1, 3, 4, 5, type == GGML_TYPE_IQ3_S ? 6 : 8} :
                (retain || grouped) ? std::vector<int>{1, 3, 4, 5} : std::vector<int>{1, 3, 4}) {
            for (int channels : grouped ? std::vector<int>{1, 2, used} : std::vector<int>{1, 2}) {
                ggml_init_params params = {8 * ggml_tensor_overhead(), nullptr, true};
                ggml_context * input_ctx[] = {ggml_init(params), ggml_init(params)};
                ggml_backend_t backends[] = {cpu, gpu};
                ggml_tensor * acts[2], * ids[2], * storage[2];
                ggml_backend_buffer_t buffers[2];
                std::vector<float> activations(input_width * channels * tokens);
                for (size_t i = 0; i < activations.size(); ++i) {
                    activations[i] = 0.2f * std::cos((float)i * 0.019f);
                }
                for (int i = 0; i < 2; ++i) {
                    acts[i] = ggml_new_tensor_3d(input_ctx[i], GGML_TYPE_F32, input_width, channels, tokens);
                    storage[i] = ggml_new_tensor_2d(input_ctx[i], GGML_TYPE_I32, 512, tokens);
                    ids[i] = ggml_view_2d(input_ctx[i], storage[i], used, tokens, storage[i]->nb[1], 0);
                    buffers[i] = ggml_backend_alloc_ctx_tensors(input_ctx[i], backends[i]);
                    ggml_backend_tensor_set(acts[i], activations.data(), 0, activations.size() * sizeof(float));
                }
                auto reference = make_fused_graph(cpu, weights[0], weights[1], acts[0], ids[0], weights[2]);
                auto candidate = make_fused_graph(gpu, weights[0], weights[1], acts[1], ids[1], weights[2]);
                std::vector<std::vector<int32_t>> routing(phases, std::vector<int32_t>(512 * tokens, 63));
                std::vector<std::vector<float>> expected(phases, std::vector<float>(ggml_nelements(candidate.out)));
                std::vector<std::vector<float>> control = expected;
                std::vector<float> actual(expected[0].size());
                for (int phase = 0; phase < phases; ++phase) {
                    for (int t = 0; t < tokens; ++t) {
                        for (int rank = 0; rank < used; ++rank) {
                            if (large_copy && (phase == 1 || phase == 2)) {
                                // Up to 64 distinct routes: exercise many expert groups,
                                // slot overflow, and several experts per small-grid CTA.
                                const int e = (used * t + rank) % n_expert;
                                routing[phase][512 * t + rank] = phase == 1 ? e : n_expert - 1 - e;
                            } else {
                                routing[phase][512 * t + rank] = patterns[(phase + t) % phases][rank];
                            }
                        }
                    }
                    ggml_backend_tensor_set(storage[0], routing[phase].data(), 0, routing[phase].size() * sizeof(int32_t));
                    ok &= ggml_backend_graph_compute(cpu, reference.graph) == GGML_STATUS_SUCCESS;
                    ggml_backend_tensor_get(reference.out, expected[phase].data(), 0, actual.size() * sizeof(float));
                }
                ggml_moe_cache_tensor_desc descriptors[3] = {};
                std::vector<uint64_t> usage[3];
                const uint64_t * counts[3];
                for (int bank = 0; bank < banks; ++bank) {
                    auto * w = weights[bank];
                    descriptors[bank] = {w->name, w->data, w->nb[2], w->ne[0], w->ne[1], w->ne[2], (int32_t)w->type};
                    usage[bank].resize(n_expert, 0);
                    usage[bank][bank] = 100;
                    counts[bank] = usage[bank].data();
                }
                for (int staged = 0; staged < 2; ++staged) {
                    unsigned long long selected_misses = 0;
                    set_env("GGML_MOE_MMV_GROUPED", grouped && staged ? "check" : nullptr);
                    set_env("GGML_MOE_CACHE_DEVICE_STATS", staged && !mask_only ? "1" : nullptr);
                    set_env("GGML_MOE_MISSJOIN_MASK", staged && mask_only ? "1" : nullptr);
                    capture.clear();
                    void * session = create_direct_session(gpu, cpu);
                    bool prepared = session && api.warm(session, descriptors, counts, banks) == (size_t)banks &&
                            api.device_prepare(session, descriptors, banks) == 1;
                    ok &= prepared;
                    const int64_t before = fusion_count(gpu, "mul_mat_glu");
                    if (prepared) {
                        for (int run = 0; run < runs; ++run) {
                            const int phase = run % phases;
                            ggml_backend_tensor_set(storage[1], routing[phase].data(), 0, routing[phase].size() * sizeof(int32_t));
                            api.session_enter(session);
                            ok &= ggml_backend_graph_compute(gpu, candidate.graph) == GGML_STATUS_SUCCESS;
                            api.session_leave(session);
                            ggml_backend_tensor_get(candidate.out, actual.data(), 0, actual.size() * sizeof(float));
                            if (grouped) {
                                auto finite_expected = expected[phase];
                                auto finite_actual = actual;
                                for (size_t i = 0; i < actual.size(); ++i) {
                                    if (std::isnan(finite_expected[i])) {
                                        ok &= std::isnan(finite_actual[i]);
                                        finite_expected[i] = finite_actual[i] = 0.0f;
                                    }
                                }
                                ok &= compare_output(finite_expected, finite_actual, 5e-4);
                            } else {
                                ok &= compare_output(expected[phase], actual, 5e-4);
                            }
                            if (!staged && run < phases) {
                                control[phase] = actual;
                            } else {
                                ok &= memcmp(control[phase].data(), actual.data(), actual.size() * sizeof(float)) == 0;
                            }
                            if (staged && !mask_only) {
                                for (int bank = 0; bank < banks; ++bank) {
                                    int uses[n_expert] = {};
                                    unsigned long long distinct = 0, misses = 0, miss_routes = 0;
                                    for (int t = 0; t < tokens; ++t) {
                                        for (int rank = 0; rank < used; ++rank) {
                                            const int e = routing[phase][512 * t + rank];
                                            miss_routes += e != bank;
                                            if (uses[e]++ == 0) {
                                                ++distinct;
                                                misses += e != bank;
                                            }
                                        }
                                    }
                                    unsigned long long bypassed = 0;
                                    if (reuse_only) {
                                        for (int e = 0; e < n_expert; ++e) {
                                            bypassed += e != bank && uses[e] == 1;
                                        }
                                    }
                                    const auto copied = std::min(misses - bypassed, (unsigned long long)slots);
                                    bypassed_experts += bypassed;
                                    bypassed_bytes += bypassed * weights[bank]->nb[2];
                                    selected_misses += misses;
                                    ++totals[0];
                                    totals[1] += distinct;
                                    totals[2] += misses;
                                    totals[3] += copied;
                                    totals[4] += miss_routes;
                                    totals[5] += misses - copied;
                                    copied_bytes += copied * weights[bank]->nb[2];
                                }
                            }
                        }
                        ok &= fused ? fusion_count(gpu, "mul_mat_glu") > before : fusion_count(gpu, "mul_mat_glu") == before;
                    }
                    if (session) {
                        api.session_destroy(session);
                    }
                    const std::string log = capture.get();
                    ok &= max_field_value(log, "replays=") > 0;
                    const bool grouped_expected = grouped && staged && tokens > 1 && type != GGML_TYPE_Q4_0;
                    ok &= (log.find("grouped MMV check:") != std::string::npos) == grouped_expected;
                    if (staged && !mask_only && !retain) {
                        std::vector<unsigned long long> got(6, 0);
                        const size_t pos = log.rfind("[moe-cache] missjoin final: ");
                        ok &= pos != std::string::npos && sscanf(log.c_str() + pos,
                                "[moe-cache] missjoin final: dispatches=%llu distinct_experts=%llu distinct_misses=%llu staged=%llu miss_routes=%llu unstaged=%llu",
                                &got[0], &got[1], &got[2], &got[3], &got[4], &got[5]) == 6 && got == totals;
                        if (large_copy && getenv("GGML_MOE_MISSJOIN_EXPERT_CTA") &&
                                strcmp(getenv("GGML_MOE_MISSJOIN_EXPERT_CTA"), "1") == 0) {
                            unsigned long long dispatches = 0, bytes = 0;
                            const size_t copy_pos = log.rfind("[moe-cache] missjoin copy layout final: ");
                            ok &= copy_pos != std::string::npos && sscanf(log.c_str() + copy_pos,
                                    "[moe-cache] missjoin copy layout final: expert_cta_dispatches=%llu staged_bytes=%llu",
                                    &dispatches, &bytes) == 2 && dispatches == totals[0] &&
                                    bytes == copied_bytes && bytes > 0;
                        }
                        const size_t reuse_pos = log.rfind("[moe-cache] missjoin reuse-only final: ");
                        if (bypassed_experts) {
                            unsigned long long experts = 0, bytes = 0;
                            ok &= reuse_pos != std::string::npos && sscanf(log.c_str() + reuse_pos,
                                    "[moe-cache] missjoin reuse-only final: bypassed_experts=%llu bypassed_bytes=%llu",
                                    &experts, &bytes) == 2 && experts == bypassed_experts && bytes == bypassed_bytes;
                        } else {
                            ok &= reuse_pos == std::string::npos;
                        }
                    }
                    if (staged && retain) {
                        unsigned long long got[7] = {};
                        const size_t pos = log.rfind("[moe-cache] missjoin retain final: ");
                        const bool parsed = pos != std::string::npos && sscanf(log.c_str() + pos,
                                "[moe-cache] missjoin retain final: dispatches=%llu hit_experts=%llu hit_routes=%llu copies=%llu copy_bytes=%llu evictions=%llu fallback_misses=%llu",
                                &got[0], &got[1], &got[2], &got[3], &got[4], &got[5], &got[6]) == 7;
                        ok &= parsed && got[0] == (unsigned long long) runs * banks && got[1] > 0 && got[2] >= got[1] &&
                              got[3] > 0 && got[4] == got[3] * weights[0]->nb[2] && got[1] + got[3] + got[6] == selected_misses;
                        saw_retained_eviction |= got[5] > 0;
                    }
                    if (staged && mask_only) {
                        ok &= log.find("resident_mask=1 route_stats=0") != std::string::npos;
                        ok &= log.find("device routes:") == std::string::npos;
                    }
                    if (staged && overlap) {
                        ok &= log.find("missjoin overlap: on, 1 complete layers") != std::string::npos;
                    }
                    if (!ok) {
                        fprintf(stderr, "fused missjoin type=%s tokens=%d channels=%d staged=%d\n%s",
                                ggml_type_name(type), tokens, channels, staged, log.c_str());
                    }
                }
                free_graph(candidate);
                free_graph(reference);
                for (int i = 0; i < 2; ++i) {
                    ggml_backend_buffer_free(buffers[i]);
                    ggml_free(input_ctx[i]);
                }
            }
        }
        ggml_backend_buffer_free(weights_buffer);
        ggml_free(weights_ctx);
    }
    set_env("GGML_MOE_MMV_GROUPED", nullptr);
    ok &= !retain || saw_retained_eviction;
    if (large_copy) {
        printf("missjoin-copy: 30 shapes, 360 staged graph runs, dispatches=%llu staged_bytes=%llu: %s\n",
                totals[0], copied_bytes, ok ? "OK" : "FAIL");
        if (reuse_only) {
            ok &= bypassed_experts > 0 && copied_bytes > 0;
            printf("missjoin-reuse-only: bypassed_experts=%llu bypassed_bytes=%llu: %s\n",
                    bypassed_experts, bypassed_bytes, ok ? "OK" : "FAIL");
        }
    }
    printf("cache-device%s-missjoin%s%s%s%s: %s\n", fused ? "-fused" : "", overlap ? "-overlap" : "", mask_only ? "-mask" : "", retain ? "-retain" : "", grouped ? "-grouped" : "", ok ? "OK" : "FAIL");
    return ok;
}

// fork VFIXMISS: a full up/gate/swiglu/down layer with mixed resident and missed experts, replayed
// with changing routes. The CPU result must match within rounding, replays of one graph must be
// bit-identical to the first run, and (via GGML_MOE_CACHE_TEST_HF_DUMP) a run with
// GGML_MOE_MISSJOIN_HITFIRST=1 must be bit-identical to a run without it.
static bool run_hitfirst_layer(ggml_backend_t gpu, ggml_backend_t cpu, log_capture & capture) {
    const char * dump_path = getenv("GGML_MOE_CACHE_TEST_HF_DUMP");
    const bool hitfirst_requested = getenv("GGML_MOE_MISSJOIN_HITFIRST") &&
            strcmp(getenv("GGML_MOE_MISSJOIN_HITFIRST"), "1") == 0;
    std::vector<float> dump;
    bool ok = true;
    std::vector<float> importance(n_in, 1.0f);
    const ggml_type types[] = {GGML_TYPE_Q4_0, GGML_TYPE_IQ4_XS, GGML_TYPE_Q4_K, GGML_TYPE_IQ3_S, GGML_TYPE_IQ2_XXS};
    constexpr int k_used = 8;
    constexpr int dim = 256; // square projections keep every quantization block size legal
    // Up/gate/swiglu/down quantizes the activations twice (the CPU in 256-wide blocks, the card in 32-wide), so the
    // CPU comparison is a sanity bound at quantization-noise level; the exact gate is the bitwise on/off comparison.
    const double chain_tolerance = 8e-3;
    long long hf_layers_total = 0;
    bool all_ok = true;
    for (const auto type : types) {
        const bool ok_before_type = ok;
        ok = true; // per type, so one failing type does not hide the others
        double type_nmse = 0.0, type_abs = 0.0, type_rel = 0.0;
        ggml_init_params weight_params = {4 * ggml_tensor_overhead(), nullptr, true};
        ggml_context * weights_ctx = ggml_init(weight_params);
        ggml_tensor * weights[3];
        const char * names[3] = {"blk.7.ffn_up_exps.weight", "blk.7.ffn_gate_exps.weight", "blk.7.ffn_down_exps.weight"};
        for (int bank = 0; bank < 3; ++bank) {
            weights[bank] = ggml_new_tensor_3d(weights_ctx, type, dim, dim, n_expert);
            ggml_set_name(weights[bank], names[bank]);
        }
        ggml_backend_buffer_t weights_buffer = ggml_backend_alloc_ctx_tensors(weights_ctx, cpu);
        ggml_backend_buffer_set_usage(weights_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        std::vector<float> values((size_t) dim * dim * n_expert);
        for (int bank = 0; bank < 3; ++bank) {
            for (size_t i = 0; i < values.size(); ++i) {
                values[i] = 0.15f * std::sin((float)(i % 997) * (0.017f + bank * 0.013f)) +
                            0.05f * std::cos((float)(i % 431) * (0.031f + bank * 0.007f));
            }
            std::vector<uint8_t> quantized(ggml_nbytes(weights[bank]));
            const float * imatrix = ggml_quantize_requires_imatrix(type) ? importance.data() : nullptr;
            const size_t bytes = ggml_quantize_chunk(type, values.data(), quantized.data(),
                    0, (int64_t) dim * n_expert, dim, imatrix);
            ok &= bytes == quantized.size();
            ggml_backend_tensor_set(weights[bank], quantized.data(), 0, quantized.size());
        }
        for (int tokens : {1, 4}) {
            configure_cache(nullptr, "8");
            set_env("GGML_CUDA_MOE_CACHE_BUDGET_MB", "16");
            set_env("GGML_MOE_CACHE_DEVICE", "1");
            set_env("GGML_MOE_CACHE_DEVICE_STATS", "1");
            set_env("GGML_MOE_CACHE_DEVICE_CAPTURE", "1");
            set_env("GGML_MOE_MISSJOIN", "stage");
            capture.clear();
            ggml_init_params params = {12 * ggml_tensor_overhead(), nullptr, true};
            ggml_context * input_ctx[] = {ggml_init(params), ggml_init(params)};
            ggml_backend_t backends[] = {cpu, gpu};
            ggml_tensor * acts[2];
            ggml_tensor * storage[2];
            ggml_tensor * ids[2];
            ggml_backend_buffer_t buffers[2];
            std::vector<float> activations((size_t) dim * tokens);
            for (size_t i = 0; i < activations.size(); ++i) {
                activations[i] = 0.2f * std::cos((float)i * 0.019f);
            }
            for (int i = 0; i < 2; ++i) {
                acts[i] = ggml_new_tensor_3d(input_ctx[i], GGML_TYPE_F32, dim, 1, tokens);
                storage[i] = ggml_new_tensor_2d(input_ctx[i], GGML_TYPE_I32, 64, tokens);
                ids[i] = ggml_view_2d(input_ctx[i], storage[i], k_used, tokens, storage[i]->nb[1], 0);
                buffers[i] = ggml_backend_alloc_ctx_tensors(input_ctx[i], backends[i]);
                ggml_backend_tensor_set(acts[i], activations.data(), 0, activations.size() * sizeof(float));
            }
            test_graph graphs[2];
            for (int i = 0; i < 2; ++i) {
                ggml_init_params gp = {24 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true};
                graphs[i].ctx = ggml_init(gp);
                ggml_context * c = graphs[i].ctx;
                ggml_tensor * up = ggml_mul_mat_id(c, weights[0], acts[i], ids[i]);
                ggml_tensor * gate = ggml_mul_mat_id(c, weights[1], acts[i], ids[i]);
                ggml_tensor * mid = ggml_swiglu_split(c, gate, up);
                graphs[i].out = ggml_mul_mat_id(c, weights[2], mid, ids[i]);
                ggml_set_name(up, "moe_cache_hf_up");
                ggml_set_name(gate, "moe_cache_hf_gate");
                ggml_set_name(graphs[i].out, "moe_cache_hf_down");
                graphs[i].graph = ggml_new_graph(c);
                ggml_build_forward_expand(graphs[i].graph, graphs[i].out);
                graphs[i].buffer = ggml_backend_alloc_ctx_tensors(c, backends[i]);
            }
            const auto api = active_api();
            void * session = create_direct_session(gpu, cpu);
            ggml_moe_cache_tensor_desc descriptors[3];
            std::vector<uint64_t> usage[3];
            const uint64_t * counts[3];
            for (int i = 0; i < 3; ++i) {
                auto * w = weights[i];
                descriptors[i] = {w->name, w->data, w->nb[2], dim, dim, n_expert, (int32_t) type};
                usage[i].assign(n_expert, 0);
                for (int e = 0; e < 6; ++e) {
                    usage[i][e] = 100; // experts 0..5 resident, everything else misses
                }
                counts[i] = usage[i].data();
            }
            const bool prepared = session && api.warm(session, descriptors, counts, 3) > 0 &&
                    api.device_prepare(session, descriptors, 3) == 1;
            ok &= prepared;
            std::vector<float> expected(ggml_nelements(graphs[0].out)), actual(expected.size());
            std::vector<std::vector<float>> first_by_route;
            for (int run = 0; run < 24 && prepared; ++run) {
                // Route patterns: all hits, all misses, mixed, duplicates, many distinct misses.
                std::vector<int32_t> routing(64 * tokens, 63);
                for (int t = 0; t < tokens; ++t) {
                    for (int j = 0; j < k_used; ++j) {
                        int e;
                        switch ((run / 3) % 6) {
                            case 0: e = (j + t) % 6; break;                       // all resident
                            case 1: e = 20 + ((j * 3 + t * 5) % 40); break;       // all missed
                            case 2: e = (j % 2) ? (j * 7 + t) % 6 : 10 + j + t; break; // mixed
                            case 3: e = (j < 4) ? 33 : (j % 3); break;            // duplicate misses and hits
                            case 4: e = (j * 8 + t * 3 + run) % n_expert; break;  // wide spread
                            default: e = 12 + ((j + t) % 2); break;               // two misses reused
                        }
                        routing[64 * t + j] = e;
                    }
                }
                for (int i = 0; i < 2; ++i) {
                    ggml_backend_tensor_set(storage[i], routing.data(), 0, routing.size() * sizeof(int32_t));
                }
                ok &= ggml_backend_graph_compute(cpu, graphs[0].graph) == GGML_STATUS_SUCCESS;
                ggml_backend_tensor_get(graphs[0].out, expected.data(), 0, expected.size() * sizeof(float));
                api.session_enter(session);
                ok &= ggml_backend_graph_compute(gpu, graphs[1].graph) == GGML_STATUS_SUCCESS;
                api.session_leave(session);
                ggml_backend_tensor_get(graphs[1].out, actual.data(), 0, actual.size() * sizeof(float));
                {
                    double se = 0.0, sr = 0.0, max_ref = 1e-12;
                    for (size_t q = 0; q < expected.size(); ++q) {
                        const double d = (double) actual[q] - expected[q];
                        se += d * d; sr += (double) expected[q] * expected[q];
                        type_abs = std::max(type_abs, std::fabs(d));
                        max_ref = std::max(max_ref, (double) std::fabs(expected[q]));
                    }
                    type_nmse = std::max(type_nmse, se / std::max(sr, 1e-12));
                    type_rel = std::max(type_rel, type_abs / max_ref);
                }
                ok &= compare_output(expected, actual, chain_tolerance);
                dump.insert(dump.end(), actual.begin(), actual.end());
                // Same route pattern (every 18 runs) must reproduce the same bits on a replayed graph.
                const size_t slot = (size_t)(run % 18);
                if (first_by_route.size() <= slot) {
                    first_by_route.resize(slot + 1);
                }
                if (first_by_route[slot].empty()) {
                    first_by_route[slot] = actual;
                } else {
                    ok &= first_by_route[slot] == actual;
                }
            }
            if (session) {
                api.session_destroy(session);
            }
            const std::string log = capture.get();
            hf_layers_total += std::max<long long>(0, max_field_value(log, "hf_layers="));
            if (!ok) {
                fprintf(stderr, "hitfirst layer failed type=%s tokens=%d\n%s", ggml_type_name(type), tokens, log.c_str());
            }
            for (int i = 0; i < 2; ++i) {
                free_graph(graphs[i]);
                ggml_backend_buffer_free(buffers[i]);
                ggml_free(input_ctx[i]);
            }
        }
        printf("cache-hitfirst-layer type=%s hitfirst=%d vs-CPU: nmse=%.3e max_abs=%.3e max_rel_to_max=%.3e (limit nmse %.0e): %s\n",
                ggml_type_name(type), hitfirst_requested ? 1 : 0, type_nmse, type_abs, type_rel, chain_tolerance, ok ? "OK" : "FAIL");
        all_ok &= ok && ok_before_type;
        ok = true;
        ggml_backend_buffer_free(weights_buffer);
        ggml_free(weights_ctx);
    }
    for (const char * name : {"GGML_MOE_CACHE_DEVICE", "GGML_MOE_CACHE_DEVICE_STATS", "GGML_MOE_CACHE_DEVICE_CAPTURE", "GGML_MOE_MISSJOIN"}) {
        set_env(name, nullptr);
    }
    ok = all_ok;
    // The path must actually have been taken (and must not be when off).
    if (hitfirst_requested) {
        ok &= hf_layers_total > 0;
    } else {
        ok &= hf_layers_total == 0;
    }
    printf("cache-hitfirst-layer hf_layers=%lld\n", hf_layers_total);
    if (dump_path) {
        FILE * in = fopen(dump_path, "rb");
        if (in) {
            std::vector<float> baseline(dump.size());
            const size_t got = fread(baseline.data(), sizeof(float), baseline.size(), in);
            fclose(in);
            const bool same = got == dump.size() && memcmp(baseline.data(), dump.data(), dump.size() * sizeof(float)) == 0;
            printf("cache-hitfirst-vs-baseline bitwise (%zu floats): %s\n", dump.size(), same ? "IDENTICAL" : "DIFFERENT");
            ok &= same;
        } else if ((in = fopen(dump_path, "wb"))) {
            fwrite(dump.data(), sizeof(float), dump.size(), in);
            fclose(in);
            printf("cache-hitfirst baseline written (%zu floats)\n", dump.size());
        }
    }
    printf("cache-hitfirst-layer: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

// Cover every arithmetic specialization used by the cache table dispatch.
static bool run_device_tables(ggml_backend_t gpu, ggml_backend_t cpu,
        ggml_tensor * initial_weights, log_capture & capture) {
    bool ok = true;
    std::vector<float> values(n_in * n_out * n_expert);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = 0.15f * std::sin((float)(i % 997) * 0.017f) +
                    0.05f * std::cos((float)(i % 431) * 0.031f);
    }
    std::vector<float> importance(n_in, 1.0f);
    for (const auto type : ggml_moe_cache_types) {
        ggml_init_params params = {ggml_tensor_overhead(), nullptr, true};
        ggml_context * ctx = ggml_init(params);
        ggml_tensor * weights = ggml_new_tensor_3d(ctx, type, n_in, n_out, n_expert);
        ggml_set_name(weights, "blk.7.ffn_up_exps.weight");
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
        ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        std::vector<uint8_t> quantized(ggml_nbytes(weights));
        const float * imatrix = ggml_quantize_requires_imatrix(type) ? importance.data() : nullptr;
        const size_t bytes = ggml_quantize_chunk(type, values.data(), quantized.data(),
                0, n_out * n_expert, n_in, imatrix);
        if (bytes != quantized.size()) {
            fprintf(stderr, "cache-device-tables quantization failed type=%s\n", ggml_type_name(type));
            ok = false;
        } else {
            ggml_backend_tensor_set(weights, quantized.data(), 0, bytes);
            ok &= run_device_tables_case(gpu, cpu, weights, capture);
        }
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
#if defined(__linux__)
    // Put a readback destination across the page-rounded expert-bank edge.
    // Registering that padding would split this CUDA copy across registered
    // and unregistered pages, although the destination is outside all weights.
    {
        const size_t page = (size_t)sysconf(_SC_PAGESIZE);
        const size_t bytes = ggml_nbytes(initial_weights);
        const size_t bank_offset = 64;
        const size_t rounded_end = (bank_offset + bytes + page - 1) & ~(page - 1);
        const size_t allocation_bytes = rounded_end + 2 * page;
        void * allocation = mmap(nullptr, allocation_bytes, PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        bool edge_ok = allocation != MAP_FAILED;
        if (edge_ok) {
            void * data = (char *)allocation + bank_offset;
            float * destination = (float *)((char *)allocation + rounded_end - 128);
            edge_ok &= (char *)destination >= (char *)data + bytes;
            memcpy(data, initial_weights->data, bytes);
            ggml_init_params params = {ggml_tensor_overhead(), nullptr, true};
            ggml_context * ctx = ggml_init(params);
            ggml_tensor * weights = ggml_dup_tensor(ctx, initial_weights);
            ggml_set_name(weights, initial_weights->name);
            ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(data, bytes);
            edge_ok &= ggml_backend_tensor_alloc(buffer, weights, data) == GGML_STATUS_SUCCESS;
            ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            edge_ok &= run_device_tables_case(gpu, cpu, weights, capture, destination);
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
            munmap(allocation, allocation_bytes);
        }
        printf("cache-device-page-edge: %s\n", edge_ok ? "OK" : "FAIL");
        ok &= edge_ok;
    }
    const char * directory = getenv("GGML_MOE_CACHE_TEST_TMPDIR");
    if (directory) {
        std::string path = std::string(directory) + "/moe-readonly-XXXXXX";
        std::vector<char> filename(path.begin(), path.end());
        filename.push_back('\0');
        int fd = mkstemp(filename.data());
        const size_t bytes = ggml_nbytes(initial_weights);
        bool mapped_ok = fd >= 0;
        size_t written = 0;
        while (mapped_ok && written < bytes) {
            const ssize_t count = write(fd, (const char *)initial_weights->data + written, bytes - written);
            mapped_ok = count > 0;
            if (mapped_ok) {
                written += (size_t)count;
            }
        }
        if (fd >= 0) {
            close(fd);
        }
        fd = mapped_ok ? open(filename.data(), O_RDONLY) : -1;
        mapped_ok &= fd >= 0;
        void * data = fd >= 0 ? mmap(nullptr, bytes, PROT_READ, MAP_SHARED, fd, 0) : MAP_FAILED;
        if (fd >= 0) {
            close(fd);
        }
        unlink(filename.data());
        mapped_ok &= data != MAP_FAILED;
        if (mapped_ok) {
            ggml_init_params params = {ggml_tensor_overhead(), nullptr, true};
            ggml_context * ctx = ggml_init(params);
            ggml_tensor * weights = ggml_dup_tensor(ctx, initial_weights);
            ggml_set_name(weights, initial_weights->name);
            ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(data, bytes);
            mapped_ok &= ggml_backend_tensor_alloc(buffer, weights, data) == GGML_STATUS_SUCCESS;
            ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            mapped_ok &= run_device_tables_case(gpu, cpu, weights, capture);
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
            munmap(data, bytes);
        }
        printf("cache-device-readonly-file: %s\n", mapped_ok ? "OK" : "FAIL");
        ok &= mapped_ok;
    } else {
        printf("cache-device-readonly-file: SKIP (set GGML_MOE_CACHE_TEST_TMPDIR)\n");
    }
#endif
    return ok;
}

// Run a test only when the provider supports the behavior it checks; on
// other providers print a skip line so CI shows the coverage gap explicitly.
// [#215/#217] expert keep policy: the provider residency probe is read-only, and a scheduler keep policy leaves out
// only non-resident experts past the keep rank (kept rows match the CPU reference, left-out rows are exactly 0).
static bool keep_rows_ok(const std::vector<float> & reference, const std::vector<float> & actual,
                         int64_t n_out, int64_t n_rows, int keep, int & n_left_out) {
    n_left_out = 0;
    for (int64_t row = 0; row < n_rows; row++) {
        std::vector<float> want(reference.begin() + row*n_out, reference.begin() + (row + 1)*n_out);
        std::vector<float> got(actual.begin() + row*n_out, actual.begin() + (row + 1)*n_out);
        bool zero = true;
        for (float v : got) {
            zero &= v == 0.0f;
        }
        const bool match = compare_output(want, got, 5e-4);
        if (row < keep && !match) {
            return false;
        }
        if (row >= keep && !match && !zero) {
            return false;
        }
        if (row >= keep && zero && !match) {
            n_left_out++;
        }
    }
    return true;
}

static bool run_expert_keep(
        ggml_backend_t gpu, ggml_backend_t cpu, ggml_tensor * weights,
        test_graph & graph, const std::vector<float> & reference,
        test_graph & fused_graph, const std::vector<float> & fused_reference) {
    configure_cache(nullptr);
    bool ok = true;

    // 1) probe: resident reports exactly the filled experts and never fills
    {
        void * session = create_direct_session(gpu, cpu);
        if (!session) {
            fprintf(stderr, "cache-expert-keep: failed to create session\n");
            return false;
        }
        const size_t expert_size = ggml_nbytes(weights) / weights->ne[2];
        const ggml_moe_cache_tensor_desc desc = {
            weights->name, weights->data, expert_size,
            weights->ne[0], weights->ne[1], weights->ne[2], (int32_t) weights->type,
        };
        active_api().session_enter(session);
        bool probe_ok = active_api().resident != nullptr &&
            wait_for_direct_pool(weights->name, weights->data, expert_size,
                    weights->ne[0], weights->ne[1], weights->type, weights->ne[2]) &&
            wait_for_direct_resident(weights, 3) && wait_for_direct_resident(weights, 7);
        if (probe_ok) {
            const int32_t ids[] = { 3, 5, 7, 9, -1, 1000 };
            uint8_t resident[6] = {};
            for (int rep = 0; rep < 40; rep++) {
                const int n = active_api().resident(&desc, ids, 6, resident);
                probe_ok &= n == 2 && resident[0] == 1 && resident[1] == 0 && resident[2] == 1 &&
                            resident[3] == 0 && resident[4] == 0 && resident[5] == 0;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            const int n = active_api().resident(&desc, ids, 6, resident);
            probe_ok &= n == 2 && resident[1] == 0 && resident[3] == 0; // 40 probes queued no fill
        }
        active_api().session_leave(session);
        if (probe_ok) {
            const int32_t ids[] = { 3 };
            uint8_t resident[1] = { 1 };
            probe_ok &= active_api().resident(&desc, ids, 1, resident) == 0 && resident[0] == 0; // no active session
        }
        active_api().session_destroy(session);
        printf("cache-expert-keep-probe: %s\n", probe_ok ? "OK" : "FAIL");
        ok &= probe_ok;
    }

    // 2) scheduler, keep 1 from the start: rank 0 runs (and caches), rank 1 is never planned so never resident -> 0
    // 3) same scheduler warmed with the policy off: once rank 1's expert is resident, keep 1 runs it
    for (int fused = 0; fused < 2; fused++) {
        test_graph & g = fused ? fused_graph : graph;
        const char * name = fused ? "cache-expert-keep-fused" : "cache-expert-keep-sched";
        (void) reference;
        (void) fused_reference;
        // earlier scenarios mutate the shared weights (invalidation tests), so take a fresh CPU reference
        // with two distinct experts {0, 1} at ranks 0 and 1
        ggml_tensor * ids_t = fused ? g.out->src[0]->src[2] : g.out->src[2];
        const int32_t ids_keep[2] = { 0, 1 };
        GGML_ASSERT(ids_t && ggml_nelements(ids_t) == 2);
        ggml_backend_tensor_set(ids_t, ids_keep, 0, sizeof(ids_keep));
        std::vector<float> ref(ggml_nelements(g.out));
        if (ggml_backend_graph_compute(cpu, g.graph) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "%s: CPU reference compute failed\n", name);
            ok = false;
            continue;
        }
        ggml_backend_tensor_get(g.out, ref.data(), 0, ref.size() * sizeof(float));
        ggml_backend_sched_t sched = make_scheduler(name, gpu, cpu, g);
        if (!sched) {
            ok = false;
            continue;
        }
        const int64_t n_out = g.out->ne[0];
        const int64_t n_rows = g.out->ne[1] * g.out->ne[2];
        std::vector<float> actual(ref.size());
        bool stage_ok = true;
        int always_left_out = 0;
        ggml_backend_sched_set_moe_expert_keep(sched, 1, nullptr, 0);
        for (int step = 0; step < 60 && stage_ok; step++) {
            stage_ok &= ggml_backend_sched_graph_compute(sched, g.graph) == GGML_STATUS_SUCCESS;
            ggml_backend_tensor_get(g.out, actual.data(), 0, actual.size() * sizeof(float));
            int left = 0;
            stage_ok &= keep_rows_ok(ref, actual, n_out, n_rows, 1, left);
            always_left_out += left == (int) (n_rows - 1) ? 1 : 0;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        stage_ok &= always_left_out == 60;
        if (!stage_ok) {
            fprintf(stderr, "%s: keep 1 cold stage failed (left out %d/60)\n", name, always_left_out);
        }

        bool resident_ran = false;
        for (int round = 0; round < 20 && stage_ok && !resident_ran; round++) {
            ggml_backend_sched_set_moe_expert_keep(sched, -1, nullptr, 0);
            for (int step = 0; step < 20 && stage_ok; step++) {
                stage_ok &= compute_matches(name, sched, g, ref, step);
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            ggml_backend_sched_set_moe_expert_keep(sched, 1, nullptr, 0);
            stage_ok &= ggml_backend_sched_graph_compute(sched, g.graph) == GGML_STATUS_SUCCESS;
            ggml_backend_tensor_get(g.out, actual.data(), 0, actual.size() * sizeof(float));
            int left = 0;
            stage_ok &= keep_rows_ok(ref, actual, n_out, n_rows, 1, left);
            resident_ran = left == 0;
        }
        stage_ok &= resident_ran;
        if (!resident_ran) {
            fprintf(stderr, "%s: a resident expert past the keep rank never ran\n", name);
        }
        // policy cleared: the same scheduler is back to the reference
        ggml_backend_sched_set_moe_expert_keep(sched, -1, nullptr, 0);
        stage_ok &= compute_matches(name, sched, g, ref, 0);
        ggml_backend_sched_free(sched);
        printf("%s: %s\n", name, stage_ok ? "OK" : "FAIL");
        ok &= stage_ok;
    }
    configure_cache(nullptr);
    return ok;
}

static bool run_optional(
        bool enabled, const char * name, std::function<bool()> test) {
    if (!enabled) {
        printf("%s: SKIP (provider-specific)\n", name);
        return true;
    }
    return test();
}

static bool run_arena_stride(ggml_backend_t gpu, ggml_backend_t cpu,
        ggml_tensor * weights, ggml_tensor * gate, test_graph & graph,
        const std::vector<float> & reference, log_capture & capture) {
    const ggml_tensor original = *weights, gate_original = *gate;
    const size_t bytes = weights->nb[2], stride = 3 * bytes;
    const size_t arena_size = stride * (size_t)weights->ne[2];
    auto * buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), arena_size);
    auto * arena = static_cast<uint8_t *>(ggml_backend_buffer_get_base(buffer));
    memset(arena, 0xa5, arena_size);
    for (int64_t e = 0; e < weights->ne[2]; ++e) {
        memcpy(arena + e * stride, (const char *)weights->data + e * bytes, bytes);
        memcpy(arena + e * stride + bytes, (const char *)gate->data + e * bytes, bytes);
    }
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    weights->buffer = gate->buffer = buffer;
    weights->data = arena;
    gate->data = arena + bytes;
    weights->nb[2] = gate->nb[2] = stride;
    weights->nb[3] = gate->nb[3] = stride * weights->ne[2];
    set_env("GGML_CUDA_MOE_CACHE", "0");
    bool ok = ggml_backend_graph_compute(cpu, graph.graph) == GGML_STATUS_SUCCESS;
    std::vector<float> actual(reference.size());
    ggml_backend_tensor_get(graph.out, actual.data(), 0, actual.size() * sizeof(float));
    ok &= actual == reference;
    ok &= run_prompt_warm(gpu, cpu, weights, graph, reference, capture);
    ok &= run_scenario("cache-arena-stride", nullptr, gpu, cpu, graph, reference, capture);
#ifdef GGML_CUDA
    const bool registered = cudaHostRegister(arena, arena_size,
            cudaHostRegisterPortable | cudaHostRegisterMapped) == cudaSuccess;
    ok &= registered;
    if (registered) {
        ok &= run_device_tables_case(gpu, cpu, weights, capture);
        // The second projection starts inside the same registered allocation.
        ok &= run_device_tables_case(gpu, cpu, gate, capture);
        // Actual padded tensors, pinned by the model owner: all paths must copy and consume.
        ok &= run_prefetch_staging(gpu, cpu, weights, capture, false);
        ok &= run_prefetch_staging(gpu, cpu, weights, capture, true);
        ok &= run_prefetch_staging(gpu, cpu, weights, capture, true, gate);
        unsigned int flags = 0;
        // Cache/table teardown must leave the model-owned registration alive.
        ok &= cudaHostGetFlags(&flags, arena) == cudaSuccess;
        ok &= capture.get().find("new_copy_bytes=0 new_registered_bytes=0") != std::string::npos;
        ok &= cudaHostUnregister(arena) == cudaSuccess;
    }
#endif
    for (int64_t e = 0; e < weights->ne[2]; ++e) {
        for (size_t i = 2 * bytes; i < stride; ++i) { ok &= arena[e * stride + i] == 0xa5; }
    }
    *weights = original;
    *gate = gate_original;
    ggml_backend_buffer_free(buffer);
    printf("cache-arena-stride: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

} // namespace

static bool test_prefetch_budget() {
    bool ok = ggml_moe_prefetch_budget(100.0, 20.0, 25000.0) == 1500000;
    ok &= ggml_moe_prefetch_budget(100.0, 20.0, 25000.0, 500000) == 1000000;
    ok &= ggml_moe_prefetch_budget(100.0, 20.0, 25000.0, 1500000) == 0;
    ok &= ggml_moe_prefetch_budget(100.0, 20.0, 25000.0, SIZE_MAX) == 0;
    ok &= ggml_moe_prefetch_budget(100.0, 100.0, 25000.0) == 0;
    ok &= ggml_moe_prefetch_budget(100.0, 101.0, 25000.0) == 0;
    ok &= ggml_moe_prefetch_budget(100.0, 0.0, 0.0) == 0;
    ok &= ggml_moe_prefetch_budget(100.0, -1.0, 25000.0) == 0;
    ok &= ggml_moe_prefetch_budget(std::numeric_limits<double>::infinity(), 0.0, 1.0) == 0;
    ok &= ggml_moe_prefetch_budget(100.0, 0.0, std::numeric_limits<double>::quiet_NaN()) == 0;
    ok &= ggml_moe_prefetch_budget(1e100, 0.0, 1e100) == (16u << 20);
    printf("cache-prefetch-budget: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

// Pure planner fixtures do not initialize a CUDA provider or allocate weights.
static bool test_profile_global_plan() {
    using item = ggml_cuda_moe_profile_item;
    ggml_cuda_moe_profile_plan plan;
    std::vector<item> items;
    std::vector<uint8_t> ranked;
    auto append32 = [&](uint32_t value) {
        for (int shift = 0; shift < 32; shift += 8) { ranked.push_back((uint8_t)(value >> shift)); }
    };
    for (uint32_t value : {UINT32_C(0x50525453), 1u, 2u, 4u, 8u, 8u,
            65537u, 0u, 196609u, 131072u, 65536u, 1u, 196608u, 131073u}) { append32(value); }
    std::vector<std::vector<uint64_t>> scores;
    bool ranked_ok = ggml_cuda_moe_cache_ranked_profile(ranked.data(), ranked.size(), 2, 4, scores);
    ranked_ok &= scores == std::vector<std::vector<uint64_t>>({{7, 4, 5, 2}, {3, 8, 1, 6}});
    const auto saved_scores = scores;
    ranked_ok &= !ggml_cuda_moe_cache_ranked_profile(ranked.data(), ranked.size() - 1, 2, 4, scores);
    ranked_ok &= !ggml_cuda_moe_cache_ranked_profile(ranked.data(), ranked.size(), 3, 4, scores);
    ranked[28] = 1; ranked[30] = 1;
    ranked_ok &= !ggml_cuda_moe_cache_ranked_profile(ranked.data(), ranked.size(), 2, 4, scores);
    ranked[28] = 0; ranked[30] = 0; ranked[24] = 2;
    ranked_ok &= !ggml_cuda_moe_cache_ranked_profile(ranked.data(), ranked.size(), 2, 4, scores);
    ranked_ok &= scores == saved_scores;
    ranked[24] = 1;
    for (uint32_t value : {1u, 4u, 3u, 6u, 5u, 0u, 7u, 2u}) { append32(value); }
    ranked_ok &= ggml_cuda_moe_cache_ranked_profile(ranked.data(), ranked.size(), 2, 4, scores);
    ranked.back() = 1;
    ranked_ok &= !ggml_cuda_moe_cache_ranked_profile(ranked.data(), ranked.size(), 2, 4, scores);
    ranked.back() = 0;
    for (size_t layer = 0; layer < 2; ++layer) {
        for (int expert = 0; expert < 4; ++expert) {
            items.push_back({layer * 2, expert, layer, layer * 2, layer ? 2u : 1u, scores[layer][expert]});
            items.push_back({layer * 2 + 1, expert, layer, layer * 2 + 1, layer ? 1u : 3u, scores[layer][expert]});
        }
    }
    ranked_ok &= ggml_cuda_moe_cache_profile_plan(items, {4, 4, 4, 4}, 7, 1, plan);
    ranked_ok &= plan.bundles == 2 && plan.selected[10] && plan.selected[0] && !plan.selected[2];
    ranked_ok &= ggml_cuda_moe_cache_profile_plan(items, {4, 4, 4, 4}, 28, 1, plan, false, 2);
    ranked_ok &= plan.bundles == 2 && plan.bytes == 7 && plan.selected[10] && plan.selected[0] && !plan.selected[2];
    // First two ranks cost 3+4: a six-byte budget cannot skip rank two and
    // spend the remainder on a cheaper later rank. Failed output is preserved.
    const auto prefix_slots = plan.slots;
    ranked_ok &= !ggml_cuda_moe_cache_profile_plan(items, {4, 4, 4, 4}, 6, 1, plan, false, 2);
    ranked_ok &= plan.slots == prefix_slots;
    ranked_ok &= ggml_cuda_moe_cache_profile_plan(items, {4, 4, 4, 4}, 13, 1, plan, false, 8);
    ranked_ok &= plan.bundles == 3 && plan.bytes == 10 && plan.selected[14] && !plan.selected[4];
    printf("cache-profile-ranked-data: %s\n", ranked_ok ? "OK" : "FAIL");
    if (!ranked_ok) { return false; }
    items.clear();
    for (int32_t expert = 0; expert < 8; ++expert) {
        items.push_back({0, expert, 0, 0, 1, (uint64_t)(80 - expert)});
        items.push_back({1, expert, 1, 1, 4, (uint64_t)(10 - expert)});
    }
    bool ok = ggml_cuda_moe_cache_profile_plan(items, {8, 8}, 20, 2, plan);
    ok &= plan.bytes == 20 && plan.slots == std::vector<size_t>({8, 3});
    ok &= plan.bundles == 11 && plan.ranks.size() == items.size();
    const auto previous_slots = plan.slots;
    ok &= !ggml_cuda_moe_cache_profile_plan(items, {8, 8}, 9, 2, plan);
    ok &= plan.slots == previous_slots;

    items.clear();
    for (size_t layer = 0; layer < 2; ++layer) {
        for (int32_t expert = 0; expert < 8; ++expert) {
            const uint64_t count = (uint64_t)((layer ? 20 : 100) - expert);
            items.push_back({layer * 3, expert, layer, 0, 1, count});
            items.push_back({layer * 3 + 1, expert, layer, 0, 1, count});
            items.push_back({layer * 3 + 2, expert, layer, 1, 4, count});
        }
    }
    ok &= ggml_cuda_moe_cache_profile_plan(items, {32, 16}, 48, 2, plan);
    ok &= plan.bytes == 48 && plan.slots == std::vector<size_t>({16, 8}) && plan.bundles == 8;
    for (size_t i = 0; i < items.size(); i += 3) {
        ok &= plan.selected[i] == plan.selected[i + 1] && plan.selected[i] == plan.selected[i + 2];
        ok &= (plan.selected[i] != 0) == (items[i].group == 0);
    }
    const auto ranks = plan.ranks;
    ok &= ggml_cuda_moe_cache_profile_plan(items, {32, 16}, 48, 2, plan) && plan.ranks == ranks;
    for (auto & entry : items) { entry.count = 0; }
    ok &= !ggml_cuda_moe_cache_profile_plan(items, {32, 16}, 48, 2, plan);
    items = {{0, 0, 0, 0, SIZE_MAX, 1}, {1, 0, 0, 1, 1, 1}};
    ok &= !ggml_cuda_moe_cache_profile_plan(items, {1, 1}, SIZE_MAX, 1, plan);

    items.clear();
    for (int32_t expert = 0; expert < 128; ++expert) { items.push_back({0, expert, 0, 0, 4, 1}); }
    ok &= ggml_cuda_moe_cache_profile_plan(items, {128}, 256, 64, plan);
    ok &= plan.bytes == 256 && plan.slots == std::vector<size_t>({64});
    for (size_t i = 0; i < items.size(); ++i) { ok &= (plan.selected[i] != 0) == (i < 64); }
    const auto minimum_ranks = plan.ranks;
    ok &= ggml_cuda_moe_cache_profile_plan(items, {128}, 256, 64, plan, true);
    ok &= plan.bytes == 256 && plan.slots == std::vector<size_t>({64}) && plan.ranks == minimum_ranks;

    items.clear();
    for (size_t layer = 0; layer < 2; ++layer) {
        for (int32_t expert = 0; expert < 2; ++expert) {
            for (size_t k = 0; k < 3; ++k) {
                items.push_back({layer * 3 + k, expert, layer, layer, layer ? 1u : 4u,
                        (uint64_t)((layer ? 10 : 20) - expert)});
            }
        }
    }
    ok &= ggml_cuda_moe_cache_profile_plan(items, {6, 6}, 27, 1, plan);
    ok &= plan.bytes == 27 && plan.slots == std::vector<size_t>({6, 3});
    ok &= ggml_cuda_moe_cache_profile_plan(items, {6, 6}, 27, 1, plan, true);
    ok &= plan.bytes == 18 && plan.slots == std::vector<size_t>({3, 6});
    for (size_t i = 0; i < items.size(); i += 3) {
        ok &= plan.selected[i] == plan.selected[i + 1] && plan.selected[i] == plan.selected[i + 2];
    }
    items = {{0, 0, 0, 0, 1, 6}, {1, 0, 0, 0, 1, 6}, {2, 0, 0, 0, 1, 6}, {3, 0, 1, 0, 1, 3}};
    ok &= ggml_cuda_moe_cache_profile_plan(items, {4}, 3, 1, plan);
    ok &= plan.bytes == 3 && plan.selected == std::vector<uint8_t>({1, 1, 1, 0});
    ok &= ggml_cuda_moe_cache_profile_plan(items, {4}, 3, 1, plan, true);
    ok &= plan.bytes == 1 && plan.selected == std::vector<uint8_t>({0, 0, 0, 1}) && plan.ranks[3] == 0;
    // Allocator-rounded pools must still select whole bundles, leaving unusable tails free.
    items.clear();
    for (int32_t expert = 0; expert < 8; ++expert) {
        items.push_back({0, expert, 0, 0, 1, (uint64_t)(8 - expert)});
        items.push_back({1, expert, 0, 0, 1, (uint64_t)(8 - expert)});
        items.push_back({2, expert, 0, 1, 4, (uint64_t)(8 - expert)});
    }
    ok &= ggml_cuda_moe_cache_profile_plan(items, {16, 8}, 47, 1, plan);
    ok &= plan.bundles == 7 && plan.bytes == 42 && plan.slots == std::vector<size_t>({14, 7});
    ok &= ggml_cuda_moe_cache_profile_plan(items, {14, 6}, 47, 1, plan);
    ok &= plan.bundles == 6 && plan.bytes == 36 && plan.slots == std::vector<size_t>({12, 6});
    for (size_t i = 0; i < items.size(); i += 3) {
        ok &= plan.selected[i] == plan.selected[i + 1] && plan.selected[i] == plan.selected[i + 2];
        ok &= (plan.selected[i] != 0) == (i < 18);
    }
    printf("cache-profile-global-plan: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

int main(int argc, char ** argv) {
    if (argc == 2 && strcmp(argv[1], "--profile-global-plan-only") == 0) {
        return test_profile_global_plan() ? 0 : 1;
    }
    const bool profile_warm_only = argc == 2 && strcmp(argv[1], "--profile-warm-only") == 0;
    if (!test_prefetch_budget()) {
        return 1;
    }
    if (argc == 2 && strcmp(argv[1], "--permanent-scheduler-rotating-cpu-only") == 0) {
        auto * cpu = ggml_backend_cpu_init();
        const bool ok = cpu && run_resident_scheduler_cpu(cpu, true, true);
        if (cpu) { ggml_backend_free(cpu); }
        return ok ? 0 : 1;
    }
    if (argc == 2 && strcmp(argv[1], "--permanent-scheduler-events-cpu-only") == 0) {
        auto * cpu = ggml_backend_cpu_init();
        const bool ok = cpu && run_resident_scheduler_cpu(cpu, true);
        if (cpu) { ggml_backend_free(cpu); }
        return ok ? 0 : 1;
    }
    if (argc == 2 && strcmp(argv[1], "--permanent-scheduler-cpu-only") == 0) {
        auto * cpu = ggml_backend_cpu_init();
        const bool ok = cpu && run_resident_scheduler_cpu(cpu);
        if (cpu) { ggml_backend_free(cpu); }
        return ok ? 0 : 1;
    }
    if (argc == 2 && strcmp(argv[1], "--permanent-host-only") == 0) {
        auto * cpu = ggml_backend_cpu_init();
        const bool ok = cpu && run_resident_host_barriers(cpu);
        if (cpu) { ggml_backend_free(cpu); }
        return ok ? 0 : 1;
    }
    if (argc == 2 && strcmp(argv[1], "--prefetch-policy-only") == 0) {
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--resume-lifecycle-cpu-only") == 0) {
        return run_resume_lifecycle_cpu() ? 0 : 1;
    }
    if (argc == 2 && strcmp(argv[1], "--prefetch-cpu-only") == 0) {
        auto cpu = ggml_backend_cpu_init();
        const bool ok = cpu && run_prefetch_router(cpu, cpu);
        if (cpu) {
            ggml_backend_free(cpu);
        }
        return ok ? 0 : 1;
    }
#ifdef GGML_CUDA
    // Pure host-budget fixtures need no CUDA device or GPU lock.
    if (argc == 2 && strcmp(argv[1], "--host-budget-only") == 0) {
        return run_device_host_budget() ? 0 : 1;
    }
#else
    (void) argc;
    (void) argv;
#endif
    log_capture capture;
    ggml_log_set(log_callback, &capture);

    ggml_backend_dev_t provider_device = find_provider_device();
    if (!provider_device) {
        printf("SKIP: no moe-cache provider device available\n");
        return 77; // ctest interprets 77 as SKIP_RETURN_CODE
    }
    ggml_backend_reg_t provider_reg =
        ggml_backend_dev_backend_reg(provider_device);
    g_test_api = ggml_moe_cache_get(provider_reg);
    const bool provider_is_cuda =
        strcmp(ggml_backend_reg_name(provider_reg), "CUDA") == 0;
    printf("test-moe-cache: provider=%s\n",
            ggml_backend_reg_name(provider_reg));

    ggml_backend_t gpu = ggml_backend_dev_init(provider_device, nullptr);
    ggml_backend_t cpu = init_cpu_backend();
    if (!gpu || !cpu) {
        fprintf(stderr, "failed to initialize provider and CPU backends\n");
        if (gpu) {
            ggml_backend_free(gpu);
        }
        if (cpu) {
            ggml_backend_free(cpu);
        }
        return 1;
    }
    if (argc == 2 && strcmp(argv[1], "--device-missjoin-retain-lifecycle-only") == 0) {
        const bool ok = provider_is_cuda && run_device_missjoin_retain_lifecycle(gpu, cpu, capture);
        ggml_backend_free(gpu);
        ggml_backend_free(cpu);
        return !provider_is_cuda ? 77 : ok ? 0 : 1;
    }
    if (argc == 2 && (strcmp(argv[1], "--device-fused-missjoin-only") == 0 ||
            strcmp(argv[1], "--device-missjoin-copy-only") == 0 ||
            strcmp(argv[1], "--device-fused-missjoin-overlap-only") == 0 ||
            strcmp(argv[1], "--device-fused-missjoin-retain-only") == 0 ||
            strcmp(argv[1], "--device-fused-missjoin-grouped-only") == 0 ||
            strcmp(argv[1], "--device-missjoin-grouped-only") == 0 ||
            strcmp(argv[1], "--device-missjoin-retain-only") == 0 ||
            strcmp(argv[1], "--device-fused-missjoin-mask-only") == 0)) {
        const bool overlap = strcmp(argv[1], "--device-fused-missjoin-overlap-only") == 0;
        const bool large_copy = strcmp(argv[1], "--device-missjoin-copy-only") == 0;
        const bool mask_only = strcmp(argv[1], "--device-fused-missjoin-mask-only") == 0;
        const bool grouped = strcmp(argv[1], "--device-fused-missjoin-grouped-only") == 0 ||
                strcmp(argv[1], "--device-missjoin-grouped-only") == 0;
        const bool fused = strcmp(argv[1], "--device-missjoin-retain-only") != 0 &&
                strcmp(argv[1], "--device-missjoin-grouped-only") != 0;
        const bool retain = strcmp(argv[1], "--device-missjoin-retain-only") == 0 ||
                strcmp(argv[1], "--device-fused-missjoin-retain-only") == 0 ||
                (grouped && getenv("GGML_MOE_MISSJOIN_RETAIN") && atoi(getenv("GGML_MOE_MISSJOIN_RETAIN")) > 0);
        const bool ok = provider_is_cuda && run_device_fused_missjoin(gpu, cpu, capture, overlap, mask_only, retain, fused, grouped, large_copy);
        ggml_backend_free(gpu);
        ggml_backend_free(cpu);
        return !provider_is_cuda ? 77 : ok ? 0 : 1;
    }
    if (argc == 2 && strcmp(argv[1], "--shared-budget-only") == 0) {
        const bool ok = run_shared_budget(gpu, cpu, capture);
        ggml_backend_free(gpu);
        ggml_backend_free(cpu);
        return ok ? 0 : 1;
    }
    if (argc == 2 && strcmp(argv[1], "--suspend-reserve-only") == 0) {
        stress_fixture fixture;
        const bool ok = provider_is_cuda && init_stress_fixture(fixture, cpu) &&
            run_suspend_resume(gpu, cpu, fixture, capture, true);
        free_stress_fixture(fixture);
        ggml_backend_free(gpu);
        ggml_backend_free(cpu);
        return ok ? 0 : 1;
    }
#if defined(GGML_CUDA) && defined(__linux__)
    if (argc == 2 && strcmp(argv[1], "--permanent-residency-only") == 0) {
        bool ok = provider_is_cuda && run_permanent_original_copy(gpu, cpu);
        ok &= provider_is_cuda && run_permanent_device_table(gpu, cpu, capture);
        ggml_backend_free(gpu);
        ggml_backend_free(cpu);
        return !provider_is_cuda ? 77 : ok ? 0 : 1;
    }
#endif
    if (argc == 2 && strcmp(argv[1], "--device-publish-async-only") == 0) {
        const char * previous = getenv("GGML_MOE_CACHE_DEVICE_PUBLISH_ASYNC");
        const std::string saved = previous ? previous : "";
        set_env("GGML_MOE_CACHE_DEVICE_PUBLISH_ASYNC", "1");
        bool ok = provider_is_cuda && run_device_adaptation(gpu, cpu, capture, true, false, true);
        ok &= capture.get().find("publish_async=1") != std::string::npos;
        ok &= provider_is_cuda && run_device_adaptation_starvation(gpu, cpu, capture);
        set_env("GGML_MOE_CACHE_DEVICE_PUBLISH_ASYNC", previous ? saved.c_str() : nullptr);
        ggml_backend_free(gpu);
        ggml_backend_free(cpu);
        return !provider_is_cuda ? 77 : ok ? 0 : 1;
    }
    if (argc == 2 && strcmp(argv[1], "--hitfirst-only") == 0) {
        const bool ok = provider_is_cuda && run_hitfirst_layer(gpu, cpu, capture);
        ggml_backend_free(gpu);
        ggml_backend_free(cpu);
        return !provider_is_cuda ? 77 : ok ? 0 : 1;
    }
    if (argc == 2 && strcmp(argv[1], "--device-adaptation-starvation-only") == 0) {
        const bool ok = provider_is_cuda && run_device_adaptation_starvation(gpu, cpu, capture);
        ggml_backend_free(gpu);
        ggml_backend_free(cpu);
        return !provider_is_cuda ? 77 : ok ? 0 : 1;
    }
    if (argc == 2 && strcmp(argv[1], "--swap-whole-only") == 0) {
        bool ok = provider_is_cuda && run_swap_whole(gpu, cpu, capture);
        if (ok) { ok &= run_swap_whole(gpu, cpu, capture, true); }
        ggml_backend_free(gpu);
        ggml_backend_free(cpu);
        ggml_log_set(nullptr, nullptr);
        return !provider_is_cuda ? 77 : ok ? 0 : 1;
    }
    if (profile_warm_only && (!provider_is_cuda || !getenv("GGML_MOE_CACHE_TEST_TMPDIR"))) {
        fprintf(stderr, "--profile-warm-only requires CUDA and GGML_MOE_CACHE_TEST_TMPDIR\n");
        ggml_backend_free(gpu);
        ggml_backend_free(cpu);
        return provider_is_cuda ? 1 : 77;
    }
    if (!profile_warm_only && !run_prefetch_router(gpu, cpu)) {
        ggml_backend_free(gpu);
        ggml_backend_free(cpu);
        return 1;
    }
    ggml_init_params static_params = {
        16 * ggml_tensor_overhead(),
        nullptr,
        true,
    };
    ggml_context * static_ctx = ggml_init(static_params);
    if (!static_ctx) {
        fprintf(stderr, "failed to create tensor context\n");
        ggml_backend_free(gpu);
        ggml_backend_free(cpu);
        return 1;
    }

    ggml_tensor * weights = ggml_new_tensor_3d(
            static_ctx, GGML_TYPE_Q4_0, n_in, n_out, n_expert);
    ggml_tensor * gate_weights = ggml_new_tensor_3d(
            static_ctx, GGML_TYPE_Q4_0, n_in, n_out, n_expert);
    ggml_tensor * ids = ggml_new_tensor_2d(
            static_ctx, GGML_TYPE_I32, n_used, n_tokens);
    ggml_tensor * activations = ggml_new_tensor_3d(
            static_ctx, GGML_TYPE_F32, n_in, 1, n_tokens);
    ggml_set_name(weights, "blk.0.ffn_up_exps.weight");
    ggml_set_name(gate_weights, "blk.0.ffn_gate_exps.weight");
    ggml_set_name(ids, "moe_cache_test_ids");
    ggml_set_name(activations, "moe_cache_test_activations");

    ggml_backend_buffer_t static_buffer =
        ggml_backend_alloc_ctx_tensors(static_ctx, cpu);
    if (!static_buffer) {
        fprintf(stderr, "failed to allocate CPU tensors\n");
        ggml_free(static_ctx);
        ggml_backend_free(gpu);
        ggml_backend_free(cpu);
        return 1;
    }
    ggml_backend_buffer_set_usage(
            static_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    std::vector<float> weights_f32(ggml_nelements(weights));
    for (size_t index = 0; index < weights_f32.size(); index++) {
        weights_f32[index] =
            0.15f * std::sin((float) (index % 997) * 0.017f) +
            0.05f * std::cos((float) (index % 431) * 0.031f);
    }
    std::vector<uint8_t> weights_q4(ggml_nbytes(weights));
    const size_t quantized = ggml_quantize_chunk(
            GGML_TYPE_Q4_0, weights_f32.data(), weights_q4.data(),
            0, n_out * n_expert, n_in, nullptr);
    if (quantized != weights_q4.size()) {
        fprintf(stderr, "unexpected quantized size: %zu != %zu\n",
                quantized, weights_q4.size());
        ggml_backend_buffer_free(static_buffer);
        ggml_free(static_ctx);
        ggml_backend_free(gpu);
        ggml_backend_free(cpu);
        return 1;
    }
    ggml_backend_tensor_set(
            weights, weights_q4.data(), 0, weights_q4.size());

    std::vector<float> gate_weights_f32(ggml_nelements(gate_weights));
    for (size_t index = 0; index < gate_weights_f32.size(); index++) {
        gate_weights_f32[index] =
            0.11f * std::sin((float) (index % 881) * 0.021f) -
            0.08f * std::cos((float) (index % 389) * 0.027f);
    }
    std::vector<uint8_t> gate_weights_q4(ggml_nbytes(gate_weights));
    const size_t gate_quantized = ggml_quantize_chunk(
            GGML_TYPE_Q4_0, gate_weights_f32.data(), gate_weights_q4.data(),
            0, n_out * n_expert, n_in, nullptr);
    if (gate_quantized != gate_weights_q4.size()) {
        fprintf(stderr, "unexpected gate quantized size: %zu != %zu\n",
                gate_quantized, gate_weights_q4.size());
        ggml_backend_buffer_free(static_buffer);
        ggml_free(static_ctx);
        ggml_backend_free(gpu);
        ggml_backend_free(cpu);
        return 1;
    }
    ggml_backend_tensor_set(
            gate_weights, gate_weights_q4.data(), 0,
            gate_weights_q4.size());

    const int32_t ids_data[n_used] = { 0, 1 };
    ggml_backend_tensor_set(ids, ids_data, 0, sizeof(ids_data));
    std::vector<float> activation_data(ggml_nelements(activations));
    for (size_t index = 0; index < activation_data.size(); index++) {
        activation_data[index] =
            0.4f * std::sin((float) index * 0.07f) -
            0.2f * std::cos((float) index * 0.11f);
    }
    ggml_backend_tensor_set(
            activations, activation_data.data(), 0,
            activation_data.size() * sizeof(float));

    test_graph graph = make_graph(cpu, weights, activations, ids);
    if (!graph.ctx || !graph.buffer) {
        fprintf(stderr, "failed to create test graph\n");
        free_graph(graph);
        ggml_backend_buffer_free(static_buffer);
        ggml_free(static_ctx);
        ggml_backend_free(gpu);
        ggml_backend_free(cpu);
        return 1;
    }

    set_env("GGML_CUDA_MOE_CACHE", "0");
    if (ggml_backend_graph_compute(cpu, graph.graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "CPU reference compute failed\n");
        free_graph(graph);
        ggml_backend_buffer_free(static_buffer);
        ggml_free(static_ctx);
        ggml_backend_free(gpu);
        ggml_backend_free(cpu);
        return 1;
    }
    std::vector<float> reference(ggml_nelements(graph.out));
    ggml_backend_tensor_get(
            graph.out, reference.data(), 0, reference.size() * sizeof(float));

    if (profile_warm_only) {
        bool ok = run_prompt_warm(gpu, cpu, weights, graph, reference, capture);
        ok &= run_prompt_warm_ranking(gpu, cpu, capture);
        ok &= run_prompt_warm_decode_heat(gpu, cpu, capture);
        ok &= run_profile_persistence(gpu, cpu, capture);
        ok &= run_prompt_cache_lending(gpu, cpu, capture);
        ok &= run_profile_swap_prior(gpu, cpu, capture);
        free_graph(graph);
        ggml_backend_buffer_free(static_buffer);
        ggml_free(static_ctx);
        ggml_quantize_free();
        ggml_backend_free(gpu);
        ggml_backend_free(cpu);
        ggml_log_set(nullptr, nullptr);
        return ok ? 0 : 1;
    }

    test_graph fused_graph = make_fused_graph(
            cpu, weights, gate_weights, activations, ids);
    if (!fused_graph.ctx || !fused_graph.buffer) {
        fprintf(stderr, "failed to create fused test graph\n");
        free_graph(fused_graph);
        free_graph(graph);
        ggml_backend_buffer_free(static_buffer);
        ggml_free(static_ctx);
        ggml_backend_free(gpu);
        ggml_backend_free(cpu);
        return 1;
    }
    set_env("GGML_CUDA_MOE_CACHE", "0");
    if (ggml_backend_graph_compute(cpu, fused_graph.graph) !=
            GGML_STATUS_SUCCESS) {
        fprintf(stderr, "fused CPU reference compute failed\n");
        free_graph(fused_graph);
        free_graph(graph);
        ggml_backend_buffer_free(static_buffer);
        ggml_free(static_ctx);
        ggml_backend_free(gpu);
        ggml_backend_free(cpu);
        return 1;
    }
    std::vector<float> fused_reference(ggml_nelements(fused_graph.out));
    ggml_backend_tensor_get(
            fused_graph.out, fused_reference.data(), 0,
            fused_reference.size() * sizeof(float));

    test_graph clamped_fused_graph = make_clamped_fused_graph(
            cpu, weights, gate_weights, activations, ids);
    if (!clamped_fused_graph.ctx || !clamped_fused_graph.buffer) {
        fprintf(stderr, "failed to create clamped fused test graph\n");
        free_graph(clamped_fused_graph);
        free_graph(fused_graph);
        free_graph(graph);
        ggml_backend_buffer_free(static_buffer);
        ggml_free(static_ctx);
        ggml_backend_free(gpu);
        ggml_backend_free(cpu);
        return 1;
    }
    set_env("GGML_CUDA_MOE_CACHE", "0");
    if (ggml_backend_graph_compute(cpu, clamped_fused_graph.graph) !=
            GGML_STATUS_SUCCESS) {
        fprintf(stderr, "clamped fused CPU reference compute failed\n");
        free_graph(clamped_fused_graph);
        free_graph(fused_graph);
        free_graph(graph);
        ggml_backend_buffer_free(static_buffer);
        ggml_free(static_ctx);
        ggml_backend_free(gpu);
        ggml_backend_free(cpu);
        return 1;
    }
    std::vector<float> clamped_fused_reference(
            ggml_nelements(clamped_fused_graph.out));
    ggml_backend_tensor_get(
            clamped_fused_graph.out, clamped_fused_reference.data(), 0,
            clamped_fused_reference.size() * sizeof(float));

    bool ok = run_optional(provider_is_cuda, "cache-arena-stride", [&] {
        return run_arena_stride(gpu, cpu, weights, gate_weights, graph, reference, capture);
    });
    ok &= run_capability_queries(provider_device, cpu);
    ok &= run_invalidation_hook_coverage(cpu);
    ok &= run_optional(provider_is_cuda, "cache-prefetch-staging-legacy", [&] {
        return run_prefetch_staging(gpu, cpu, weights, capture, false);
    });
    ok &= run_optional(provider_is_cuda, "cache-prefetch-staging-device", [&] {
        return run_prefetch_staging(gpu, cpu, weights, capture, true);
    });
    ok &= run_optional(provider_is_cuda, "cache-prefetch-staging-fused", [&] {
        return run_prefetch_staging(gpu, cpu, weights, capture, true, gate_weights);
    });
    ok &= run_optional(provider_is_cuda, "cache-device-tables", [&] {
        return run_device_tables(gpu, cpu, weights, capture);
    });
    ok &= run_optional(provider_is_cuda, "cache-device-fused", [&] {
        return run_device_fused_tables(gpu, cpu, capture);
    });
    ok &= run_optional(provider_is_cuda, "cache-hitfirst-layer", [&] {
        return run_hitfirst_layer(gpu, cpu, capture);
    });
    ok &= run_optional(provider_is_cuda, "cache-device-host-budget", [&] {
        return run_device_host_budget();
    });
    ok &= run_optional(provider_is_cuda, "cache-device-adaptation", [&] {
        return run_device_adaptation(gpu, cpu, capture);
    });
    ok &= run_optional(provider_is_cuda, "cache-device-adaptation-tuning", [&] {
        return run_device_adaptation(gpu, cpu, capture, true, false, true);
    });
    ok &= run_optional(provider_is_cuda, "cache-device-adaptation-starvation", [&] {
        return run_device_adaptation_starvation(gpu, cpu, capture);
    });
    ok &= run_optional(provider_is_cuda, "cache-device-capture", [&] {
        return run_device_adaptation(gpu, cpu, capture, true);
    });
    ok &= run_optional(provider_is_cuda, "cache-device-source-reuse", [&] {
        return run_device_adaptation(gpu, cpu, capture, true, true);
    });
    ok &= run_optional(provider_is_cuda, "cache-device-argsort-view", [&] {
        return run_device_argsort_view(gpu, cpu, weights, capture);
    });
    ok &= run_optional(provider_is_cuda, "cache-device-capture-lifetime", [&] {
        return run_device_capture_lifetime(gpu, cpu, weights, capture);
    });
    ok &= run_optional(provider_is_cuda, "cache-prompt-warm", [&] {
        return run_prompt_warm(gpu, cpu, weights, graph, reference, capture);
    });
    ok &= run_optional(provider_is_cuda, "cache-prompt-warm-ranking", [&] {
        return run_prompt_warm_ranking(gpu, cpu, capture);
    });
    ok &= run_optional(provider_is_cuda, "cache-prompt-warm-decode-heat", [&] {
        return run_prompt_warm_decode_heat(gpu, cpu, capture);
    });
    ok &= run_optional(provider_is_cuda, "cache-profile-persistence", [&] {
        return run_profile_persistence(gpu, cpu, capture);
    });
    ok &= run_optional(provider_is_cuda, "cache-prompt-lending", [&] {
        return run_prompt_cache_lending(gpu, cpu, capture);
    });
    ok &= run_scenario("cache-hit", nullptr, gpu, cpu, graph, reference, capture);
    ok &= run_scenario(
            "cache-generic-mmv", nullptr, gpu, cpu, graph, reference,
            capture, "1", nullptr, "0");
    ok &= run_scenario("dispatch-fallback", "dispatch", gpu, cpu, graph, reference, capture);
    ok &= run_scenario("collect-fallback", "collect", gpu, cpu, graph, reference, capture);
    ok &= run_scenario("insert-fallback", "insert", gpu, cpu, graph, reference, capture);
    ok &= run_scenario("slab-fallback", "slab", gpu, cpu, graph, reference, capture);
    ok &= run_optional(provider_is_cuda, "cache-fused-swiglu", [&] {
        return run_scenario(
                "cache-fused-swiglu", nullptr, gpu, cpu,
                fused_graph, fused_reference, capture,
                "1", "fusion-nodes=");
    });
    ok &= run_optional(provider_is_cuda, "cache-fused-dispatch-fallback", [&] {
        return run_scenario(
                "cache-fused-dispatch-fallback", "dispatch", gpu, cpu,
                fused_graph, fused_reference, capture,
                "1", "fusion-attempts=");
    });
    ok &= run_optional(provider_is_cuda, "cache-fused-collect-fallback", [&] {
        return run_scenario(
                "cache-fused-collect-fallback", "collect", gpu, cpu,
                fused_graph, fused_reference, capture,
                "1", "fusion-nodes=");
    });
    ok &= run_optional(provider_is_cuda, "cache-fused-clamped-swiglu", [&] {
        return run_scenario(
                "cache-fused-clamped-swiglu", nullptr, gpu, cpu,
                clamped_fused_graph, clamped_fused_reference, capture,
                "1", "fusion-nodes=");
    });
    ok &= run_optional(provider_is_cuda, "cache-fused-clamped-collect-fallback", [&] {
        return run_scenario(
                "cache-fused-clamped-collect-fallback", "collect", gpu, cpu,
                clamped_fused_graph, clamped_fused_reference, capture,
                "1", "fusion-nodes=");
    });
    ok &= run_optional(provider_is_cuda, "cache-fused-partial-duplicate", [&] {
        return run_fused_partial_invalidation(
                gpu, cpu, weights, gate_weights, activations,
                weights_q4.data(), gate_weights_q4.data(),
                fused_reference, capture);
    });
    ok &= run_optional(provider_is_cuda, "cache-fused-concurrent", [&] {
        return run_fused_concurrent_sessions(
                provider_device, gpu, cpu, weights, gate_weights,
                activations, ids, fused_graph, fused_reference, capture);
    });
    ok &= run_optional(provider_is_cuda, "cache-fused-lifecycle", [&] {
        return run_fused_repeated_lifecycle(
                gpu, cpu, fused_graph, fused_reference, capture);
    });
    ok &= run_multi_token_scenario(
            gpu, cpu, weights, gate_weights, capture,
            provider_is_cuda, provider_is_cuda);
    ok &= run_mxfp4_shared_pool(gpu, cpu, capture);
    const size_t expert_size = ggml_nbytes(weights) / n_expert;
    ok &= run_precensus_invalidation(
            gpu, cpu, graph, weights,
            weights_q4.data() + (n_expert - 1) * expert_size,
            expert_size, reference, capture);

    const int32_t repeated_ids[n_used] = { 0, 0 };
    ggml_backend_tensor_set(ids, repeated_ids, 0, sizeof(repeated_ids));
    set_env("GGML_CUDA_MOE_CACHE", "0");
    if (ggml_backend_graph_compute(cpu, graph.graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "cache-invalidate: initial CPU reference compute failed\n");
        ok = false;
    } else {
        std::vector<float> old_reference(ggml_nelements(graph.out));
        ggml_backend_tensor_get(
                graph.out, old_reference.data(), 0,
                old_reference.size() * sizeof(float));

        std::vector<float> replacement_f32(n_in);
        for (size_t index = 0; index < replacement_f32.size(); index++) {
            replacement_f32[index] =
                1.25f + 0.3f * std::sin((float) index * 0.041f);
        }
        std::vector<uint8_t> replacement_q4(
                ggml_row_size(GGML_TYPE_Q4_0, n_in));
        const size_t replacement_size = ggml_quantize_chunk(
                GGML_TYPE_Q4_0, replacement_f32.data(), replacement_q4.data(),
                0, 1, n_in, nullptr);
        if (replacement_size != replacement_q4.size()) {
            fprintf(stderr, "cache-invalidate: unexpected replacement size\n");
            ok = false;
        } else {
            ok &= run_invalidation_scenario(
                    gpu, cpu, graph, weights, replacement_q4,
                    old_reference, capture);
        }
    }

    stress_fixture stress;
    if (!init_stress_fixture(stress, cpu)) {
        fprintf(stderr, "failed to initialize cache stress fixture\n");
        ok = false;
    } else {
        ok &= run_concurrent_sessions(
                provider_device, gpu, cpu, stress, capture);
        ok &= run_repeated_lifecycle(gpu, cpu, stress, capture);
        ok &= run_suspend_resume(gpu, cpu, stress, capture);
        ok &= run_fill_invalidation(gpu, cpu, stress, capture);
    }
    free_stress_fixture(stress);
    ok &= run_optional(provider_is_cuda, "cache-scope", [&] {
        return run_scope_isolation(gpu, cpu, weights);
    });
    ok &= run_explicit_session_config(gpu, cpu);
    ok &= run_optional(provider_is_cuda, "cache-policy-diagnostics", [&] {
        return run_policy_diagnostics(gpu, cpu, weights, capture);
    });
    ok &= run_optional(provider_is_cuda, "cache-cpu-overlap", [&] {
        return run_cpu_overlap_policy(gpu, cpu, weights, capture);
    });
    ok &= run_optional(provider_is_cuda, "cache-cpu-overlap-auto", [&] {
        return run_adaptive_cpu_overlap_policy(gpu, cpu, weights, capture);
    });
    ok &= run_optional(provider_is_cuda, "cache-cpu-overlap-calibration", [&] {
        return run_adaptive_cpu_overlap_calibration(gpu, cpu, weights, capture);
    });
    ok &= run_shape_liveness(gpu, cpu);
    ok &= run_optional(provider_is_cuda, "cache-shape-inventory", [&] {
        return run_exact_shape_inventory(gpu, cpu, capture);
    });
    ok &= run_complete_pool_allocation(gpu, cpu);
    ok &= run_optional(provider_is_cuda, "cache-shared-budget", [&] {
        return run_shared_budget(gpu, cpu, capture);
    });
    ok &= run_optional(provider_is_cuda, "cache-route-override", [&] {
        return run_route_override(provider_device, gpu, cpu, capture);
    });
    ok &= run_single_gpu_auto(provider_device, gpu, cpu, capture);
    ok &= run_optional(provider_is_cuda, "cache-admission-policy", [&] {
        return run_admission_policy(gpu, cpu, capture);
    });
    ok &= run_optional(provider_is_cuda, "cache-expert-keep", [&] {
        return run_expert_keep(gpu, cpu, weights, graph, reference, fused_graph, fused_reference);
    });

    free_graph(clamped_fused_graph);
    free_graph(fused_graph);
    free_graph(graph);
    ggml_backend_free(gpu);
#ifdef GGML_BACKEND_DL
    ggml_backend_unload(provider_reg);
    ggml_backend_buffer_free(static_buffer);
    static_buffer = nullptr;
    printf("cache-backend-unload: OK\n");

    ggml_backend_dev_t reloaded_device = find_provider_device();
    ggml_backend_t reloaded_gpu = reloaded_device
        ? ggml_backend_dev_init(reloaded_device, nullptr) : nullptr;
    ggml_backend_reg_t reloaded_reg = reloaded_gpu
        ? ggml_backend_dev_backend_reg(reloaded_device) : nullptr;
    configure_cache(nullptr);
    void * reloaded_session = nullptr;
    if (reloaded_gpu) {
        // The registry re-registers on load; use the fresh api so the
        // destroy callback comes from the reloaded library, not the
        // pre-unload snapshot in g_test_api.
        const ggml_moe_cache_api reloaded_api =
            ggml_moe_cache_get(reloaded_reg);
        void * backends[] = { reloaded_gpu, cpu };
        reloaded_session = reloaded_api.session_create
            ? reloaded_api.session_create(backends, 2, nullptr) : nullptr;
        if (reloaded_session) {
            reloaded_api.session_destroy(reloaded_session);
        }
    }
    const bool reload_ok = reloaded_session != nullptr;
    if (reloaded_gpu) {
        ggml_backend_free(reloaded_gpu);
        ggml_backend_unload(reloaded_reg);
    }
    printf("cache-backend-reload: %s\n", reload_ok ? "OK" : "FAIL");
    ok &= reload_ok;
#else
    (void) provider_reg;
    printf("cache-backend-unload: SKIP (static backend)\n");
#endif
    if (static_buffer) {
        ggml_backend_buffer_free(static_buffer);
    }
    ggml_free(static_ctx);
    ggml_quantize_free();
    ggml_backend_free(cpu);
    ggml_log_set(nullptr, nullptr);
    return ok ? 0 : 1;
}
