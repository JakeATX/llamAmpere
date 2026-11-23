#include "llama-ink-stream.h"

#include "llama-impl.h"
#include "llama-model.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

// gate_up, gate, up, down (whichever the layer has)
static std::vector<ggml_tensor *> ink_stream_layer_experts(const llama_layer & layer) {
    std::vector<ggml_tensor *> res;
    for (ggml_tensor * t : { layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps }) {
        if (t != nullptr) {
            res.push_back(t);
        }
    }
    return res;
}

static bool ink_stream_host_weight(const ggml_tensor * t) {
    if (t->buffer == nullptr || t->data == nullptr || t->view_src != nullptr || t->extra != nullptr ||
            ggml_backend_buffer_get_usage(t->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS ||
            !ggml_backend_buffer_is_host(t->buffer) || !ggml_is_contiguous(t)) {
        return false;
    }
    // repacked CPU layouts differ from the device layout
    const char * bname = ggml_backend_buft_name(ggml_backend_buffer_get_type(t->buffer));
    return !strstr(bname, "REPACK") && !strstr(bname, "AMX") && !strstr(bname, "KLEIDI");
}

static int ink_stream_env(const char * name, int def) {
    const char * v = getenv(name);
    return v && *v ? atoi(v) : def;
}

bool llama_ink_stream::wanted(const llama_model & model) {
    if (ink_stream_env("LLAMA_INK_STREAM", 1) == 0) {
        return false;
    }
    return model.arch == LLM_ARCH_INKLING2 || model.arch == LLM_ARCH_INKLING2_LOOP;
}

llama_ink_stream::llama_ink_stream(const llama_model & model, ggml_backend_t backend_gpu) : backend_gpu(backend_gpu) {
    dev = ggml_backend_get_device(backend_gpu);
    if (dev == nullptr || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU ||
            model.devices.size() != 1 || model.devices[0].dev != dev) {
        return; // one GPU: every visit computes on it
    }
    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(dev, &props);
    if (!props.caps.events || !props.caps.async) {
        LLAMA_LOG_WARN("%s: [INKSTREAM] the device has no events or async copies, the streamer is off\n", __func__);
        return;
    }
    buft_dev = ggml_backend_dev_buffer_type(dev);
    const size_t align = std::max<size_t>(256, ggml_backend_buft_get_alignment(buft_dev));

    const int n_layer = (int) model.layers.size();
    visits.resize(n_layer);
    view_of.resize(n_layer);
    size_t n_views = 0;
    for (int il = 0; il < n_layer; ++il) {
        auto experts = ink_stream_layer_experts(model.layers[il]);
        if (experts.empty() || !std::all_of(experts.begin(), experts.end(), ink_stream_host_weight)) {
            continue; // resident (or no) experts: the normal path
        }
        bool ok = true;
        for (const ggml_tensor * t : experts) {
            const char * tname = ggml_type_name(t->type);
            // these types are converted on upload, so their device bytes differ from the host bytes
            ok = ok && strncmp(tname, "tq", 2) != 0 && strncmp(tname, "turbo", 5) != 0;
        }
        if (!ok) {
            continue;
        }
        visit & v = visits[il];
        size_t off = 0;
        for (const ggml_tensor * t : experts) {
            off = GGML_PAD(off, align);
            v.host.push_back(t);
            v.offs.push_back(off);
            // the device alloc size (MMQ row padding) plus a zeroed tail, as the cache banks and the op offload copies keep
            off += ggml_backend_buft_get_alloc_size(buft_dev, t) + 512;
        }
        slot_size = std::max(slot_size, GGML_PAD(off, align));
        v.pos = (int) order.size();
        order.push_back(il);
        n_views += experts.size();
    }
    if (order.size() < 2) {
        if (!order.empty()) {
            LLAMA_LOG_WARN("%s: [INKSTREAM] only one visit has host experts, the streamer is off\n", __func__);
        }
        return;
    }
    // a slot is fixed per visit, alternating in graph order; an odd count wraps two same-slot visits, handled in copy()
    for (int il : order) {
        visits[il].slot = visits[il].pos % 2;
    }

    {
        ggml_init_params params = { (n_views + 2)*ggml_tensor_overhead(), nullptr, true };
        ctx_meta = ggml_init(params);
        ggml_init_params mparams = { order.size()*ggml_tensor_overhead(), nullptr, true };
        ctx_map = ggml_init(mparams);
    }
    if (!ctx_meta || !ctx_map) {
        return;
    }
    const int64_t n_expert = visits[order[0]].host[0]->ne[2];
    for (int il : order) {
        visit & v = visits[il];
        for (const ggml_tensor * t : v.host) {
            if (t->ne[2] != n_expert) {
                return;
            }
            ggml_tensor * view = ggml_new_tensor(ctx_meta, t->type, GGML_MAX_DIMS, t->ne);
            GGML_ASSERT(view->nb[1] == t->nb[1] && view->nb[2] == t->nb[2]);
            ggml_format_name(view, "ink_stream.%d.%s", il, t->name);
            v.views.push_back(view);
            view_of[il][t] = view;
        }
        v.map = ggml_new_tensor_2d(ctx_map, GGML_TYPE_I32, 1, n_expert);
        ggml_format_name(v.map, "ink_stream.map-%d", il);
        map_visit[v.map] = il;
    }
    for (int s = 0; s < 2; ++s) {
        raw[s] = ggml_new_tensor_1d(ctx_meta, GGML_TYPE_I8, (int64_t) slot_size);
        ggml_format_name(raw[s], "ink_stream.slot%d", s);
    }
    buf_map = ggml_backend_alloc_ctx_tensors_from_buft(ctx_map, ggml_backend_cpu_buffer_type());
    if (!buf_map) {
        return;
    }
    // a host weight: the scheduler starts a split at its lookup and copies it with the copy callback
    ggml_backend_buffer_set_usage(buf_map, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (int il : order) {
        int32_t * d = (int32_t *) visits[il].map->data;
        for (int64_t e = 0; e < n_expert; ++e) {
            d[e] = (int32_t) e;
        }
    }

    n_workers = std::clamp(ink_stream_env("LLAMA_INK_STREAM_THREADS", 4), 1, 32);
    piece_max = (size_t) std::clamp(ink_stream_env("LLAMA_INK_STREAM_PIECE_MIB", 8), 1, 256) << 20;
    init_ok = true;
    LLAMA_LOG_INFO("%s: [INKSTREAM] %zu of %d visits stream their experts through 2 slots of %.1f MiB (%d copy workers)\n",
            __func__, order.size(), n_layer, slot_size/1048576.0, n_workers);
}

llama_ink_stream::~llama_ink_stream() {
    disarm();
    if (buf_map) {
        ggml_backend_buffer_free(buf_map);
    }
    if (ctx_meta) {
        ggml_free(ctx_meta);
    }
    if (ctx_map) {
        ggml_free(ctx_map);
    }
}

void llama_ink_stream::free_all() {
    for (auto & w : wst) {
        for (int b = 0; b < 2; ++b) {
            if (w.ev[b]) { ggml_backend_event_free(w.ev[b]); }
            if (w.buf[b]) { ggml_backend_buffer_free(w.buf[b]); }
        }
    }
    wst.clear();
    for (int s = 0; s < 2; ++s) {
        if (ev_free[s])  { ggml_backend_event_free(ev_free[s]);  ev_free[s]  = nullptr; }
        if (ev_ready[s]) { ggml_backend_event_free(ev_ready[s]); ev_ready[s] = nullptr; }
        issued[s] = -1;
    }
    if (backend_copy) {
        ggml_backend_free(backend_copy);
        backend_copy = nullptr;
    }
    if (buf_slot) {
        ggml_backend_buffer_free(buf_slot);
        buf_slot = nullptr;
    }
    // the views and raw slots point into the freed buffer: detach them so arm() can place them again
    for (ggml_tensor * t = ggml_get_first_tensor(ctx_meta); t; t = ggml_get_next_tensor(ctx_meta, t)) {
        t->data   = nullptr;
        t->buffer = nullptr;
    }
}

bool llama_ink_stream::arm() {
    if (!init_ok) {
        return false;
    }
    if (is_armed) {
        return true;
    }
    buf_slot = ggml_backend_buft_alloc_buffer(buft_dev, 2*slot_size);
    if (!buf_slot) {
        LLAMA_LOG_WARN("%s: [INKSTREAM] cannot allocate 2 x %.1f MiB of expert slots\n", __func__, slot_size/1048576.0);
        free_all();
        return false;
    }
    ggml_backend_buffer_clear(buf_slot, 0); // the padding after each expert tensor stays zero
    ggml_backend_buffer_set_usage(buf_slot, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    char * base = (char *) ggml_backend_buffer_get_base(buf_slot);
    bool ok = true;
    for (int s = 0; s < 2 && ok; ++s) {
        ok = ggml_backend_tensor_alloc(buf_slot, raw[s], base + s*slot_size) == GGML_STATUS_SUCCESS;
    }
    for (int il : order) {
        visit & v = visits[il];
        for (size_t i = 0; i < v.views.size() && ok; ++i) {
            ok = ggml_backend_tensor_alloc(buf_slot, v.views[i], base + v.slot*slot_size + v.offs[i]) == GGML_STATUS_SUCCESS;
        }
    }
    backend_copy = ok ? ggml_backend_dev_init(dev, nullptr) : nullptr;
    ok = ok && backend_copy != nullptr;
    for (int s = 0; s < 2 && ok; ++s) {
        ev_free[s]  = ggml_backend_event_new(dev);
        ev_ready[s] = ggml_backend_event_new(dev);
        ok = ev_free[s] && ev_ready[s];
    }
    // pinned staging: two pieces per worker
    ggml_backend_buffer_type_t buft_host = ggml_backend_dev_host_buffer_type(dev);
    ok = ok && buft_host != nullptr;
    wst.resize(ok ? n_workers : 0);
    for (auto & w : wst) {
        for (int b = 0; b < 2 && ok; ++b) {
            w.buf[b] = ggml_backend_buft_alloc_buffer(buft_host, piece_max);
            w.ev[b]  = ggml_backend_event_new(dev);
            ok = w.buf[b] && w.ev[b];
            if (ok) {
                w.data[b] = (char *) ggml_backend_buffer_get_base(w.buf[b]);
            }
        }
    }
    if (!ok) {
        LLAMA_LOG_WARN("%s: [INKSTREAM] cannot set up the copy pipeline\n", __func__);
        free_all();
        return false;
    }
    ggml_backend_synchronize(backend_gpu); // the cleared slots are visible to both streams

    stop = false;
    n_busy = 0;
    req_gen = done_gen = 0;
    req_slot = -1;
    pieces.clear();
    for (int w = 0; w < n_workers; ++w) {
        threads.emplace_back(&llama_ink_stream::worker, this, w);
    }
    is_armed = true;
    return true;
}

void llama_ink_stream::disarm() {
    if (!is_armed) {
        return;
    }
    wait_issued();
    {
        std::lock_guard<std::mutex> lk(mtx);
        stop = true;
    }
    cv.notify_all();
    for (auto & t : threads) {
        t.join();
    }
    threads.clear();
    ggml_backend_synchronize(backend_copy);
    ggml_backend_synchronize(backend_gpu);
    free_all();
    is_armed = false;
}

ggml_tensor * llama_ink_stream::get_map(int il) const {
    if (!is_armed || il < 0 || il >= (int) visits.size() || visits[il].slot < 0) {
        return nullptr;
    }
    return visits[il].map;
}

ggml_tensor * llama_ink_stream::get_experts(const ggml_tensor * w, int il) const {
    if (!is_armed || il < 0 || il >= (int) visits.size()) {
        return nullptr;
    }
    const auto it = view_of[il].find(w);
    return it != view_of[il].end() ? it->second : nullptr;
}

llama_ink_stream_stats llama_ink_stream::take_stats() {
    llama_ink_stream_stats res = st;
    st = {};
    return res;
}

// queue the upload of visit il into its slot; no request is in flight here (copy() waits for each before the next)
void llama_ink_stream::issue(int il, ggml_backend_t backend) {
    const visit & v = visits[il];
    const int s = v.slot;
    // the compute stream has queued every read of this slot by its previous visit
    ggml_backend_event_record(ev_free[s], backend);
    std::lock_guard<std::mutex> lk(mtx);
    GGML_ASSERT(done_gen == req_gen && n_busy == 0);
    {
        std::lock_guard<std::mutex> ls(mtx_stream);
        ggml_backend_event_wait(backend_copy, ev_free[s]);
    }
    pieces.clear();
    for (size_t i = 0; i < v.host.size(); ++i) {
        const ggml_tensor * t = v.host[i];
        const size_t nb = ggml_nbytes(t);
        for (size_t o = 0; o < nb; o += piece_max) {
            pieces.push_back({ (const char *) t->data + o, v.offs[i] + o, std::min(piece_max, nb - o) });
        }
    }
    req_slot = s;
    pieces_done = 0;
    next_piece.store(0);
    ++req_gen;
    issued[s] = il;
    st.uploads++;
    cv.notify_all();
}

void llama_ink_stream::wait_issued() {
    std::unique_lock<std::mutex> lk(mtx);
    cv.wait(lk, [&] { return done_gen == req_gen && n_busy == 0; });
}

void llama_ink_stream::worker(int w) {
    worker_state & ws = wst[w];
    uint64_t seen = 0;
    for (;;) {
        size_t n;
        int s;
        {
            std::unique_lock<std::mutex> lk(mtx);
            cv.wait(lk, [&] { return stop || req_gen != seen; });
            if (stop) {
                return;
            }
            // issue() changes the request only when no worker is busy, so this snapshot stays valid
            seen = req_gen;
            n    = pieces.size();
            s    = req_slot;
            n_busy++;
        }
        size_t mine = 0;
        for (size_t i = next_piece.fetch_add(1); i < n; i = next_piece.fetch_add(1)) {
            const piece & p = pieces[i];
            const int b = ws.next;
            ws.next ^= 1;
            if (ws.used[b]) {
                ggml_backend_event_synchronize(ws.ev[b]); // the previous upload from this staging buffer is done
            }
            memcpy(ws.data[b], p.src, p.len);
            {
                std::lock_guard<std::mutex> ls(mtx_stream);
                ggml_backend_tensor_set_async(backend_copy, raw[s], ws.data[b], p.dst_offs, p.len);
                ggml_backend_event_record(ws.ev[b], backend_copy);
            }
            ws.used[b] = true;
            ++mine;
        }
        std::lock_guard<std::mutex> lk(mtx);
        pieces_done += mine;
        if (pieces_done == n && done_gen != seen) {
            // every piece is on the copy stream: the slot is ready when the stream reaches this event
            {
                std::lock_guard<std::mutex> ls(mtx_stream);
                ggml_backend_event_record(ev_ready[s], backend_copy);
            }
            for (const piece & p : pieces) {
                st.bytes_h2d += p.len;
            }
            done_gen = seen;
        }
        n_busy--;
        cv.notify_all();
    }
}

bool llama_ink_stream::copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst) {
    const auto it = map_visit.find(src);
    if (it == map_visit.end()) {
        return false;
    }
    if (is_armed) {
        GGML_ASSERT(backend == backend_gpu);
        const int il = it->second;
        const visit & v = visits[il];
        const int64_t t0 = ggml_time_us();
        if (issued[v.slot] != il) {
            wait_issued();
            issue(il, backend);
        }
        wait_issued();                                    // its ready event is recorded
        st.t_wait_us += ggml_time_us() - t0;
        ggml_backend_event_wait(backend, ev_ready[v.slot]); // the compute stream waits for the slot
        // the next streamed visit goes into the other slot while this one computes (wrapping into the next ubatch)
        const int nx = order[(v.pos + 1) % order.size()];
        if (visits[nx].slot != v.slot && issued[visits[nx].slot] != nx) {
            issue(nx, backend);
        }
    }
    ggml_backend_tensor_set_async(backend, dst, src->data, 0, ggml_nbytes(src));
    return true;
}
