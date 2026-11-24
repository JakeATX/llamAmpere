#pragma once

// [#250] PARSAMPLE: exact parallel sampled-token selection across slots (LLAMA_SERVER_PAR_SAMPLE=N, default off).
//
// After a multi-user decode call every generating slot samples (or verifies a draft) from its own rows of the same
// logits. Each slot has its own sampler chain, RNG and grammar, and the logits are read-only once the context is
// synchronized, so the per-slot work can run side by side. The server runs only the sampler calls in this pool, then
// does every side effect (accept bookkeeping, checkpoint restores, streaming, stop checks) on its own thread in the
// usual slot order. The tokens are the ones the sequential loop picks: same sampler state, same RNG draws.
//
// N = worker threads (the server thread is one of them); N <= 1 or unset = the sequential loop. Jobs smaller than
// two slots run inline.

#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

struct server_par_sample_pool {
    // LLAMA_SERVER_PAR_SAMPLE=N: N threads in total (the caller included); unset/0/1 = off
    static int env_threads() {
        static const int n = [] {
            const char * e = std::getenv("LLAMA_SERVER_PAR_SAMPLE");
            const int v = e && *e ? std::atoi(e) : 0;
            return v > 1 ? std::min(v, 64) : 0;
        }();
        return n;
    }

    explicit server_par_sample_pool(int n_threads) {
        for (int i = 1; i < n_threads; ++i) {
            workers.emplace_back([this] { worker_loop(); });
        }
    }

    ~server_par_sample_pool() {
        {
            std::lock_guard<std::mutex> lk(mtx);
            stop = true;
        }
        cv_work.notify_all();
        for (auto & t : workers) {
            t.join();
        }
    }

    server_par_sample_pool(const server_par_sample_pool &) = delete;
    server_par_sample_pool & operator=(const server_par_sample_pool &) = delete;

    // run fn(0) .. fn(n - 1); job i runs on exactly one thread; returns when all are done
    void run(int n, const std::function<void(int)> & fn) {
        if (n <= 0) {
            return;
        }
        if (n == 1 || workers.empty()) {
            for (int i = 0; i < n; ++i) {
                fn(i);
            }
            return;
        }
        {
            std::lock_guard<std::mutex> lk(mtx);
            job      = &fn;
            n_jobs   = n;
            next     = 0;
            n_done   = 0;
            ++epoch;
        }
        cv_work.notify_all();
        drain(fn, epoch_now());
        std::unique_lock<std::mutex> lk(mtx);
        cv_done.wait(lk, [&] { return n_done == n_jobs; });
        job = nullptr;
    }

    int n_threads() const { return (int) workers.size() + 1; }

private:
    uint64_t epoch_now() {
        std::lock_guard<std::mutex> lk(mtx);
        return epoch;
    }

    // take jobs of run `ep` until none are left (each index handed out once under the lock); a worker that wakes late
    // never takes an index of a later run with this run's function
    void drain(const std::function<void(int)> & fn, uint64_t ep) {
        for (;;) {
            int i;
            {
                std::lock_guard<std::mutex> lk(mtx);
                if (epoch != ep || next >= n_jobs) {
                    return;
                }
                i = next++;
            }
            fn(i);
            bool last;
            {
                std::lock_guard<std::mutex> lk(mtx);
                last = ++n_done == n_jobs;
            }
            if (last) {
                cv_done.notify_all();
            }
        }
    }

    void worker_loop() {
        uint64_t seen = 0;
        for (;;) {
            const std::function<void(int)> * fn = nullptr;
            {
                std::unique_lock<std::mutex> lk(mtx);
                cv_work.wait(lk, [&] { return stop || (epoch != seen && job != nullptr); });
                if (stop) {
                    return;
                }
                seen = epoch;
                fn   = job;
            }
            drain(*fn, seen);
        }
    }

    std::vector<std::thread> workers;
    std::mutex               mtx;
    std::condition_variable  cv_work;
    std::condition_variable  cv_done;
    const std::function<void(int)> * job = nullptr;
    int      n_jobs = 0;
    int      next   = 0;
    int      n_done = 0;
    uint64_t epoch  = 0;
    bool     stop   = false;
};
