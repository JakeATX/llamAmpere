// Equivalence + timing test for llama_grammar_clone_impl.
//
// The clone has to redirect every stack pointer from the source rules to the
// same (rule, offset) in the copied rules. The old code did this with a nested
// search over every rule element; the new code uses a sorted rule-start table.
// This test keeps a copy of the old search as the reference and checks that
// both give the same (rule, offset) for every stack pointer.

#ifdef NDEBUG
#undef NDEBUG
#endif

#include "../src/llama-grammar.h"

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <utility>
#include <vector>

// the clone as it was before the lookup table (reference)
static llama_grammar * clone_ref(const llama_grammar & grammar) {
    auto * result = new llama_grammar {
        grammar.vocab,
        grammar.rules,
        grammar.stacks,
        grammar.partial_utf8,
        grammar.lazy,
        grammar.awaiting_trigger,
        grammar.trigger_buffer,
        grammar.trigger_buffer_positions,
        grammar.trigger_tokens,
        grammar.trigger_patterns,
    };
    for (size_t is = 0; is < result->stacks.size(); is++) {
        for (size_t ie = 0; ie < result->stacks[is].size(); ie++) {
            for (size_t ir0 = 0; ir0 < grammar.rules.size(); ir0++) {
                for (size_t ir1 = 0; ir1 < grammar.rules[ir0].size(); ir1++) {
                    if (grammar.stacks[is][ie] == &grammar.rules[ir0][ir1]) {
                        result->stacks[is][ie] = &result->rules[ir0][ir1];
                    }
                }
            }
        }
    }
    return result;
}

static const std::pair<size_t, size_t> NOT_FOUND = { SIZE_MAX, SIZE_MAX };

// brute force (rule, offset) of a pointer inside rules
static std::pair<size_t, size_t> locate(const llama_grammar_rules & rules, const llama_grammar_element * p) {
    for (size_t ir0 = 0; ir0 < rules.size(); ir0++) {
        for (size_t ir1 = 0; ir1 < rules[ir0].size(); ir1++) {
            if (p == &rules[ir0][ir1]) {
                return { ir0, ir1 };
            }
        }
    }
    return NOT_FOUND;
}

static size_t n_checked = 0;

static void check_clone(const llama_grammar & g, const char * what) {
    llama_grammar * a = llama_grammar_clone_impl(g);
    llama_grammar * b = clone_ref(g);

    assert(a->rules.size()  == g.rules.size());
    assert(a->stacks.size() == g.stacks.size());
    assert(b->stacks.size() == g.stacks.size());

    for (size_t is = 0; is < g.stacks.size(); is++) {
        assert(a->stacks[is].size() == g.stacks[is].size());
        for (size_t ie = 0; ie < g.stacks[is].size(); ie++) {
            const auto src = locate(g.rules,  g.stacks[is][ie]);
            const auto ref = locate(b->rules, b->stacks[is][ie]);
            const auto got = locate(a->rules, a->stacks[is][ie]);
            if (src == NOT_FOUND) {
                // not a pointer into the rules: both versions leave it as it is
                assert(b->stacks[is][ie] == g.stacks[is][ie]);
                assert(a->stacks[is][ie] == g.stacks[is][ie]);
            } else if (got != src || ref != src) {
                fprintf(stderr, "%s: stack %zu elem %zu: src (%zu,%zu) ref (%zu,%zu) got (%zu,%zu)\n",
                        what, is, ie, src.first, src.second, ref.first, ref.second, got.first, got.second);
                assert(false);
            }
            n_checked++;
        }
    }

    llama_grammar_free_impl(a);
    llama_grammar_free_impl(b);
}

static llama_grammar * build(const std::string & gbnf) {
    llama_grammar * g = llama_grammar_init_impl(nullptr, gbnf.c_str(), "root", false, nullptr, 0, nullptr, 0);
    assert(g != nullptr);
    return g;
}

static void feed(llama_grammar * g, const std::string & text) {
    for (unsigned char c : text) {
        llama_grammar_accept(g, c);
        assert(!llama_grammar_get_stacks(g).empty());
    }
}

// a tool-call grammar shaped like the ones the server builds from tool schemas:
// one alternative and a few rules per tool, shared value rules
static std::string tool_grammar(int n_tools) {
    std::string s;
    s += "ws ::= [ \\t\\n]*\n";
    s += "string ::= \"\\\"\" [a-zA-Z0-9_ ]* \"\\\"\"\n";
    s += "number ::= \"-\"? [0-9]+ (\".\" [0-9]+)?\n";
    s += "boolean ::= \"true\" | \"false\"\n";
    s += "value ::= string | number | boolean\n";
    s += "freeform ::= \"{\" ws \"\\\"text\\\":\" ws string ws \"}\"\n";
    std::string root = "root ::= freeform";
    for (int i = 0; i < n_tools; i++) {
        const std::string id = std::to_string(i);
        s += "tool-" + id + "-args ::= \"{\" ws \"\\\"path_" + id + "\\\":\" ws string "
             "(\",\" ws \"\\\"count_" + id + "\\\":\" ws number)? "
             "(\",\" ws \"\\\"flag_" + id + "\\\":\" ws boolean)? "
             "(\",\" ws \"\\\"extra_" + id + "\\\":\" ws value)? ws \"}\"\n";
        s += "tool-" + id + " ::= \"{\" ws \"\\\"name\\\":\" ws \"\\\"tool_" + id + "\\\"\" ws \",\" ws "
             "\"\\\"arguments\\\":\" ws tool-" + id + "-args ws \"}\"\n";
        root += " | tool-" + id;
    }
    s += root + "\n";
    return s;
}

// random grammar with single-char literals and forward rule references only
// (no left recursion); every alternative starts with a literal
static std::string random_grammar(std::mt19937 & rng, int n_rules) {
    std::string s;
    for (int r = 0; r < n_rules; r++) {
        s += (r == 0 ? std::string("root") : "r" + std::to_string(r)) + " ::= ";
        const int n_alt = 1 + rng() % 3;
        for (int a = 0; a < n_alt; a++) {
            if (a > 0) {
                s += " | ";
            }
            s += std::string("\"") + (char) ('a' + rng() % 26) + "\"";
            const int n_sym = rng() % 5;
            for (int k = 0; k < n_sym; k++) {
                if (r + 1 < n_rules && rng() % 2 == 0) {
                    s += " r" + std::to_string(r + 1 + rng() % (n_rules - r - 1));
                } else {
                    s += std::string(" \"") + (char) ('a' + rng() % 26) + "\"";
                }
            }
        }
        s += "\n";
    }
    return s;
}

template <typename F>
static double ns_per_call(int iters, F && f) {
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; i++) {
        f();
    }
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::nano>(t1 - t0).count() / iters;
}

int main() {
    // 1. tool-call grammars with 0 / 20 / 107 / 200 tools, at several points of a call
    for (int n_tools : { 0, 20, 107, 200 }) {
        const std::string gbnf = tool_grammar(n_tools);
        const int last = n_tools - 1;
        std::vector<std::string> prefixes = { "", "{", "{ \"" };
        if (n_tools > 0) {
            const std::string id = std::to_string(last);
            prefixes.push_back("{\"name\": \"tool_");
            prefixes.push_back("{\"name\": \"tool_" + id + "\", \"arguments\": {\"path_" + id + "\": \"src_ma");
            prefixes.push_back("{\"name\": \"tool_" + id + "\", \"arguments\": {\"path_" + id + "\": \"a\", \"count_" + id + "\": 12");
            prefixes.push_back("{\"name\": \"tool_" + id + "\", \"arguments\": {\"path_" + id + "\": \"a\", \"extra_" + id + "\": ");
        } else {
            prefixes.push_back("{\"text\": \"hel");
        }
        for (const auto & prefix : prefixes) {
            llama_grammar * g = build(gbnf);
            feed(g, prefix);
            check_clone(*g, "tools");
            // a clone of a clone must hold too (the server clones the sampler again each round)
            llama_grammar * c = llama_grammar_clone_impl(*g);
            check_clone(*c, "tools/clone-of-clone");
            llama_grammar_free_impl(c);
            llama_grammar_free_impl(g);
        }
    }

    // 2. random grammars, random accepted walks
    std::mt19937 rng(20261111);
    for (int it = 0; it < 300; it++) {
        llama_grammar * g = build(random_grammar(rng, 1 + rng() % 40));
        const int n_steps = rng() % 12;
        for (int st = 0; st <= n_steps; st++) {
            check_clone(*g, "random");
            const auto & stacks = llama_grammar_get_stacks(g);
            std::vector<uint32_t> next;
            for (const auto & stack : stacks) {
                if (!stack.empty() && stack.back()->type == LLAMA_GRETYPE_CHAR) {
                    next.push_back(stack.back()->value);
                }
            }
            if (next.empty()) {
                break;
            }
            llama_grammar_accept(g, next[rng() % next.size()]);
        }
        llama_grammar_free_impl(g);
    }

    // 3. a pointer that is not inside the rules stays untouched in both versions
    {
        llama_grammar * g = build(tool_grammar(5));
        static const llama_grammar_element foreign = { LLAMA_GRETYPE_END, 0 };
        auto & stacks = llama_grammar_get_stacks(g);
        assert(!stacks.empty());
        stacks[0].push_back(&foreign);
        check_clone(*g, "foreign");
        llama_grammar_free_impl(g);
    }

    printf("clone equivalence: %zu stack pointers checked, all equal\n", n_checked);

    // 4. timing: old search vs lookup table (whole clone, rules copy included)
    for (int n_tools : { 0, 20, 107, 200 }) {
        llama_grammar * g = build(tool_grammar(n_tools));
        feed(g, n_tools > 0 ? "{\"name\": \"tool_" : "{\"text\": \"hel");
        size_t n_elems = 0;
        for (const auto & rule : g->rules) {
            n_elems += rule.size();
        }
        size_t n_ptrs = 0;
        for (const auto & stack : g->stacks) {
            n_ptrs += stack.size();
        }
        const int iters = 200;
        const double ns_old = ns_per_call(iters, [&]() { llama_grammar_free_impl(clone_ref(*g)); });
        const double ns_new = ns_per_call(iters, [&]() { llama_grammar_free_impl(llama_grammar_clone_impl(*g)); });
        printf("timing tools=%3d rules=%5zu elems=%6zu stacks=%4zu ptrs=%5zu  old %10.1f us  new %10.1f us  x%.2f\n",
               n_tools, g->rules.size(), n_elems, g->stacks.size(), n_ptrs, ns_old / 1e3, ns_new / 1e3, ns_old / ns_new);
        llama_grammar_free_impl(g);
    }

    return 0;
}
