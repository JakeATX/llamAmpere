#!/usr/bin/env python3
"""DSPARKQAT long-context train docs: pick N LongBench-v2 contexts (>= MINCH chars, so ~100K tokens fit), stratified by
domain, deterministic. Writes one jsonl row per doc: id, domain, context (cut to MAXCH chars), and two tasks
(the item's own multiple-choice question with worked reasoning + one long-form task that suits the domain).
Never uses wikitext-2 test (the gate's prompt text).
usage: prep_long.py longbench_v2.json out.jsonl [N=20]"""
import json, random, sys

MINCH, MAXCH = 400_000, 700_000
QUOTA = {'Single-Document QA': 5, 'Multi-Document QA': 4, 'Code Repository Understanding': 4,
         'Long-dialogue History Understanding': 3, 'Long In-context Learning': 2, 'Long Structured Data Understanding': 2}
LONGFORM = {
    'Single-Document QA': "Write a detailed study guide for the document above: go through it section by section, list the key facts, "
                          "names and dates, explain how they connect, and finish with twenty exam questions with worked answers.",
    'Multi-Document QA': "Write a long, detailed briefing on the documents above: summarise each one in turn, then compare them, "
                         "point out where they agree and disagree, and finish with fifteen questions with worked answers.",
    'Code Repository Understanding': "Explain how the code above is organised: walk through its main modules and functions in detail, "
                                     "describe the data flow, then propose three concrete improvements and write the code for each.",
    'Long-dialogue History Understanding': "Write a detailed account of the conversation above: who says what, how the topics develop, "
                                           "the key decisions and facts, and finish with fifteen questions about it with worked answers.",
    'Long In-context Learning': "Using the material above, write a detailed tutorial that teaches its content step by step, "
                                "with many worked examples, then give fifteen practice exercises with full solutions.",
    'Long Structured Data Understanding': "Describe the data above in detail: its structure, the main entities and relations, notable "
                                          "patterns and outliers, then answer fifteen analytical questions about it with worked reasoning.",
}


def main():
    src, out = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    items = [x for x in json.load(open(src)) if len(x['context']) >= MINCH and 'wikitext' not in x['context'][:2000].lower()]
    rng = random.Random(20261123)
    rng.shuffle(items)
    quota = dict(QUOTA); scale = n / sum(quota.values())
    quota = {k: max(1, round(v * scale)) for k, v in quota.items()}
    pick = []
    for x in items:
        if quota.get(x['domain'], 0) > 0:
            quota[x['domain']] -= 1; pick.append(x)
    for x in items:                       # fill up if a domain ran short
        if len(pick) >= n:
            break
        if x not in pick:
            pick.append(x)
    with open(out, 'w') as f:
        for x in pick[:n]:
            mc = (f"Question: {x['question']}\nA) {x['choice_A']}\nB) {x['choice_B']}\nC) {x['choice_C']}\nD) {x['choice_D']}\n\n"
                  "Answer the question above about the material. Reason it through step by step and in detail, quoting the relevant "
                  "passages, rule out the wrong choices one by one, then state the letter of the correct choice.")
            f.write(json.dumps(dict(id=x['_id'], domain=x['domain'], context=x['context'][:MAXCH],
                                    tasks=[mc, LONGFORM[x['domain']]])) + '\n')
    print('docs', len(pick[:n]), {d: sum(1 for x in pick[:n] if x['domain'] == d) for d in QUOTA})


if __name__ == '__main__':
    main()
