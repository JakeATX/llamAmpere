#!/bin/bash
# DRAFT-WORDWT 24ink 100K acceptance replay (WRITTEN, NOT RUN): shipped vs WORD-weighted drafter on the Inkling ship file,
# --preset 24ink (the ship 24 GB config: 4/4 KV, ub 1024, DSpark d4 mb), drafter swapped with -md, everything else equal.
# Same 3 chat cells at ~100K fill as the acceptance-by-class study (q88 coding/agentic/rag trajectories cut at an assistant
# turn end, + follow-up turns, >= 5,120-token replies, temp 1 / top_k 20 / top_p 0.95), per-round LLAMA_SPEC_TOKLOG.
# Boots ABBA: ship s1, new s1, new s2, ship s2 (one boot = 3 cells, ~14 min on a 3090 -> ~1.0 GPU-h for 4 boots).
# WORD here = class of the DRAFTED token (as in the study); the training loss weights by the target's token, the two agree
# on every accepted slot. Output per arm: tok/round, accepted/drafted, WORD-class acceptance (pooled + per k) + lift (the number the loss targets), decode tok/s
# (secondary; the speed verdict is the inkbench ABBA x6 gate, gate.sh, run only if this replay shows a gain).
# env (under cardq/rentq): NEW=<...-Q8_0-fold.gguf> [GPU=1] [SEEDS="1 2"] [BIN=toklog build] [TAG]
set -u
[[ ${PARENT_HOLDS_GPU_LOCK:-0} = 1 ]] || { echo "run under cardq/rentq"; exit 9; }
SHIP=/mnt/data/inkling2/Inkling-2-Mini-DSpark-Q8_0-fold.gguf; NEW=${NEW:?NEW drafter gguf}
for f in $SHIP $NEW; do sz=$(stat -c %s $f); [ "$sz" = 453522464 ] || { echo "ABORT $f is $sz B, not the shipped 453,522,464 B Q8_0 fold"; exit 3; }; done
H=/mnt/data/mtppunct/h; V=/home/jakek/mtppunct/vocab_ink.json; O=/home/jakek/mtppunct/out; T=${TAG:-ww}
CELLS=100k:coding,100k:agentic,100k:rag; SEEDS=${SEEDS:-1 2}
for s in $SEEDS; do
  if (( s % 2 )); then ord="ship new"; else ord="new ship"; fi
  for a in $ord; do
    f=$SHIP; [ $a = new ] && f=$NEW
    M=ink CELLS=$CELLS TAG=$T-$a-s$s SEED=$s XARGS="-md $f" GPU=${GPU:-1} timeout 1800 bash $H/job.sh > $O/$T-$a-s$s.job.log 2>&1
    grep -q 'JOB_DONE rc=0' $O/$T-$a-s$s.job.log || { echo "ABORT $a s$s: boot failed: $(tail -2 $O/$T-$a-s$s.job.log | tr '\n' ' ')"; exit 6; }
    grep -a -m1 "card preset 24ink: drafter\|draft-dspark" $O/$T-$a-s$s/server.log | cut -c1-200
    grep -aq "$(basename $f)" $O/$T-$a-s$s/server.log || { echo "ABORT $a s$s: server did not load $(basename $f)"; exit 4; }
    echo "BOOT $a s$s $(tail -1 $O/$T-$a-s$s.job.log)"
  done
done
for a in ship new; do
  runs=$(for s in $SEEDS; do printf "%s," $O/$T-$a-s$s; done); runs=${runs%,}
  python3 -I $H/ana.py --vocab $V --runs $runs --json $O/$T-$a.ana.json > $O/$T-$a.ana.txt 2>&1 || { echo "ana $a failed"; exit 5; }
done
python3 -I - $O $T "$SEEDS" <<'PY'
import json, sys, statistics as st
O, T, seeds = sys.argv[1], sys.argv[2], sys.argv[3].split()
R = {}
for a in ('ship', 'new'):
    j = json.load(open(f'{O}/{T}-{a}.ana.json')); sm = j['summary']
    w = next(x for x in j['table_oc_mb'] if x['cls'] == 'WORD')
    dh = {int(k): v for k, v in sm['depth_hist'].items()}; depth = sum(k * v for k, v in dh.items()) / sum(dh.values())
    n = sum(x['n'] for x in j['table_oc_mb'])
    tps = [c['decode_tok_s'] for s in seeds for c in map(json.loads, open(f'{O}/{T}-{a}-s{s}/cells.jsonl'))]
    R[a] = dict(tok_round=sm['tokens'] / sm['rounds'], acc=(sm['tokens'] / sm['rounds'] - 1) / depth, word_byk=w['byk'], word_acc=w['rate'], word_ci=(w['lo'], w['hi']), word_lift=w['lift'],
                word_share=w['n'] / n, tps=st.mean(tps))
    r = R[a]
    print(f"ARM {a:4s} tok/round {r['tok_round']:.3f} accepted/drafted {r['acc']:.3f} WORD acc {r['word_acc']:.3f} "
          f"[{r['word_ci'][0]:.3f},{r['word_ci'][1]:.3f}] lift {r['word_lift']:.3f} share {r['word_share']:.2f} decode {r['tps']:.1f} tok/s"
          f" | WORD by k {{k: round(v[0], 3) for k, v in r['word_byk'].items()}}")
s, n = R['ship'], R['new']
print(f"DELTA new vs ship: tok/round {100 * (n['tok_round'] / s['tok_round'] - 1):+.2f}%  acc {n['acc'] - s['acc']:+.3f}  "
      f"WORD acc {n['word_acc'] - s['word_acc']:+.3f}  decode {100 * (n['tps'] / s['tps'] - 1):+.2f}% (secondary)")
print('VERDICT', 'GATE (run gate.sh ABBA x6 at 100K)' if n['tok_round'] > s['tok_round'] else 'DROP (no tok/round gain at 100K)')
PY
echo JOB_DONE
