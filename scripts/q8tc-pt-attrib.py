#!/usr/bin/env python3
# Q8TC-PT attribution (SURGE3-Q8TC2): which Q8_0 dense launches can run mode 1 (plain f16 activations) under the mode 2
# default without raising E21 KLD. Two subcommands:
#   arms  BIN [N_LAYERS=24] [SPLIT=4]   -> one kld.sh ARMS string: mode 2 reference, mode 1 everywhere, and for each kind
#                                          (g = grouped QKV launch, o = plain attn_output launch) the layer blocks of SPLIT
#                                          layers run in mode 1 (GGML_CUDA_Q8TC_M1), the rest mode 2.
#   score DIR [REF=m2]                   -> per arm over the per-token E21 csvs (<arm>.ub<UB>.<domain>.csv): mean KLD,
#                                          delta vs REF with a paired standard error, tokens that moved most, and a
#                                          suggested GGML_CUDA_Q8TC_M1 = the blocks whose delta is within 1 SE of 0.
import csv, glob, math, os, sys, collections

def arms(b, n=24, split=4):
    out = [f"m2|{b}|GGML_CUDA_Q8TC=2", f"m1|{b}|GGML_CUDA_Q8TC=1"]
    for kind in "go":
        for lo in range(0, n, split):
            hi = min(n, lo + split) - 1
            out.append(f"{kind}{lo}_{hi}|{b}|GGML_CUDA_Q8TC=2,GGML_CUDA_Q8TC_M1={kind}{lo}-{hi}")
    print(" ".join(out))

def load(d):
    R = collections.defaultdict(dict)   # (arm, ub) -> {(domain, pos): kld}
    for f in glob.glob(os.path.join(d, "*.csv")):
        arm, ub, dom = os.path.basename(f)[:-4].rsplit(".", 2)
        for r in csv.DictReader(open(f)):
            R[(arm, ub)][(dom, int(r["pos"]))] = float(r["kld"])
    return R

def score(d, ref="m2"):
    R = load(d); keep = []
    for (arm, ub), v in sorted(R.items()):
        rv = R.get((ref, ub))
        if rv is None or arm == ref:
            print(f"ARM {arm} {ub} n={len(v)} kld={sum(v.values())/max(1,len(v)):.6f}" + (" (ref)" if arm == ref else " (no ref)"))
            continue
        ks = [k for k in v if k in rv]
        dl = [v[k] - rv[k] for k in ks]; n = len(dl)
        if n < 2:
            continue
        m = sum(dl)/n; se = math.sqrt(sum((x - m)**2 for x in dl)/(n - 1)/n)
        top = sorted(ks, key=lambda k: -(v[k] - rv[k]))[:3]
        print(f"ARM {arm} {ub} n={n} kld={sum(v[k] for k in ks)/n:.6f} d={m:+.6f} se={se:.6f} "
              f"top=" + ",".join(f"{k[0]}@{k[1]}:{v[k]-rv[k]:+.4f}" for k in top))
        if arm[0] in "go" and arm[1:2].isdigit() and m <= se:
            keep.append(arm.replace("_", "-"))
    if keep:
        print("SUGGEST GGML_CUDA_Q8TC_M1=" + ",".join(keep) + "   (then gate the combined arm: E21 ub5 must stay <= base)")

if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] not in ("arms", "score"):
        sys.exit(__doc__ if __doc__ else "usage: q8tc-pt-attrib.py arms BIN [N] [SPLIT] | score DIR [REF]")
    if a[0] == "arms":
        arms(a[1], *(int(x) for x in a[2:4]))
    else:
        score(*a[1:3])
