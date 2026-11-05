#!/usr/bin/env python3
# encoding: utf-8
"""
Full analysis of the cells of the cross-GPU variance study.

Equivalent to `scripts/variance_check/compare_and_decompose.py` of cpc18-learning-playground, adapted: there the cells
were CPU "tiers", here they are GPU architectures (and optionally stacks).

Produces, per condition:
  1. bit-exactness per pair (control: if it were 100% there would be nothing to study)
  2. shape statistics  (mean, median, std, skewness, kurtosis)
  3. equality tests (Mann-Whitney for centre, Levene for dispersion, KS for shape)
  4. shift per percentile, in % relative to a reference cell
  5. bootstrap CI of the p90 per cell, to see whether the intervals overlap
  6. transferability: range between cells vs sampling noise (calibrated test)
  7. seed-by-seed ranking (Spearman, % of inversions)
  8. variance decomposition: how much is seed and how much is machine

Usage:
    analyze_cells.py --cells name=pattern [name=pattern ...] [--ref name]
"""
import argparse, glob, sys
from collections import defaultdict
import numpy as np, yaml
from scipy import stats

def load(pats):
    m = {}
    for p in pats.split(","):
        for f in sorted(glob.glob(p)):
            m.update(yaml.safe_load(open(f)) or {})
    return m

def h(t): print(f"\n{'='*78}\n{t}\n{'='*78}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", required=True, help="name=pattern,pattern")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--boot", type=int, default=40000)
    a = ap.parse_args()
    rng = np.random.default_rng(20260904)

    cells = {}
    for spec in a.cells:
        n, pat = spec.split("=", 1)
        cells[n] = load(pat)
        if not cells[n]: sys.exit(f"ABORT: cell {n} is empty ({pat})")
    ref = a.ref or list(cells)[0]

    h("CELLS")
    for n, d in cells.items():
        r = next(iter(d.values()))
        print(f"  {n:16} {len(d):5} trials  {r.get('gpu','?'):26} {r.get('capability','?'):8} torch {r.get('torch','?')}")
    cks = {v.get("pool_checksum") for d in cells.values() for v in d.values()}
    if len(cks) > 1: sys.exit(f"ABORT: different pools {cks}; not comparable")
    print(f"  common pool_checksum: {cks.pop()}   reference: {ref}")

    conds = sorted({v["model"] for v in cells[ref].values()})
    names = list(cells)

    h("1. BIT-EXACTNESS per pair  (control, not a result)")
    print(f"  {'condition':24}" + "".join(f"{x+'/'+y:>16}" for i,x in enumerate(names) for y in names[i+1:]))
    for c in conds:
        row = f"  {c:24}"
        for i, x in enumerate(names):
            for y in names[i+1:]:
                ks = [k for k,v in cells[x].items() if v["model"]==c and k in cells[y]]
                eq = sum(1 for k in ks if cells[x][k]["final_weights_sha256"]==cells[y][k]["final_weights_sha256"])
                row += f"{eq:>7}/{len(ks):<8}"
        print(row)

    h("2. SHAPE of the accuracy distribution")
    print(f"  {'condition':22}{'cell':12}{'n':>5}{'mean':>11}{'median':>11}{'std':>10}{'skew':>8}{'kurt':>8}")
    for c in conds:
        for n in names:
            x = np.array([v["accuracy"] for v in cells[n].values() if v["model"]==c])
            print(f"  {c if n==names[0] else '':22}{n:12}{len(x):>5}{x.mean():>11.6f}"
                  f"{np.median(x):>11.6f}{x.std(ddof=1):>10.6f}{stats.skew(x):>8.2f}{stats.kurtosis(x):>8.2f}")

    h("3. EQUALITY TESTS against the reference")
    print(f"  {'condition':22}{'cell':12}{'MannWhitney':>13}{'Levene':>10}{'KS':>10}")
    for c in conds:
        xr = np.array([v["accuracy"] for v in cells[ref].values() if v["model"]==c])
        for n in names:
            if n == ref: continue
            x = np.array([v["accuracy"] for v in cells[n].values() if v["model"]==c])
            mw = stats.mannwhitneyu(xr, x).pvalue
            lv = stats.levene(xr, x).pvalue
            ks = stats.ks_2samp(xr, x).pvalue
            print(f"  {c if n==names[1] else '':22}{n:12}{mw:>13.3f}{lv:>10.3f}{ks:>10.3f}")

    h("4. SHIFT PER PERCENTILE  (% relative to the reference)")
    ps = [50, 75, 90, 95, 99]
    print(f"  {'condition':22}{'cell':12}" + "".join(f"{'p'+str(p):>10}" for p in ps))
    for c in conds:
        xr = np.array([v["accuracy"] for v in cells[ref].values() if v["model"]==c])
        for n in names:
            if n == ref: continue
            x = np.array([v["accuracy"] for v in cells[n].values() if v["model"]==c])
            row = f"  {c if n==names[1] else '':22}{n:12}"
            for p in ps:
                b = np.percentile(xr, p); row += f"{100*(np.percentile(x,p)-b)/b:>9.3f}%"
            print(row)

    h("5. BOOTSTRAP CI of the p90 per cell  (do they overlap?)")
    print(f"  {'condition':22}{'cell':12}{'p90':>11}{'95% CI':>26}")
    for c in conds:
        cis = {}
        for n in names:
            x = np.array([v["accuracy"] for v in cells[n].values() if v["model"]==c])
            bs = np.percentile(rng.choice(x, size=(a.boot, len(x)), replace=True), 90, axis=1)
            cis[n] = (np.percentile(x,90), np.percentile(bs,2.5), np.percentile(bs,97.5))
        overlap = max(v[1] for v in cis.values()) <= min(v[2] for v in cis.values())
        for n in names:
            v = cis[n]
            print(f"  {c if n==names[0] else '':22}{n:12}{v[0]:>11.6f}   [{v[1]:.6f}, {v[2]:.6f}]")
        print(f"  {'':22}{'-> ' + ('the three OVERLAP' if overlap else 'they do NOT overlap'):50}")

    h("6. TRANSFERABILITY  p = P(sampling range >= range between cells)")
    ests = [("mean",np.mean),("median",np.median),("p90",lambda z:np.percentile(z,90)),
            ("p99",lambda z:np.percentile(z,99)),("max",np.max)]
    print(f"  {'condition':24}{'n':>5}" + "".join(f"{e:>11}" for e,_ in ests))
    acc = defaultdict(list)
    for c in conds:
        ks = sorted(k for k,v in cells[ref].items() if v["model"]==c)
        X = {n: np.array([cells[n][k]["accuracy"] for k in ks]) for n in names}
        base = np.concatenate(list(X.values())); nn = len(ks); k3 = len(names)
        row = f"  {c:24}{nn:>5}"
        for e, f in ests:
            obs = max(f(X[n]) for n in X) - min(f(X[n]) for n in X)
            idx = rng.integers(0, len(base), size=(a.boot, k3, nn))
            nul = np.array([[f(base[idx[b,j]]) for j in range(k3)] for b in range(a.boot)])
            p = float(((nul.max(1)-nul.min(1)) >= obs).mean())
            acc[e].append(p); row += f"{p:>11.3f}"
        print(row)
    print("  " + "-"*76)
    print(f"  {'median p':24}{'':>5}" + "".join(f"{np.median(acc[e]):>11.3f}" for e,_ in ests))

    h("7. seed-by-seed RANKING  (Spearman / % inversions)")
    print(f"  {'condition':24}" + "".join(f"{x+'/'+y:>18}" for i,x in enumerate(names) for y in names[i+1:]))
    for c in conds:
        row = f"  {c:24}"
        for i, x in enumerate(names):
            for y in names[i+1:]:
                ks = sorted(k for k,v in cells[x].items() if v["model"]==c and k in cells[y])
                u = np.array([cells[x][k]["accuracy"] for k in ks]); w = np.array([cells[y][k]["accuracy"] for k in ks])
                rho = stats.spearmanr(u,w).statistic
                inv = tot = 0
                for p_ in range(len(ks)):
                    for q in range(p_+1, len(ks)):
                        if u[p_]!=u[q] and w[p_]!=w[q]:
                            tot += 1
                            if (u[p_]<u[q]) != (w[p_]<w[q]): inv += 1
                row += f"{rho:>10.2f}{100*inv/tot if tot else 0:>7.0f}%"
        print(row)

    h("8. VARIANCE DECOMPOSITION  (seed vs machine)")
    print(f"  {'condition':24}{'sigma seed':>15}{'sigma machine':>15}{'% machine':>12}")
    for c in conds:
        ks = sorted(k for k,v in cells[ref].items() if v["model"]==c)
        M = np.array([[cells[n][k]["accuracy"] for k in ks] for n in names])   # cells x seeds
        gm = M.mean()
        v_seed = M.mean(0).var(ddof=1)
        v_mach = M.mean(1).var(ddof=1) if len(names) > 1 else 0.0
        v_res = ((M - M.mean(0) - M.mean(1)[:,None] + gm)**2).sum()/((M.shape[0]-1)*(M.shape[1]-1))
        tot = v_seed + v_mach + v_res
        print(f"  {c:24}{np.sqrt(v_seed):>15.6f}{np.sqrt(v_mach+v_res):>15.6f}"
              f"{100*(v_mach+v_res)/tot if tot else 0:>11.1f}%")
    print("\n  'sigma machine' groups the systematic cell effect and the seed x cell")
    print("  interaction, which is where the ranking disorder lives.")

if __name__ == "__main__":
    main()
