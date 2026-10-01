#!/usr/bin/env python3
"""LocalScore, computed two ways on the llama.cpp build we actually ship.

  reference : run all nine of LocalScore's scenarios properly. Slow, the thing we validate against.
  fast      : measure four numbers, project all nine scenarios, apply the same formula.

NOT comparable to scores on localscore.ai: the official binary is pinned to llamafile 0.9.3
(llama.cpp build 1500) and current llama.cpp is 1.7-2.5x faster on identical hardware. The formula
and the scenarios are LocalScore's; the engine is ours, and is reported alongside the score.
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time

# LocalScore's nine scenarios, verbatim from localscore.cpp (prompt, generate).
PAIRS = [(1024,16),(4096,256),(2048,256),(2048,768),(1024,1024),(1280,3072),(384,1152),(64,1024),(16,1536)]

def score(pp_list, gen_list, ttft_list):
    """localscore.cpp: arithmetic mean of each metric, then 10 * cuberoot of their product."""
    n = len(pp_list)
    avg_pp, avg_gen, avg_ttft = sum(pp_list)/n, sum(gen_list)/n, sum(ttft_list)/n
    return 10*(avg_pp*avg_gen*(1000/avg_ttft))**(1/3), avg_pp, avg_gen, avg_ttft

def run_bench(binary, model, args, timeout):
    cmd = [binary, "-m", model, "--offline", "-o", "json"] + args
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"llama-bench exited {proc.returncode}: {proc.stderr[-800:]}")
    out = proc.stdout
    return json.JSONDecoder().raw_decode(out[out.index("["):])[0]

def reference(binary, model, reps, timeout):
    """Every scenario measured: pp from a prompt-only test, generation from a -pg test with the
    prefill time subtracted out, which is exactly how LocalScore splits them."""
    prompts = sorted({p for p, _ in PAIRS})
    args = ["-p", ",".join(map(str, prompts)), "-n", "0", "-d", "0", "-r", str(reps)]
    for p, g in PAIRS:
        args += ["-pg", f"{p},{g}"]
    rows = run_bench(binary, model, args, timeout)

    pp_by_p, pg_by_pair = {}, {}
    for r in rows:
        if r["n_gen"] == 0:
            pp_by_p[r["n_prompt"]] = r["avg_ts"]
        else:
            pg_by_pair[(r["n_prompt"], r["n_gen"])] = r["avg_ts"]

    pps, gens, ttfts = [], [], []
    for p, g in PAIRS:
        pp = pp_by_p[p]
        prefill_ms = p / pp * 1000.0
        total_ms = (p + g) / pg_by_pair[(p, g)] * 1000.0
        gen_ms = max(total_ms - prefill_ms, 1e-6)
        pps.append(pp); gens.append(g / gen_ms * 1000.0); ttfts.append(prefill_ms)
    return score(pps, gens, ttfts), list(zip(PAIRS, pps, gens, ttfts))

def fit(x0, y0, x1, y1):
    if x1 == x0:
        return y0, 0.0
    slope = (y1 - y0) / (x1 - x0)
    return y0 - slope*x0, max(slope, 0.0)

def fast(binary, model, probes, gen_tokens, reps, timeout):
    """Four measurements -> a prefill line and a decode line -> all nine scenarios in closed form."""
    p_lo, p_hi = probes
    args = ["-p", f"{p_lo},{p_hi}", "-n", "0", "-d", "0", "-r", str(reps)]
    args += ["-pg", f"{p_lo},{gen_tokens}", "-pg", f"{p_hi},{gen_tokens}"]
    rows = run_bench(binary, model, args, timeout)

    pp, pg = {}, {}
    for r in rows:
        (pp if r["n_gen"] == 0 else pg)[r["n_prompt"]] = r["avg_ts"]

    # prefill cost per token at prompt length P averages c + e*(P/2)
    c, e_half = fit(p_lo/2, 1000.0/pp[p_lo], p_hi/2, 1000.0/pp[p_hi])
    def prefill_ms(P): return (c + e_half*(P/2)) * P

    # decode cost per token at depth d is a + b*d; recover it by removing the prefill time
    def gen_ms_per_tok(P):
        total = (P + gen_tokens) / pg[P] * 1000.0
        return max(total - prefill_ms(P), 1e-6) / gen_tokens
    a, b = fit(p_lo + gen_tokens/2, gen_ms_per_tok(p_lo),
               p_hi + gen_tokens/2, gen_ms_per_tok(p_hi))

    pps, gens, ttfts = [], [], []
    for P, G in PAIRS:
        pre = prefill_ms(P)
        pps.append(P / pre * 1000.0)
        gens.append(1000.0 / (a + b*(P + G/2)))
        ttfts.append(pre)
    return score(pps, gens, ttfts), {"c": c, "e": e_half*2, "a": a, "b": b}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["reference", "fast"], required=True)
    ap.add_argument("--probes", default="256,2048")
    ap.add_argument("--gen-tokens", type=int, default=32)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=3600)
    a = ap.parse_args()

    t0 = time.monotonic()
    if a.mode == "reference":
        (s, pp, gen, ttft), detail = reference(a.binary, a.model, a.reps, a.timeout)
        extra = {"scenarios": [{"prompt": p, "gen": g, "pp_tps": x, "gen_tps": y, "ttft_ms": z}
                               for (p, g), x, y, z in detail]}
    else:
        lo, hi = (int(v) for v in a.probes.split(","))
        (s, pp, gen, ttft), extra = fast(a.binary, a.model, (lo, hi), a.gen_tokens, a.reps, a.timeout)
        extra = {"fit": extra}
    print(json.dumps({"mode": a.mode, "score": s, "avg_pp_tps": pp, "avg_gen_tps": gen,
                      "avg_ttft_ms": ttft, "wall_seconds": time.monotonic()-t0, **extra}, indent=2))

if __name__ == "__main__":
    sys.exit(main())
