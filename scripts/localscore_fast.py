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
    """Every scenario measured directly. prompt_tps from a prompt-only test; gen_tps from a
    generation test with the KV cache pre-filled to that scenario's prompt length via -d, which is
    where its generation actually happens."""
    prompts = sorted({p for p, _ in PAIRS})
    pp_rows = run_bench(binary, model, ["-p", ",".join(map(str, prompts)), "-n", "0", "-d", "0",
                                        "-r", str(reps)], timeout)
    pp_by_p = {r["n_prompt"]: r["avg_ts"] for r in pp_rows if r["n_gen"] == 0}

    gen_by_pair = {}
    for P, G in PAIRS:
        rows = run_bench(binary, model, ["-p", "0", "-n", str(G), "-d", str(P), "-r", str(reps)], timeout)
        gen_by_pair[(P, G)] = next(r["avg_ts"] for r in rows if r["n_gen"])

    pps, gens, ttfts = [], [], []
    for P, G in PAIRS:
        pp = pp_by_p[P]
        pps.append(pp)
        gens.append(gen_by_pair[(P, G)])
        ttfts.append(P / pp * 1000.0)
    return score(pps, gens, ttfts), list(zip(PAIRS, pps, gens, ttfts))

def fit(x0, y0, x1, y1):
    if x1 == x0:
        return y0, 0.0
    slope = (y1 - y0) / (x1 - x0)
    return y0 - slope*x0, max(slope, 0.0)

def fast(binary, model, prompt_len, depths, gen_tokens, reps, timeout):
    """One model load, four measurements: a prompt test and a generation test at each of two KV
    depths. Both costs are linear in depth, so two points fix each line and all nine scenarios
    follow in closed form.

    Generation is measured with -d rather than derived by subtracting prefill from a combined -pg
    run. The subtraction is badly conditioned -- at a 2048-token prompt the prefill is ~20s against
    ~1s of generation, so a 5% prefill error lands as an ~80% error on the generation estimate, and
    it did: it put this laptop at 13 tok/s where it really does 26-55."""
    d_lo, d_hi = depths
    rows = run_bench(binary, model, [
        "-p", str(prompt_len), "-n", str(gen_tokens), "-d", f"{d_lo},{d_hi}", "-r", str(reps),
    ], timeout)

    pp, gen = {}, {}
    for r in rows:
        (pp if r["n_gen"] == 0 else gen)[r["n_depth"]] = r["avg_ts"]

    # A prompt test at depth d spans depths d..d+prompt_len, so its representative depth is the
    # midpoint; same for a generation test over its own span.
    c, e = fit(d_lo + prompt_len/2, 1000.0/pp[d_lo], d_hi + prompt_len/2, 1000.0/pp[d_hi])
    a, b = fit(d_lo + gen_tokens/2, 1000.0/gen[d_lo], d_hi + gen_tokens/2, 1000.0/gen[d_hi])

    def prefill_ms(P):
        return (c + e*(P/2)) * P      # integrate c + e*d over 0..P

    pps, gens, ttfts = [], [], []
    for P, G in PAIRS:
        pre = prefill_ms(P)
        pps.append(P / pre * 1000.0)
        gens.append(1000.0 / (a + b*(P + G/2)))
        ttfts.append(pre)
    return score(pps, gens, ttfts), {"c": c, "e": e, "a": a, "b": b}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["reference", "fast"], required=True)
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--depths", default="0,1024")
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
        lo, hi = (int(v) for v in a.depths.split(","))
        (s, pp, gen, ttft), extra = fast(a.binary, a.model, a.prompt_len, (lo, hi),
                                         a.gen_tokens, a.reps, a.timeout)
        extra = {"fit": extra}
    print(json.dumps({"mode": a.mode, "score": s, "avg_pp_tps": pp, "avg_gen_tps": gen,
                      "avg_ttft_ms": ttft, "wall_seconds": time.monotonic()-t0, **extra}, indent=2))

if __name__ == "__main__":
    sys.exit(main())
