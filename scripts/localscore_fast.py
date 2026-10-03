#!/usr/bin/env python3
"""LocalScore, computed two ways on the llama.cpp build we actually ship.

  reference : run all nine of LocalScore's scenarios properly. Slow, the thing we validate against.
  fast      : measure four numbers, project all nine scenarios, apply the same formula.

NOT comparable to scores on localscore.ai: the official binary is pinned to llamafile 0.9.3
(llama.cpp build 1500) and current llama.cpp is 1.7-2.5x faster on identical hardware. The formula
and the scenarios are LocalScore's; the engine is ours, and is reported alongside the score.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from aipotluck.installer import model_localscore  # noqa: E402


def _rate(r):
    """Median of the per-repetition samples rather than llama-bench's mean. On a thermally
    unstable laptop, three single-rep probes of one model scored 62/51/56 against a reference of
    64 -- that spread is the machine, and a mean carries its outlier into the fit."""
    samples = r.get("samples_ns") or []
    if not samples:
        return r["avg_ts"]
    ordered = sorted(float(x) for x in samples)
    mid = len(ordered)//2
    median_ns = ordered[mid] if len(ordered) % 2 else (ordered[mid-1]+ordered[mid])/2
    return (r["n_prompt"] or r["n_gen"]) / (median_ns / 1e9)

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

def fast(binary, model, prefill_anchors, depths, gen_tokens, reps, timeout):
    """One model load. Prefill is MEASURED at several prompt lengths and interpolated between them;
    decode is fitted as a line in KV depth, which it genuinely is.

    Prefill is not fitted to a line because it is not one: fixed per-call overhead inflates tiny
    prompts (Jetson: 6.93 ms/token at 16 against 1.22 at 384) and CPU prefill steps at llama.cpp's
    default n_ubatch of 512 (laptop: 4.33 at 384 against 9.37 at 1024). A straight line through two
    points gave -12% on a laptop and +3% on a Jetson -- wrong on both, cancelling on one.
    """
    d_lo, d_hi = depths
    rows = run_bench(binary, model, [
        "-p", ",".join(str(a) for a in prefill_anchors), "-n", "0", "-d", "0", "-r", str(reps),
    ], timeout)
    prefill = {r["n_prompt"]: _rate(r) for r in rows if r["n_gen"] == 0}

    rows = run_bench(binary, model, [
        "-p", "0", "-n", str(gen_tokens), "-d", f"{d_lo},{d_hi}", "-r", str(reps),
    ], timeout)
    gen = {r["n_depth"]: _rate(r) for r in rows if r["n_gen"]}

    y0, y1 = 1000.0/gen[d_lo], 1000.0/gen[d_hi]
    x0, x1 = d_lo + gen_tokens/2, d_hi + gen_tokens/2
    b = max((y1 - y0) / (x1 - x0), 0.0) if x1 != x0 else 0.0
    a = y0 - b*x0

    cost = model_localscore.CostModel({P: 1000.0/t for P, t in prefill.items()}, a, b)
    r = model_localscore.localscore(cost)
    return (r.score, r.avg_prompt_tps, r.avg_gen_tps, r.avg_ttft_ms), {
        "prefill_ms_per_token": cost.prefill_ms_per_token,
        "decode_base_ms": a, "decode_depth_ms": b, "band": r.band,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["reference", "fast"], required=True)
    ap.add_argument("--prefill-anchors", default="16,64,1024")
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
        anchors = [int(v) for v in a.prefill_anchors.split(",")]
        (s, pp, gen, ttft), extra = fast(a.binary, a.model, anchors, (lo, hi),
                                         a.gen_tokens, a.reps, a.timeout)

    print(json.dumps({"mode": a.mode, "score": s, "avg_pp_tps": pp, "avg_gen_tps": gen,
                      "avg_ttft_ms": ttft, "wall_seconds": time.monotonic()-t0, **extra}, indent=2))

if __name__ == "__main__":
    sys.exit(main())
