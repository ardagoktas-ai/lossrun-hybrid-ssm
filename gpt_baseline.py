"""How the GPT baseline was run (reference; needs the private test-set generator lossrun_gen.py and an OpenAI key).
Stage 5 (optional, needs an OpenAI API key): run GPT on the SAME held-out loss runs, score it with the
SAME metric, and write a side-by-side comparison with our model.
Usage:
  export OPENAI_API_KEY=...            # never paste the key into code or chat
  python gpt_baseline.py --model <openai-model-name> --price_in <USD per 1M input tokens> --price_out <USD per 1M output tokens>
"""
import os, sys, json, time, argparse
from concurrent.futures import ThreadPoolExecutor

sys.path.append("/workspace/v1")                       # fallback only: files next to this script win
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lossrun_gen, score

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--price_in", type=float, required=True)
ap.add_argument("--price_out", type=float, required=True)
ap.add_argument("--out", default=os.environ.get("OUT", "/workspace/out"))
ap.add_argument("--n_docs", type=int, default=300)
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--no_temperature", action="store_true", help="for models that reject the temperature parameter")
ap.add_argument("--reasoning_effort", default="low", help="for reasoning models; dropped automatically if rejected")
args = ap.parse_args()

from openai import OpenAI
client = OpenAI()

SYSTEM = ("You extract structured data from commercial insurance loss run reports. Return ONLY one JSON object, "
          "no prose, with exactly this schema: {\"insured\": str, \"carrier\": str, \"line\": str, "
          "\"valuation_date\": \"YYYY-MM-DD\", \"policies\": [{\"policy\": str, \"start\": \"YYYY-MM-DD\", "
          "\"end\": \"YYYY-MM-DD\", \"claims\": [{\"id\": str, \"date\": \"YYYY-MM-DD\", \"status\": str, "
          "\"paid\": \"0.00\", \"reserve\": \"0.00\", \"incurred\": \"0.00\"}]}]}. Rules: keep names exactly as written; "
          "status exactly as written (Open, Closed or Reopened); money as plain numbers with two decimals, no $ or commas; "
          "two-digit years are 20xx; list every policy and every claim in document order; use an empty claims list "
          "when a policy has no losses.")
shot = next(lossrun_gen.generate(1, seed=5))
docs = [json.loads(l) for l in open(f"{args.out}/eval_heldout.jsonl")][: args.n_docs]


KW = {} if args.no_temperature else {"temperature": 0}
if args.reasoning_effort:
    KW["reasoning_effort"] = args.reasoning_effort
FAILED = []


def run(ex):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": shot["doc"]}, {"role": "assistant", "content": shot["target"]},
            {"role": "user", "content": ex["doc"]}]
    for attempt in range(6):
        try:
            t = time.time()
            r = client.chat.completions.create(model=args.model, messages=msgs,
                                               response_format={"type": "json_object"}, **dict(KW))
            return r.choices[0].message.content or "", r.usage.prompt_tokens, r.usage.completion_tokens, time.time() - t
        except Exception as e:
            msg = str(e)
            for k in ("temperature", "reasoning_effort"):        # model does not support this option: drop it
                if k in msg and k in KW:
                    KW.pop(k, None); print(f"  note: {args.model} rejects '{k}', retrying without it", flush=True)
                    break
            else:
                print(f"  retry {attempt + 1}: {type(e).__name__}: {msg[:200]}", flush=True)
                time.sleep(5 * (attempt + 1))
    FAILED.append(1)
    return "", 0, 0, 0.0


# one test call first, so a wrong model name or key fails fast instead of 300 times
print("test call...", flush=True)
_t = run(docs[0])
if not _t[0]:
    raise SystemExit("test call failed - check the model name / API key (no money was spent on the full run)")
print(f"test OK: {_t[1]} input + {_t[2]} output tokens", flush=True)

t0 = time.time()
with ThreadPoolExecutor(args.threads) as pool:
    outs = list(pool.map(run, docs))
rows = [score.score_one(o[0], ex["target"]) for o, ex in zip(outs, docs)]
with open(f"{args.out}/preds_{args.model}.jsonl", "w") as f:              # keep every answer for inspection
    for o, ex in zip(outs, docs):
        f.write(json.dumps({"pred": o[0], "gold": ex["target"]}) + "\n")
res = score.aggregate(rows)
tin, tout = sum(o[1] for o in outs), sum(o[2] for o in outs)
res.update({"model": args.model, "input_tokens": tin, "output_tokens": tout,
            "avg_seconds_per_document": sum(o[3] for o in outs) / len(outs),
            "usd_per_1000_documents": (tin * args.price_in + tout * args.price_out) / 1e6 / len(outs) * 1000})
res["failed_calls"] = len(FAILED)
json.dump(res, open(f"{args.out}/gpt_results_{args.model}.json", "w"), indent=1)

ours = json.load(open(f"{args.out}/eval_results.json"))
keys = ["json_valid_rate", "perfect_document_rate", "header_accuracy", "policy_accuracy", "claim_recall",
        "claim_precision", "claim_field_accuracy"]
md = ["# Loss-run extraction: our hybrid SSM vs " + args.model, "",
      f"Same {len(docs)} held-out documents, same scoring script.", "",
      f"| Metric | Ours ({ours['model_params_M']}M, from scratch) | {args.model} |", "|---|---|---|"]
for k in keys:
    md.append(f"| {k.replace('_', ' ')} | {100*ours[k]:.1f}% | {100*res[k]:.1f}% |")
md.append(f"| seconds per document | {ours['seconds_per_document_one_gpu']:.2f} (1 GPU) | "
          f"{res['avg_seconds_per_document']:.2f} (API) |")
md.append(f"| cost per 1,000 documents | ${ours['usd_per_1000_documents']:.2f} | ${res['usd_per_1000_documents']:.2f} |")
ratio = res["usd_per_1000_documents"] / max(ours["usd_per_1000_documents"], 1e-9)
md += ["", (f"Cost ratio: {args.model} is {ratio:.1f}x MORE expensive per document than ours." if ratio >= 1 else
            f"Cost ratio: {args.model} is {1/ratio:.1f}x CHEAPER per document than ours (single-stream, unbatched)."),
       f"Failed API calls: {len(FAILED)}"]
open(f"{args.out}/comparison_vs_{args.model}.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))
print("\nformat-insensitive perfect rate:", f"{100*res['perfect_document_rate_format_insensitive']:.1f}%")
print("header field accuracy:", {k: f"{100*res[f'header_{k}_accuracy']:.1f}%" for k in score.HEADER_FIELDS})
print(f"done in {(time.time()-t0)/60:.1f} min")
