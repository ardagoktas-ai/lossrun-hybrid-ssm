"""Scoring for loss-run extraction (shared by our model and the GPT baseline)."""
import json

CLAIM_FIELDS = ["date", "status", "paid", "reserve", "incurred"]
HEADER_FIELDS = ["insured", "carrier", "line", "valuation_date"]


def parse(text):
    """Parse model output into a dict; tolerant to text around the JSON object."""
    try:
        return json.loads(text)
    except Exception:
        pass
    a, b = text.find("{"), text.rfind("}")
    if a >= 0 and b > a:
        try:
            return json.loads(text[a:b + 1])
        except Exception:
            return None
    return None


def _norm_money(v):
    try:
        return f"{float(str(v).replace('$', '').replace(',', '')):.2f}"
    except Exception:
        return str(v)


def _claims(doc):
    out = {}
    for p in (doc or {}).get("policies", []) or []:
        for c in (p or {}).get("claims", []) or []:
            if isinstance(c, dict) and "id" in c:
                c = dict(c)
                for k in ("paid", "reserve", "incurred"):
                    if k in c:
                        c[k] = _norm_money(c[k])
                out[str(c["id"])] = c
    return out


def _normalize(doc):
    """Format-insensitive view: money as 2-decimal strings, text stripped, header names case-insensitive
    (e.g. a carrier printed in capitals is not an error). Used for a fairer 'perfect' score."""
    if not isinstance(doc, dict):
        return doc
    d = {k: (str(v).strip().lower() if k in HEADER_FIELDS else v) for k, v in doc.items()}   # case-insensitive names
    pols = []
    for p in d.get("policies") or []:
        if not isinstance(p, dict):
            continue
        p = {k: (str(v).strip() if k != "claims" else v) for k, v in p.items()}
        cl = []
        for c in p.get("claims") or []:
            if isinstance(c, dict):
                c = {k: (_norm_money(v) if k in ("paid", "reserve", "incurred") else str(v).strip()) for k, v in c.items()}
            cl.append(c)
        p["claims"] = cl
        pols.append(p)
    d["policies"] = pols
    return d


def score_one(pred_text, gold_text):
    gold = json.loads(gold_text)
    pred = parse(pred_text)
    r = {"json_valid": pred is not None and isinstance(pred, dict)}
    if not r["json_valid"]:
        pred = {}
    r["header_correct"] = sum(str(pred.get(k)) == str(gold.get(k)) for k in HEADER_FIELDS)
    r["header_total"] = len(HEADER_FIELDS)
    for k in HEADER_FIELDS:
        r["hdr_" + k] = int(str(pred.get(k)) == str(gold.get(k)))
    gp = {p["policy"]: p for p in gold["policies"]}
    pp = {str(p.get("policy")): p for p in (pred.get("policies") or []) if isinstance(p, dict)}
    r["policy_correct"] = sum(1 for k, p in gp.items() if k in pp and pp[k].get("start") == p["start"]
                              and pp[k].get("end") == p["end"])
    r["policy_total"] = len(gp)
    gc, pc = _claims(gold), _claims(pred)
    r["claims_gold"], r["claims_pred"] = len(gc), len(pc)
    r["claims_found"] = len(set(gc) & set(pc))
    r["field_correct"] = sum(1 for cid, g in gc.items() for f in CLAIM_FIELDS
                             if cid in pc and str(pc[cid].get(f)) == str(g[f]))
    r["field_total"] = len(gc) * len(CLAIM_FIELDS)
    r["perfect"] = r["json_valid"] and pred == gold
    r["perfect_normalized"] = r["json_valid"] and _normalize(pred) == _normalize(gold)
    return r


def aggregate(rows):
    n = len(rows)
    s = lambda k: sum(r[k] for r in rows)
    return {
        "documents": n,
        "json_valid_rate": s("json_valid") / max(n, 1),
        "perfect_document_rate": s("perfect") / max(n, 1),
        "header_accuracy": s("header_correct") / max(s("header_total"), 1),
        "policy_accuracy": s("policy_correct") / max(s("policy_total"), 1),
        "claim_recall": s("claims_found") / max(s("claims_gold"), 1),
        "claim_precision": s("claims_found") / max(s("claims_pred"), 1),
        "claim_field_accuracy": s("field_correct") / max(s("field_total"), 1),
        "perfect_document_rate_format_insensitive": s("perfect_normalized") / max(n, 1),
        **{f"header_{k}_accuracy": s("hdr_" + k) / max(n, 1) for k in HEADER_FIELDS},
    }
