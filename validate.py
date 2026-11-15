"""Consistency checks on an extracted loss run (no model needed).

Flags rows a human should look at before the data is used. These are rules every loss run follows, so a wrong
digit copied by the model (or by a person) usually breaks one of them:
  * incurred = paid + reserve  (to the cent)
  * a Closed claim has no open reserve
  * dates are real dates; policy start < end; loss date inside the policy period and not after the valuation date
  * money is never negative

Usage:  from validate import check;  flags = check(extracted_dict)   ->  list of human-readable flags ([] = clean)
"""
import datetime as dt
import re

_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numbers_in(text):
    out = set()
    for t in _NUM.findall(text or ""):
        try:
            out.add(round(float(t.replace(",", "")), 2))
        except ValueError:
            pass
    return out


def _d(s):
    try:
        return dt.date.fromisoformat(str(s))
    except Exception:
        return None


def _m(v):
    try:
        return round(float(str(v).replace("$", "").replace(",", "")), 2)
    except Exception:
        return None


def check(doc, source=None):
    """source (optional): the original document text. Then every amount and claim number must literally appear in it,
    which catches copying errors that still add up (e.g. a dropped digit in both paid and incurred)."""
    flags = []
    nums = _numbers_in(source) if source else None
    if not isinstance(doc, dict):
        return ["output is not a JSON object"]
    val = _d(doc.get("valuation_date"))
    if val is None:
        flags.append("valuation_date is not a valid date")
    for p in doc.get("policies") or []:
        pol = p.get("policy", "?")
        ps, pe = _d(p.get("start")), _d(p.get("end"))
        if ps is None or pe is None or ps >= pe:
            flags.append(f"policy {pol}: period start/end invalid ({p.get('start')} -> {p.get('end')})")
        for c in p.get("claims") or []:
            cid = c.get("id", "?")
            paid, res, inc = _m(c.get("paid")), _m(c.get("reserve")), _m(c.get("incurred"))
            if None in (paid, res, inc):
                flags.append(f"claim {cid}: unreadable amount")
                continue
            if nums is not None:
                for name, v in (("paid", paid), ("reserve", res), ("incurred", inc)):
                    if v not in nums:
                        flags.append(f"claim {cid}: {name} {v:,.2f} does not appear in the document")
                if str(cid) not in source:
                    flags.append(f"claim {cid}: claim number not found in the document")
            if abs(paid + res - inc) > 0.011:
                flags.append(f"claim {cid}: paid {paid:,.2f} + reserve {res:,.2f} != incurred {inc:,.2f}")
            if min(paid, res, inc) < 0:
                flags.append(f"claim {cid}: negative amount")
            if str(c.get("status", "")).lower() == "closed" and res > 0:
                flags.append(f"claim {cid}: Closed but reserve {res:,.2f} is still open")
            dol = _d(c.get("date"))
            if dol is None:
                flags.append(f"claim {cid}: loss date invalid ({c.get('date')})")
            else:
                if ps and pe and not (ps <= dol < pe):
                    flags.append(f"claim {cid}: loss date {dol} outside policy period {ps} -> {pe}")
                if val and dol > val:
                    flags.append(f"claim {cid}: loss date {dol} after valuation date {val}")
    return flags


if __name__ == "__main__":
    # self-test; needs the (private) synthetic data generator
    import json, sys, copy
    sys.path.insert(0, ".")
    try:
        import lossrun_gen
    except ImportError:
        raise SystemExit("self-test needs lossrun_gen.py (private). Use: from validate import check")
    clean = 0
    for ex in lossrun_gen.generate(300, seed=777):
        clean += check(json.loads(ex["target"]), ex["doc"]) == []
    print(f"correct documents with no false alarm (incl. source check): {clean}/300")
    src = next(lossrun_gen.generate(1, seed=777))["doc"]
    # the two digit errors our model actually made in testing
    ex = json.loads(next(lossrun_gen.generate(1, seed=777))["target"])
    bad = copy.deepcopy(ex)
    c = next(c for p in bad["policies"] for c in p["claims"])
    c["paid"] = f"{float(c['paid']) + 0.21:.2f}"                     # 29365.17 -> 29365.38 style error
    print("paid digit error caught:", check(bad))
    bad = copy.deepcopy(ex)
    c = next(c for p in bad["policies"] for c in p["claims"])
    c["incurred"] = f"{float(c['incurred']) + 400:.2f}"              # 4375.32 -> 4775.32 style error
    print("incurred digit error caught:", check(bad))
    bad = copy.deepcopy(ex)                                           # dropped digit in BOTH paid and incurred
    c = next(c for p in bad["policies"] for c in p["claims"] if float(c["paid"]) > 1000)
    c["paid"] = c["incurred"] = c["paid"][:1] + c["paid"][2:]
    c["reserve"] = "0.00" if float(c["reserve"]) == 0 else c["reserve"]
    print("consistent dropped digit, arithmetic check only:", check(bad))
    print("consistent dropped digit, with source check:", check(bad, src))
