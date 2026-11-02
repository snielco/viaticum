#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 snielco
"""KV slot-move test client (stdlib only). One step per subcommand; each appends a JSON line to
runs/<run>/results.jsonl. Talks straight to a host's llama-swap (:9091) through /upstream/<route>/.

  build   HOST ROUTE NTOK        tokenize a fixed long document with 3 needles -> runs/<run>/doc-<NTOK>.json
  prefill HOST ROUTE NTOK SLOT   process the document into SLOT (cache_prompt)
  save    HOST ROUTE SLOT FILE   POST /slots/SLOT?action=save
  restore HOST ROUTE SLOT FILE   POST /slots/SLOT?action=restore
  query   HOST ROUTE NTOK SLOT LABEL   document + question, temperature 0, 48 tokens; records answer and timings
  erase   HOST ROUTE SLOT
  unload  HOST ROUTE

HOST = a name from KV_HOSTS or a host/IP. KV_HOSTS="strix=10.0.0.11,observer=10.0.0.12" maps names to
addresses. Run dir from env KVRUN (default: today's date).
"""
import glob
import json
import os
import sys
import time
import urllib.request

HOSTS = dict(kv.split("=", 1) for kv in os.environ.get("KV_HOSTS", "").split(",") if "=" in kv)
HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "runs", os.environ.get("KVRUN", time.strftime("%Y%m%d")))
NEEDLES = [("red", "4417093"), ("green", "8820516"), ("blue", "3069472")]
QUESTION = ("\n\nQuestion: The text above contains three notes giving vault codes for the red, green and blue "
            "doors. List the three codes.\nAnswer: red door")


def base(host, route):
    return "http://%s:9091/upstream/%s" % (HOSTS.get(host, host), route)


def call(url, body=None, timeout=3600):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"},
                                 method="POST" if body is not None else "GET")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = r.read().decode()
            status = r.status
    except urllib.error.HTTPError as e:
        out, status = e.read().decode(), e.code
    try:
        out = json.loads(out)
    except ValueError:
        pass
    return status, out, time.time() - t0


def log(rec):
    os.makedirs(RUN, exist_ok=True)
    rec = dict(t=time.strftime("%F %T"), **rec)
    with open(os.path.join(RUN, "results.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec, indent=1)[:3000])


def source_text(nchars):
    # fixed, deterministic filler: the Python stdlib sources in sorted order
    files = sorted(glob.glob("/usr/lib/python3*/[a-z]*.py"))
    parts, n = [], 0
    for p in files:
        t = open(p, errors="replace").read()
        parts.append(t)
        n += len(t)
        if n >= nchars:
            break
    return "".join(parts)[:nchars]


def doc_path(ntok):
    return os.path.join(RUN, "doc-%d.json" % ntok)


def build(host, route, ntok):
    ntok = int(ntok)
    nchars = ntok * 3
    for _ in range(6):
        text = source_text(nchars)
        L = len(text)
        for i, (c, code) in enumerate(NEEDLES):  # needles at 10 / 50 / 90 %
            pos = text.rfind("\n", 0, int(L * (0.1 + 0.4 * i))) + 1
            text = text[:pos] + "# NOTE: the vault code for the %s door is %s.\n" % (c, code) + text[pos:]
        st, out, _ = call(base(host, route) + "/tokenize", {"content": text})
        toks = out["tokens"]
        if abs(len(toks) - ntok) < ntok * 0.02:
            break
        nchars = int(nchars * ntok / len(toks))
    st, q, _ = call(base(host, route) + "/tokenize", {"content": QUESTION})
    os.makedirs(RUN, exist_ok=True)
    json.dump({"doc": toks, "question": q["tokens"]}, open(doc_path(ntok), "w"))
    log(dict(step="build", host=host, route=route, ntok=ntok, doc_tokens=len(toks), q_tokens=len(q["tokens"])))


def prefill(host, route, ntok, slot):
    d = json.load(open(doc_path(int(ntok))))
    st, out, dt = call(base(host, route) + "/completion",
                       {"prompt": d["doc"], "n_predict": 1, "id_slot": int(slot), "cache_prompt": True, "temperature": 0})
    log(dict(step="prefill", host=host, route=route, ntok=int(ntok), slot=int(slot), status=st, wall_s=round(dt, 2),
             timings=out.get("timings") if isinstance(out, dict) else out))


def slot_action(action, host, route, slot, fname=None):
    body = {"filename": fname} if fname else {}
    st, out, dt = call(base(host, route) + "/slots/%s?action=%s" % (slot, action), body)
    log(dict(step=action, host=host, route=route, slot=int(slot), file=fname, status=st, wall_s=round(dt, 2), resp=out))


QUESTION2 = "\nQuestion: Which of the three doors has the largest code?\nAnswer: the"


def query(host, route, ntok, slot, label, extend=None):
    """extend = label of an earlier query in this run: continue that session (its prompt + its answer tokens)
    with a second question, the way a follow-up turn extends a saved session."""
    d = json.load(open(doc_path(int(ntok))))
    prompt = d["doc"] + d["question"]
    if extend:
        prev = [json.loads(l) for l in open(os.path.join(RUN, "results.jsonl"))
                if '"query"' in l and json.loads(l).get("label") == extend][-1]
        st, q2, _ = call(base(host, route) + "/tokenize", {"content": QUESTION2})
        prompt = prompt + prev["tokens"] + q2["tokens"]
    st, out, dt = call(base(host, route) + "/completion",
                       {"prompt": prompt, "n_predict": 48, "id_slot": int(slot),
                        "cache_prompt": True, "temperature": 0, "top_k": 1, "return_tokens": True, "seed": 1})
    if not isinstance(out, dict) or st != 200:
        return log(dict(step="query", label=label, host=host, route=route, status=st, resp=out))
    ans = out.get("content", "")
    log(dict(step="query", label=label, extend=extend, host=host, route=route, ntok=int(ntok), slot=int(slot), status=st,
             wall_s=round(dt, 2), needles={c: (code in ans) for c, code in NEEDLES}, answer=ans,
             tokens=out.get("tokens"), tokens_cached=out.get("tokens_cached"), timings=out.get("timings")))


def unload(host, route):
    for path in ("/api/models/unload/%s" % route, "/unload"):
        st, out, dt = call("http://%s:9091%s" % (HOSTS.get(host, host), path), {})
        if st == 200:
            break
    log(dict(step="unload", host=host, route=route, via=path, status=st, resp=out))


if __name__ == "__main__":
    cmd, a = sys.argv[1], sys.argv[2:]
    {"build": build, "prefill": prefill, "query": query, "unload": unload,
     "save": lambda *x: slot_action("save", *x), "restore": lambda *x: slot_action("restore", *x),
     "erase": lambda *x: slot_action("erase", *x)}[cmd](*a)
