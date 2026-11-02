#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 snielco
"""Add (or with --remove, strip) the KV slot-move test routes in a llama-swap config.yaml.

A text edit, not a YAML
round-trip, so the comments and layout of the live file are kept. Routes go in just before the
"# ---- ... Admission control" banner; each gets a matrix var added to one set (Strix `heavy`,
observer `big`), so a test route is never resident next to Strix's qwen4exp.

Run on the host:  python3 add_kv_routes.py <config.yaml> --host strix|observer [--remove]
Back up config.yaml first. Saving it makes llama-swap (--watch-config) reload and stop running models.
"""
import os
import re
import sys

HOME = os.path.expanduser("~")

HOSTS = {
    "strix": dict(
        set="heavy",
        anchor="# -----------------------------------------------------------------------------\n# Admission control",
        bin=HOME + "/llamacpp-vk/llama-b11960/llama-b11960",
        env="",
        model=HOME + "/llamacpp-vk/models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf",
        slots=HOME + "/llamacpp-vk/slot-cache/",
        # var, route id, ctx, parallel, kv type, cache-ram, extra flags
        routes=[("tkv1", "test-kv-q38", 262144, 1, "q8_0", 0, ""),
                ("tkv2", "test-kv-q38-p2", 524288, 2, "q8_0", 22528, ""),
                ("tkv3", "test-kv-q38-f16", 262144, 1, "f16", 0, ""),
                # 2 slots on ONE unified KV stream: n_stream = 1, like a 1-slot server, so slot files move
                ("tkv4", "test-kv-q38-p2u", 524288, 2, "q8_0", 22528, "--kv-unified --kv-unified-per-slot 262144")],
    ),
    "observer": dict(
        set="big",
        anchor="# Exactly one ~30B-class model resident",
        bin=HOME + "/llm/bin/llama-b11960/llama-b11960",   # check against your live config before use
        env="GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1 ",
        model=HOME + "/llm/models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf",
        slots=HOME + "/llm/slot-cache/",
        routes=[("tkv1", "test-kv-q38", 262144, 1, "q8_0", 0, "")],
    ),
}
TAG = "kvmove test"
BEGIN, END = "  # >>> kvmove test routes (2026-11-01, remove after test)\n", "  # <<< kvmove test routes\n"


def block(h):
    out = [BEGIN]
    for var, rid, ctx, par, kv, cram, extra in h["routes"]:
        out.append(
            f'  "{rid}":\n    cmd: |\n      env {h["env"]}LD_LIBRARY_PATH={h["bin"]}\n      {h["bin"]}/llama-server\n'
            f'      -m {h["model"]}\n      --host 127.0.0.1 --port ${{PORT}}\n'
            f"      -ngl 999 -fa on -c {ctx} --parallel {par} --cache-type-k {kv} --cache-type-v {kv}"
            f" --jinja --reasoning off --load-mode none{(' ' + extra) if extra else ''}\n"
            f"      --cache-ram {cram} --ctx-checkpoints 8\n"
            f"      --spec-type draft-mtp --spec-draft-n-max 3 -ctkd q8_0 -ctvd q8_0\n"
            f'      --slot-save-path {h["slots"]}\n    ttl: 600\n\n')
    out.append(END)
    return "".join(out)


def strip(s, h):
    if BEGIN in s:
        a, b = s.index(BEGIN), s.index(END) + len(END)
        s = s[:a] + s[b:].lstrip("\n")
    s = "\n".join(l for l in s.split("\n") if not (TAG in l and l.strip().split(":")[0] in {r[0] for r in h["routes"]}))
    for v in sorted((r[0] for r in h["routes"]), key=len, reverse=True):
        s = s.replace(f" | {v}", "")
    return s


def main():
    path = sys.argv[1]
    h = HOSTS[sys.argv[sys.argv.index("--host") + 1]]
    s = strip(open(path).read(), h)
    if "--remove" not in sys.argv:
        assert h["anchor"] in s, "anchor not found: %r" % h["anchor"]
        s = s.replace(h["anchor"], block(h) + "\n" + h["anchor"], 1)
        vars_add = "".join(f"    {v}: {rid}  # {TAG}\n" for v, rid, *_ in h["routes"])
        assert "  evict_costs:\n" in s, "evict_costs not found"
        s = s.replace("  evict_costs:\n", vars_add + "  evict_costs:\n", 1)
        sets_at = s.index("  sets:\n")
        m = re.compile(r'^(    %s:\s+"\()([^)]*)(\))' % h["set"], re.M).search(s, sets_at)
        assert m, "set %r not found under sets:" % h["set"]
        alts = " | ".join(v for v, *_ in h["routes"])
        s = s[:m.end(2)] + " | " + alts + s[m.end(2):]
    open(path, "w").write(s)
    print("removed" if "--remove" in sys.argv else "added %d test routes" % len(h["routes"]))


if __name__ == "__main__":
    main()
