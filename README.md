# Viaticum

**A small, priority-aware router for a homelab of mixed llama.cpp GPU hosts, where a session's KV cache can move
between machines so a conversation picks up where it left off.**

> *Viaticum* (Latin, "provisions for the journey"; Greek *ephodion*): what a traveller carries for the road.
> In this project the session's KV cache is the provision: it goes with the conversation from one machine to another,
> so the conversation never has to start over.

**Status: design and experiments. There is no router code in this repository yet.** What is here today is the goal,
the design direction, and one tested building block (cross-host KV moves, `experiments/kvmove/`).

---

## The problem

A typical "serious homelab" for local LLMs is not a cluster. It is a few unrelated machines: an AMD APU with lots of
unified memory, a discrete AMD card, an NVIDIA card in a small box, each running [llama.cpp] behind
[llama-swap]. They are used in two very different ways at once:

- **Live** traffic: a person or an agent waiting on a reply, often with a long context (100k-256k tokens) that took
  minutes to prefill.
- **Batch** traffic: embedding a library, overnight summarization, evaluations: millions of short requests that
  should use every idle GPU, and get out of the way the moment someone needs one.

The existing tools don't cover this mix:

- Gateways (LiteLLM, Olla, Bifrost, ...) balance requests but don't have request priority classes that make batch
  work give way to live traffic across hosts.
- Datacenter stacks with KV/session management (llm-d, NVIDIA Dynamo, LMCache, SGLang's gateway) assume vLLM or
  SGLang, NVIDIA or datacenter AMD, and usually Kubernetes. None of them drive llama.cpp on consumer Vulkan/ROCm hardware.
- llama-swap does model swapping and priority **per host** very well, but a single llama-swap doesn't know about the
  other machines.

The result: batch jobs evict a live model, a long chat loses its cache and pays a multi-minute re-prefill, and a
crashed backend gets hammered by retry storms.

## Goals

1. **Live goes first, batch fills the gaps.** Two request classes (header `X-LLM-Priority: batch`, default
   interactive). Live requests wait in a short in-memory priority queue; batch requests are never held. When there
   is no batch capacity the router answers `429` + `Retry-After` and the batch worker sleeps, so a router restart
   loses nothing.
2. **Hosts give way.** A host that served live traffic recently is "hot" and gets no new batch work. Batch shrinks
   from N hosts to 1 as people start using machines, and grows back when they stop. Reserved live slots per model,
   quiet windows (e.g. nightly maintenance), exponential backoff on backend errors.
3. **Slot- and memory-aware placement.** Know how many concurrent slots each host can hold for each model, and what is
   actually loaded (poll llama-swap's `/running`), so the router prefers warm hosts and never sends what a host would
   have to evict for.
4. **Sessions that survive.** Using llama.cpp's own slot save/restore:
   - **Park** a long session to local NVMe when its model is about to be evicted or it goes idle; restore it in under
     a second on the next turn, instead of re-prefilling for minutes.
   - **Move** a session to another host (different GPU, same model and build) when its home host is busy or down.
     Tested: a 100k-token session moves in ~42 s on 1 GbE versus a 530 s re-prefill, with token-identical output
     (`experiments/kvmove/KV_MOVE_TEST.md`). On 10 GbE the copy part drops to seconds.
   - Enforce the compatibility rules the experiment found (same GGUF hash, same build, same KV types, same KV stream
     count, idle destination slot).
5. **Later: general GPU jobs.** Expose host state (hot / cold / window / backoff) at `GET /router/hosts` so a job
   runner can place ffmpeg/whisper/other GPU jobs on idle machines. Design the hook only; build after the router has
   run in production.

## Design principles

- **Python standard library only.** No third-party packages, no supply-chain surface to keep updated. Meant to run on
  a small always-on box for years without attention.
- **Use what llama.cpp and llama-swap already do.** llama-swap stays the per-host model manager (pinned version,
  per-model priority). llama.cpp's `--slot-save-path` and `/slots/<id>?action=save|restore` are the session primitives.
  Viaticum adds the cross-host policy.
- **OpenAI-compatible front end** (`/v1/chat/completions`, `/v1/embeddings`, `/v1/models`), so clients only change a URL.
- **Two small services, not one big one:** the router (classes, slots, hot/cold state) and a session store
  (save/restore, index of sessions: host, model hash, build, tokens, bytes, last use; retention). Each can restart alone.
- **Honest logging:** every request to a JSONL spool with class, backend tried, timings and the reason a batch request
  was refused.

## Roadmap

Each phase runs in production before the next one starts.

| Phase | Scope | State |
|---|---|---|
| P0 | Cross-host KV move experiment | **done** (`experiments/kvmove/`) |
| P1 | Priority classes, batch yield, slot-aware capacity, backoff, `429` + `Retry-After` | design |
| P2 | Session parking on the serving host's local NVMe | design |
| P3 | Session store and cross-host moves (best with 10 GbE) | design; building block tested |
| P4 | `/router/hosts` for general GPU jobs | hook designed only |

## Non-goals

- Not a multi-provider cloud gateway (no API keys, billing, or hosted-provider adapters).
- Not a replacement for llama-swap or llama.cpp.
- Not for datacenters or Kubernetes; aimed at 2-5 heterogeneous machines on one LAN.
- No converting KV caches between quantization types (looked at, judged not worth it: see the experiment notes).

## Prior art and credit

- [llama.cpp] (slot save/restore, `--kv-unified`, context checkpoints) and [llama-swap] (per-host model swapping).
- llama.cpp discussion [#20572](https://github.com/ggml-org/llama.cpp/discussions/20572): per-session KV
  save/restore with hooks, on one host.
- Design ideas borrowed from [llm-d] flow control (priority bands; shed the low band rather than queue it) and SGLang's
  cache-aware routing.

## License

Copyright (C) 2026 snielco. Licensed under the [GNU General Public License v3.0 or later](LICENSE)
(`GPL-3.0-or-later`). You may use, study, change and share it, including commercially; if you distribute it or a
modified version, you must do so under the same license with its source code.

If you use Viaticum or its results in published work, please cite it ([CITATION.cff](CITATION.cff)).

[llama.cpp]: https://github.com/ggml-org/llama.cpp
[llama-swap]: https://github.com/mostlygeek/llama-swap
[llm-d]: https://github.com/llm-d/llm-d
