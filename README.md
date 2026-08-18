# vllm-progress-pi

A context / prefill / decode meter in the [pi](https://pi.dev) footer, fed by a
local vLLM server.

```
████████░░░░░░░░░░ pp 491/s 12.4k   3 req   mtp 2.4
```

## Install

```bash
pi install https://github.com/BlackCatTB/vllm-progress-pi
```

Then `/vllm` to toggle it on, or `/vllm 192.168.1.50:8003` to point it at another
box.

## It needs a feed

The meter reads a small read-only service that polls vLLM's `/metrics`. It never
sits in the request path, so it cannot slow down or break serving.

```bash
python3 progress_api.py --port 8003
```

`progress_api.py` is included in this repo under `server/`.

## Reading it

| element | meaning |
|---|---|
| `████░░░░` | KV pool occupancy across **all** clients. This is what predicts eviction. |
| `pp 491/s` | prefill running, at that many tokens/s |
| `12.4k` | tokens computed since this turn started |
| `⠹ decode` | decode running; pulse speed tracks tok/s, so a starved agent visibly crawls |
| `3 req` | three requests share the GPU — the usual reason a turn feels slow |
| `mtp 2.4` | speculative-decoding acceptance, tokens per step |

## Why prefill is derived from KV occupancy

Every token counter vLLM exposes is credited only when a request **completes**.
Measured on a 4×MI50 box, sampling every 3 s through a 36-second prefill:

```
kv_cache_usage_perc  0.0212 -> 0.0354   climbing every sample
iteration_tokens_sum 107      FROZEN
prompt_tokens_total  66551    FROZEN
generation_tokens    44391    FROZEN
```

So `prompt_tokens_total` reads 0 tok/s for the whole prefill and then spikes to
21,736 tok/s in a single sample — useless for a live bar. KV occupancy is the
only value that moves while a prompt is being processed, so the feed publishes
`kv_tokens` and its derivative.

Verified live on a 45k prompt: `kv_tokens` 22,373 → 40,272 with `prefill_tok_s`
holding **491 ±1.5**.

A consequence worth knowing: a prefix-cache hit appears as an instant jump rather
than a sweep. That is correct — those tokens genuinely were not recomputed.

## Decode tok/s

Measure it client-side from streaming chunk timings; it is exact there, and the
server cannot provide it live for the same reason as above.

**Gotcha:** if your server runs a reasoning parser (e.g. `--reasoning-parser
qwen3`), the thinking phase arrives as `delta.reasoning_content`, not
`delta.content`. A client that counts only `delta.content` shows a frozen bar for
the entire thinking phase.

## License

MIT
