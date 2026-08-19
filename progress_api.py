#!/usr/bin/env python3
"""
Progress feed for harness plugins (VSCode, pi, Hermes desktop).

Polls vLLM's /metrics and republishes a small JSON/SSE view that a client can
drive a context bar from. Read-only: it never sits in the request path, so it
cannot slow down or break serving.

    python3 diag/progress_api.py            # serve on :8003
    curl localhost:8003/stats               # one JSON sample
    curl -N localhost:8003/stream           # SSE, ~4 Hz

WHAT THE SERVER CAN AND CANNOT TELL YOU

Available honestly, server-wide:
    running, waiting          how many requests share the GPU right now
    prefill_tok_s             instantaneous prompt throughput
    decode_tok_s              instantaneous generation throughput
    kv_usage_pct              KV cache occupancy
    prefix_hit_pct            cumulative prefix cache hit rate
    acceptance_len            MTP tokens accepted per step

NOT available per request: vLLM exposes no per-request "prefill is 47% done".
The scheduler knows num_computed_tokens but never publishes it. So a per-request
bar has to be driven CLIENT-side, which works because the client knows its own
prompt size and the server publishes the current prefill rate:

    progress += prefill_tok_s * dt        # integrate the live rate
    fraction  = progress / prompt_tokens

That is an ESTIMATE and should be labelled as one in the UI. It is exact at the
two moments that matter: 0 when the request is sent, and 1.0 when the first
token arrives (prefill is over by definition once decode starts). With several
agents active the rate is shared across them, so the bar slows down honestly
when the box is contended -- which is the effect worth showing.

DO NOT drive the bar from vllm:prompt_tokens_total. It is credited when a
prefill COMPLETES: measured here reading 0.0 for a whole 28 s prefill and then
21,736 tok/s in a single sample. It is exposed as prompt_done_tok_s for
completeness only.

STREAMING GOTCHA: this server runs --reasoning-parser qwen3, so deltas may carry
reasoning_content instead of content. A plugin that counts only delta.content
shows a frozen bar for the whole thinking phase. Count both.

Everything else the bar needs is already client-side and exact:
    prompt_tokens   from the usage block, or tokenize locally before sending
    prefill done    the instant the first stream chunk arrives (= TTFT)
    decode rate     count chunk tokens over time between chunks
Ask for usage on the stream with:  "stream_options": {"include_usage": true}

PULSE RATE: pulse the decode bar at decode_tok_s / 50, clamped to [0.15, 1.0].
On this box a healthy single agent is ~20-23 tok/s and 4+ concurrent agents can
push one victim near 5 tok/s, so the pulse doubles as a contention indicator.
"""
import json, time, urllib.request, argparse, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VLLM = "http://127.0.0.1:8001"

# name -> (metric, kind). counters are differentiated into a rate.
WANT = {
    "running":       ("vllm:num_requests_running",  "gauge"),
    "waiting":       ("vllm:num_requests_waiting",  "gauge"),
    "kv_usage_pct":  ("vllm:kv_cache_usage_perc",   "gauge"),
    # MEASURED, not assumed. During a 36 s prefill these were sampled every 3 s:
    #   kv_cache_usage_perc  0.0212 -> 0.0354   climbing every sample
    #   iteration_tokens_sum 107      FROZEN
    #   prompt_tokens_total  66551    FROZEN
    #   generation_tokens    44391    FROZEN
    # Every token counter is credited when a request COMPLETES, so none of them
    # can drive a live bar. KV occupancy is the only signal that moves.
    "_kv":           ("vllm:kv_cache_usage_perc",     "raw"),
    "decode_done_tok_s": ("vllm:generation_tokens_total","counter"),
    "prompt_done_tok_s": ("vllm:prompt_tokens_total",   "counter"),
    "_hits":         ("vllm:prefix_cache_hits",     "raw"),
    "_queries":      ("vllm:prefix_cache_queries",  "raw"),
    "_acc":          ("vllm:spec_decode_num_accepted_tokens_total", "raw"),
    "_drafts":       ("vllm:spec_decode_num_draft_tokens_total",    "raw"),
}


def scrape(base):
    """Sum each metric family across label sets (per-engine splits).

    Every call is a full Prometheus serialization on the API server, so this is
    driven by ONE background poller, never per client request. See State.
    """
    raw = urllib.request.urlopen(base + "/metrics", timeout=5).read().decode()
    vals = {}
    for line in raw.splitlines():
        if not line or line[0] == "#":
            continue
        name = line.split("{")[0].split(" ")[0]
        try:
            v = float(line.rsplit(" ", 1)[1])
        except (ValueError, IndexError):
            continue
        vals[name] = vals.get(name, 0.0) + v
    return vals


class State:
    """Publishes a cached view, refreshed by one background poller.

    Originally every /stats request and every SSE tick scraped vLLM directly, so
    the cost scaled with the number of connected clients: a single SSE stream at
    4 Hz was 4 full /metrics serializations per second on the API server, plus a
    new TCP connection each time, plus an access-log line each time. That is what
    flooded the serve log with GET /metrics.

    Now: one poller at POLL_S, and any number of clients read the cache for free.
    Client-side smoothness is therefore decoupled from server load.
    """

    def __init__(self, base, capacity, poll_s=1.0):
        self.base, self.prev, self.prev_t = base, {}, None
        self.capacity = capacity
        self._prev_kv_tok = None
        self.poll_s = poll_s
        self._lock = threading.Lock()
        self._cached = {"ts": 0.0, "stale": True}
        self._err = None
        t = threading.Thread(target=self._poll_loop, daemon=True)
        t.start()

    def _poll_loop(self):
        while True:
            try:
                s = self._sample_locked()
                with self._lock:
                    self._cached, self._err = s, None
            except Exception as e:
                with self._lock:
                    self._err = str(e)
                    # Keep serving the last good value, but mark it stale so a
                    # client shows a dead meter instead of a frozen-looking one.
                    self._cached = dict(self._cached, stale=True, error=str(e))
            time.sleep(self.poll_s)

    def sample(self):
        """Cheap: returns the cached view. Never touches the network."""
        with self._lock:
            out = dict(self._cached)
        out["age_s"] = round(time.time() - out.get("ts", 0), 2) if out.get("ts") else None
        return out

    def _sample_locked(self):
        now = time.monotonic()
        v = scrape(self.base)
        out, dt = {}, (now - self.prev_t) if self.prev_t else None
        for key, (metric, kind) in WANT.items():
            cur = v.get(metric, 0.0)
            if kind == "counter":
                # A restart resets counters; a negative delta means "no rate yet".
                d = cur - self.prev.get(metric, cur)
                out[key] = round(d / dt, 1) if (dt and dt > 0 and d >= 0) else 0.0
            else:
                out[key] = cur
            self.prev[metric] = cur
        self.prev_t = now

        # Resident tokens, and its derivative: the live prefill rate. This is the
        # one number that moves while a prompt is being processed.
        kv = out.pop("_kv", 0.0)
        out["kv_tokens"] = int(kv * self.capacity)
        dk = out["kv_tokens"] - self._prev_kv_tok if self._prev_kv_tok is not None else 0
        out["prefill_tok_s"] = round(dk / dt, 1) if (dt and dt > 0 and dk > 0) else 0.0
        self._prev_kv_tok = out["kv_tokens"]

        q, h = out.pop("_queries", 0), out.pop("_hits", 0)
        out["prefix_hit_pct"] = round(100.0 * h / q, 1) if q > 0 else None
        dr, ac = out.pop("_drafts", 0), out.pop("_acc", 0)
        # tokens per step = accepted/drafted ratio scaled by draft depth, +1 for
        # the always-emitted base token. num_speculative_tokens is 2 here.
        out["acceptance_len"] = round(1.0 + 2.0 * (ac / dr), 2) if dr > 0 else None
        out["kv_usage_pct"] = round(100.0 * out.get("kv_usage_pct", 0.0), 2)
        out["running"], out["waiting"] = int(out["running"]), int(out["waiting"])
        # UI hints so every harness pulses identically without re-deriving it.
        d = out["decode_done_tok_s"]
        out["pulse_hz"] = round(min(1.0, max(0.15, d / 50.0)), 2)
        out["busy"] = out["running"] > 1 or out["waiting"] > 0
        out["ts"] = time.time()
        out["stale"] = False
        return out


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # keep stdout clean
        pass

    def do_GET(self):
        if self.path.startswith("/stats"):
            try:
                body = json.dumps(self.server.state.sample()).encode()
            except Exception as e:
                self.send_response(502); self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode()); return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers(); self.wfile.write(body)
        elif self.path.startswith("/stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                while True:
                    s = self.server.state.sample()
                    self.wfile.write(b"data: " + json.dumps(s).encode() + b"\n\n")
                    self.wfile.flush()
                    # Reads the cache, so this costs the vLLM server nothing.
                    time.sleep(0.25)
            except Exception:
                pass  # client hung up
        else:
            self.send_response(404); self.end_headers()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8003)
    # 1.0s, not the original per-request scrape. Each scrape is a full Prometheus
    # serialization plus a TCP connect plus an access-log line on the API server.
    ap.add_argument("--poll", type=float, default=1.0,
                    help="seconds between vLLM /metrics scrapes (one poller total)")
    ap.add_argument("--vllm", default=VLLM)
    ap.add_argument("--kv-capacity", type=int, default=0,
                    help="KV tokens; auto-read from newest runs/serve_*.log if 0")
    a = ap.parse_args()
    cap = a.kv_capacity
    if not cap:
        import glob, re
        logs = sorted(glob.glob("runs/serve_*.log"), key=lambda f: -__import__("os").path.getmtime(f))
        for f in logs[:3]:
            m = re.findall(r"GPU KV cache size: ([\d,]+)", open(f, errors="ignore").read())
            if m:
                cap = int(m[-1].replace(",", "")); break
    if not cap:
        cap = 1_054_533
        print(f"WARNING: could not read KV capacity from runs/serve_*.log, assuming {cap:,}")
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), H)
    srv.state = State(a.vllm, cap, poll_s=a.poll)
    print(f"kv capacity: {cap:,} tokens")
    print(f"progress api on :{a.port}  ->  {a.vllm}")
    print(f"  curl localhost:{a.port}/stats")
    print(f"  curl -N localhost:{a.port}/stream")
    srv.serve_forever()
