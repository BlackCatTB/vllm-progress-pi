/**
 * vLLM progress meter for pi.
 *
 * Shows a context/prefill/decode bar in the footer, fed by diag/progress_api.py
 * (default http://127.0.0.1:8003).
 *
 *   /vllm         toggle the meter on/off
 *   /vllm 10.42.42.2:8003   point it at a remote box
 *
 * WHAT THE BAR MEANS
 *   dim block   context already resident in KV (prefix cache hit -- free)
 *   blue block  prefill in progress, i.e. tokens being computed now
 *   green pulse decode; pulse rate scales with tok/s, so a stalled agent is
 *               visibly slower than a healthy one
 *   N req       how many requests share the GPU right now. When this is >1 your
 *               decode is being shared, which is the usual reason a turn feels
 *               slow.
 *
 * The prefill rate comes from KV occupancy, because every token counter vLLM
 * exposes is credited only when a request COMPLETES and therefore sits frozen
 * for the entire prefill. See diag/progress_api.py for the measurement.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

interface Stats {
	running: number;
	waiting: number;
	kv_tokens: number;
	kv_usage_pct: number;
	prefill_tok_s: number;
	decode_done_tok_s: number;
	prefix_hit_pct: number | null;
	acceptance_len: number | null;
	pulse_hz: number;
	busy: boolean;
}

const KEY = "vllm-progress";
const BAR_W = 18;

export default function (pi: ExtensionAPI) {
	let host = process.env.VLLM_PROGRESS_URL || "http://127.0.0.1:8003";
	let timer: ReturnType<typeof setInterval> | undefined;
	let phase = 0;
	let turnActive = false;
	let kvAtTurnStart: number | null = null;
	let lastCtx: Ctx | undefined;

	type Ctx = Parameters<Parameters<ExtensionAPI["on"]>[1]>[1];

	async function poll(): Promise<Stats | undefined> {
		try {
			const c = new AbortController();
			const t = setTimeout(() => c.abort(), 1500);
			const r = await fetch(`${host}/stats`, { signal: c.signal });
			clearTimeout(t);
			if (!r.ok) return undefined;
			return (await r.json()) as Stats;
		} catch {
			return undefined; // server down or feed not running: stay quiet
		}
	}

	function render(ctx: Ctx, s: Stats | undefined) {
		const theme = ctx.ui.theme;
		if (!s) {
			ctx.ui.setStatus(KEY, theme.fg("dim", "vllm ?"));
			return;
		}

		// Tokens computed since this turn began. The feed reports KV occupancy,
		// so a prefix-cache hit shows up as an instant jump -- correct, because
		// those tokens genuinely were not recomputed.
		const delta = kvAtTurnStart === null ? 0 : Math.max(0, s.kv_tokens - kvAtTurnStart);

		const prefilling = s.prefill_tok_s > 5;
		const decoding = turnActive && !prefilling;

		// Bar shows KV pool occupancy: how much of the server's context budget is
		// in use across ALL agents, which is the number that predicts eviction.
		const filled = Math.round((s.kv_usage_pct / 100) * BAR_W);
		let bar = "";
		for (let i = 0; i < BAR_W; i++) {
			if (i < filled) {
				bar += prefilling && i >= filled - 2 ? theme.fg("accent", "█") : theme.fg("dim", "█");
			} else {
				bar += theme.fg("dim", "░");
			}
		}

		const bits: string[] = [bar];

		if (prefilling) {
			bits.push(theme.fg("accent", `pp ${Math.round(s.prefill_tok_s)}/s`));
			if (delta > 0) bits.push(theme.fg("dim", `${(delta / 1000).toFixed(1)}k`));
		} else if (decoding) {
			// Pulse so a slow agent looks slow. pulse_hz is supplied by the feed
			// so every harness animates identically.
			const frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧"];
			const step = Math.max(1, Math.round(1 / Math.max(0.15, s.pulse_hz)));
			const f = frames[Math.floor(phase / step) % frames.length];
			bits.push(theme.fg("success", `${f} decode`));
		}

		if (s.running > 1 || s.waiting > 0) {
			// The single most useful number when a turn feels slow.
			bits.push(theme.fg("warning", `${s.running} req${s.waiting ? `+${s.waiting}w` : ""}`));
		}
		if (s.acceptance_len !== null) bits.push(theme.fg("dim", `mtp ${s.acceptance_len.toFixed(1)}`));

		ctx.ui.setStatus(KEY, bits.join(" "));
	}

	function start(ctx: Ctx) {
		if (timer) return;
		timer = setInterval(async () => {
			phase++;
			const s = await poll();
			if (s && kvAtTurnStart === null && turnActive) kvAtTurnStart = s.kv_tokens;
			render(ctx, s);
		}, 250);
		// Do not hold the process open on exit.
		(timer as unknown as { unref?: () => void }).unref?.();
	}

	function stop(ctx: Ctx) {
		if (timer) clearInterval(timer);
		timer = undefined;
		ctx.ui.setStatus(KEY, undefined);
	}

	pi.registerCommand("vllm", {
		description: "Toggle the vLLM progress meter (optionally: /vllm host:port)",
		handler: async (args, ctx) => {
			lastCtx = ctx;
			const arg = (args || "").trim();
			if (arg) {
				host = arg.startsWith("http") ? arg : `http://${arg}`;
				ctx.ui.notify(`vllm progress: ${host}`, "info");
			}
			if (timer && !arg) {
				stop(ctx);
				ctx.ui.notify("vllm progress meter off", "info");
				return;
			}
			const s = await poll();
			if (!s) {
				ctx.ui.notify(
					`No progress feed at ${host}. Start it with: python3 diag/progress_api.py --port 8003`,
					"warn",
				);
				return;
			}
			start(ctx);
			ctx.ui.notify("vllm progress meter on", "info");
		},
	});

	pi.on("turn_start", async (_e, ctx) => {
		lastCtx = ctx;
		turnActive = true;
		kvAtTurnStart = null; // re-baseline on the next poll
	});

	pi.on("turn_end", async (_e, ctx) => {
		lastCtx = ctx;
		turnActive = false;
		kvAtTurnStart = null;
	});

	pi.on("agent_end", async (_e, ctx) => {
		if (timer) stop(ctx);
	});
}
