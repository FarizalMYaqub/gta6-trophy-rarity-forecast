// GTA VI trophy earn-rate forecaster (make.fun: gta6-trophy-rarity-forecast-01).
//
// Reads every case file in $MAKEFUN_INPUT_DIR (default /input/) and writes
// one forecast per case to $MAKEFUN_OUTPUT_DIR (default /output/):
//   {"trophies":[{"id":n,"q":[q05,q25,q50,q75,q95]}]}
// for each of the case's trophies, five quantiles of its Exophase earn rate
// (percent of tracked players) at the case's target instant.
//
// Method, per trophy:
//   * Work on the logit scale against x = ln(1 + hours since release
//     2026-11-19T00:00:00Z). Fit a weighted least-squares line y = a + b*x
//     over the reads in the last 36 hours before the checkpoint (all reads
//     when fewer than 6), with weight 1 for the last 12 hours decaying
//     linearly to 0.25 at 36 hours.
//   * Shrink the slope toward the median slope of trophies in the same tier
//     (b' = (n*b + 3*b_tier) / (n + 3)), clamp b' to [-0.2, 2.0], and cap the
//     extrapolated logit move at 2.2.
//   * The median is the line read at the target, clamped to [0.02, 99.5].
//     A trophy already at 100% stays at 100%.
//   * The spread combines the fit's residual variance, the slope's standard
//     error over the extrapolation distance, a random-walk term that grows
//     with the horizon (0.30 * sqrt(horizon_days)), and a "stale page" term
//     (0.35^2) when every rate was unchanged over the last 6 hours, which
//     on launch night signalled an Exophase refresh (and dilution) was due.
//     Sigma is clamped to [0.25, 1.6].
//   * Quantiles are the normal quantiles of the logit median with that
//     spread, rounded to 4 decimals, non-decreasing, clamped to [0, 100].
//
// Node 22, standard library only, no network, fully deterministic: two runs
// on the same case write byte-identical output.

import { readdirSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";

const RELEASE_MS = Date.parse("2026-11-19T00:00:00Z");
const Z = [-1.64485, -0.67449, 0, 0.67449, 1.64485];

interface Trophy {
  id: number;
  tier: string;
}

interface Read {
  at: string;
  rates: (number | null)[];
}

interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  trophies: Trophy[];
  history: Read[];
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function logitPct(rate: number): number {
  const p = clamp(rate, 0.01, 99.99) / 100;
  return Math.log(p / (1 - p));
}

function sigmoid(z: number): number {
  if (z >= 0) {
    const e = Math.exp(-z);
    return 1 / (1 + e);
  }
  const e = Math.exp(z);
  return e / (1 + e);
}

function hoursSinceRelease(ms: number): number {
  return Math.max((ms - RELEASE_MS) / 3600000, 0);
}

function median(values: number[]): number {
  const s = [...values].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

interface Fit {
  a: number;
  b: number;
  n: number;
  residVar: number;
  seB: number;
  xLast: number;
  lastRate: number;
  stale: boolean;
  varY: number;
}

// Weighted least squares of logit(rate) on ln(1 + hours since release).
function fitTrophy(c: Case, idx: number): Fit {
  const atMs = Date.parse(c.at);
  const pts: { x: number; y: number; w: number; at: number; rate: number }[] = [];
  for (const r of c.history) {
    const rate = r.rates[idx];
    if (rate === null || rate === undefined) continue;
    const t = Date.parse(r.at);
    const ageH = (atMs - t) / 3600000;
    const w = ageH <= 12 ? 1 : ageH >= 36 ? 0.25 : 1 - 0.75 * ((ageH - 12) / 24);
    pts.push({
      x: Math.log(1 + hoursSinceRelease(t)),
      y: logitPct(rate),
      w,
      at: t,
      rate,
    });
  }

  // Window: reads in the last 36h before the checkpoint; all reads when < 6.
  let use = pts.filter((p) => (atMs - p.at) / 3600000 <= 36);
  if (use.length < 6) use = pts;

  const lastRate = pts.length > 0 ? pts[pts.length - 1].rate : 0;
  const xLast = pts.length > 0 ? pts[pts.length - 1].x : 0;

  if (use.length === 0) {
    return { a: logitPct(0.5), b: 0, n: 0, residVar: 0.02, seB: 0, xLast, lastRate, stale: false, varY: 0 };
  }

  let sw = 0;
  let sx = 0;
  let sy = 0;
  for (const p of use) {
    sw += p.w;
    sx += p.w * p.x;
    sy += p.w * p.y;
  }
  const xBar = sx / sw;
  const yBar = sy / sw;

  let sxx = 0;
  let sxy = 0;
  for (const p of use) {
    sxx += p.w * (p.x - xBar) * (p.x - xBar);
    sxy += p.w * (p.x - xBar) * (p.y - yBar);
  }

  let b = 0;
  let a = yBar;
  if (sxx > 1e-12) {
    b = sxy / sxx;
    a = yBar - b * xBar;
  }

  let resid = 0;
  for (const p of use) {
    const e = p.y - (a + b * p.x);
    resid += p.w * e * e;
  }
  const residVar = Math.max(resid / sw, 0.02);
  const seB = sxx > 1e-12 ? Math.sqrt(residVar / sxx) : 0;

  let varY = 0;
  for (const p of use) varY += p.w * (p.y - yBar) * (p.y - yBar);
  varY /= sw;

  // Stale page: every rate unchanged over the last 6 hours before checkpoint.
  let stale = false;
  const recent = pts.filter((p) => (atMs - p.at) / 3600000 <= 6);
  if (recent.length >= 2) {
    stale = recent.every((p) => p.rate === lastRate);
  }

  return { a, b, n: use.length, residVar, seB, xLast, lastRate, stale, varY };
}

function forecastCase(c: Case): { trophies: { id: number; q: number[] }[] } {
  const fits = c.trophies.map((_, i) => fitTrophy(c, i));

  // Tier median slopes, from trophies with >= 6 reads and real variance.
  const tierSlopes = new Map<string, number>();
  const tiers = [...new Set(c.trophies.map((t) => t.tier))];
  for (const tier of tiers) {
    const slopes: number[] = [];
    c.trophies.forEach((t, i) => {
      if (t.tier === tier && fits[i].n >= 6 && fits[i].varY > 0) slopes.push(fits[i].b);
    });
    if (slopes.length > 0) tierSlopes.set(tier, median(slopes));
  }

  const xTarget = Math.log(1 + hoursSinceRelease(Date.parse(c.target)));
  const horizonDays = c.horizon_minutes / 1440;

  const rows = c.trophies.map((t, i) => {
    const f = fits[i];

    // A trophy already at 100% stays there.
    if (f.lastRate >= 99.99) {
      return { id: t.id, q: [99.9, 99.97, 100, 100, 100] };
    }

    const bTier = tierSlopes.get(t.tier);
    const bShrunk =
      bTier === undefined ? f.b : (f.n * f.b + 3 * bTier) / (f.n + 3);
    // Always clamp the shrunk slope, even when no tier median exists
    // (e.g. practice, where every same-tier slope has zero variance).
    const bPrime = clamp(bShrunk, -0.2, 2.0);
    const dx = xTarget - f.xLast;
    const move = clamp(bPrime * dx, -2.2, 2.2);
    const logitMedian = f.a + bPrime * f.xLast + move;
    const m = clamp(100 * sigmoid(logitMedian), 0.02, 99.5);

    const staleTerm = f.stale ? 0.35 * 0.35 : 0;
    const sigmaSq =
      f.residVar +
      (f.seB * dx) * (f.seB * dx) +
      (0.3 * Math.sqrt(horizonDays)) * (0.3 * Math.sqrt(horizonDays)) +
      staleTerm;
    const sigma = clamp(Math.sqrt(sigmaSq), 0.25, 1.6);

    const lm = logitPct(m);
    const q: number[] = [];
    for (const z of Z) {
      const v = clamp(100 * sigmoid(lm + z * sigma), 0, 100);
      q.push(Math.round(v * 10000) / 10000);
    }
    // Enforce non-decreasing after rounding.
    for (let k = 1; k < q.length; k++) {
      if (q[k] < q[k - 1]) q[k] = q[k - 1];
    }
    return { id: t.id, q };
  });

  return { trophies: rows };
}

function main(): void {
  const inDir = process.env.MAKEFUN_INPUT_DIR ?? "/input/";
  const outDir = process.env.MAKEFUN_OUTPUT_DIR ?? "/output/";
  mkdirSync(outDir, { recursive: true });

  const files = readdirSync(inDir)
    .filter((f) => f.endsWith(".in"))
    .sort();
  for (const f of files) {
    const c: Case = JSON.parse(readFileSync(join(inDir, f), "utf8"));
    const out = forecastCase(c);
    writeFileSync(join(outDir, `${c.id}.out`), JSON.stringify(out));
  }
}

main();
