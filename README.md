# GTA VI Trophy Rarity Forecaster

Entry for the make.fun bounty
[gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01):
forecast how many GTA VI players earn each trophy in week one.

`main.ts` (Node 22, standard library only, no network) reads one case file
from `$MAKEFUN_INPUT_DIR` (default `/input/`) and writes
`$MAKEFUN_OUTPUT_DIR/<id>.out` (default `/output/`): for each of the game's
55 PlayStation 5 trophies, five quantiles (5%, 25%, 50%, 75%, 95%) of its
Exophase earn rate at the case's target instant.

## Model

Per trophy, on the logit scale against `x = ln(1 + hours since release)`:

- Weighted least squares `y = a + b*x` over the reads in the last 36 hours
  before the checkpoint (all reads when fewer than 6), weight 1 for the last
  12 hours decaying linearly to 0.25 at 36 hours.
- The slope is shrunk toward the median slope of trophies in the same tier
  (`b' = (n*b + 3*b_tier) / (n + 3)`), clamped to `[-0.2, 2.0]`, and the
  extrapolated logit move is capped at 2.2.
- The median is the fitted line read at the target, clamped to
  `[0.02, 99.5]`. A trophy already at 100% stays at 100%.
- The spread combines the fit's residual variance, the slope's standard
  error over the extrapolation distance, a random-walk term growing with the
  horizon (`0.30 * sqrt(horizon_days)`), and a "stale page" term (`0.35^2`)
  when every rate was unchanged over the last 6 hours — on launch night
  that staleness signalled an Exophase refresh (and dilution) was due.
  Sigma is clamped to `[0.25, 1.6]`.
- Quantiles are normal quantiles of the logit median with that spread,
  rounded to 4 decimals, non-decreasing, clamped to `[0, 100]`.

The program is fully deterministic: two runs on the same case write
byte-identical output.

## Run locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

## License

MIT — see `LICENSE`.
