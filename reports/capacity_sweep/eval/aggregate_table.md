| Metric (validation, mean ± sd over seeds) | 6×256 | 6×384 | 6×512 |
|---|---|---|---|
| Parameters | 4,115,846 | 9,000,006 | 15,768,326 |
| First in-window STOP recall | 0.347 ± 0.159 | 0.396 ± 0.178 | 0.165 ± 0.286 |
| Never stopped (censored rate) | 0.653 ± 0.159 | 0.604 ± 0.178 | 0.835 ± 0.286 |
| Premature stop — during overlap | 0.028 ± 0.026 | 0.021 ± 0.018 | 0.032 ± 0.055 |
| Premature stop — during non-overlap | 0.084 ± 0.032 | 0.112 ± 0.052 | 0.042 ± 0.073 |
| Hesitation-commit false stops (segment rate) | 0.579 ± 0.091 | 0.596 ± 0.061 | 0.211 ± 0.365 |
| Noise false stops (segment rate) | 0.147 ± 0.051 | 0.157 ± 0.034 | 0.059 ± 0.102 |
| Background-speech false stops (segment rate) | 0.233 ± 0.115 | 0.289 ± 0.204 | 0.100 ± 0.173 |
| Backchannel false stops (segment rate) | 0.217 ± 0.059 | 0.287 ± 0.117 | 0.085 ± 0.148 |
| Total false stops (frames) | 4035.0 ± 1279.6 | 4837.0 ± 1859.0 | 1637.0 ± 2835.4 |
| Decision latency median (ms) | — | 381.0 ± 0.0 | — |
| Decision latency p95 (ms) | — | — | — |
| Stop-decision F-beta (β=2) | 0.285 ± 0.080 | 0.294 ± 0.067 | 0.360 ± 0.000 |
