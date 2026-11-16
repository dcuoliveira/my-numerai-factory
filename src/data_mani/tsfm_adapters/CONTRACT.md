# Zero-shot TSFM adapter contract (E154-E163)

One file per model, `src/data_mani/tsfm_adapters/<name>.py`, importable from that model's own
environment with `PYTHONPATH=<repo>`. It must not import anything from the repo except numpy/pandas
and its own model code. It defines:

```python
class Forecaster:
    name: str              # e.g. "tempopfn"
    point: str             # what forecast() returns: "median" (q0.5 of a quantile head), "mean", or "point" (a single regression output)
    max_context: int|None  # longest context the model accepts (None = unbounded); the runner only asks for <= this
    max_horizon: int|None
    handles_nan: bool      # True if missing values inside the context are passed to the model natively
    checkpoint: str        # exact weights used: path + sha256 of the file, or repo@revision

    def __init__(self, device: str = "cuda:0"): ...

    def forecast(self, ctx: np.ndarray, n_steps: int, last_dates: np.ndarray) -> np.ndarray:
        """
        ctx        float32 [n, L]; row k is ONE univariate daily series, oldest -> newest, the last column is
                   the day last_dates[k]. NaN = missing. Short histories are LEFT-padded with NaN.
        n_steps    forecast steps 1..n_steps = the days last_dates[k]+1 .. last_dates[k]+n_steps (daily frequency).
        last_dates numpy datetime64[D] [n]; may differ per row.
        returns    float32 [n, n_steps] point forecast in the ORIGINAL units of ctx. NaN only if the model
                   genuinely cannot forecast that row (document why).
        """
```

Rules:
- Every row is forecast independently: no information may flow between rows (per-series scaling, no
  cross-series attention). Batching for speed is fine if it provably does not change a row's output
  (check: a row forecast alone vs inside a batch of 256 must agree to <= 1e-4 relative).
- Weights frozen, eval mode, no fine-tuning, no test-time adaptation, fixed seeds if anything is stochastic.
- If the model cannot take NaN: drop the leading NaN padding per row, forward-fill interior NaN (and
  back-fill nothing). Document it in a docstring. If after that a row has fewer points than the model's
  minimum, return NaN for it.
- If the model needs dates/time features, build them from last_dates and the daily frequency.
- If a context longer than max_context is passed, keep the most recent max_context points.
