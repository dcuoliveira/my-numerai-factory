# Numerai Agentic Research

An agent-driven research framework for the [Numerai tournament](https://numer.ai/): a
config-driven training pipeline with era-aware cross-validation, BMC scoring, and a set of
skills that let a coding agent run research rounds end to end.

Derived from [numerai/example-scripts](https://github.com/numerai/example-scripts), stripped
down to the agentic framework. Need help? [Find us on Discord.](https://discord.gg/numerai)

## Setup

```bash
pip install numerapi numerai-tools pandas pyarrow lightgbm matplotlib cloudpickle
curl -sL https://numer.ai/install-mcp.sh | bash     # Numerai MCP server (optional)
```

Skills live in `.claude/skills/` and Claude Code picks them up automatically. For Codex CLI,
symlink them in: `ln -s $PWD/.claude/skills/* ~/.codex/skills/`

Build the datasets (downloads ~v5.3 train + validation, then writes full and every-4th-era
downsampled parquets under `numerai/v5.3/`):

```bash
PYTHONPATH=numerai python3 -m agents.code.data.build_full_datasets
```

## Run

Train and score a config. Results land in `<output_dir>/results/<name>.json` and
`<output_dir>/predictions/<name>.parquet`:

```bash
PYTHONPATH=numerai python3 -m agents.code.modeling --config numerai/agents/baselines/configs/deep_lgbm_ender20_baseline_downsampled.py
```

Or hand the whole loop to an agent:

```bash
codex exec --yolo "find the best neural network architecture to predict target ender"
```

## Layout

```
.claude/skills/            # agent workflows, one SKILL.md each
numerai/
├── agents/
│   ├── AGENTS.md          # modeling philosophy and conventions
│   ├── baselines/         # baseline configs + their results
│   ├── code/
│   │   ├── analysis/      # show_experiment, plot_benchmark_corrs, summarize_results
│   │   ├── data/          # dataset builders
│   │   ├── metrics/       # BMC / corr scoring
│   │   └── modeling/      # training pipeline (models/ + model-agnostic utils/)
│   ├── experiments/       # one folder per line of inquiry (gitignored)
│   └── tests/
└── v5.3/                  # dataset files (gitignored)
```

## Runners

One shell script per study, in `experiments/`. Each documents in its own header
what it runs, what it is measured against, and what would count as a result.
All take `DRY_RUN=1` to print the commands without running them.

| Runner | What it measures |
|---|---|
| `lgb_6m6m_reg_crypto.sh` | LightGBM regression, 9 arms: raw 22 -> allfeat++ 812 -> the 7 raw feature groups alone |
| `lgb_6m6m_clf_crypto.sh` | The same 9 arms as classification |
| `ridge_comparison_crypto.sh` | Fixed-grid ridge vs the glmnet lambda path, across tasks and objectives |
| `refit_cadence_crypto.sh` | How often to refit, and over how long a training window |
| `feature_groups_crypto.sh` | Each of the 7 raw feature groups fitted alone, both ridge variants |
| `group_blends_crypto.sh` | Best trio and quintet of groups: blend of models vs one model on the union |
| `feature_engineering_crypto.sh` | Target encoding, one-hot, rolling means, spreads, interactions |
| `all_features_crypto.sh` | Ceiling check: 788 features at once |
| `allfeat_plus_crypto.sh` | The genuine union, 812 features (needs the chain from `setup.sh --with-derived`) |
| `ablation_crypto.sh` | Feature importance by leave-one-block-out against allfeat++ |
| `weighting_crypto.sh` | How to weight CURUPIRA_05's five members |
| `weighting_sweep_crypto.sh` | Tunes the four surviving weighting schemes; read with `analysis.weighting_report` |
| `target60_crypto.sh` | Every deployed model refit from scratch on the 60-day target |
| `e01_blocks60_lgb_reg_crypto.sh` | **E01** — the LightGBM feature-block table on the 60-day target (see `experiments/EXPERIMENTS.md`) |
| `autoencoder_crypto.sh` | Bottleneck-autoencoder codes as extra features, against a PCA control and an untuned variant (needs torch) |

`lgb_6m6m_crypto.sh` is the shared implementation behind the two `lgb_6m6m_*`
runners; invoke those, not it. Knobs common to the LightGBM runners:
`ONLY=`, `SKIP=`, `SHARD=/OF=`, `SMOKE=1`, `LGB_N_ITER=`, `LGB_SEARCH_TYPE=`.

## Skills

| Skill | Use it for |
|---|---|
| `numerai-research` | End-to-end: the meta-workflow that sequences the rest |
| `numerai-experiment-design` | Plan and run experiment rounds, sweep, scale winners |
| `numerai-model-implementation` | Add a new model type to the pipeline |
| `report-research` | Write `experiment.md` and generate the standard plots |
| `numerai-model-upload` | Build a submission pickle and deploy it via MCP |

Read [AGENTS.md](AGENTS.md) for the full agent brief: tournament rules, MCP tools, metrics,
and the pickle upload workflow.
