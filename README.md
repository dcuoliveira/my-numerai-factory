# Numerai Agentic Research

An agent-driven research framework for the [Numerai tournament](https://numer.ai/): a
config-driven training pipeline with era-aware cross-validation, BMC scoring, and a set of
skills that let a coding agent run research rounds end to end.

Derived from [numerai/example-scripts](https://github.com/numerai/example-scripts), stripped
down to the agentic framework. Need help? [Find us on Discord.](https://discord.gg/numerai)

## Setup

```bash
pip install numerapi numerai-tools pandas pyarrow lightgbm matplotlib cloudpickle
curl -sL https://numer.ai/install-mcp.sh | bash          # Numerai MCP server (optional)
ln -s $PWD/numerai/agents/skills/* ~/.codex/skills/      # register skills with Codex CLI
```

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
│   ├── skills/            # agent workflows, one SKILL.md each
│   └── tests/
└── v5.3/                  # dataset files (gitignored)
```

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
