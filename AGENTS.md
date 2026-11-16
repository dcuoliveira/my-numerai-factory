# Repository instructions for Codex

## Keep Claude Code and Codex in sync

When the user asks to add repository context or skills, make equivalent
additions for both agents in the same change. Put context in both
`CLAUDE.md` and `AGENTS.md`, and add matching skills under
`.claude/skills/` and `.agents/skills/`. Use each agent's own paths and
invocation syntax, then verify that both versions give the same guidance.
If the user explicitly limits the request to one agent, follow that scope.

## Commit and push repository changes

After verifying each requested repository change, commit it and push it to the
branch's configured remote. Stage only files that belong to the requested
work. If the user explicitly asks to keep a change local, follow that scope.
Report any commit or push failure.

## Numerai Crypto missing values

When creating, extending, rebuilding, testing or evaluating a Numerai Crypto
dataset with potentially missing candidate features, use
[`$numerai-crypto-missing-values`](.agents/skills/numerai-crypto-missing-values/SKILL.md)
before implementation. Before accepting or interpreting the resulting experiment,
use [`$numerai-crypto-missing-values-audit`](.agents/skills/numerai-crypto-missing-values-audit/SKILL.md).
