# AWorld 1.0

A minimal Python agent kernel: **Agent + Context + Tool**, with Session/Run as the application entry.

```bash
python -m pip install '.[llm,skills]'
aworld run --demo --task hello
aworld run --model MODEL --base-url URL --task 'Your task' --skill-path ~/.agents/skills
```

Credentials come from `AWORLD_API_KEY` or `OPENAI_API_KEY`; they are never CLI arguments. The core package has no mandatory third-party dependencies. `llm` adds HTTPX, and `skills` adds YAML parsing.

`aworld` is the single Python namespace and distribution, containing the kernel and CLI. The optional `aworld-cli` distribution is only a thin wrapper around that CLI. `aworldv1` identifies the new **Lingguang Bench Runtime Harness**, selected with `--harness aworldv1`; it is not a Python package.

Each Session owns its context history; each submitted Run drives a direct model/tool loop. LocalSandbox provides local filesystem and process execution. Defaults include read, write, bash, read_session, search_sessions and session_query. Subagents are implemented as tools with child Session/Run lifecycles. Skills are discovered as metadata and read when needed.

This branch replaces the old framework and CLI implementations. It provides no compatibility layer for LLMAgent, Runners, event bus, automatic dependency installation or global MemoryFactory. History uses a small storage interface; optional MemoryStoreAdapter accepts an explicit store and item factory.

Build with `python -m build`; packaging is declared in `pyproject.toml` using Hatchling. No setup.py executes application code or installs packages during a build.

The first version uses in-process Sessions and full context history. It does not yet provide cross-process session persistence, token-budget compaction, streaming or provider retry policies.

[Session/Run contract](aworld/docs/aworld-1.0-session-run-contract.md) · [CLI and packaging](aworld/docs/aworld-1.0-packaging-cli.md)
