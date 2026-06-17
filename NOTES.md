# Running on uv
Run `uv sync`.

Activate environment with `source .venv/bin/activate`.

`uv pip install denario-app`

Run `denario run`.

# Custom OpenAI-compatible endpoint (e.g. Edinburgh ELM)

Lets Denario route LLM calls to a custom OpenAI-compatible proxy with no source edits going forward.

- Base URL: set via env vars (zero code) — `OPENAI_API_KEY`, `OPENAI_BASE_URL` (openai SDK / cmbagent path) and `OPENAI_API_BASE` (LangChain core readers). See `.env.example`.
- Model-id remap: `DENARIO_MODEL_OVERRIDES` env var (JSON map of dated id -> id the proxy accepts), since ELM exposes bare names (`gpt-4o`, `gpt-4.1`, `o3-mini`) not Denario's dated defaults.
  - `denario/utils.py`: `_model_overrides()` + remap in `llm_parser` (covers all Denario-selected models, both paths).
  - `denario/utils.py` `install_cmbagent_model_overrides()`, called from `denario/__init__.py`: wraps `cmbagent.cmbagent.get_model_config` so cmbagent's internal-default agents (`plot_judge`, `camb_context`, …) are remapped too.
- Use OpenAI-type models only (`gpt-*`, `o3-*`); Gemini/Claude are not redirected.
