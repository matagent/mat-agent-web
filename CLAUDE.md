# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

MatAgent is a materials science AI platform. It exposes materials science tools (Materials Project/OQMD queries, crystal structure building/visualization, ML bandgap prediction, VASP job management) via MCP protocol, wraps them in a LangChain agent, and serves everything through a Streamlit web UI.

## Commands

```bash
# Dependency management (Python 3.13.4 required)
uv sync
uv pip install -r requirements.txt

# Start services (run in separate terminals, in order)
uv run mcp_server.py             # MCP tool server (port 8000)
uv run agent_mcp_server.py       # FastAPI agent server (port 8766)
streamlit run web_mcp_app.py     # Web UI (port 8501)
python flask_server.py           # 2D/3D file server (port 6750)
```

## Architecture

```
Streamlit UI (web_mcp_app.py) → FastAPI Agent Server (agent_mcp_server.py) → Agent (agent/langchain_mcp_agent.py) → MCP Server (mcp_server.py)
                                                                                  │                                        │
                                                                                  ▼                                        ▼
                                                                           LangChain Agent                          MCP tools layer
                                                                           (langchain-mcp-adapters)          (FastMCP + @mcp.tool() decorators)
```

- **`agent/langchain_mcp_agent.py`** — `MatAgentMCP` class. Connects to the MCP server via SSE, loads tools through `MultiServerMCPClient`, wraps them (MCP returns `list[dict]` but LangChain needs strings), creates a LangChain agent with `ChatOpenAI`. Supports `deepseek-v4-flash`, `deepseek-v4-pro`, `kimi-k2.6`, `kimi-k2.5`, and `glm-5` models. Provides sync (`chat`) and async (`chat_stream`) interfaces.
- **`agent_mcp_server.py`** — FastAPI on port 8766. Manages a global `MatAgentMCP` singleton. API endpoints: `/chat`, `/chat/stream`, `/tools`, `/sessions/`, `/materials/search`, `/materials/structure/{id}`, `/predict_bandgap`. Persists chat history to `matagent_server_history.db` via module-level SQLite functions (no shared connections — critical for Streamlit multi-threading).
- **`mcp_server.py`** — FastMCP server on port 8000. ~20 `@mcp.tool()` functions: `search_materials_from_mp`, `search_materials_from_oqmd`, `get_material_structure_from_mp`, `build_structure`, `predict_bandgap`, `create_task`, `submit_mission`, `extract_result`, `get_time`, etc. Uses MPRester (Materials Project API) and pymatgen/ASE for structure manipulation.
- **`flask_server.py`** — Flask on port 6750. Serves 2D structure images from `temp_images/` and 3D HTML visualizations from `temp_3d/`. Loads/saves structure metadata via `structure_info.json`.
- **`tryssh.py`** — `VaspTaskInitializer` class. SSH connection via paramiko, creates VASP input files, submits/manages calculations (relax, SCF, band, DOS), extracts results.
- **`oqmd.py`** — Scrapes OQMD materials database via HTTP/BeautifulSoup.
- **`databasemanage.py`** — `DatabaseManager` class. SQLite storage for materials (structure as pickle BLOB) and chat history. Used by the Streamlit frontend (`matagent.db`).
- **`myml/`** — ML models: `bandgap_predict.py` (XGBoost model from `xgb_model.json`), `featurizer.py`, atomic orbital calculations.

## Environment variables (.env)

| Variable | Required | Purpose |
|---|---|---|
| `DEEPSEEK_API_KEY` | Yes | DeepSeek API key |
| `ZAI_API_KEY` | No | Zhipu GLM-5 API key |
| `mp_API_KEY` / `MP_API_KEY` | No | Materials Project API key |
| `local_HOST` | For VASP | Local IP for file server URLs |
| `HOST`, `PORT`, `USERNAME`, `PASSWORD` | For VASP | SSH connection |
| `base_dir` | For VASP | Remote VASP tasks directory |
| `MATAGENT_SYSTEM_PROMPT` | No | Custom system prompt for the agent |

## Important patterns and pitfalls

- **StructuredTool is not directly callable.** MCP `@tool` functions become `StructuredTool` objects. Use `MatAgentMCP` class methods (which wrap them), not the raw `@tool` functions.
- **SQLite + Streamlit = module-level functions only.** Streamlit's multi-threading breaks shared DB connections. `agent_mcp_server.py` uses module-level functions that create a new connection each call — follow that pattern, never use a class with a shared `self.conn`.
- **MCP tool results are `list[dict]`.** The `_wrap_tool` method in `agent/langchain_mcp_agent.py` converts MCP's `[{"text": "..."}]` format to strings that LangChain can process.
- **`flask_server.py` persists structure metadata** to `structure_info.json` so 3D visualizations survive restarts.
- **Chat history** lives in `matagent_server_history.db` (agent server) or `matagent_history.db`. Sessions are keyed by `session_id` UUID.
