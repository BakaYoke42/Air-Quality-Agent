# MCP Inspector verification

The production server was started at `http://127.0.0.1:8000/mcp` and exercised
with MCP Inspector CLI `0.17.2` over Streamable HTTP.

```text
npx -y @modelcontextprotocol/inspector@0.17.2 --cli \
  http://127.0.0.1:8000/mcp --transport http --method tools/list
```

`tools/list` exited successfully and returned exactly:

1. `search_air_quality_evidence`
2. `get_country_air_quality`
3. `compare_countries`
4. `find_station_extremes`

All four entries exposed JSON input schemas and complete selection
descriptions containing `Use when`, `Do NOT use`, `Returns`, preference
guidance, and an example.

## Inspector calls

| Tool/case | Inspector result |
| --- | --- |
| Documentary search, `top_k=1`, HyDE disabled | `status: ok`; one parent-expanded result |
| France 2024 PM2.5 country summary | `status: ok`; 239 retained points and full statistics |
| FR/DE/IT 2024 NO2 comparison | `status: ok`; three ranked country rows |
| Two highest Italian 2024 NO2 points | `status: ok`; two sampling-point records |
| Invalid country `ES` | Controlled `status: error`, `error_type: invalid_arguments`; no uncaught server exception |

The documentary call was run with Hugging Face/Transformers offline mode after
the already-downloaded models were cached. This avoids a nonessential Hub
metadata request and does not change retrieval.

The host has Node 20.19.2, while this Inspector release declares Node
`>=22.7.5`; npm emitted an engine warning, but all CLI commands exited with
status 0. Upgrade Node for warning-free use of current Inspector releases. The
project’s Python runtime does not depend on Node.
