# Codex Proxy Plugin for [Opencode](https://github.com/sst/opencode)

An OpenCode plugin that wires the ChatGPT provider to a Codex-compatible proxy.

## Installation

```bash
mv plugin_codexproxy.ts ~/.config/opencode/plugin/
pip3 install .
```

This exposes the `codex-proxy` CLI and Install plugin for opencode.

### Usage
1. Sign in to Codex via OAuth using the Codex VSCode Extension or the Codex-CLI to generate auth.json.

2. Quick run:
```bash
codex-proxy

✓ Loaded configuration host=127.0.0.1 port=8111 auth_path=~/.codex/auth.json debug=disabled
INFO:     Started server process [85563]
INFO:     Waiting for application startup.
Loaded authentication data from ~/.codex/auth.json
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8111 (Press CTRL+C to quit)
INFO:     127.0.0.1:56709 - "POST /v1/chat/completions HTTP/1.1" 200 OK
```

3. Or with custom host/port:
```bash
codex-proxy --auth-path ~/.config/codex/auth.json --host 127.0.0.1 --port 8111
```

## Chatgpt provider plugin

1. Ensure the proxy is running at `http://127.0.0.1:8111/v1/` or set `CHATGPT_CODEX_PROXY_BASE_URL` (`plugin_codexproxy.ts`) to your endpoint.

2. The provider registers itself at startup; use the `/models` command to list them—the names are prefixed with `Chatgpt`.

The plugin attempts to load available Codex models from `/v1/models` (from proxy) and falls back to the bundled `gpt-5` and `gpt-5-codex` entries when none are returned.

## Uninstall

To remove the package from your environment:
```bash
rm -f ~/.config/opencode/plugin/plugin_codexproxy.ts
pip3 uninstall codex-proxy
```

## References

- [ChatMock](https://github.com/RayBytes/ChatMock) — Codex-compatible mock proxy you can run while configuring the local service.
- [just-every/code](https://github.com/just-every/code) — Reference implementation of Codex tooling that inspired this proxy integration.

