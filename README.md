# Codex Proxy Plugin

A lightweight OpenCode plugin that wires the ChatGPT provider to a Codex-compatible proxy.

## Usage

1. Install the package that contains this plugin in your opencode environment.

```
mv plugin_codexproxy.ts ~/.config/opencode/plugin/
```

2. Ensure the proxy is running at `http://127.0.0.1:8111/v1/` or set `CHATGPT_CODEX_PROXY_BASE_URL` to your endpoint.  
*You can use [ChatMock](https://github.com/RayBytes/ChatMock) as a drop-in Codex-compatible proxy endpoint.*

3. The provider will register itself during startup.

The plugin attempts to load available Codex models from `/v1/models` (from proxy) and falls back to the bundled `gpt-5` and `gpt-5-codex` entries when none are returned.
