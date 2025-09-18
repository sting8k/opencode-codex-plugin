## Codex Proxy & Plugin for OpenCode

This project connects [OpenCode](https://github.com/sst/opencode) to a Codex-compatible API.

It consists of two main parts:
1.  **Proxy Server (`codex-proxy`)**: A Python-based proxy server that translates OpenAI-compatible requests into the format required by the Codex backend.
2.  **OpenCode Plugin (`plugin_codexproxy.ts`)**: A plugin for OpenCode that directs the ChatGPT provider in OpenCode to use this local proxy.

---

## Installation Guide

### 1. Quick install codex-proxy as a cli tool
### Step 1: Get Codex Credentials

The proxy needs an `auth.json` file to authenticate with Codex services.

1.  Sign in to Codex using either the Codex VS Code Extension or the Codex CLI.
2.  This action will generate an `auth.json` file. By default, the proxy looks for this file at `~/.codex/auth.json`.
3.  If your file is in a different location, you can specify the path when running the proxy (see Usage section).

### Step 2: Run the Proxy Server

```bash
uvx --from git+https://github.com/sting8k/opencode-codex-plugin.git codex-proxy
```

### Step 3: Install the OpenCode Plugin

Download the plugin file to your OpenCode configuration directory:

* Linux and Mac OS
```bash
mkdir -p "$HOME/.config/opencode/plugin" && curl -fsSL "https://raw.githubusercontent.com/sting8k/opencode-codex-plugin/master/plugin_codexproxy.ts" -o "$HOME/.config/opencode/plugin/plugin_codexproxy.ts"
```

* Windows
```powershell
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.config\opencode\plugin" > $null; Invoke-WebRequest 'https://raw.githubusercontent.com/sting8k/opencode-codex-plugin/master/plugin_codexproxy.ts' -OutFile "$env:USERPROFILE\.config\opencode\plugin\plugin_codexproxy.ts"
```

---

## How to Use

### 1. Run the Proxy

Start the proxy server from your terminal (Same as step 2 above):
```bash
uvx --from git+https://github.com/sting8k/opencode-codex-plugin.git codex-proxy
```

You will see output similar to this, indicating the server is running:
```
✓ Loaded configuration host=127.0.0.1 port=8111 auth_path=~/.codex/auth.json debug=disabled
INFO:     Started server process [85563]
INFO:     Waiting for application startup.
Loaded authentication data from ~/.codex/auth.json
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8111 (Press CTRL+C to quit)
```

To run on a different host or port, or with a custom `auth.json` path:
```bash
codex-proxy --host 127.0.0.1 --port 8111 --auth-path /path/to/your/auth.json
```

### 2. Use in OpenCode

With codex-proxy running, the OpenCode plugin will automatically register the available models. You can list them with the `/models` command in OpenCode. The models provided by this proxy will be prefixed with `Chatgpt`.

### 3. Use as subagent

To config as subagent in `opencode.json`, use `codex-proxy` as provider:
```json
"agent": {
    "test-agent": {
      "mode": "subagent",
      "model": "codex-proxy/gpt-5-codex-low",
      "prompt": "You will act as Joe Billy. A funny joking guy."
    }
}
```

---

## Uninstall

To remove completely, delete the plugin file.

```bash
rm -f ~/.config/opencode/plugin/plugin_codexproxy.ts
```

## References

- [ChatMock](https://github.com/RayBytes/ChatMock) — A mock Codex-compatible proxy you can run while configuring the local service.
- [just-every/code](https://github.com/just-every/code) — A reference implementation of Codex tooling that inspired this proxy integration.