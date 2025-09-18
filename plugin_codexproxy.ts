/**
 * Minimal Codex Proxy plugin for OpenCode.
 */

import type { Plugin, PluginInput } from "@opencode-ai/plugin";

// Minimal process typing for build environments without Node declarations
declare const process: {
  env: Record<string, string | undefined>;
};

const CODEX_PROXY_PROVIDER_NAME = "codex-proxy";
const DEFAULT_CODEX_PROXY_BASE_URL = "http://127.0.0.1:8111/v1/";
const CODEX_PROXY_SCHEMA = "https://opencode.ai/config.json";
const CODEX_PROXY_NPM_PACKAGE = "@ai-sdk/openai-compatible";

const CODEX_PROXY_MODELS: Record<string, { name: string }> = {
  "gpt-5": { name: "ChatGPT GPT-5" },
  "gpt-5-codex": { name: "ChatGPT GPT-5 Codex" },
};

function normalizeBaseURL(baseURL: string): string {
  return baseURL.endsWith("/") ? baseURL : `${baseURL}/`;
}

function buildCodexProxyProvider(baseURL: string) {
  return {
    schema: CODEX_PROXY_SCHEMA,
    npm: CODEX_PROXY_NPM_PACKAGE,
    name: "Codex Proxy",
    options: {
      baseURL: normalizeBaseURL(baseURL),
      headers: {
        "Content-Type": "application/json",
      },
    },
    models: CODEX_PROXY_MODELS,
  };
}

async function configureChatGptProviders(config: any) {
  if (!config.provider) {
    config.provider = {};
  }

  const baseURL = process.env.CHATGPT_CODEX_PROXY_BASE_URL || DEFAULT_CODEX_PROXY_BASE_URL;
  config.provider[CODEX_PROXY_PROVIDER_NAME] = buildCodexProxyProvider(baseURL);
  console.log("✅ ChatGPT Codex proxy provider ready");
}

export const CodexProxyPlugin: Plugin = async (_input: PluginInput) => {
  return {
    config: configureChatGptProviders,
  };
};

export default CodexProxyPlugin;
