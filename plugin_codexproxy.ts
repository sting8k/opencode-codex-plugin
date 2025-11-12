/**
 * Codex Proxy Plugin
 * Provides Codex proxy-backed ChatGPT provider for OpenCode.
 */

// @ts-ignore - OpenCode injects types at runtime
import type { Plugin, PluginInput } from "@opencode-ai/plugin";

// Minimal process typing for build environments without Node declarations
declare const process: {
  env: Record<string, string | undefined>;
};

// === Model utilities ===
type ModelInfo = { name: string };
type AvailableModels = Record<string, ModelInfo>;

const codexProxyModelCache = new Map<string, AvailableModels>();

async function fetchCodexProxyModels(baseURL: string): Promise<AvailableModels> {
  const normalized = normalizeCodexProxyBaseURL(baseURL);
  const cacheKey = normalized;

  if (codexProxyModelCache.has(cacheKey)) {
    return codexProxyModelCache.get(cacheKey)!;
  }

  const modelsEndpoint = `${normalized}models`;

  try {
    const response = await fetch(modelsEndpoint, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      throw new Error(`Failed to fetch Codex proxy models: ${response.status}`);
    }

    const payload = await response.json();
    const models: AvailableModels = {};

    if (Array.isArray(payload?.data)) {
      for (const model of payload.data) {
        if (model?.id) {
          const remoteName =
            typeof model?.name === "string" && model.name.trim().length
              ? model.name
              : deriveCodexProxyModelName(model.id);
          models[model.id] = { name: remoteName };
        }
      }
    }

    if (!Object.keys(models).length) {
      throw new Error("Codex proxy returned an empty model list");
    }

    codexProxyModelCache.set(cacheKey, models);
    console.warn("✅ Codex proxy models fetched successfully!");
    return models;
  }  catch (error) {
    let errMessage = 'unknown error';
    if (error instanceof Error) {
      errMessage = error.message;
    } else if (typeof error === 'string') {
      errMessage = error;
    } else if (error && typeof error === 'object' && 'code' in error) {
      errMessage = (error as any).code || 'connection error';
    }

    console.warn("[!] Failed to fetch models from Codex proxy at %s (reason: %s). Using fallback.", modelsEndpoint, errMessage);
    const fallback = buildCodexProxyModels();
    codexProxyModelCache.set(cacheKey, fallback);
    return fallback;
  }
}

async function getCodexProxyModels(baseURL: string): Promise<AvailableModels> {
  return fetchCodexProxyModels(baseURL);
}

// === Codex Proxy provider wiring ===
const CODEX_PROXY_PROVIDER_NAME = "codex-proxy";
const DEFAULT_CODEX_PROXY_BASE_URL = "http://127.0.0.1:8111/v1/";
const CODEX_PROXY_SCHEMA = "https://opencode.ai/config.json";
const CODEX_PROXY_NPM_PACKAGE = "@ai-sdk/openai-compatible";

function normalizeCodexProxyBaseURL(baseURL: string): string {
  return baseURL.endsWith("/") ? baseURL : `${baseURL}/`;
}

const CODEX_PROXY_BASE_MODELS: Record<string, string> = {
  "gpt-5": "ChatGPT GPT-5",
  "gpt-5-codex": "ChatGPT GPT-5 Codex",
  "gpt-5-codex-mini": "ChatGPT GPT-5 Codex Mini",
};

const CODEX_PROXY_VARIANT_SUFFIXES = new Set(["minimal", "low", "medium", "high"]);

function deriveCodexProxyModelName(modelId: string): string {
  if (CODEX_PROXY_BASE_MODELS[modelId]) {
    return CODEX_PROXY_BASE_MODELS[modelId];
  }

  const parts = modelId.split("-");
  if (parts.length >= 2) {
    const variant = parts[parts.length - 1];
    if (CODEX_PROXY_VARIANT_SUFFIXES.has(variant)) {
      const baseId = parts.slice(0, -1).join("-");
      const baseName = CODEX_PROXY_BASE_MODELS[baseId] || baseId;
      return `${baseName} (${variant})`;
    }
  }

  return modelId;
}

function envFlag(name: string, defaultValue: boolean): boolean {
  const raw = process.env[name];
  if (raw === undefined) return defaultValue;
  return ["1", "true", "yes", "on"].includes(raw.trim().toLowerCase());
}

function buildCodexProxyModels() {
  const models: Record<string, { name: string }> = {};

  for (const [modelId, label] of Object.entries(CODEX_PROXY_BASE_MODELS)) {
    models[modelId] = { name: label };
    for (const suffix of CODEX_PROXY_VARIANT_SUFFIXES) {
      const variantId = `${modelId}-${suffix}`;
      models[variantId] = { name: `${label} (${suffix})` };
    }
  }

  return models;
}

function buildCodexProxyProvider(baseURL: string, models: AvailableModels) {
  return {
    schema: CODEX_PROXY_SCHEMA,
    npm: CODEX_PROXY_NPM_PACKAGE,
    name: "Codex Proxy (Local)",
    options: {
      baseURL: normalizeCodexProxyBaseURL(baseURL),
      headers: {
        "Content-Type": "application/json",
      },
    },
    models,
  };
}

async function configureChatGptProviders(config: any) {
  if (!config.provider) {
    config.provider = {};
  }

  // Only configure Codex proxy for ChatGPT access
  const enableCodexProxy = envFlag("CHATGPT_ENABLE_CODEX_PROXY", true);
  if (enableCodexProxy) {
    if (!config.provider[CODEX_PROXY_PROVIDER_NAME]) {
      const baseURL = process.env.CHATGPT_CODEX_PROXY_BASE_URL || DEFAULT_CODEX_PROXY_BASE_URL;
      const codexProxyModels = await getCodexProxyModels(baseURL);
      config.provider[CODEX_PROXY_PROVIDER_NAME] = buildCodexProxyProvider(baseURL, codexProxyModels);
    }
  }
}

export const ChatGptPlugin: Plugin = async (_input: PluginInput) => {
  return {
    config: configureChatGptProviders,
  };
};

export default ChatGptPlugin;