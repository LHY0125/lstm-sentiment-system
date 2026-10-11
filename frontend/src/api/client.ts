/** 唯一进行网络请求的模块。组件一律通过这里访问后端。 */

import type { BatchItem, ExplainResponse, ModelInfo, PredictResponse } from "../types";

/** 相对路径，由 Vite 代理转发到 Flask（见 vite.config.ts）。 */
const BASE = "/api";

/** 后端错误响应统一为 { error: string }。 */
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });

  if (!res.ok) {
    let message = `请求失败（HTTP ${res.status}）`;
    try {
      const body = (await res.json()) as { error?: string };
      if (body.error) message = body.error;
    } catch {
      // 响应不是 JSON，沿用默认提示
    }
    throw new Error(message);
  }
  return (await res.json()) as T;
}

export function fetchHealth(): Promise<{ status: string }> {
  return request("/health");
}

export function fetchModels(): Promise<ModelInfo[]> {
  return request("/models");
}

export function fetchPredict(
  text: string,
  modelId: string,
  maxLen?: number,
): Promise<PredictResponse> {
  return request("/predict", {
    method: "POST",
    body: JSON.stringify({ text, model_id: modelId, max_len: maxLen }),
  });
}

export function fetchExplain(
  text: string,
  modelId: string,
  maxLen?: number,
): Promise<ExplainResponse> {
  return request("/explain", {
    method: "POST",
    body: JSON.stringify({ text, model_id: modelId, max_len: maxLen }),
  });
}

export function fetchBatch(
  texts: string[],
  modelId: string,
  maxLen?: number,
): Promise<BatchItem[]> {
  return request("/batch", {
    method: "POST",
    body: JSON.stringify({ texts, model_id: modelId, max_len: maxLen }),
  });
}
