import fs from "node:fs/promises";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { OpenClawConfig } from "../config/config.js";
import { withTempHome as withTempHomeBase } from "../../test/helpers/temp-home.js";

async function withTempHome<T>(fn: (home: string) => Promise<T>): Promise<T> {
  return withTempHomeBase(fn, { prefix: "openclaw-models-vertex-" });
}

describe("models-config vertex-ai provider", () => {
  let previousHome: string | undefined;

  beforeEach(() => {
    previousHome = process.env.HOME;
  });

  afterEach(() => {
    process.env.HOME = previousHome;
  });

  it("auto-injects vertex-ai provider when VERTEX_AI_API_KEY is set", async () => {
    await withTempHome(async () => {
      vi.resetModules();
      const prevKey = process.env.VERTEX_AI_API_KEY;
      process.env.VERTEX_AI_API_KEY = "test-vertex-ai-key";
      try {
        const { ensureOpenClawModelsJson } = await import("./models-config.js");
        const { resolveOpenClawAgentDir } = await import("./agent-paths.js");

        const cfg: OpenClawConfig = {};

        await ensureOpenClawModelsJson(cfg);

        const modelPath = path.join(resolveOpenClawAgentDir(), "models.json");
        const raw = await fs.readFile(modelPath, "utf8");
        const parsed = JSON.parse(raw) as {
          providers: Record<string, { apiKey?: string; models?: Array<{ id: string }> }>;
        };

        expect(parsed.providers["vertex-ai"]).toBeDefined();
        expect(parsed.providers["vertex-ai"]?.apiKey).toBe("VERTEX_AI_API_KEY");

        const ids = parsed.providers["vertex-ai"]?.models?.map((model) => model.id);
        expect(ids).toContain("gemini-3-flash-preview");
        expect(ids).toContain("gemini-3-pro-preview");
      } finally {
        if (prevKey === undefined) {
          delete process.env.VERTEX_AI_API_KEY;
        } else {
          process.env.VERTEX_AI_API_KEY = prevKey;
        }
      }
    });
  });

  it("fills missing vertex-ai provider.apiKey from env var when models exist", async () => {
    await withTempHome(async () => {
      vi.resetModules();
      const prevKey = process.env.VERTEX_AI_API_KEY;
      process.env.VERTEX_AI_API_KEY = "test-vertex-ai-key";
      try {
        const { ensureOpenClawModelsJson } = await import("./models-config.js");
        const { resolveOpenClawAgentDir } = await import("./agent-paths.js");

        const cfg: OpenClawConfig = {
          models: {
            providers: {
              "vertex-ai": {
                baseUrl:
                  "https://aiplatform.googleapis.com/v1/projects/test-project/locations/global/publishers/google/models",
                api: "vertex-ai",
                models: [
                  {
                    id: "gemini-3-flash-preview",
                    name: "Gemini 3 Flash Preview",
                    reasoning: false,
                    input: ["text", "image"],
                    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
                    contextWindow: 1048576,
                    maxTokens: 65536,
                  },
                ],
              },
            },
          },
        };

        await ensureOpenClawModelsJson(cfg);

        const modelPath = path.join(resolveOpenClawAgentDir(), "models.json");
        const raw = await fs.readFile(modelPath, "utf8");
        const parsed = JSON.parse(raw) as {
          providers: Record<string, { apiKey?: string; models?: Array<{ id: string }> }>;
        };

        expect(parsed.providers["vertex-ai"]?.apiKey).toBe("VERTEX_AI_API_KEY");
        const ids = parsed.providers["vertex-ai"]?.models?.map((model) => model.id);
        expect(ids).toContain("gemini-3-flash-preview");
      } finally {
        if (prevKey === undefined) {
          delete process.env.VERTEX_AI_API_KEY;
        } else {
          process.env.VERTEX_AI_API_KEY = prevKey;
        }
      }
    });
  });

  it("normalizes gemini-3 model IDs for vertex-ai provider", async () => {
    await withTempHome(async () => {
      vi.resetModules();
      const prevKey = process.env.VERTEX_AI_API_KEY;
      process.env.VERTEX_AI_API_KEY = "test-vertex-ai-key";
      try {
        const { ensureOpenClawModelsJson } = await import("./models-config.js");
        const { resolveOpenClawAgentDir } = await import("./agent-paths.js");

        // Config with non-preview model IDs
        const cfg: OpenClawConfig = {
          models: {
            providers: {
              "vertex-ai": {
                baseUrl:
                  "https://aiplatform.googleapis.com/v1/projects/test-project/locations/global/publishers/google/models",
                api: "vertex-ai",
                models: [
                  {
                    id: "gemini-3-flash", // Should be normalized to gemini-3-flash-preview
                    name: "Gemini 3 Flash",
                    reasoning: false,
                    input: ["text", "image"],
                    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
                    contextWindow: 1048576,
                    maxTokens: 65536,
                  },
                  {
                    id: "gemini-3-pro", // Should be normalized to gemini-3-pro-preview
                    name: "Gemini 3 Pro",
                    reasoning: true,
                    input: ["text", "image"],
                    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
                    contextWindow: 1048576,
                    maxTokens: 65536,
                  },
                ],
              },
            },
          },
        };

        await ensureOpenClawModelsJson(cfg);

        const modelPath = path.join(resolveOpenClawAgentDir(), "models.json");
        const raw = await fs.readFile(modelPath, "utf8");
        const parsed = JSON.parse(raw) as {
          providers: Record<string, { models?: Array<{ id: string }> }>;
        };

        const ids = parsed.providers["vertex-ai"]?.models?.map((model) => model.id);
        expect(ids).toContain("gemini-3-flash-preview");
        expect(ids).toContain("gemini-3-pro-preview");
        expect(ids).not.toContain("gemini-3-flash");
        expect(ids).not.toContain("gemini-3-pro");
      } finally {
        if (prevKey === undefined) {
          delete process.env.VERTEX_AI_API_KEY;
        } else {
          process.env.VERTEX_AI_API_KEY = prevKey;
        }
      }
    });
  });
});
