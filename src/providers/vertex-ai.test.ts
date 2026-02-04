import type { Context, Model } from "@mariozechner/pi-ai";
import { describe, expect, it, vi } from "vitest";
import { streamVertexAI } from "./vertex-ai.js";

const makeVertexAIModel = (id: string, baseUrl?: string): Model<"vertex-ai"> =>
  ({
    id,
    name: id,
    api: "vertex-ai",
    provider: "vertex-ai",
    baseUrl,
    reasoning: false,
    input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 1048576,
    maxTokens: 65536,
  }) as Model<"vertex-ai">;

describe("vertex-ai URL construction", () => {
  it("constructs URL correctly with default baseUrl (no duplicate /models/)", async () => {
    const model = makeVertexAIModel("gemini-3-flash-preview");
    const context: Context = {
      messages: [{ role: "user", content: "test" }],
    };

    let capturedEndpoint = "";

    // Mock fetch to capture the endpoint
    global.fetch = vi.fn(async (url) => {
      capturedEndpoint = url.toString();
      // Return a mock response that will cause the stream to error out quickly
      return new Response(null, { status: 400 });
    }) as typeof fetch;

    try {
      const stream = streamVertexAI(model, context, { apiKey: "test-key" });
      // Consume the stream to trigger the fetch
      for await (const _event of stream) {
        // Just consume events until error
      }
    } catch {
      // Expected to fail - we just want to capture the URL
    }

    // Verify the URL doesn't have duplicate /models/
    expect(capturedEndpoint).toContain(
      "https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-3-flash-preview:streamGenerateContent",
    );
    expect(capturedEndpoint).not.toContain("/models/models/");
    expect(capturedEndpoint).toContain("?key=test-key");
  });

  it("constructs URL correctly with custom baseUrl (no duplicate /models/)", async () => {
    const model = makeVertexAIModel(
      "gemini-3-flash-preview",
      "https://aiplatform.googleapis.com/v1/publishers/google/models",
    );
    const context: Context = {
      messages: [{ role: "user", content: "test" }],
    };

    let capturedEndpoint = "";

    // Mock fetch to capture the endpoint
    global.fetch = vi.fn(async (url) => {
      capturedEndpoint = url.toString();
      return new Response(null, { status: 400 });
    }) as typeof fetch;

    try {
      const stream = streamVertexAI(model, context, { apiKey: "test-key" });
      for await (const _event of stream) {
        // Just consume events until error
      }
    } catch {
      // Expected to fail
    }

    // Verify the URL doesn't have duplicate /models/
    expect(capturedEndpoint).toContain(
      "https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-3-flash-preview:streamGenerateContent",
    );
    expect(capturedEndpoint).not.toContain("/models/models/");
    expect(capturedEndpoint).toContain("?key=test-key");
  });

  it("constructs URL correctly with custom baseUrl with different path", async () => {
    const model = makeVertexAIModel(
      "gemini-3-flash-preview",
      "https://aiplatform.googleapis.com/v1/projects/test-project/locations/global/publishers/google/models",
    );
    const context: Context = {
      messages: [{ role: "user", content: "test" }],
    };

    let capturedEndpoint = "";

    // Mock fetch to capture the endpoint
    global.fetch = vi.fn(async (url) => {
      capturedEndpoint = url.toString();
      return new Response(null, { status: 400 });
    }) as typeof fetch;

    try {
      const stream = streamVertexAI(model, context, { apiKey: "test-key" });
      for await (const _event of stream) {
        // Just consume events until error
      }
    } catch {
      // Expected to fail
    }

    // Verify the URL doesn't have duplicate /models/
    expect(capturedEndpoint).toContain(
      "https://aiplatform.googleapis.com/v1/projects/test-project/locations/global/publishers/google/models/gemini-3-flash-preview:streamGenerateContent",
    );
    expect(capturedEndpoint).not.toContain("/models/models/");
    expect(capturedEndpoint).toContain("?key=test-key");
  });
});

describe("vertex-ai request body format", () => {
  it("formats systemInstruction correctly as object with parts array", async () => {
    const model = makeVertexAIModel("gemini-3-flash-preview");
    const context: Context = {
      messages: [{ role: "user", content: "test" }],
      systemPrompt: "You are a helpful assistant",
    };

    let capturedBody: unknown = null;

    // Mock fetch to capture the request body
    global.fetch = vi.fn(async (_url, options) => {
      if (options?.body) {
        capturedBody = JSON.parse(options.body.toString());
      }
      return new Response(null, { status: 400 });
    }) as typeof fetch;

    try {
      const stream = streamVertexAI(model, context, { apiKey: "test-key" });
      for await (const _event of stream) {
        // Just consume events until error
      }
    } catch {
      // Expected to fail
    }

    // Verify systemInstruction format
    expect(capturedBody).toBeDefined();
    expect(capturedBody).toHaveProperty("systemInstruction");
    expect((capturedBody as any).systemInstruction).toEqual({
      parts: [{ text: "You are a helpful assistant" }],
    });
  });

  it("formats contents correctly with role and parts", async () => {
    const model = makeVertexAIModel("gemini-3-flash-preview");
    const context: Context = {
      messages: [
        { role: "user", content: "hello" },
        { role: "assistant", content: [{ type: "text", text: "hi there" }] },
        { role: "user", content: "how are you?" },
      ],
    };

    let capturedBody: unknown = null;

    global.fetch = vi.fn(async (_url, options) => {
      if (options?.body) {
        capturedBody = JSON.parse(options.body.toString());
      }
      return new Response(null, { status: 400 });
    }) as typeof fetch;

    try {
      const stream = streamVertexAI(model, context, { apiKey: "test-key" });
      for await (const _event of stream) {
        // Just consume events until error
      }
    } catch {
      // Expected to fail
    }

    // Verify contents format
    expect(capturedBody).toBeDefined();
    expect((capturedBody as any).contents).toEqual([
      { role: "user", parts: [{ text: "hello" }] },
      { role: "model", parts: [{ text: "hi there" }] },
      { role: "user", parts: [{ text: "how are you?" }] },
    ]);
  });

  it("formats tools correctly with functionDeclarations", async () => {
    const model = makeVertexAIModel("gemini-3-flash-preview");
    const context: Context = {
      messages: [{ role: "user", content: "test" }],
      tools: [
        {
          name: "get_weather",
          description: "Get weather information",
          parameters: {
            type: "object",
            properties: {
              location: { type: "string" },
            },
            required: ["location"],
          },
        },
      ],
    };

    let capturedBody: unknown = null;

    global.fetch = vi.fn(async (_url, options) => {
      if (options?.body) {
        capturedBody = JSON.parse(options.body.toString());
      }
      return new Response(null, { status: 400 });
    }) as typeof fetch;

    try {
      const stream = streamVertexAI(model, context, { apiKey: "test-key" });
      for await (const _event of stream) {
        // Just consume events until error
      }
    } catch {
      // Expected to fail
    }

    // Verify tools format
    expect(capturedBody).toBeDefined();
    expect((capturedBody as any).tools).toEqual([
      {
        functionDeclarations: [
          {
            name: "get_weather",
            description: "Get weather information",
            parameters: {
              type: "object",
              properties: {
                location: { type: "string" },
              },
              required: ["location"],
            },
          },
        ],
      },
    ]);
  });

  it("formats tool calls without id field in functionCall", async () => {
    const model = makeVertexAIModel("gemini-3-flash-preview");
    const context: Context = {
      messages: [
        { role: "user", content: "What's the weather in NYC?" },
        {
          role: "assistant",
          content: [
            {
              type: "toolCall",
              id: "call_123",
              name: "get_weather",
              arguments: { location: "NYC" },
            },
          ],
        },
      ],
    };

    let capturedBody: unknown = null;

    global.fetch = vi.fn(async (_url, options) => {
      if (options?.body) {
        capturedBody = JSON.parse(options.body.toString());
      }
      return new Response(null, { status: 400 });
    }) as typeof fetch;

    try {
      const stream = streamVertexAI(model, context, { apiKey: "test-key" });
      for await (const _event of stream) {
        // Just consume events until error
      }
    } catch {
      // Expected to fail
    }

    // Verify functionCall does not contain id field
    expect(capturedBody).toBeDefined();
    expect((capturedBody as any).contents).toEqual([
      { role: "user", parts: [{ text: "What's the weather in NYC?" }] },
      {
        role: "model",
        parts: [
          {
            functionCall: {
              name: "get_weather",
              args: { location: "NYC" },
              // id should NOT be present
            },
          },
        ],
      },
    ]);
    // Explicitly verify id is not in the functionCall
    expect((capturedBody as any).contents[1].parts[0].functionCall).not.toHaveProperty("id");
  });

  it("formats tool results with role 'user' (not 'function')", async () => {
    const model = makeVertexAIModel("gemini-3-flash-preview");
    const context: Context = {
      messages: [
        { role: "user", content: "What's the weather in NYC?" },
        {
          role: "assistant",
          content: [
            {
              type: "toolCall",
              id: "call_123",
              name: "get_weather",
              arguments: { location: "NYC" },
            },
          ],
        },
        {
          role: "toolResult",
          toolCallId: "call_123",
          toolName: "get_weather",
          content: [{ type: "text", text: "Sunny, 75°F" }],
        },
      ],
    };

    let capturedBody: unknown = null;

    global.fetch = vi.fn(async (_url, options) => {
      if (options?.body) {
        capturedBody = JSON.parse(options.body.toString());
      }
      return new Response(null, { status: 400 });
    }) as typeof fetch;

    try {
      const stream = streamVertexAI(model, context, { apiKey: "test-key" });
      for await (const _event of stream) {
        // Just consume events until error
      }
    } catch {
      // Expected to fail
    }

    // Verify tool result uses role "user" not "function"
    expect(capturedBody).toBeDefined();
    const contents = (capturedBody as any).contents;
    expect(contents).toHaveLength(3);
    expect(contents[2]).toEqual({
      role: "user", // Should be "user" not "function"
      parts: [
        {
          functionResponse: {
            name: "get_weather",
            response: { result: "Sunny, 75°F" },
          },
        },
      ],
    });
  });
});
