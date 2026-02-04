import type { Context, Model } from "@mariozechner/pi-ai";
import { afterEach, describe, expect, it, vi } from "vitest";
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

  it("formats tool calls without id field in functionCall but injects __openclaw_tool_call_id in args", async () => {
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

    // Verify functionCall does not contain id field at top level but has __openclaw_tool_call_id in args
    expect(capturedBody).toBeDefined();
    expect((capturedBody as any).contents).toEqual([
      { role: "user", parts: [{ text: "What's the weather in NYC?" }] },
      {
        role: "model",
        parts: [
          {
            functionCall: {
              name: "get_weather",
              args: { location: "NYC", __openclaw_tool_call_id: "call_123" },
              // id should NOT be present at top level
            },
          },
        ],
      },
    ]);
    // Explicitly verify id is not in the functionCall at top level
    expect((capturedBody as any).contents[1].parts[0].functionCall).not.toHaveProperty("id");
    // But verify __openclaw_tool_call_id is in args
    expect((capturedBody as any).contents[1].parts[0].functionCall.args).toHaveProperty(
      "__openclaw_tool_call_id",
      "call_123",
    );
  });

  it("formats tool results with role 'user' (not 'model' or 'function')", async () => {
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

    // Verify tool result uses role "user" not "model" or "function"
    expect(capturedBody).toBeDefined();
    const contents = (capturedBody as any).contents;
    expect(contents).toHaveLength(3);
    expect(contents[2]).toEqual({
      role: "user", // Should be "user" not "model" or "function"
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

describe("vertex-ai debug logging", () => {
  const originalLog = console.log;

  afterEach(() => {
    // Restore original console.log
    console.log = originalLog;
  });

  it("always logs debug information", async () => {
    const model = makeVertexAIModel("gemini-3-flash-preview");
    const context: Context = {
      messages: [{ role: "user", content: "test message" }],
    };

    const logCalls: unknown[] = [];
    console.log = vi.fn((...args: unknown[]) => {
      logCalls.push(args);
    });

    global.fetch = vi.fn(async () => {
      return new Response(null, { status: 400 });
    }) as typeof fetch;

    try {
      const stream = streamVertexAI(model, context, { apiKey: "test-key" });
      for await (const _event of stream) {
        // Consume stream
      }
    } catch {
      // Expected to fail
    }

    // Should have log calls since logging is always enabled
    expect(logCalls.length).toBeGreaterThanOrEqual(1);
  });

  it("logs concise request summary instead of verbose JSON dumps", async () => {
    const model = makeVertexAIModel("gemini-3-flash-preview");
    const context: Context = {
      messages: [
        { role: "user", content: "What is the weather?" },
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

    const logCalls: unknown[] = [];
    console.log = vi.fn((...args: unknown[]) => {
      logCalls.push(args);
    });

    global.fetch = vi.fn(async () => {
      return new Response(null, { status: 400 });
    }) as typeof fetch;

    try {
      const stream = streamVertexAI(model, context, { apiKey: "test-key" });
      for await (const _event of stream) {
        // Consume stream
      }
    } catch {
      // Expected to fail
    }

    // Should have log calls
    expect(logCalls.length).toBeGreaterThanOrEqual(1);

    // Check concise request summary log
    const requestSummaryLog = logCalls.find(
      (log) =>
        Array.isArray(log) && typeof log[0] === "string" && log[0].includes("[vertex-ai] Request:"),
    );
    expect(requestSummaryLog).toBeDefined();
    expect(requestSummaryLog).toEqual(["[vertex-ai] Request: 3 messages, 1 tools"]);

    // Verify verbose logs are NOT present
    const verboseMessageLog = logCalls.find(
      (log) => Array.isArray(log) && log[0] === "[vertex-ai] context.messages summary:",
    );
    expect(verboseMessageLog).toBeUndefined();

    const verboseBodyLog = logCalls.find(
      (log) => Array.isArray(log) && log[0] === "[vertex-ai] redacted requestBody:",
    );
    expect(verboseBodyLog).toBeUndefined();
  });

  it("logs response status for successful and failed requests", async () => {
    const model = makeVertexAIModel("gemini-3-flash-preview");
    const context: Context = {
      messages: [{ role: "user", content: "test" }],
    };

    const logCalls: unknown[] = [];
    const errorCalls: unknown[] = [];
    console.log = vi.fn((...args: unknown[]) => {
      logCalls.push(args);
    });
    console.error = vi.fn((...args: unknown[]) => {
      errorCalls.push(args);
    });

    global.fetch = vi.fn(async () => {
      return new Response("Error message", { status: 400 });
    }) as typeof fetch;

    try {
      const stream = streamVertexAI(model, context, { apiKey: "test-key" });
      for await (const _event of stream) {
        // Consume stream
      }
    } catch {
      // Expected to fail
    }

    // Check response status log
    const responseStatusLog = logCalls.find(
      (log) =>
        Array.isArray(log) &&
        typeof log[0] === "string" &&
        log[0].includes("[vertex-ai] Response status:"),
    );
    expect(responseStatusLog).toBeDefined();
    expect(responseStatusLog).toEqual(["[vertex-ai] Response status: 400"]);

    // Check error log for failed request
    const errorLog = errorCalls.find(
      (log) =>
        Array.isArray(log) &&
        typeof log[0] === "string" &&
        log[0].includes("[vertex-ai] API error:"),
    );
    expect(errorLog).toBeDefined();
    expect(errorLog).toEqual(["[vertex-ai] API error: 400 Error message"]);
  });

  it("logs SSE stream start indicator", async () => {
    const model = makeVertexAIModel("gemini-3-flash-preview");
    const context: Context = {
      messages: [{ role: "user", content: "test" }],
    };

    const logCalls: unknown[] = [];
    console.log = vi.fn((...args: unknown[]) => {
      logCalls.push(args);
    });

    // Mock successful response
    const mockSSE = `data: {"candidates":[{"content":{"parts":[{"text":"Hello"}]}}]}\n\ndata: [DONE]\n\n`;
    global.fetch = vi.fn(async () => {
      const stream = new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode(mockSSE));
          controller.close();
        },
      });
      return new Response(stream, { status: 200 });
    }) as typeof fetch;

    const stream = streamVertexAI(model, context, { apiKey: "test-key" });
    for await (const _event of stream) {
      // Consume stream
    }

    // Check SSE stream start log
    const sseStartLog = logCalls.find(
      (log) => Array.isArray(log) && log[0] === "[vertex-ai] Starting SSE stream parsing...",
    );
    expect(sseStartLog).toBeDefined();
  });
});

describe("vertex-ai SSE response parsing with tool call id", () => {
  it("prefers echoed __openclaw_tool_call_id from args over server id", async () => {
    const model = makeVertexAIModel("gemini-3-flash-preview");
    const context: Context = {
      messages: [{ role: "user", content: "What's the weather?" }],
    };

    // Mock SSE response with echoed __openclaw_tool_call_id in args
    const mockSSE = `data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"get_weather","args":{"location":"NYC","__openclaw_tool_call_id":"call_123"},"id":"server_generated_id"}}]}}]}\n\ndata: [DONE]\n\n`;

    global.fetch = vi.fn(async () => {
      const stream = new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode(mockSSE));
          controller.close();
        },
      });
      return new Response(stream, { status: 200 });
    }) as typeof fetch;

    const stream = streamVertexAI(model, context, { apiKey: "test-key" });
    const events = [];
    for await (const event of stream) {
      events.push(event);
    }

    // Find the toolcall_end event
    const toolcallEnd = events.find((e) => e.type === "toolcall_end");
    expect(toolcallEnd).toBeDefined();
    expect(toolcallEnd).toHaveProperty("toolCall");

    // Verify the id is the echoed one, not the server one
    expect((toolcallEnd as any).toolCall.id).toBe("call_123");
    expect((toolcallEnd as any).toolCall.name).toBe("get_weather");

    // Verify the __openclaw_tool_call_id marker is removed from arguments
    expect((toolcallEnd as any).toolCall.arguments).not.toHaveProperty("__openclaw_tool_call_id");
    expect((toolcallEnd as any).toolCall.arguments).toEqual({ location: "NYC" });
  });

  it("falls back to server id when __openclaw_tool_call_id is not present", async () => {
    const model = makeVertexAIModel("gemini-3-flash-preview");
    const context: Context = {
      messages: [{ role: "user", content: "What's the weather?" }],
    };

    // Mock SSE response without __openclaw_tool_call_id
    const mockSSE = `data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"get_weather","args":{"location":"NYC"},"id":"server_generated_id"}}]}}]}\n\ndata: [DONE]\n\n`;

    global.fetch = vi.fn(async () => {
      const stream = new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode(mockSSE));
          controller.close();
        },
      });
      return new Response(stream, { status: 200 });
    }) as typeof fetch;

    const stream = streamVertexAI(model, context, { apiKey: "test-key" });
    const events = [];
    for await (const event of stream) {
      events.push(event);
    }

    // Find the toolcall_end event
    const toolcallEnd = events.find((e) => e.type === "toolcall_end");
    expect(toolcallEnd).toBeDefined();

    // Verify the id is the server one
    expect((toolcallEnd as any).toolCall.id).toBe("server_generated_id");
    expect((toolcallEnd as any).toolCall.name).toBe("get_weather");
    expect((toolcallEnd as any).toolCall.arguments).toEqual({ location: "NYC" });
  });

  it("synthesizes id when neither echoed nor server id is present", async () => {
    const model = makeVertexAIModel("gemini-3-flash-preview");
    const context: Context = {
      messages: [{ role: "user", content: "What's the weather?" }],
    };

    // Mock SSE response without any id
    const mockSSE = `data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"get_weather","args":{"location":"NYC"}}}]}}]}\n\ndata: [DONE]\n\n`;

    global.fetch = vi.fn(async () => {
      const stream = new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode(mockSSE));
          controller.close();
        },
      });
      return new Response(stream, { status: 200 });
    }) as typeof fetch;

    const stream = streamVertexAI(model, context, { apiKey: "test-key" });
    const events = [];
    for await (const event of stream) {
      events.push(event);
    }

    // Find the toolcall_end event
    const toolcallEnd = events.find((e) => e.type === "toolcall_end");
    expect(toolcallEnd).toBeDefined();

    // Verify a synthesized id is present (should start with get_weather_)
    expect((toolcallEnd as any).toolCall.id).toMatch(/^get_weather_\d+_\d+$/);
    expect((toolcallEnd as any).toolCall.name).toBe("get_weather");
    expect((toolcallEnd as any).toolCall.arguments).toEqual({ location: "NYC" });
  });
});

describe("vertex-ai thoughtSignature handling", () => {
  it("stores thoughtSignature when present in SSE response", async () => {
    const model = makeVertexAIModel("gemini-3-flash-preview");
    const context: Context = {
      messages: [{ role: "user", content: "What's the weather?" }],
    };

    // Mock SSE response with thoughtSignature
    const mockSSE = `data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"get_weather","args":{"location":"NYC"}},"thoughtSignature":"test_signature_123"}]}}]}\n\ndata: [DONE]\n\n`;

    global.fetch = vi.fn(async () => {
      const stream = new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode(mockSSE));
          controller.close();
        },
      });
      return new Response(stream, { status: 200 });
    }) as typeof fetch;

    const stream = streamVertexAI(model, context, { apiKey: "test-key" });
    const events = [];
    for await (const event of stream) {
      events.push(event);
    }

    // Find the toolcall_end event
    const toolcallEnd = events.find((e) => e.type === "toolcall_end");
    expect(toolcallEnd).toBeDefined();
    expect(toolcallEnd).toHaveProperty("toolCall");

    // Verify thoughtSignature is stored
    expect((toolcallEnd as any).toolCall.thoughtSignature).toBe("test_signature_123");
  });

  it("includes thoughtSignature when sending functionCall in request", async () => {
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
              // Add thoughtSignature using type assertion
            } as any & { thoughtSignature: string },
          ],
        },
      ],
    };

    // Add thoughtSignature to the toolCall
    (context.messages[1].content[0] as any).thoughtSignature = "test_signature_123";

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

    // Verify thoughtSignature is included in the request
    expect(capturedBody).toBeDefined();
    const contents = (capturedBody as any).contents;
    expect(contents).toHaveLength(2);
    expect(contents[1].parts[0]).toHaveProperty("thoughtSignature", "test_signature_123");
  });

  it("handles missing thoughtSignature gracefully", async () => {
    const model = makeVertexAIModel("gemini-3-flash-preview");
    const context: Context = {
      messages: [{ role: "user", content: "What's the weather?" }],
    };

    // Mock SSE response without thoughtSignature
    const mockSSE = `data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"get_weather","args":{"location":"NYC"}}}]}}]}\n\ndata: [DONE]\n\n`;

    global.fetch = vi.fn(async () => {
      const stream = new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode(mockSSE));
          controller.close();
        },
      });
      return new Response(stream, { status: 200 });
    }) as typeof fetch;

    const stream = streamVertexAI(model, context, { apiKey: "test-key" });
    const events = [];
    for await (const event of stream) {
      events.push(event);
    }

    // Find the toolcall_end event
    const toolcallEnd = events.find((e) => e.type === "toolcall_end");
    expect(toolcallEnd).toBeDefined();
    expect(toolcallEnd).toHaveProperty("toolCall");

    // Verify thoughtSignature is undefined when not present
    expect((toolcallEnd as any).toolCall.thoughtSignature).toBeUndefined();
  });
});
