/**
 * Vertex AI provider implementation with API key as query parameter.
 *
 * This provider is similar to google-generative-ai but sends the API key
 * as a query parameter (?key=API_KEY) instead of as a header (x-goog-api-key).
 *
 * Endpoint format: https://aiplatform.googleapis.com/v1/publishers/google/models/{model}:generateContent?key={apiKey}
 */

import type {
  AssistantMessage,
  Context,
  Model,
  SimpleStreamOptions,
  StreamFunction,
  StreamOptions,
  TextContent,
  ThinkingContent,
  ToolCall,
} from "@mariozechner/pi-ai";
import {
  calculateCost,
  createAssistantMessageEventStream,
  getEnvApiKey,
} from "@mariozechner/pi-ai";

interface VertexAIOptions extends StreamOptions {
  toolChoice?: "auto" | "none" | "any";
  thinking?: {
    enabled: boolean;
    budgetTokens?: number;
    level?: "minimal" | "low" | "medium" | "high";
  };
}

// Google Generative AI types (matching the API response structure)
interface GooglePart {
  text?: string;
  functionCall?: {
    name: string;
    args?: Record<string, unknown>;
    id?: string;
  };
  functionResponse?: {
    name: string;
    response: Record<string, unknown>;
  };
  inlineData?: {
    mimeType: string;
    data: string;
  };
}

interface GoogleContent {
  role: "user" | "model" | "function";
  parts: GooglePart[];
}

interface GoogleCandidate {
  content?: GoogleContent;
  finishReason?: string;
  index?: number;
  safetyRatings?: Array<{
    category: string;
    probability: string;
  }>;
}

interface GoogleUsageMetadata {
  promptTokenCount?: number;
  candidatesTokenCount?: number;
  totalTokenCount?: number;
}

interface GoogleResponse {
  candidates?: GoogleCandidate[];
  usageMetadata?: GoogleUsageMetadata;
  promptFeedback?: {
    blockReason?: string;
    safetyRatings?: Array<{
      category: string;
      probability: string;
    }>;
  };
}

interface GoogleRequest {
  contents: GoogleContent[];
  systemInstruction?: string | { parts: GooglePart[] };
  generationConfig?: {
    temperature?: number;
    maxOutputTokens?: number;
    topP?: number;
    topK?: number;
    candidateCount?: number;
    stopSequences?: string[];
  };
  safetySettings?: Array<{
    category: string;
    threshold: string;
  }>;
  tools?: Array<{
    functionDeclarations?: Array<{
      name: string;
      description: string;
      parameters?: Record<string, unknown>;
    }>;
  }>;
  toolConfig?: {
    functionCallingConfig?: {
      mode?: string;
      allowedFunctionNames?: string[];
    };
  };
}

let toolCallCounter = 0;

function convertMessagesToGoogleFormat(context: Context): GoogleContent[] {
  const contents: GoogleContent[] = [];

  for (const msg of context.messages) {
    if (msg.role === "user") {
      const parts: GooglePart[] = [];
      const content = msg.content;
      if (typeof content === "string") {
        parts.push({ text: content });
      } else {
        for (const block of content) {
          if (block.type === "text") {
            parts.push({ text: block.text });
          } else if (block.type === "image") {
            parts.push({
              inlineData: {
                mimeType: block.mimeType || "image/jpeg",
                data: block.data,
              },
            });
          }
        }
      }
      contents.push({ role: "user", parts });
    } else if (msg.role === "assistant") {
      const parts: GooglePart[] = [];
      for (const block of msg.content) {
        if (block.type === "text") {
          parts.push({ text: block.text });
        } else if (block.type === "toolCall") {
          parts.push({
            functionCall: {
              name: block.name,
              args: block.arguments,
              id: block.id,
            },
          });
        }
      }
      if (parts.length > 0) {
        contents.push({ role: "model", parts });
      }
    } else if (msg.role === "toolResult") {
      const parts: GooglePart[] = [];
      for (const block of msg.content) {
        if (block.type === "text") {
          parts.push({
            functionResponse: {
              name: msg.toolName,
              response: { result: block.text },
            },
          });
        }
      }
      if (parts.length > 0) {
        contents.push({ role: "function", parts });
      }
    }
  }

  return contents;
}

function convertTools(tools: Context["tools"]): GoogleRequest["tools"] {
  if (!tools || tools.length === 0) {
    return undefined;
  }

  const functionDeclarations = tools.map((tool) => ({
    name: tool.name,
    description: tool.description,
    parameters: tool.parameters,
  }));

  return [{ functionDeclarations }];
}

function mapStopReason(finishReason?: string): AssistantMessage["stopReason"] {
  if (!finishReason) return "stop";
  switch (finishReason) {
    case "STOP":
      return "stop";
    case "MAX_TOKENS":
      return "length";
    case "SAFETY":
    case "RECITATION":
      return "error";
    default:
      return "stop";
  }
}

export const streamVertexAI: StreamFunction<"vertex-ai", VertexAIOptions> = (
  model,
  context,
  options,
) => {
  const stream = createAssistantMessageEventStream();

  (async () => {
    const output: AssistantMessage = {
      role: "assistant",
      content: [],
      api: "vertex-ai",
      provider: model.provider,
      model: model.id,
      usage: {
        input: 0,
        output: 0,
        cacheRead: 0,
        cacheWrite: 0,
        totalTokens: 0,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
      },
      stopReason: "stop",
      timestamp: Date.now(),
    };

    try {
      const apiKey = options?.apiKey || getEnvApiKey(model.provider) || "";
      if (!apiKey) {
        throw new Error("Vertex AI requires an API key");
      }

      // Build the request body
      const requestBody: GoogleRequest = {
        contents: convertMessagesToGoogleFormat(context),
      };

      // Add system instruction if present
      if (context.systemPrompt) {
        requestBody.systemInstruction = context.systemPrompt;
      }

      // Add generation config
      const generationConfig: GoogleRequest["generationConfig"] = {};
      if (options?.temperature !== undefined) {
        generationConfig.temperature = options.temperature;
      }
      if (options?.maxTokens !== undefined) {
        generationConfig.maxOutputTokens = options.maxTokens;
      }
      if (Object.keys(generationConfig).length > 0) {
        requestBody.generationConfig = generationConfig;
      }

      // Add tools if present
      if (context.tools && context.tools.length > 0) {
        requestBody.tools = convertTools(context.tools);
        if (options?.toolChoice) {
          requestBody.toolConfig = {
            functionCallingConfig: {
              mode:
                options.toolChoice === "auto"
                  ? "AUTO"
                  : options.toolChoice === "any"
                    ? "ANY"
                    : "NONE",
            },
          };
        }
      }

      // Build the URL with API key as query parameter
      const baseUrl = model.baseUrl || "https://generativelanguage.googleapis.com/v1beta";
      const endpoint = `${baseUrl}/models/${model.id}:streamGenerateContent?key=${apiKey}&alt=sse`;

      // Call onPayload if provided
      options?.onPayload?.(requestBody);

      // Make the request
      const headers: Record<string, string> = {
        "Content-Type": "application/json",
        ...(model.headers || {}),
        ...(options?.headers || {}),
      };

      const response = await fetch(endpoint, {
        method: "POST",
        headers,
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Vertex AI request failed: ${response.status} ${errorText}`);
      }

      if (!response.body) {
        throw new Error("Response body is null");
      }

      stream.push({ type: "start", partial: output });

      // Parse the SSE stream
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let currentBlock: TextContent | ThinkingContent | null = null;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.trim() || line.startsWith(":")) continue;
          if (!line.startsWith("data: ")) continue;

          const data = line.slice(6);
          if (data === "[DONE]") continue;

          try {
            const chunk: GoogleResponse = JSON.parse(data);
            const candidate = chunk.candidates?.[0];

            if (candidate?.content?.parts) {
              for (const part of candidate.content.parts) {
                if (part.text !== undefined) {
                  // End previous block if it's thinking
                  if (currentBlock !== null) {
                    if (currentBlock.type === "thinking") {
                      stream.push({
                        type: "thinking_end",
                        contentIndex: output.content.length - 1,
                        content: (currentBlock as ThinkingContent).thinking,
                        partial: output,
                      });
                      currentBlock = null;
                    }
                  }
                  // Start new text block if needed
                  if (currentBlock === null) {
                    currentBlock = { type: "text", text: "" };
                    output.content.push(currentBlock);
                    stream.push({
                      type: "text_start",
                      contentIndex: output.content.length - 1,
                      partial: output,
                    });
                  }
                  // Append text to current block
                  if (currentBlock.type === "text") {
                    (currentBlock as TextContent).text += part.text;
                    stream.push({
                      type: "text_delta",
                      contentIndex: output.content.length - 1,
                      delta: part.text,
                      partial: output,
                    });
                  }
                }

                if (part.functionCall) {
                  if (currentBlock !== null) {
                    if (currentBlock.type === "text") {
                      stream.push({
                        type: "text_end",
                        contentIndex: output.content.length - 1,
                        content: (currentBlock as TextContent).text,
                        partial: output,
                      });
                    } else if (currentBlock.type === "thinking") {
                      stream.push({
                        type: "thinking_end",
                        contentIndex: output.content.length - 1,
                        content: (currentBlock as ThinkingContent).thinking,
                        partial: output,
                      });
                    }
                    currentBlock = null;
                  }

                  const toolCallId =
                    part.functionCall.id ||
                    `${part.functionCall.name}_${Date.now()}_${++toolCallCounter}`;
                  const toolCall: ToolCall = {
                    type: "toolCall",
                    id: toolCallId,
                    name: part.functionCall.name,
                    arguments: part.functionCall.args ?? {},
                  };
                  output.content.push(toolCall);
                  stream.push({
                    type: "toolcall_start",
                    contentIndex: output.content.length - 1,
                    partial: output,
                  });
                  stream.push({
                    type: "toolcall_delta",
                    contentIndex: output.content.length - 1,
                    delta: JSON.stringify(toolCall.arguments),
                    partial: output,
                  });
                  stream.push({
                    type: "toolcall_end",
                    contentIndex: output.content.length - 1,
                    toolCall,
                    partial: output,
                  });
                }
              }
            }

            if (candidate?.finishReason) {
              output.stopReason = mapStopReason(candidate.finishReason);
              if (output.content.some((b) => b.type === "toolCall")) {
                output.stopReason = "toolUse";
              }
            }

            if (chunk.usageMetadata) {
              output.usage.input = chunk.usageMetadata.promptTokenCount || 0;
              output.usage.output = chunk.usageMetadata.candidatesTokenCount || 0;
              output.usage.totalTokens = chunk.usageMetadata.totalTokenCount || 0;
              const cost = calculateCost(model, output.usage);
              output.usage.cost = cost;
            }
          } catch (err) {
            console.warn("Failed to parse SSE chunk:", err);
          }
        }
      }

      // End any remaining block
      if (currentBlock !== null) {
        if (currentBlock.type === "text") {
          stream.push({
            type: "text_end",
            contentIndex: output.content.length - 1,
            content: (currentBlock as TextContent).text,
            partial: output,
          });
        } else if (currentBlock.type === "thinking") {
          stream.push({
            type: "thinking_end",
            contentIndex: output.content.length - 1,
            content: (currentBlock as ThinkingContent).thinking,
            partial: output,
          });
        }
      }

      // Send done event only if stopReason is not error/aborted
      if (
        output.stopReason === "stop" ||
        output.stopReason === "length" ||
        output.stopReason === "toolUse"
      ) {
        stream.push({ type: "done", reason: output.stopReason, message: output });
      } else {
        stream.push({
          type: "error",
          reason: output.stopReason,
          error: { ...output },
        });
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      stream.push({
        type: "error",
        reason: "error",
        error: { ...output, errorMessage, stopReason: "error" },
      });
    } finally {
      stream.end(output);
    }
  })();

  return stream;
};

export const streamSimpleVertexAI: StreamFunction<"vertex-ai", SimpleStreamOptions> =
  streamVertexAI;
