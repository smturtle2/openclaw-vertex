/**
 * Vertex AI provider implementation with API key as query parameter.
 *
 * This provider uses the Vertex AI public endpoint and sends the API key
 * as a query parameter (?key=API_KEY) instead of as a header (x-goog-api-key).
 *
 * Vertex AI API differences from standard Google Generative AI:
 * - Endpoint format: https://aiplatform.googleapis.com/v1/publishers/google/models/{model}:streamGenerateContent?key={apiKey}&alt=sse
 * - Does not accept `id` field in functionCall requests (though it may return ids in responses)
 * - Only accepts `role: "user"` or `role: "model"` in contents (no `role: "function"`)
 * - Tool results (functionResponse) must use `role: "user"` to work correctly with continuous tool execution
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
    // Note: id is not sent in requests but may be present in responses
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
  thoughtSignature?: string;
}

interface GoogleContent {
  // Vertex AI only accepts "user" and "model" roles in contents
  role: "user" | "model";
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
  systemInstruction?: { parts: GooglePart[] };
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
          // Vertex AI does not accept id field in functionCall at the top level,
          // so we inject the internal toolCall id into the args with a marker key
          const argsWithMarker = { ...block.arguments };
          if (block.id) {
            argsWithMarker.__openclaw_tool_call_id = block.id;
            // Debug logging: injecting internal id
            console.log(
              `[vertex-ai] outgoing toolCall — injecting id: ${block.id} into functionCall.args for ${block.name}`,
            );
          }

          const part: GooglePart = {
            functionCall: {
              name: block.name,
              args: argsWithMarker,
            },
          };

          // Include thoughtSignature if present
          if ((block as any).thoughtSignature) {
            part.thoughtSignature = (block as any).thoughtSignature;
          }

          parts.push(part);
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
      // Use "user" role for tool results
      if (parts.length > 0) {
        contents.push({
          role: "user",
          parts,
        });
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

      // Debug logging: streamVertexAI entry point
      console.log(`[vertex-ai] streamVertexAI entered — model: ${model.id}`);

      // Build the request body
      const requestBody: GoogleRequest = {
        contents: convertMessagesToGoogleFormat(context),
      };

      // Add system instruction if present
      if (context.systemPrompt) {
        requestBody.systemInstruction = {
          parts: [{ text: context.systemPrompt }],
        };
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
      const baseUrl =
        model.baseUrl || "https://aiplatform.googleapis.com/v1/publishers/google/models";
      const endpoint = `${baseUrl}/${model.id}:streamGenerateContent?key=${apiKey}&alt=sse`;

      // Debug logging: concise request summary
      console.log(
        `[vertex-ai] Request: ${context.messages.length} messages, ${context.tools?.length || 0} tools`,
      );

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

      // Debug logging: HTTP response status
      console.log(`[vertex-ai] Response status: ${response.status}`);

      if (!response.ok) {
        const errorText = await response.text();
        console.error(`[vertex-ai] API error: ${response.status} ${errorText}`);
        throw new Error(`Vertex AI request failed: ${response.status} ${errorText}`);
      }

      if (!response.body) {
        throw new Error("Response body is null");
      }

      stream.push({ type: "start", partial: output });

      // Debug logging: SSE stream start
      console.log("[vertex-ai] Starting SSE stream parsing...");

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

            // Log each SSE chunk received
            console.log("[vertex-ai] SSE chunk received:", JSON.stringify(chunk, null, 2));

            // Log promptFeedback if present (can indicate safety blocks)
            if (chunk.promptFeedback) {
              console.log(
                "[vertex-ai] promptFeedback:",
                JSON.stringify(chunk.promptFeedback, null, 2),
              );
            }

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

                  // Prefer the echoed internal id from args, fall back to server id, or synthesize
                  const args = part.functionCall.args ?? {};
                  const echoedId = args.__openclaw_tool_call_id as string | undefined;
                  const serverId = part.functionCall.id;
                  const toolCallId =
                    echoedId ||
                    serverId ||
                    `${part.functionCall.name}_${Date.now()}_${++toolCallCounter}`;

                  // Debug logging: tool call id resolution
                  console.log(
                    `[vertex-ai] functionCall received — name: ${part.functionCall.name}, echoedId: ${echoedId || "(none)"}, serverId: ${serverId || "(none)"}, resolved: ${toolCallId}`,
                  );

                  // Remove the internal marker from arguments before creating the toolCall
                  const cleanArgs = { ...args };
                  delete cleanArgs.__openclaw_tool_call_id;

                  const toolCall: ToolCall & { thoughtSignature?: string } = {
                    type: "toolCall",
                    id: toolCallId,
                    name: part.functionCall.name,
                    arguments: cleanArgs,
                    thoughtSignature: part.thoughtSignature,
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
              console.log(`[vertex-ai] finishReason: ${candidate.finishReason}`);
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

      // Log stream completion
      console.log(
        `[vertex-ai] Stream completed. Content blocks: ${output.content.length}, stopReason: ${output.stopReason}`,
      );

      // Warn if no content was generated
      if (output.content.length === 0) {
        console.warn("[vertex-ai] Warning: No content generated in response");
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
