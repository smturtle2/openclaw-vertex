/**
 * Vertex AI provider implementation using @google/genai SDK.
 *
 * This provider uses the @google/genai SDK with Vertex AI mode to stream responses.
 * It maintains all existing event patterns and internal mechanisms.
 *
 * Vertex AI API differences from standard Google Generative AI:
 * - Does not accept `id` field in functionCall requests (though it may return ids in responses)
 * - Only accepts `role: "user"` or `role: "model"` in contents (no `role: "function"`)
 * - Tool results (functionResponse) must use `role: "user"` to work correctly with continuous tool execution
 */

import type { Candidate, Content, GenerateContentResponse, Part } from "@google/genai";
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
import { GoogleGenAI } from "@google/genai";
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

let toolCallCounter = 0;

function convertMessagesToGoogleFormat(context: Context): Content[] {
  const contents: Content[] = [];

  for (const msg of context.messages) {
    if (msg.role === "user") {
      const parts: Part[] = [];
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
      const parts: Part[] = [];
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

          parts.push({
            functionCall: {
              name: block.name,
              args: argsWithMarker,
            },
          });
        }
      }
      if (parts.length > 0) {
        contents.push({ role: "model", parts });
      }
    } else if (msg.role === "toolResult") {
      const parts: Part[] = [];
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

function convertTools(tools: Context["tools"]) {
  if (!tools || tools.length === 0) {
    return undefined;
  }

  return tools.map((tool) => ({
    functionDeclarations: [
      {
        name: tool.name,
        description: tool.description,
        parameters: tool.parameters,
      },
    ],
  }));
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

      // Debug logging: concise request summary
      console.log(
        `[vertex-ai] Request: ${context.messages.length} messages, ${context.tools?.length || 0} tools`,
      );

      // Create GoogleGenAI client
      const client = new GoogleGenAI({
        apiKey,
        vertexai: true,
      });

      // Build the request parameters
      const contents = convertMessagesToGoogleFormat(context);

      const config: Record<string, any> = {};

      // Add generation config
      if (options?.temperature !== undefined) {
        config.temperature = options.temperature;
      }
      if (options?.maxTokens !== undefined) {
        config.maxOutputTokens = options.maxTokens;
      }

      // Add tools if present
      if (context.tools && context.tools.length > 0) {
        config.tools = convertTools(context.tools);
        if (options?.toolChoice) {
          config.toolConfig = {
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

      // Build the request body for onPayload
      const requestBody: any = {
        contents,
        ...(context.systemPrompt && {
          systemInstruction: { parts: [{ text: context.systemPrompt }] },
        }),
        ...(Object.keys(config).length > 0 && { config }),
      };

      // Call onPayload if provided
      options?.onPayload?.(requestBody);

      // Prepare request parameters
      const params: any = {
        model: model.id,
        contents,
        config,
      };

      // Add system instruction if present
      if (context.systemPrompt) {
        params.systemInstruction = { parts: [{ text: context.systemPrompt }] };
      }

      // Debug logging: Starting stream
      console.log("[vertex-ai] Starting SDK stream...");

      stream.push({ type: "start", partial: output });

      // Use SDK to stream content
      const streamingResp = await client.models.generateContentStream(params);

      let currentBlock: TextContent | ThinkingContent | null = null;

      for await (const chunk of streamingResp) {
        // Log each chunk received
        console.log("[vertex-ai] SDK chunk received:", JSON.stringify(chunk, null, 2));

        // Log promptFeedback if present (can indicate safety blocks)
        if (chunk.promptFeedback) {
          console.log("[vertex-ai] promptFeedback:", JSON.stringify(chunk.promptFeedback, null, 2));
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
              const serverId = (part.functionCall as any).id;
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

              const toolCall: ToolCall = {
                type: "toolCall",
                id: toolCallId,
                name: part.functionCall.name,
                arguments: cleanArgs,
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
          output.usage.input = (chunk.usageMetadata as any).promptTokenCount || 0;
          output.usage.output = (chunk.usageMetadata as any).candidatesTokenCount || 0;
          output.usage.totalTokens = (chunk.usageMetadata as any).totalTokenCount || 0;
          const cost = calculateCost(model, output.usage);
          output.usage.cost = cost;
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
      console.error("[vertex-ai] Error:", errorMessage);
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
