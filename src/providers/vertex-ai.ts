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
import { AssistantMessageEventStream, calculateCost, getEnvApiKey } from "@mariozechner/pi-ai";
import {
  convertMessages,
  convertTools,
  isThinkingPart,
  mapStopReason,
  mapToolChoice,
  retainThoughtSignature,
} from "@mariozechner/pi-ai/dist/providers/google-shared.js";

interface VertexAIOptions extends StreamOptions {
  toolChoice?: "auto" | "none" | "any";
  thinking?: {
    enabled: boolean;
    budgetTokens?: number;
    level?: "minimal" | "low" | "medium" | "high";
  };
}

let toolCallCounter = 0;

export const streamVertexAI: StreamFunction<"vertex-ai", VertexAIOptions> = (
  model,
  context,
  options,
) => {
  const stream = new AssistantMessageEventStream();

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
      const contents = convertMessages(model, context);

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
              mode: mapToolChoice(options.toolChoice),
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
              // Determine if this is a thinking block
              const isThinking = isThinkingPart(part);

              // Check if we need to start a new block
              if (
                !currentBlock ||
                (isThinking && currentBlock.type !== "thinking") ||
                (!isThinking && currentBlock.type !== "text")
              ) {
                // End previous block if exists
                if (currentBlock) {
                  if (currentBlock.type === "text") {
                    stream.push({
                      type: "text_end",
                      contentIndex: output.content.length - 1,
                      content: currentBlock.text,
                      partial: output,
                    });
                  } else {
                    stream.push({
                      type: "thinking_end",
                      contentIndex: output.content.length - 1,
                      content: (currentBlock as ThinkingContent).thinking,
                      partial: output,
                    });
                  }
                }

                // Start new block
                if (isThinking) {
                  currentBlock = { type: "thinking", thinking: "", thinkingSignature: undefined };
                  output.content.push(currentBlock);
                  stream.push({
                    type: "thinking_start",
                    contentIndex: output.content.length - 1,
                    partial: output,
                  });
                } else {
                  currentBlock = { type: "text", text: "" };
                  output.content.push(currentBlock);
                  stream.push({
                    type: "text_start",
                    contentIndex: output.content.length - 1,
                    partial: output,
                  });
                }
              }

              // Append text to current block
              if (currentBlock.type === "thinking") {
                (currentBlock as ThinkingContent).thinking += part.text;
                (currentBlock as ThinkingContent).thinkingSignature = retainThoughtSignature(
                  (currentBlock as ThinkingContent).thinkingSignature,
                  part.thoughtSignature,
                );
                stream.push({
                  type: "thinking_delta",
                  contentIndex: output.content.length - 1,
                  delta: part.text,
                  partial: output,
                });
              } else {
                (currentBlock as TextContent).text += part.text;
                (currentBlock as any).textSignature = retainThoughtSignature(
                  (currentBlock as any).textSignature,
                  part.thoughtSignature,
                );
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

              // Use provided id or generate a new one if missing or duplicate
              const providedId = part.functionCall.id;
              const needsNewId =
                !providedId ||
                output.content.some((b) => b.type === "toolCall" && b.id === providedId);
              const toolCallId = needsNewId
                ? `${part.functionCall.name}_${Date.now()}_${++toolCallCounter}`
                : providedId;

              const toolCall: ToolCall & { thoughtSignature?: string } = {
                type: "toolCall",
                id: toolCallId,
                name: part.functionCall.name || "",
                arguments: part.functionCall.args ?? {},
                ...(part.thoughtSignature && { thoughtSignature: part.thoughtSignature }),
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
      stream.end();
    }
  })();

  return stream;
};

export const streamSimpleVertexAI: StreamFunction<"vertex-ai", SimpleStreamOptions> =
  streamVertexAI;
