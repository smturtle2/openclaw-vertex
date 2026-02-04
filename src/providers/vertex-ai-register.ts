/**
 * Register the Vertex AI custom API provider.
 *
 * This must be imported and executed before any pi-ai usage to ensure
 * the custom "vertex-ai" API type is available.
 */

import { registerApiProvider } from "@mariozechner/pi-ai";
import { streamSimpleVertexAI, streamVertexAI } from "./vertex-ai.js";

export function registerVertexAIProvider(): void {
  registerApiProvider(
    {
      api: "vertex-ai",
      stream: streamVertexAI,
      streamSimple: streamSimpleVertexAI,
    },
    "openclaw-vertex-ai",
  );
}
