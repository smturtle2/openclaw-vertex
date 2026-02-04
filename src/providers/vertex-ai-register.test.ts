import { describe, expect, it } from "vitest";

describe("vertex-ai-register", () => {
  it("exports registerVertexAIProvider function", async () => {
    const { registerVertexAIProvider } = await import("./vertex-ai-register.js");
    expect(typeof registerVertexAIProvider).toBe("function");
  });

  it("registerVertexAIProvider is called in attempt.ts", async () => {
    // Verify that the registration is imported and called in attempt.ts
    // This is the key fix - the provider must be registered before use
    const attemptModule = await import("fs/promises");
    const attemptSource = await attemptModule.readFile(
      new URL("../agents/pi-embedded-runner/run/attempt.ts", import.meta.url),
      "utf-8",
    );

    // Verify the import exists
    expect(attemptSource).toContain('import { registerVertexAIProvider }');
    expect(attemptSource).toContain('from "../../../providers/vertex-ai-register.js"');
    
    // Verify the function is called
    expect(attemptSource).toContain("registerVertexAIProvider()");
  });
});
