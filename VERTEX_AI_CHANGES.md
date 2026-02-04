# Vertex AI Provider Tool-Call Flow Fix and Debug Logging

> **Note:** This file documents the Vertex AI provider improvements for stable tool-call id mapping and debug logging capabilities.

## Latest Changes: Comprehensive SSE Debug Logging (This PR)

### Problem
The Vertex AI provider had two critical debugging issues:

1. **Empty responses with no tool calls** - The agent received empty responses from Vertex AI and the loop would restart without any tool execution, making it impossible to diagnose why no content was being generated.

2. **Missing debug logging for SSE responses** - There was no visibility into what Vertex AI was actually returning in the SSE stream, making it impossible to debug issues.

### Solution
Added comprehensive always-on debug logging to capture the complete SSE response flow:

1. **SSE chunk logging** - Every SSE chunk received from Vertex AI is now logged with full JSON details
2. **finishReason logging** - Logs when the model indicates completion and why (STOP, MAX_TOKENS, SAFETY, etc.)
3. **Empty response detection** - Warns when no content is generated, helping identify empty response issues
4. **Stream completion logging** - Logs summary when stream ends, including content block count and stopReason
5. **promptFeedback logging** - Logs safety blocks and other prompt-level feedback from the API

### Implementation Details

All logging is **always enabled** (no environment variables or debug flags required) using `console.log` and `console.warn`:

#### 1. SSE Chunk Logging
```typescript
const chunk: GoogleResponse = JSON.parse(data);

// Log each SSE chunk received
console.log("[vertex-ai] SSE chunk received:", JSON.stringify(chunk, null, 2));
```

#### 2. Prompt Feedback Logging
```typescript
// Log promptFeedback if present (can indicate safety blocks)
if (chunk.promptFeedback) {
  console.log("[vertex-ai] promptFeedback:", JSON.stringify(chunk.promptFeedback, null, 2));
}
```

#### 3. Finish Reason Logging
```typescript
if (candidate?.finishReason) {
  console.log(`[vertex-ai] finishReason: ${candidate.finishReason}`);
  output.stopReason = mapStopReason(candidate.finishReason);
  // ... existing code
}
```

#### 4. Stream Completion and Empty Response Detection
```typescript
// Log stream completion
console.log(`[vertex-ai] Stream completed. Content blocks: ${output.content.length}, stopReason: ${output.stopReason}`);

// Warn if no content was generated
if (output.content.length === 0) {
  console.warn("[vertex-ai] Warning: No content generated in response");
}
```

### Example Output

When a request is made, you'll now see comprehensive logging like:

```
[vertex-ai] streamVertexAI entered — model: gemini-3-flash-preview
[vertex-ai] context.messages summary: [...]
[vertex-ai] redacted requestBody: {...}
[vertex-ai] SSE chunk received: {
  "candidates": [{
    "content": {
      "parts": [{"text": "Hello, how can I help?"}]
    }
  }]
}
[vertex-ai] SSE chunk received: {
  "candidates": [{
    "finishReason": "STOP"
  }],
  "usageMetadata": {...}
}
[vertex-ai] finishReason: STOP
[vertex-ai] Stream completed. Content blocks: 1, stopReason: stop
```

For empty responses (the original problem):
```
[vertex-ai] streamVertexAI entered — model: gemini-3-flash-preview
[vertex-ai] SSE chunk received: {"candidates": []}
[vertex-ai] Stream completed. Content blocks: 0, stopReason: stop
[vertex-ai] Warning: No content generated in response
```

For safety blocks:
```
[vertex-ai] promptFeedback: {
  "blockReason": "SAFETY",
  "safetyRatings": [...]
}
```

### Testing
All 15 existing tests pass with the new logging:
```bash
npx vitest run --config vitest.unit.config.ts src/providers/vertex-ai.test.ts
```

### Impact
- **Debugging**: Full visibility into SSE stream responses
- **Empty response detection**: Clear warnings when no content is generated
- **Safety block visibility**: Can now see when requests are blocked for safety reasons
- **Non-breaking**: All logging is additive and doesn't affect functionality
- **Always-on**: No flags or environment variables needed - logging is always active

---

## Previous Changes: Tool-Call ID Mapping

### Problem
The Vertex AI provider had an intermittent tool-call flow issue:
- Vertex AI rejects `function_call.id` in request payloads at the top level
- The runtime lost stable mapping between assistant toolCall ids and incoming function_call responses
- This caused runs to succeed once but fail when multiple tools were used in sequence

### Solution
We stabilized the toolCall id mapping by:
1. **Injecting** the internal assistant toolCall id into function call arguments using `__openclaw_tool_call_id` marker
2. **Parsing** incoming SSE responses to prefer the echoed internal id
3. **Cleaning** arguments by removing the internal marker before passing to tool execution
4. **Logging** the flow to make debugging visible

### Implementation Details

#### 1. Outgoing Function Calls
When converting assistant messages to Google format, we inject the internal id:

```typescript
else if (block.type === "toolCall") {
  const args = { ...block.arguments };
  if (block.id) {
    args.__openclaw_tool_call_id = block.id;
  }
  parts.push({
    functionCall: {
      name: block.name,
      args,  // Contains the marker
    },
  });
}
```

#### 2. Incoming SSE Responses
When parsing functionCall from Vertex SSE responses, we prioritize the echoed id:

```typescript
const args = part.functionCall.args ?? {};
const echoedId = args.__openclaw_tool_call_id as string | undefined;
const serverId = part.functionCall.id;
const toolCallId = echoedId || serverId || `${part.functionCall.name}_${Date.now()}_${++toolCallCounter}`;

// Remove marker before creating ToolCall
const cleanArgs = { ...args };
delete cleanArgs.__openclaw_tool_call_id;
```

#### 3. Debug Logging
Added three debug log points (enabled via `VERTEX_AI_DEBUG_PAYLOAD=1`):

1. **Entry point**: Logs when streamVertexAI is entered
   ```
   [vertex-ai] streamVertexAI entered — model: gemini-3-flash-preview
   ```

2. **Outgoing injection**: Logs when injecting internal id into args
   ```
   [vertex-ai] outgoing toolCall — injecting id: call_123 into functionCall.args for get_weather
   ```

3. **Incoming resolution**: Logs the id resolution process
   ```
   [vertex-ai] functionCall received — name: get_weather, echoedId: call_123, serverId: (none), resolved: call_123
   ```

### Testing

#### Unit Tests
All 16 tests pass, including 3 new tests for:
- Echoed id preference over server id
- Server id fallback when no echoed id
- ID synthesis when neither id is present

Run tests with:
```bash
npx vitest run --config vitest.unit.config.ts src/providers/vertex-ai.test.ts
```

#### Manual Testing
A manual test script demonstrates:
- ✓ Original id preserved through round-trip
- ✓ Server id ignored when echoed id present
- ✓ Marker removed from arguments
- ✓ Tool execution sees clean arguments

### Impact
- **Stability**: Sequential tool calls now work reliably
- **Debugging**: Clear log trail for tool call id resolution
- **Correctness**: Tools receive clean arguments without internal markers
- **Non-breaking**: Existing behavior preserved, only stabilizes id mapping

---

## Previous Changes: Debug Logging and Experimental Toggle

This PR adds experimental debug logging and a toggle to control the role used for tool results in the Vertex AI provider (`src/providers/vertex-ai.ts`).

## Changes Made

### 1. Added Experimental Toggle Constant
```typescript
// Experimental toggle: when true, toolResult uses role "model"; when false, uses role "user"
const FORCE_TOOLRESULT_AS_MODEL = true;
```
- Default: `true` (uses role "model" for tool results)
- Can be changed to `false` to experiment with role "user"

### 2. Added redactedClone Utility Function
```typescript
function redactedClone(obj: unknown): unknown {
  return redactSecrets(obj);
}
```
- Wraps the existing `redactSecrets` function
- Provides a convenient API for safe logging as requested in the problem statement

### 3. Updated convertMessagesToGoogleFormat
Modified the tool result handling to use the toggle:
```typescript
// Use experimental toggle to control toolResult role
if (parts.length > 0) {
  contents.push({
    role: FORCE_TOOLRESULT_AS_MODEL ? "model" : "user",
    parts,
  });
}
```

### 4. Enhanced Debug Logging
- Changed from `console.debug` to `console.log` for better visibility
- Added try/catch block to prevent runtime interference
- Uses `redactedClone` instead of `redactSecrets` for consistency
- Updated label from "requestBody:" to "redacted requestBody:" for clarity

Example output when `VERTEX_AI_DEBUG_PAYLOAD=1`:
```
[vertex-ai] context.messages summary: [
  { "role": "user", "preview": "What is the weather?" },
  { "role": "assistant", "preview": "" },
  { "role": "toolResult", "preview": "Sunny, 75°F" }
]
[vertex-ai] redacted requestBody: {
  "contents": [...],
  "systemInstruction": {...}
}
```

### 5. Updated All Tests
Updated all test cases in `src/providers/vertex-ai.test.ts`:
- Changed `console.debug` mocks to `console.log`
- Updated test expectations to match new log labels
- All existing tests continue to pass with the new implementation

## How to Use

### Enable Debug Logging
Set the environment variable before running:
```bash
VERTEX_AI_DEBUG_PAYLOAD=1 node your-app.js
```

### Toggle Tool Result Role
To experiment with different roles, modify the constant in `src/providers/vertex-ai.ts`:
```typescript
const FORCE_TOOLRESULT_AS_MODEL = false;  // Use "user" role instead of "model"
```

## Testing

All changes have been validated:
- ✅ FORCE_TOOLRESULT_AS_MODEL constant added
- ✅ redactedClone function added
- ✅ Toggle controls toolResult role correctly
- ✅ console.log used for debug output
- ✅ Try/catch wraps debug logging
- ✅ All tests updated and passing

## Impact

These changes are:
- **Minimal**: Only affects the Vertex AI provider file and its tests
- **Safe**: Debug logging is opt-in via environment variable
- **Non-breaking**: Default behavior (role: "model") matches the previous implementation
- **Experimental**: The toggle allows quick experimentation without code changes

## Next Steps

After collecting debug output from production:
1. Analyze the request payload structure
2. Determine if role "model" or "user" works better for tool results
3. Either keep the current default or change it based on findings
4. Consider removing the toggle and hardcoding the working approach
