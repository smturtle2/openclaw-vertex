# Sample Debug Output

When `VERTEX_AI_DEBUG_PAYLOAD=1` is set, the Vertex AI provider will output the following:

## Example 1: Simple User Message

```json
[vertex-ai] context.messages summary: [
  {
    "role": "user",
    "preview": "What is the weather in New York?"
  }
]
[vertex-ai] redacted requestBody: {
  "contents": [
    {
      "role": "user",
      "parts": [
        {
          "text": "What is the weather in New York?"
        }
      ]
    }
  ]
}
```

## Example 2: Tool Call and Tool Result Flow

```json
[vertex-ai] context.messages summary: [
  {
    "role": "user",
    "preview": "What is the weather in New York?"
  },
  {
    "role": "assistant",
    "preview": ""
  },
  {
    "role": "toolResult",
    "preview": "Sunny, 75°F with light breeze"
  }
]
[vertex-ai] redacted requestBody: {
  "contents": [
    {
      "role": "user",
      "parts": [
        {
          "text": "What is the weather in New York?"
        }
      ]
    },
    {
      "role": "model",
      "parts": [
        {
          "functionCall": {
            "name": "get_weather",
            "args": {
              "location": "New York"
            }
          }
        }
      ]
    },
    {
      "role": "model",
      "parts": [
        {
          "functionResponse": {
            "name": "get_weather",
            "response": {
              "result": "Sunny, 75°F with light breeze"
            }
          }
        }
      ]
    }
  ]
}
```

## Example 3: Long Message Preview

When a message exceeds 120 characters, it will be truncated:

```json
[vertex-ai] context.messages summary: [
  {
    "role": "user",
    "preview": "This is a very long message that will be truncated to 120 characters so that the log output remains readable and doesn't ..."
  }
]
```

## Example 4: API Key Redaction

API keys and secrets are automatically redacted:

```json
[vertex-ai] redacted requestBody: {
  "contents": [...],
  "apiKey": "[REDACTED]",
  "systemInstruction": {
    "parts": [
      {
        "text": "You are a helpful assistant"
      }
    ]
  }
}
```

## Toggle Behavior

### With FORCE_TOOLRESULT_AS_MODEL = true (default)
Tool results use `"role": "model"`:
```json
{
  "role": "model",
  "parts": [{"functionResponse": {...}}]
}
```

### With FORCE_TOOLRESULT_AS_MODEL = false
Tool results use `"role": "user"`:
```json
{
  "role": "user",
  "parts": [{"functionResponse": {...}}]
}
```

## Error Handling

If debug logging fails, a warning is shown but the request proceeds:
```
[vertex-ai] debug logging failed: Error: ...
```
