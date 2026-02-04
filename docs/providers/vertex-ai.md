# GCP Vertex AI

Use Google Cloud Vertex AI to access Gemini 3 models.

## Supported models

- `vertex-ai/gemini-3-flash-preview` - Fast, efficient model for most tasks
- `vertex-ai/gemini-3-pro-preview` - Most capable model with reasoning

## Quick start

1. Set your API key:

```bash
export VERTEX_AI_API_KEY="your-api-key"
```

2. Configure with your GCP project ID in the baseUrl:

```json5
{
  agents: {
    defaults: {
      model: { primary: "vertex-ai/gemini-3-flash-preview" },
    },
  },
  models: {
    providers: {
      "vertex-ai": {
        baseUrl: "https://aiplatform.googleapis.com/v1/projects/YOUR_PROJECT_ID/locations/global/publishers/google/models",
        apiKey: "${VERTEX_AI_API_KEY}",
        api: "google-generative-ai",
        models: [
          {
            id: "gemini-3-flash-preview",
            name: "Gemini 3 Flash Preview",
            reasoning: false,
            input: ["text", "image"],
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
            contextWindow: 1048576,
            maxTokens: 65536,
          },
          {
            id: "gemini-3-pro-preview",
            name: "Gemini 3 Pro Preview",
            reasoning: true,
            input: ["text", "image"],
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
            contextWindow: 1048576,
            maxTokens: 65536,
          },
        ],
      },
    },
  },
}
```

## Important notes

- Vertex AI Gemini 3 preview models are only available via the **global endpoint** (not regional endpoints)
- Replace `YOUR_PROJECT_ID` with your actual GCP project ID in the baseUrl
- You can get an API key from the Google Cloud Console
