- An embedding `api_key: ${NAME}` may name only `OPENAI_API_KEY`, `GEMINI_API_KEY` or `OPENROUTER_API_KEY`, each for its provider's host, or the `NAME=host` entries in `OMNIGRAPH_EMBED_API_KEY_VARS`. Clusters using another name or host must list it before upgrading. See [providers][embed-key-vars].

[embed-key-vars]: ../docs/user/clusters/config.md#embedding-providers
