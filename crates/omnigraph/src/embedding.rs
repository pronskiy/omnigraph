use std::future::Future;
use std::time::Duration;

use reqwest::Client;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::time::sleep;

use crate::error::{OmniError, Result};

const DEFAULT_OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";
const DEFAULT_OPENROUTER_MODEL: &str = "openai/text-embedding-3-large";
const DEFAULT_OPENAI_BASE_URL: &str = "https://api.openai.com/v1";
const DEFAULT_OPENAI_MODEL: &str = "text-embedding-3-large";
const DEFAULT_GEMINI_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";
const DEFAULT_GEMINI_MODEL: &str = "gemini-embedding-2";
const DEFAULT_TIMEOUT_MS: u64 = 30_000;
const DEFAULT_RETRY_ATTEMPTS: usize = 4;
const DEFAULT_RETRY_BACKOFF_MS: u64 = 200;
const DEFAULT_DEADLINE_MS: u64 = 60_000;
/// The shortest time a client waits after a 429 before it calls the provider
/// again, whatever `Retry-After` said (including none, or `0`).
const RATE_LIMIT_COOLDOWN_FLOOR: Duration = Duration::from_secs(1);
/// The longest `Retry-After` a client honours; a longer one is capped (the
/// call's own deadline still bounds every wait).
const MAX_RETRY_AFTER: Duration = Duration::from_secs(300);
const GEMINI_QUERY_TASK_TYPE: &str = "RETRIEVAL_QUERY";
const GEMINI_DOCUMENT_TASK_TYPE: &str = "RETRIEVAL_DOCUMENT";

/// Which embedding API a client speaks. Each variant owns its request shape,
/// auth, and response parsing; everything else (retry, deadline, normalization,
/// tracing) is provider-independent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Provider {
    /// OpenAI-compatible (`POST {base}/embeddings`, bearer auth,
    /// `{model, input, dimensions}`). Covers OpenRouter (the default gateway),
    /// OpenAI direct, and self-hosted endpoints (vLLM/Ollama/LM Studio).
    OpenAiCompatible,
    /// Google Gemini `generativelanguage` (`POST {base}/models/{model}:embedContent`,
    /// `x-goog-api-key`), with `RETRIEVAL_QUERY` / `RETRIEVAL_DOCUMENT` task types.
    Gemini,
    /// Deterministic, offline. No network, no key.
    Mock,
}

/// Whether the text being embedded is a search query or a stored document.
/// Only Gemini distinguishes these (`RETRIEVAL_QUERY` vs `RETRIEVAL_DOCUMENT`);
/// OpenAI-compatible providers and Mock produce the identical request for both,
/// which is also the same-space property a query relies on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EmbedRole {
    Query,
    Document,
}

/// The single source of truth for how embedding text becomes a vector:
/// provider + model + endpoint + key. Resolved once (from env for direct
/// engine/CLI callers, or from an applied cluster `providers.embedding` profile
/// at server boot) and shared by the query path and the offline CLI so stored
/// and query vectors stay same-space by construction.
#[derive(Clone, Debug)]
pub struct EmbeddingConfig {
    pub provider: Provider,
    pub model: String,
    pub base_url: String,
    pub api_key: String,
}

impl EmbeddingConfig {
    /// Resolve from the environment. Precedence:
    /// 1. `OMNIGRAPH_EMBEDDINGS_MOCK` → Mock.
    /// 2. `OMNIGRAPH_EMBED_PROVIDER` (`openai-compatible`|`openai`|`gemini`|`mock`);
    ///    unset defaults to `openai-compatible` (OpenRouter).
    /// 3. `OMNIGRAPH_EMBED_BASE_URL` else the provider default.
    /// 4. `OMNIGRAPH_EMBED_MODEL` else the provider default.
    /// 5. provider api-key env (`OPENROUTER_API_KEY`/`OPENAI_API_KEY`, or `GEMINI_API_KEY`).
    pub fn from_env() -> Result<Self> {
        if env_flag("OMNIGRAPH_EMBEDDINGS_MOCK") {
            // The mock flag deliberately wins (pinned by
            // from_env_mock_flag_wins) — but overriding an EXPLICITLY
            // configured real provider must be loud: mock vectors are
            // indistinguishable from real ones (correct dimension, unit
            // norm), so a leaked test env var would otherwise silently
            // poison persisted embeds and query-time nearest().
            if let Some(provider) = env_string("OMNIGRAPH_EMBED_PROVIDER")
                && provider != "mock"
            {
                tracing::warn!(
                    target: "omnigraph::embedding",
                    overridden_provider = %provider,
                    "OMNIGRAPH_EMBEDDINGS_MOCK is set and overrides the \
                     explicitly configured OMNIGRAPH_EMBED_PROVIDER — all \
                     embeddings in this process are deterministic mock \
                     vectors, not real ones"
                );
            }
            return Ok(Self::mock());
        }

        let alias = env_string("OMNIGRAPH_EMBED_PROVIDER");
        if alias.as_deref() == Some("mock") {
            return Ok(Self::mock());
        }

        let (provider, default_base, default_model, key_envs) = provider_profile(alias.as_deref())?;
        let base_url = env_string("OMNIGRAPH_EMBED_BASE_URL")
            .unwrap_or_else(|| default_base.to_string())
            .trim_end_matches('/')
            .to_string();
        let model =
            env_string("OMNIGRAPH_EMBED_MODEL").unwrap_or_else(|| default_model.to_string());

        let api_key = key_envs
            .iter()
            .copied()
            .find_map(env_string)
            .ok_or_else(|| {
                OmniError::manifest_internal(format!(
                    "{} is required for the {} embedding provider",
                    key_envs.join(" or "),
                    alias.as_deref().unwrap_or("openai-compatible")
                ))
            })?;

        Ok(Self {
            provider,
            model,
            base_url,
            api_key,
        })
    }

    /// Build a config from explicit parts — the cluster `providers.embedding` profile path
    /// (RFC-012 Phase 5). `provider`/`base_url`/`model` default exactly as
    /// `from_env` does (shared `provider_profile`); `api_key` is already resolved
    /// (the cluster path resolves a `${NAME}` ref before calling this).
    pub fn from_parts(
        provider: Option<&str>,
        base_url: Option<String>,
        model: Option<String>,
        api_key: String,
    ) -> Result<Self> {
        if provider == Some("mock") {
            // An explicit `model` (e.g. a cluster `providers.embedding` profile) is
            // authoritative — it is what the same-space check compares against —
            // so honor it; fall back to `mock()`'s env-based model only when the
            // caller supplied none. Without this, a profile's `model` is silently
            // dropped and the same-space check resolves to OMNIGRAPH_EMBED_MODEL.
            let mut config = Self::mock();
            if let Some(model) = model {
                config.model = model;
            }
            return Ok(config);
        }
        let (provider, default_base, default_model, _key_envs) = provider_profile(provider)?;
        let base_url = base_url
            .unwrap_or_else(|| default_base.to_string())
            .trim_end_matches('/')
            .to_string();
        let model = model.unwrap_or_else(|| default_model.to_string());
        Ok(Self {
            provider,
            model,
            base_url,
            api_key,
        })
    }

    fn mock() -> Self {
        Self {
            provider: Provider::Mock,
            // Honor OMNIGRAPH_EMBED_MODEL so the same-space check is exercisable
            // under mock; the mock vectors themselves don't depend on the model.
            model: env_string("OMNIGRAPH_EMBED_MODEL").unwrap_or_default(),
            base_url: String::new(),
            api_key: String::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct EmbeddingClient {
    config: EmbeddingConfig,
    http: Client,
    retry_attempts: usize,
    retry_backoff_ms: u64,
    /// Total wall-clock budget for one embed call, across all retries
    /// (`OMNIGRAPH_EMBED_DEADLINE_MS`). `0` = unbounded.
    deadline_ms: u64,
    /// End of the cooldown a provider rate refusal (429) started, shared by
    /// every clone of this client.
    rate_limited_until: std::sync::Arc<std::sync::Mutex<Option<tokio::time::Instant>>>,
}

#[derive(Default)]
struct EmbedCallError {
    message: String,
    /// A transient failure (timeout, connect error, 5xx): retried with
    /// jittered exponential backoff inside the deadline.
    retryable: bool,
    /// The provider refused for rate (429). Never retried blind: only after
    /// the provider's own `Retry-After`, and it starts a client-wide cooldown
    /// that every concurrent and later call on this client honours.
    rate_limited: bool,
    retry_after: Option<Duration>,
}

#[derive(Debug, Deserialize)]
struct GeminiEmbedResponse {
    embedding: GeminiContentEmbedding,
}

#[derive(Debug, Deserialize)]
struct GeminiContentEmbedding {
    values: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct GoogleErrorEnvelope {
    error: GoogleErrorBody,
}

#[derive(Debug, Deserialize)]
struct GoogleErrorBody {
    message: String,
}

#[derive(Debug, Deserialize)]
struct OpenAiEmbeddingResponse {
    data: Vec<OpenAiEmbeddingDatum>,
}

#[derive(Debug, Deserialize)]
struct OpenAiEmbeddingDatum {
    index: usize,
    embedding: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct OpenAiErrorEnvelope {
    error: OpenAiErrorBody,
}

#[derive(Debug, Deserialize)]
struct OpenAiErrorBody {
    message: String,
}

impl EmbeddingClient {
    pub fn from_env() -> Result<Self> {
        Self::new(EmbeddingConfig::from_env()?)
    }

    pub fn new(config: EmbeddingConfig) -> Result<Self> {
        let retry_attempts =
            parse_env_usize("OMNIGRAPH_EMBED_RETRY_ATTEMPTS", DEFAULT_RETRY_ATTEMPTS);
        let retry_backoff_ms =
            parse_env_u64("OMNIGRAPH_EMBED_RETRY_BACKOFF_MS", DEFAULT_RETRY_BACKOFF_MS);
        let deadline_ms =
            parse_env_u64_allow_zero("OMNIGRAPH_EMBED_DEADLINE_MS", DEFAULT_DEADLINE_MS);
        let timeout_ms = parse_env_u64("OMNIGRAPH_EMBED_TIMEOUT_MS", DEFAULT_TIMEOUT_MS);
        let http = Client::builder()
            .timeout(Duration::from_millis(timeout_ms))
            .build()
            .map_err(|e| {
                OmniError::manifest_internal(format!("failed to initialize HTTP client: {}", e))
            })?;

        Ok(Self {
            config,
            http,
            retry_attempts,
            retry_backoff_ms,
            deadline_ms,
            rate_limited_until: Default::default(),
        })
    }

    pub fn config(&self) -> &EmbeddingConfig {
        &self.config
    }

    #[cfg(test)]
    fn mock_for_tests() -> Self {
        Self::new(EmbeddingConfig::mock()).expect("mock client builds")
    }

    /// Embed a text `nearest()` query. `input` is the query string, unchanged.
    pub async fn embed_query_text(&self, input: &str, expected_dim: usize) -> Result<Vec<f32>> {
        self.embed_text(input, expected_dim, EmbedRole::Query).await
    }

    /// Embed stored text. `input` is a record's `@embed` source value,
    /// unchanged, as `embed_query_text` takes the query: a template on either
    /// side would leave a record whose source equals the query text away from
    /// that query's vector.
    pub async fn embed_document_text(&self, input: &str, expected_dim: usize) -> Result<Vec<f32>> {
        self.embed_text(input, expected_dim, EmbedRole::Document)
            .await
    }

    async fn embed_text(
        &self,
        input: &str,
        expected_dim: usize,
        role: EmbedRole,
    ) -> Result<Vec<f32>> {
        if expected_dim == 0 {
            return Err(OmniError::manifest_internal(
                "embedding dimension must be greater than zero",
            ));
        }

        let started = std::time::Instant::now();
        let result = self
            .run_with_deadline(self.embed_text_inner(input, expected_dim, role))
            .await;
        let elapsed_ms = started.elapsed().as_millis() as u64;

        match &result {
            Ok(_) => tracing::info!(
                target: "omnigraph::embedding",
                provider = ?self.config.provider,
                model = %self.config.model,
                dim = expected_dim,
                elapsed_ms,
                outcome = "ok",
                "embedding succeeded"
            ),
            Err(err) => tracing::warn!(
                target: "omnigraph::embedding",
                provider = ?self.config.provider,
                model = %self.config.model,
                dim = expected_dim,
                elapsed_ms,
                outcome = "error",
                error = %err,
                "embedding failed"
            ),
        }
        result
    }

    /// Bound the whole embed operation (all retries + backoff) by `deadline_ms`,
    /// so a degraded provider can never hang the caller for the full retry
    /// envelope. Applies to every embed call (query and document). `0` =
    /// unbounded. Embedding has no Lance/manifest side effects, so cancelling the
    /// in-flight request future on elapse is safe.
    async fn run_with_deadline<F>(&self, fut: F) -> Result<Vec<f32>>
    where
        F: Future<Output = Result<Vec<f32>>>,
    {
        if self.deadline_ms == 0 {
            return fut.await;
        }
        match tokio::time::timeout(Duration::from_millis(self.deadline_ms), fut).await {
            Ok(res) => res,
            Err(_elapsed) => Err(OmniError::manifest_internal(format!(
                "embedding deadline exceeded after {} ms (provider={:?}, model={})",
                self.deadline_ms, self.config.provider, self.config.model
            ))),
        }
    }

    async fn embed_text_inner(
        &self,
        input: &str,
        expected_dim: usize,
        role: EmbedRole,
    ) -> Result<Vec<f32>> {
        match self.config.provider {
            Provider::Mock => Ok(mock_embedding(input, expected_dim)),
            Provider::Gemini => {
                self.with_retry(role, || self.embed_gemini_once(input, expected_dim, role))
                    .await
            }
            Provider::OpenAiCompatible => {
                self.with_retry(role, || self.embed_openai_once(input, expected_dim))
                    .await
            }
        }
    }

    async fn with_retry<T, F, Fut>(&self, role: EmbedRole, mut operation: F) -> Result<T>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = std::result::Result<T, EmbedCallError>>,
    {
        let max_attempt = self.retry_attempts.max(1);
        // A document embedding (`omnigraph embed`, a load's `@embed`) is a
        // batch the operator started: it waits out a rate refusal and goes on.
        // A query embedding answers a caller, who is told when to come back.
        let waits_out = matches!(role, EmbedRole::Document);
        let deadline = (self.deadline_ms > 0)
            .then(|| tokio::time::Instant::now() + Duration::from_millis(self.deadline_ms));
        let fits = |wait: Duration| {
            deadline.is_none_or(|deadline| tokio::time::Instant::now() + wait < deadline)
        };
        let mut attempt = 0usize;
        let mut rate_retried = false;
        loop {
            // A provider that refused for rate is not asked again before the
            // time it named (or the cooldown floor): no call on this client
            // adds load to a provider that is refusing.
            if let Some(wait) = self.rate_limit_remaining() {
                if waits_out && fits(wait) {
                    sleep(wait).await;
                    continue;
                }
                return Err(rate_limited(wait));
            }
            attempt += 1;
            match operation().await {
                Ok(value) => return Ok(value),
                Err(err) if err.rate_limited => {
                    let cooldown = err
                        .retry_after
                        .unwrap_or_default()
                        .max(RATE_LIMIT_COOLDOWN_FLOOR);
                    self.start_rate_limit_cooldown(cooldown);
                    // A query retries a rate refusal once, only when the
                    // provider said when; a document waits within its attempts.
                    let may_wait = if waits_out {
                        attempt < max_attempt
                    } else {
                        err.retry_after.is_some() && !rate_retried && attempt < max_attempt
                    };
                    // Callers refused together do not come back together.
                    let spread = Duration::from_millis(jittered(cooldown.as_millis() as u64 / 4));
                    if !may_wait || !fits(cooldown + spread) {
                        return Err(rate_limited(cooldown));
                    }
                    rate_retried = true;
                    tracing::warn!(
                        target: "omnigraph::embedding",
                        provider = ?self.config.provider,
                        model = %self.config.model,
                        attempt,
                        retry_after_ms = cooldown.as_millis() as u64,
                        error = %err.message,
                        "embedding rate limited, waiting out the refusal"
                    );
                    sleep(cooldown + spread).await;
                }
                Err(err) => {
                    if !err.retryable || attempt >= max_attempt {
                        return Err(OmniError::manifest_internal(err.message));
                    }
                    tracing::warn!(
                        target: "omnigraph::embedding",
                        provider = ?self.config.provider,
                        model = %self.config.model,
                        attempt,
                        error = %err.message,
                        "embedding attempt failed, retrying"
                    );
                    let shift = (attempt - 1).min(10) as u32;
                    let ceiling = self.retry_backoff_ms.saturating_mul(1u64 << shift);
                    sleep(Duration::from_millis(jittered(ceiling))).await;
                }
            }
        }
    }

    /// Time left before this client may call its provider again, if a rate
    /// refusal started a cooldown that has not ended.
    fn rate_limit_remaining(&self) -> Option<Duration> {
        let until = (*self
            .rate_limited_until
            .lock()
            .unwrap_or_else(|e| e.into_inner()))?;
        until
            .checked_duration_since(tokio::time::Instant::now())
            .filter(|remaining| !remaining.is_zero())
    }

    fn start_rate_limit_cooldown(&self, cooldown: Duration) {
        let until = tokio::time::Instant::now() + cooldown;
        let mut slot = self
            .rate_limited_until
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if slot.is_none_or(|current| current < until) {
            *slot = Some(until);
        }
    }

    async fn embed_gemini_once(
        &self,
        input: &str,
        expected_dim: usize,
        role: EmbedRole,
    ) -> std::result::Result<Vec<f32>, EmbedCallError> {
        let task_type = match role {
            EmbedRole::Query => GEMINI_QUERY_TASK_TYPE,
            EmbedRole::Document => GEMINI_DOCUMENT_TASK_TYPE,
        };

        let response = self
            .http
            .post(gemini_endpoint(&self.config.base_url, &self.config.model))
            .header("x-goog-api-key", &self.config.api_key)
            .json(&build_gemini_request(
                &self.config.model,
                input,
                expected_dim,
                task_type,
            ))
            .send()
            .await;
        let response = match response {
            Ok(response) => response,
            Err(err) => {
                let retryable = err.is_timeout() || err.is_connect() || err.is_request();
                return Err(EmbedCallError {
                    message: format!("embedding request failed: {}", err),
                    retryable,
                    ..Default::default()
                });
            }
        };

        let status = response.status();
        let retry_after = retry_after_header(response.headers());
        let body = match response.text().await {
            Ok(body) => body,
            Err(err) => {
                return Err(EmbedCallError {
                    message: format!(
                        "embedding response read failed (status {}): {}",
                        status, err
                    ),
                    retryable: status.is_server_error(),
                    rate_limited: status.as_u16() == 429,
                    retry_after,
                });
            }
        };

        if !status.is_success() {
            let message = parse_google_error_message(&body).unwrap_or(body);
            return Err(EmbedCallError {
                message: format!(
                    "embedding request failed with status {}: {}",
                    status, message
                ),
                retryable: status.is_server_error(),
                rate_limited: status.as_u16() == 429,
                retry_after,
            });
        }

        let parsed: GeminiEmbedResponse =
            serde_json::from_str(&body).map_err(|err| EmbedCallError {
                message: format!("embedding response decode failed: {}", err),
                retryable: false,
                ..Default::default()
            })?;

        validate_and_normalize_embedding(parsed.embedding.values, expected_dim).map_err(|message| {
            EmbedCallError {
                message,
                retryable: false,
                ..Default::default()
            }
        })
    }

    async fn embed_openai_once(
        &self,
        input: &str,
        expected_dim: usize,
    ) -> std::result::Result<Vec<f32>, EmbedCallError> {
        let response = self
            .http
            .post(format!("{}/embeddings", self.config.base_url))
            .bearer_auth(&self.config.api_key)
            .json(&build_openai_request(
                &self.config.model,
                input,
                expected_dim,
            ))
            .send()
            .await;
        let response = match response {
            Ok(response) => response,
            Err(err) => {
                let retryable = err.is_timeout() || err.is_connect() || err.is_request();
                return Err(EmbedCallError {
                    message: format!("embedding request failed: {}", err),
                    retryable,
                    ..Default::default()
                });
            }
        };

        let status = response.status();
        let retry_after = retry_after_header(response.headers());
        let body = match response.text().await {
            Ok(body) => body,
            Err(err) => {
                return Err(EmbedCallError {
                    message: format!(
                        "embedding response read failed (status {}): {}",
                        status, err
                    ),
                    retryable: status.is_server_error(),
                    rate_limited: status.as_u16() == 429,
                    retry_after,
                });
            }
        };

        if !status.is_success() {
            let message = parse_openai_error_message(&body).unwrap_or(body);
            return Err(EmbedCallError {
                message: format!(
                    "embedding request failed with status {}: {}",
                    status, message
                ),
                retryable: status.is_server_error(),
                rate_limited: status.as_u16() == 429,
                retry_after,
            });
        }

        let parsed: OpenAiEmbeddingResponse =
            serde_json::from_str(&body).map_err(|err| EmbedCallError {
                message: format!("embedding response decode failed: {}", err),
                retryable: false,
                ..Default::default()
            })?;

        // The query path embeds exactly one string, so expect one datum at index 0.
        let datum = parsed
            .data
            .into_iter()
            .find(|d| d.index == 0)
            .ok_or_else(|| EmbedCallError {
                message: "embedding response missing data[0]".to_string(),
                retryable: false,
                ..Default::default()
            })?;

        validate_and_normalize_embedding(datum.embedding, expected_dim).map_err(|message| {
            EmbedCallError {
                message,
                retryable: false,
                ..Default::default()
            }
        })
    }
}

/// The typed refusal of a call made while the provider refuses for rate:
/// nothing was sent, and the caller is told when to try again.
fn rate_limited(wait: Duration) -> OmniError {
    OmniError::EmbeddingProviderRateLimited {
        retry_after_ms: u64::try_from(wait.as_millis()).unwrap_or(u64::MAX).max(1),
    }
}

/// A `Retry-After` given in seconds. The HTTP-date form is not honoured: the
/// call then gets the default cooldown and is not retried.
fn retry_after_header(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    let seconds: u64 = headers
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse()
        .ok()?;
    Some(Duration::from_secs(seconds).min(MAX_RETRY_AFTER))
}

/// Equal jitter: half the ceiling plus a random share of the other half, so
/// clients that failed together do not retry together.
fn jittered(ceiling_ms: u64) -> u64 {
    if ceiling_ms < 2 {
        return ceiling_ms;
    }
    let half = ceiling_ms / 2;
    // A process-random hasher seeded per call: no clock correlation between
    // callers that failed together.
    let entropy = {
        use std::hash::BuildHasher;
        std::collections::hash_map::RandomState::new().hash_one(std::time::Instant::now())
    };
    half + entropy % (half + 1)
}

fn gemini_endpoint(base_url: &str, model: &str) -> String {
    format!(
        "{}/models/{}:embedContent",
        base_url.trim_end_matches('/'),
        model
    )
}

fn build_gemini_request(model: &str, input: &str, expected_dim: usize, task_type: &str) -> Value {
    json!({
        "model": format!("models/{}", model),
        "content": {
            "parts": [
                {
                    "text": input
                }
            ]
        },
        "taskType": task_type,
        "outputDimensionality": expected_dim,
    })
}

fn build_openai_request(model: &str, input: &str, expected_dim: usize) -> Value {
    json!({
        "model": model,
        "input": [input],
        "dimensions": expected_dim,
    })
}

fn validate_and_normalize_embedding(
    values: Vec<f32>,
    expected_dim: usize,
) -> std::result::Result<Vec<f32>, String> {
    if values.len() != expected_dim {
        return Err(format!(
            "embedding dimension mismatch: expected {}, got {}",
            expected_dim,
            values.len()
        ));
    }
    // Reject poisoned vectors BEFORE normalizing: a NaN component makes the
    // norm NaN (whose `> EPSILON` comparison is false, silently skipping
    // normalization), an Inf component normalizes the rest to 0s and itself
    // to NaN, and a zero vector has no direction — all three would persist
    // as "successful" embeddings and degrade ANN/IVF for unrelated rows.
    if let Some(idx) = values.iter().position(|v| !v.is_finite()) {
        return Err(format!(
            "embedding component {} is not finite ({})",
            idx, values[idx]
        ));
    }
    let norm = values
        .iter()
        .map(|v| (*v as f64) * (*v as f64))
        .sum::<f64>()
        .sqrt() as f32;
    if norm <= f32::EPSILON {
        return Err("embedding has zero norm (no direction)".to_string());
    }
    // Normalize in place with the norm just computed — `normalize_vector`
    // would recompute it (and its zero guard is unreachable here, since a
    // zero norm already returned Err above).
    let mut values = values;
    for value in &mut values {
        *value /= norm;
    }
    Ok(values)
}

fn normalize_vector(mut values: Vec<f32>) -> Vec<f32> {
    let norm = values
        .iter()
        .map(|v| (*v as f64) * (*v as f64))
        .sum::<f64>()
        .sqrt() as f32;
    if norm > f32::EPSILON {
        for value in &mut values {
            *value /= norm;
        }
    }
    values
}

fn parse_google_error_message(body: &str) -> Option<String> {
    serde_json::from_str::<GoogleErrorEnvelope>(body)
        .ok()
        .map(|e| e.error.message)
        .filter(|msg| !msg.trim().is_empty())
}

fn parse_openai_error_message(body: &str) -> Option<String> {
    serde_json::from_str::<OpenAiErrorEnvelope>(body)
        .ok()
        .map(|e| e.error.message)
        .filter(|msg| !msg.trim().is_empty())
}

/// Map a provider alias to `(provider, default base URL, default model, ordered
/// api-key envs)`. Shared by `from_env` and `from_parts` so both apply identical
/// defaults: `openai-compatible`/unset → the OpenRouter gateway, `openai` →
/// OpenAI's own host. `mock` is handled by callers before this is reached. The
/// `Provider` enum alone would collapse the two openai aliases, so the alias
/// (not the enum) determines the key-env order here.
fn provider_profile(
    alias: Option<&str>,
) -> Result<(
    Provider,
    &'static str,
    &'static str,
    &'static [&'static str],
)> {
    Ok(match alias {
        None | Some("openai-compatible") => (
            Provider::OpenAiCompatible,
            DEFAULT_OPENROUTER_BASE_URL,
            DEFAULT_OPENROUTER_MODEL,
            &["OPENROUTER_API_KEY", "OPENAI_API_KEY"],
        ),
        Some("openai") => (
            Provider::OpenAiCompatible,
            DEFAULT_OPENAI_BASE_URL,
            DEFAULT_OPENAI_MODEL,
            &["OPENAI_API_KEY"],
        ),
        Some("gemini") => (
            Provider::Gemini,
            DEFAULT_GEMINI_BASE_URL,
            DEFAULT_GEMINI_MODEL,
            &["GEMINI_API_KEY"],
        ),
        Some(other) => {
            return Err(OmniError::manifest_internal(format!(
                "unknown embedding provider '{}' (expected openai-compatible|openai|gemini|mock)",
                other
            )));
        }
    })
}

fn env_string(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

fn parse_env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(default)
}

fn parse_env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(default)
}

/// Like [`parse_env_u64`] but accepts `0` as a meaningful value (the deadline
/// uses `0` for "unbounded").
fn parse_env_u64_allow_zero(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(default)
}

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| {
            let s = v.trim().to_ascii_lowercase();
            s == "1" || s == "true" || s == "yes" || s == "on"
        })
        .unwrap_or(false)
}

fn mock_embedding(input: &str, dim: usize) -> Vec<f32> {
    let mut seed = fnv1a64(input.as_bytes());
    let mut out = Vec::with_capacity(dim);
    for _ in 0..dim {
        seed = xorshift64(seed);
        let ratio = (seed as f64 / u64::MAX as f64) as f32;
        out.push((ratio * 2.0) - 1.0);
    }
    normalize_vector(out)
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 14695981039346656037u64;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(1099511628211u64);
    }
    hash
}

fn xorshift64(mut x: u64) -> u64 {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use serial_test::serial;

    use super::*;

    /// A loopback provider that answers every request with the given status
    /// and headers in turn (the last one repeats) and counts the requests.
    async fn provider_stub(answers: Vec<(u16, &'static str)>) -> (String, Arc<AtomicUsize>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}/v1", listener.local_addr().unwrap());
        let calls = Arc::new(AtomicUsize::new(0));
        let seen = Arc::clone(&calls);
        tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    return;
                };
                let n = seen.fetch_add(1, Ordering::SeqCst);
                let (status, headers) = answers[n.min(answers.len() - 1)];
                let mut buf = vec![0u8; 16 * 1024];
                let _ = socket.read(&mut buf).await;
                let body = if status == 200 {
                    r#"{"data":[{"index":0,"embedding":[0.6,0.8]}]}"#
                } else {
                    r#"{"error":{"message":"slow down"}}"#
                };
                let response = format!(
                    "HTTP/1.1 {status} X\r\ncontent-type: application/json\r\n{headers}content-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = socket.write_all(response.as_bytes()).await;
            }
        });
        (base, calls)
    }

    fn stub_client(base: String) -> EmbeddingClient {
        let mut client = EmbeddingClient::new(
            EmbeddingConfig::from_parts(
                Some("openai-compatible"),
                Some(base),
                Some("stub".to_string()),
                "dummy".to_string(),
            )
            .unwrap(),
        )
        .unwrap();
        client.retry_attempts = 4;
        client.retry_backoff_ms = 1;
        client
    }

    /// A query's 429 with no `Retry-After` is not retried, and the cooldown
    /// it starts makes the next call on the same client a typed refusal sent
    /// without a request: a rate refusal is never multiplied by the retry loop
    /// or by concurrent callers.
    #[tokio::test]
    async fn rate_refusal_without_retry_after_is_not_retried_and_cools_down() {
        let (base, calls) = provider_stub(vec![(429, "")]).await;
        let client = stub_client(base);
        for _ in 0..2 {
            let refused = client.embed_query_text("q", 2).await.unwrap_err();
            assert!(
                matches!(refused, OmniError::EmbeddingProviderRateLimited { retry_after_ms } if retry_after_ms >= 1),
                "{refused:?}"
            );
            assert_eq!(
                calls.load(Ordering::SeqCst),
                1,
                "the cooldown sent no request"
            );
        }
    }

    /// A query retries a 429 that names `Retry-After` once, after at least the
    /// one-second floor (`Retry-After: 0` included), and a transient 5xx is
    /// still retried with backoff. A second 429 is not retried again.
    #[tokio::test]
    async fn query_retries_a_named_rate_refusal_once() {
        let (base, calls) =
            provider_stub(vec![(429, "retry-after: 0\r\n"), (503, ""), (200, "")]).await;
        let client = stub_client(base);
        let started = std::time::Instant::now();
        let vector = client.embed_query_text("q", 2).await.unwrap();
        assert_eq!(vector.len(), 2);
        assert_eq!(calls.load(Ordering::SeqCst), 3);
        assert!(started.elapsed() >= RATE_LIMIT_COOLDOWN_FLOOR);

        let (base, calls) = provider_stub(vec![
            (429, "retry-after: 1\r\n"),
            (429, "retry-after: 1\r\n"),
        ])
        .await;
        let client = stub_client(base);
        let refused = client.embed_query_text("q", 2).await.unwrap_err();
        assert!(
            matches!(refused, OmniError::EmbeddingProviderRateLimited { .. }),
            "{refused:?}"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    /// A document embedding, a batch the operator started, waits out a rate
    /// refusal with no `Retry-After` and goes on.
    #[tokio::test]
    async fn document_embedding_waits_out_a_rate_refusal() {
        let (base, calls) = provider_stub(vec![(429, ""), (200, "")]).await;
        let client = stub_client(base);
        let vector = client.embed_document_text("d", 2).await.unwrap();
        assert_eq!(vector.len(), 2);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    /// A wait longer than what is left of the call's deadline is refused at
    /// once, never slept into the deadline.
    #[tokio::test]
    async fn a_wait_past_the_deadline_is_refused_at_once() {
        let (base, calls) = provider_stub(vec![(429, "retry-after: 120\r\n")]).await;
        let mut client = stub_client(base);
        client.deadline_ms = 5_000;
        let started = std::time::Instant::now();
        let refused = client.embed_document_text("d", 2).await.unwrap_err();
        assert!(
            matches!(
                refused,
                OmniError::EmbeddingProviderRateLimited {
                    retry_after_ms: 120_000
                }
            ),
            "{refused:?}"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(started.elapsed() < Duration::from_secs(2));
    }

    #[test]
    fn retry_after_is_whole_seconds_capped_at_five_minutes() {
        let header = |value: &str| {
            let mut headers = reqwest::header::HeaderMap::new();
            headers.insert(reqwest::header::RETRY_AFTER, value.parse().unwrap());
            retry_after_header(&headers)
        };
        assert_eq!(header("5"), Some(Duration::from_secs(5)));
        assert_eq!(header(" 7 "), Some(Duration::from_secs(7)));
        assert_eq!(header("86400"), Some(MAX_RETRY_AFTER));
        assert_eq!(header("Wed, 21 Oct 2015 07:28:00 GMT"), None);
        assert_eq!(header("-1"), None);
    }

    struct EnvGuard {
        saved: Vec<(&'static str, Option<String>)>,
    }

    impl EnvGuard {
        fn set(vars: &[(&'static str, Option<&str>)]) -> Self {
            let saved = vars
                .iter()
                .map(|(name, _)| (*name, std::env::var(name).ok()))
                .collect::<Vec<_>>();
            for (name, value) in vars {
                unsafe {
                    match value {
                        Some(value) => std::env::set_var(name, value),
                        None => std::env::remove_var(name),
                    }
                }
            }
            Self { saved }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (name, value) in self.saved.drain(..) {
                unsafe {
                    match value {
                        Some(value) => std::env::set_var(name, value),
                        None => std::env::remove_var(name),
                    }
                }
            }
        }
    }

    // Every test that calls `EmbeddingConfig::from_env` clears the full set of
    // embedding env vars first so the host environment can't leak in.
    const EMBED_ENV: &[&str] = &[
        "OMNIGRAPH_EMBEDDINGS_MOCK",
        "OMNIGRAPH_EMBED_PROVIDER",
        "OMNIGRAPH_EMBED_BASE_URL",
        "OMNIGRAPH_EMBED_MODEL",
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
    ];

    fn cleared_env(extra: &[(&'static str, Option<&str>)]) -> EnvGuard {
        let mut vars: Vec<(&'static str, Option<&str>)> =
            EMBED_ENV.iter().map(|n| (*n, None)).collect();
        vars.extend_from_slice(extra);
        EnvGuard::set(&vars)
    }

    #[tokio::test]
    async fn mock_embeddings_are_deterministic() {
        let client = EmbeddingClient::mock_for_tests();
        let a = client.embed_query_text("alpha", 8).await.unwrap();
        let b = client.embed_query_text("alpha", 8).await.unwrap();
        let c = client.embed_query_text("beta", 8).await.unwrap();
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(a.len(), 8);
    }

    #[test]
    fn gemini_request_uses_model_retrieval_query_and_dimension() {
        let request =
            build_gemini_request("gemini-embedding-2", "alpha", 4, GEMINI_QUERY_TASK_TYPE);
        assert_eq!(request["model"], "models/gemini-embedding-2");
        assert_eq!(request["taskType"], GEMINI_QUERY_TASK_TYPE);
        assert_eq!(request["outputDimensionality"], 4);
        assert_eq!(request["content"]["parts"][0]["text"], "alpha");
    }

    #[test]
    fn gemini_document_request_uses_retrieval_document_task_type() {
        let request =
            build_gemini_request("gemini-embedding-2", "alpha", 4, GEMINI_DOCUMENT_TASK_TYPE);
        assert_eq!(request["taskType"], GEMINI_DOCUMENT_TASK_TYPE);
    }

    #[test]
    fn openai_request_uses_model_input_array_and_dimensions() {
        let request = build_openai_request("openai/text-embedding-3-large", "alpha", 4);
        assert_eq!(request["model"], "openai/text-embedding-3-large");
        assert_eq!(request["input"][0], "alpha");
        assert!(request["input"].is_array());
        assert_eq!(request["dimensions"], 4);
        assert!(request.get("taskType").is_none());
    }

    #[test]
    fn validate_and_normalize_embedding_rejects_non_finite_and_zero() {
        // iss-embedding-nan-validation: a NaN component must be rejected —
        // not silently returned un-normalized (NaN > EPSILON is false, so the
        // normalize guard inverts and the raw NaN vector came back Ok).
        let err = validate_and_normalize_embedding(vec![f32::NAN, 1.0], 2).unwrap_err();
        assert!(err.contains("finite"), "NaN must be rejected, got: {err}");
        // An Inf component must be rejected — not normalized into 0s + NaN.
        let err = validate_and_normalize_embedding(vec![f32::INFINITY, 1.0], 2).unwrap_err();
        assert!(err.contains("finite"), "Inf must be rejected, got: {err}");
        let err = validate_and_normalize_embedding(vec![1.0, f32::NEG_INFINITY], 2).unwrap_err();
        assert!(err.contains("finite"), "-Inf must be rejected, got: {err}");
        // An all-zero vector has no direction — undefined under cosine/ANN.
        let err = validate_and_normalize_embedding(vec![0.0, 0.0], 2).unwrap_err();
        assert!(
            err.contains("zero"),
            "zero vector must be rejected, got: {err}"
        );
    }

    #[test]
    fn validate_and_normalize_embedding_enforces_dimension() {
        let normalized = validate_and_normalize_embedding(vec![3.0, 4.0], 2).unwrap();
        assert!((normalized[0] - 0.6).abs() < 1e-6);
        assert!((normalized[1] - 0.8).abs() < 1e-6);

        let err = validate_and_normalize_embedding(vec![1.0, 2.0], 3).unwrap_err();
        assert!(err.contains("expected 3, got 2"));
    }

    #[tokio::test]
    async fn with_retry_retries_retryable_failures() {
        let client = EmbeddingClient::mock_for_tests();
        let attempts = Arc::new(AtomicUsize::new(0));
        let attempts_for_call = Arc::clone(&attempts);

        let value = client
            .with_retry(EmbedRole::Query, || {
                let attempts_for_call = Arc::clone(&attempts_for_call);
                async move {
                    let attempt = attempts_for_call.fetch_add(1, Ordering::SeqCst);
                    if attempt == 0 {
                        Err(EmbedCallError {
                            message: "retry me".to_string(),
                            retryable: true,
                            ..Default::default()
                        })
                    } else {
                        Ok("ok")
                    }
                }
            })
            .await
            .unwrap();

        assert_eq!(value, "ok");
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn with_retry_stops_on_non_retryable_failures() {
        let client = EmbeddingClient::mock_for_tests();
        let err = client
            .with_retry(EmbedRole::Query, || async {
                Err::<(), _>(EmbedCallError {
                    message: "do not retry".to_string(),
                    retryable: false,
                    ..Default::default()
                })
            })
            .await
            .unwrap_err();

        assert!(err.to_string().contains("do not retry"));
    }

    #[tokio::test]
    async fn run_with_deadline_aborts_slow_future() {
        let mut client = EmbeddingClient::mock_for_tests();
        client.deadline_ms = 20;
        let slow = async {
            tokio::time::sleep(Duration::from_secs(5)).await;
            Ok(vec![0.0_f32])
        };
        let err = client.run_with_deadline(slow).await.unwrap_err();
        assert!(err.to_string().contains("deadline exceeded"));
    }

    #[tokio::test]
    async fn run_with_deadline_passes_through_fast_future() {
        let client = EmbeddingClient::mock_for_tests();
        let ok = client
            .run_with_deadline(async { Ok(vec![1.0_f32, 2.0]) })
            .await
            .unwrap();
        assert_eq!(ok, vec![1.0, 2.0]);
    }

    #[tokio::test]
    async fn run_with_deadline_zero_is_unbounded() {
        let mut client = EmbeddingClient::mock_for_tests();
        client.deadline_ms = 0;
        let ok = client
            .run_with_deadline(async { Ok(vec![3.0_f32]) })
            .await
            .unwrap();
        assert_eq!(ok, vec![3.0]);
    }

    #[test]
    #[serial]
    fn from_env_defaults_to_openai_compatible_openrouter() {
        let _guard = cleared_env(&[("OPENROUTER_API_KEY", Some("sk-test"))]);
        let config = EmbeddingConfig::from_env().unwrap();
        assert_eq!(config.provider, Provider::OpenAiCompatible);
        assert_eq!(config.base_url, DEFAULT_OPENROUTER_BASE_URL);
        assert_eq!(config.model, DEFAULT_OPENROUTER_MODEL);
        assert_eq!(config.api_key, "sk-test");
    }

    #[test]
    #[serial]
    fn from_env_openai_alias_uses_openai_host_not_openrouter() {
        let _guard = cleared_env(&[
            ("OMNIGRAPH_EMBED_PROVIDER", Some("openai")),
            ("OPENAI_API_KEY", Some("k")),
        ]);
        let config = EmbeddingConfig::from_env().unwrap();
        assert_eq!(config.provider, Provider::OpenAiCompatible);
        assert_eq!(config.base_url, DEFAULT_OPENAI_BASE_URL); // api.openai.com, not OpenRouter
        assert_eq!(config.model, DEFAULT_OPENAI_MODEL); // text-embedding-3-large, no openai/ prefix
        assert_eq!(config.api_key, "k");
    }

    #[test]
    #[serial]
    fn from_env_openai_alias_prefers_openai_key_over_openrouter() {
        // `openai` targets api.openai.com, so an OpenRouter key must not be sent there.
        let _guard = cleared_env(&[
            ("OMNIGRAPH_EMBED_PROVIDER", Some("openai")),
            ("OPENROUTER_API_KEY", Some("router")),
            ("OPENAI_API_KEY", Some("openai")),
        ]);
        let config = EmbeddingConfig::from_env().unwrap();
        assert_eq!(config.base_url, DEFAULT_OPENAI_BASE_URL);
        assert_eq!(config.api_key, "openai");
    }

    #[test]
    #[serial]
    fn from_env_openai_alias_errors_when_only_openrouter_key_is_set() {
        let _guard = cleared_env(&[
            ("OMNIGRAPH_EMBED_PROVIDER", Some("openai")),
            ("OPENROUTER_API_KEY", Some("router")),
        ]);
        let err = EmbeddingConfig::from_env().unwrap_err();
        assert!(err.to_string().contains("OPENAI_API_KEY"), "got: {err}");
    }

    #[test]
    fn from_parts_applies_provider_defaults_and_overrides() {
        let openrouter = EmbeddingConfig::from_parts(None, None, None, "k".to_string()).unwrap();
        assert_eq!(openrouter.provider, Provider::OpenAiCompatible);
        assert_eq!(openrouter.base_url, DEFAULT_OPENROUTER_BASE_URL);
        assert_eq!(openrouter.model, DEFAULT_OPENROUTER_MODEL);
        assert_eq!(openrouter.api_key, "k");

        let gemini =
            EmbeddingConfig::from_parts(Some("gemini"), None, None, "g".to_string()).unwrap();
        assert_eq!(gemini.provider, Provider::Gemini);
        assert_eq!(gemini.base_url, DEFAULT_GEMINI_BASE_URL);

        let overridden = EmbeddingConfig::from_parts(
            Some("openai"),
            Some("https://x/v1/".to_string()),
            Some("custom".to_string()),
            "k".to_string(),
        )
        .unwrap();
        assert_eq!(overridden.base_url, "https://x/v1"); // trailing slash trimmed
        assert_eq!(overridden.model, "custom");

        let err =
            EmbeddingConfig::from_parts(Some("cohere"), None, None, "k".to_string()).unwrap_err();
        assert!(
            err.to_string().contains("unknown embedding provider"),
            "got: {err}"
        );
    }

    #[test]
    #[serial]
    fn from_parts_mock_honors_an_explicit_model() {
        // A cluster `providers.embedding` profile that sets `kind: mock, model: X`
        // must resolve to model X — it is what the query-time same-space check
        // compares against. Env cleared so the assertion isolates the arg.
        let _guard = cleared_env(&[]);
        let pinned = EmbeddingConfig::from_parts(
            Some("mock"),
            None,
            Some("recorded-x".to_string()),
            String::new(),
        )
        .unwrap();
        assert_eq!(pinned.provider, Provider::Mock);
        assert_eq!(pinned.model, "recorded-x");
        // With no explicit model, mock falls back to its env-based default (here
        // empty, since the env is cleared).
        let bare = EmbeddingConfig::from_parts(Some("mock"), None, None, String::new()).unwrap();
        assert_eq!(bare.provider, Provider::Mock);
        assert_eq!(bare.model, "");
    }

    #[test]
    #[serial]
    fn from_env_openai_compatible_prefers_openrouter_key() {
        let _guard = cleared_env(&[
            ("OPENROUTER_API_KEY", Some("router")),
            ("OPENAI_API_KEY", Some("openai")),
        ]);
        let config = EmbeddingConfig::from_env().unwrap();
        assert_eq!(config.api_key, "router");
    }

    #[test]
    #[serial]
    fn from_env_explicit_gemini_provider() {
        let _guard = cleared_env(&[
            ("OMNIGRAPH_EMBED_PROVIDER", Some("gemini")),
            ("GEMINI_API_KEY", Some("g-key")),
        ]);
        let config = EmbeddingConfig::from_env().unwrap();
        assert_eq!(config.provider, Provider::Gemini);
        assert_eq!(config.base_url, DEFAULT_GEMINI_BASE_URL);
        assert_eq!(config.model, DEFAULT_GEMINI_MODEL);
        assert_eq!(config.api_key, "g-key");
    }

    #[test]
    #[serial]
    fn from_env_base_url_and_model_overrides_apply() {
        let _guard = cleared_env(&[
            ("OMNIGRAPH_EMBED_PROVIDER", Some("openai-compatible")),
            ("OMNIGRAPH_EMBED_BASE_URL", Some("https://example.test/v1/")),
            ("OMNIGRAPH_EMBED_MODEL", Some("custom/model")),
            ("OPENAI_API_KEY", Some("k")),
        ]);
        let config = EmbeddingConfig::from_env().unwrap();
        assert_eq!(config.base_url, "https://example.test/v1"); // trailing slash trimmed
        assert_eq!(config.model, "custom/model");
    }

    #[test]
    #[serial]
    fn from_env_unknown_provider_errors() {
        let _guard = cleared_env(&[("OMNIGRAPH_EMBED_PROVIDER", Some("cohere"))]);
        let err = EmbeddingConfig::from_env().unwrap_err();
        assert!(err.to_string().contains("unknown embedding provider"));
    }

    #[test]
    #[serial]
    fn from_env_errors_when_no_key_present() {
        let _guard = cleared_env(&[]);
        let err = EmbeddingConfig::from_env().unwrap_err();
        assert!(
            err.to_string()
                .contains("OPENROUTER_API_KEY or OPENAI_API_KEY")
        );
    }

    #[test]
    #[serial]
    fn from_env_mock_flag_wins() {
        let _guard = cleared_env(&[
            ("OMNIGRAPH_EMBEDDINGS_MOCK", Some("1")),
            ("OMNIGRAPH_EMBED_PROVIDER", Some("gemini")),
        ]);
        let config = EmbeddingConfig::from_env().unwrap();
        assert_eq!(config.provider, Provider::Mock);
    }
}
