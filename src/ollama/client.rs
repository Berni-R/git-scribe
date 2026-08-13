use anyhow::Result;
use reqwest::{StatusCode, blocking::RequestBuilder};
use serde::{Deserialize, de::DeserializeOwned};
use thiserror::Error;

const DEFAULT_BASE_URL: &str = "http://localhost:11434";

/// A client for the (local) ollama instance.
#[derive(Debug)]
pub struct Client {
    pub(super) client: reqwest::blocking::Client,
    pub(super) base_url: String,
}

impl Default for Client {
    fn default() -> Self {
        Self {
            client: reqwest::blocking::Client::default(),
            base_url: DEFAULT_BASE_URL.to_owned(),
        }
    }
}

#[derive(Debug, Error)]
pub enum OllamaError {
    #[error("failed to communicate with Ollama: {0}")]
    Request(#[from] reqwest::Error),

    #[error("Ollama returned HTTP {status}: {message}")]
    Api { status: StatusCode, message: String },

    #[error("model `{model}` not found; available models: {available_models:?}")]
    ModelNotFound {
        model: String,
        available_models: Vec<String>,
    },

    #[error("failed to decode Ollama response: {0}")]
    Decode(#[from] serde_json::Error),
}

#[derive(Debug, Deserialize)]
struct ErrorResponse {
    error: String,
}

impl Client {
    /// Send the given request and parse the result, handling (HTTP and API) error in the process.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// let response = self.client
    ///     .get(format!("{}/api/tags", self.base_url))
    ///     .timeout(Duration::from_secs(3));
    /// let models: ListModelsResponse = Client::call(response)?;
    /// ```
    pub(super) fn call<T>(request: RequestBuilder) -> Result<T, OllamaError>
    where
        T: DeserializeOwned,
    {
        let response = request.send()?;
        let status = response.status();
        let body = response.bytes()?;

        if !status.is_success() {
            let message = serde_json::from_slice::<ErrorResponse>(&body).map_or_else(
                |_| String::from_utf8_lossy(&body).into_owned(),
                |response| response.error,
            );

            return Err(OllamaError::Api { status, message });
        }

        Ok(serde_json::from_slice(&body)?)
    }

    pub(super) fn resolve_model_not_found(
        &self,
        model: &str,
        original_error: OllamaError,
    ) -> OllamaError {
        let Ok(models) = self.list_models() else {
            return original_error;
        };

        if models
            .iter()
            .any(|available| normalize_model_name(&available.model) == normalize_model_name(model))
        {
            // The model exists, so this 404 apparently meant something else.
            return original_error;
        }

        let available_models = models.into_iter().map(|model| model.model).collect();

        OllamaError::ModelNotFound {
            model: model.to_owned(),
            available_models,
        }
    }
}

fn normalize_model_name(model: &str) -> &str {
    model.strip_suffix(":latest").unwrap_or(model)
}
