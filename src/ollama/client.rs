use anyhow::Result;
use reqwest::{
    StatusCode,
    blocking::{RequestBuilder, Response},
};
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

    #[error("failed to read Ollama's streamed response: {0}")]
    Stream(#[from] std::io::Error),

    #[error("Ollama returned HTTP {status}: {message}")]
    Api { status: StatusCode, message: String },

    #[error("model `{model}` not found; available models: {available_models:?}")]
    ModelNotFound {
        model: String,
        available_models: Vec<String>,
    },

    #[error("failed to decode Ollama response: {0}")]
    Decode(#[from] serde_json::Error),

    #[error("Ollama ended a streamed response before reporting completion")]
    IncompleteStream,
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
    /// ```rust,ignore
    /// let response = self.client
    ///     .get(format!("{}/api/tags", self.base_url))
    ///     .timeout(Duration::from_secs(3));
    /// let models: ListModelsResponse = Client::call(response)?;
    /// ```
    pub(super) fn call<T>(request: RequestBuilder) -> Result<T, OllamaError>
    where
        T: DeserializeOwned,
    {
        let response = Self::send(request)?;
        let body = response.bytes()?;

        Ok(serde_json::from_slice(&body)?)
    }

    /// Send a request and return a successful response body without consuming it.
    ///
    /// Streaming endpoints use this to check API errors before incrementally reading the body.
    pub(super) fn send(request: RequestBuilder) -> Result<Response, OllamaError> {
        let response = request.send()?;
        let status = response.status();

        if !status.is_success() {
            let body = response.bytes()?;
            let message = serde_json::from_slice::<ErrorResponse>(&body).map_or_else(
                |_| String::from_utf8_lossy(&body).into_owned(),
                |response| response.error,
            );

            return Err(OllamaError::Api { status, message });
        }

        Ok(response)
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

pub(super) fn normalize_model_name(model: &str) -> &str {
    model.strip_suffix(":latest").unwrap_or(model)
}
