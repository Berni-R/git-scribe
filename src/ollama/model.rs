use std::time::Duration;

use serde::Deserialize;

use crate::ollama::client::{Client, OllamaError};

/// A Ollama model description.
#[derive(Debug, Deserialize)]
pub struct Model {
    /// The model's name.
    pub model: String,
    // URL of the upstream Ollama host, if the model is remote.
    remote_host: Option<String>,
    /// Total size of the model on disk in bytes.
    pub size: u64,
}

impl Model {
    /// `true` if this is a local model.
    #[must_use]
    pub fn is_local(&self) -> bool {
        self.remote_host.is_none()
    }
}

#[derive(Debug, Deserialize)]
struct ListModelsResponse {
    models: Vec<Model>,
}

impl Client {
    /// Request the list of available Ollama models (with a timeout of 3 seconds).
    pub fn list_models(&self) -> Result<Vec<Model>, OllamaError> {
        let response = self
            .client
            .get(format!("{}/api/tags", self.base_url))
            .timeout(Duration::from_secs(3));

        Ok(Self::call::<ListModelsResponse>(response)?.models)
    }
}
