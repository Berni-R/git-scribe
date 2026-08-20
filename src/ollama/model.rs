use std::time::Duration;

use serde::Deserialize;

use crate::ollama::{
    client::{Client, OllamaError},
    normalize_model_name,
};

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

/// A model currently resident in Ollama memory.
#[derive(Debug, Deserialize)]
pub struct RunningModel {
    /// Model name as reported by Ollama.
    pub model: String,
    /// Size of the model in bytes.
    pub size: u64,
    /// Time when the model will be unloaded.
    pub expires_at: String, // TODO: conver to proper date and time / timestamp
    /// VRAM usage in bytes
    pub size_vram: u64,
    /// Context window allocated to the running model.
    pub context_length: u32,
}

#[derive(Debug, Deserialize)]
struct ListRunningModelsResponse {
    models: Vec<RunningModel>,
}

/// Returns `true` if there is the `model` contained in `models` with the given `context_length`.
#[must_use]
pub fn is_model_contained(model: &str, context_length: u32, models: &[RunningModel]) -> bool {
    models.iter().any(|running| {
        normalize_model_name(&running.model) == normalize_model_name(model)
            && running.context_length == context_length
    })
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

    /// Request models currently loaded in Ollama memory (with a timeout of 3 seconds).
    pub fn list_running_models(&self) -> Result<Vec<RunningModel>, OllamaError> {
        let response = self
            .client
            .get(format!("{}/api/ps", self.base_url))
            .timeout(Duration::from_secs(3));

        Ok(Self::call::<ListRunningModelsResponse>(response)?.models)
    }
}
