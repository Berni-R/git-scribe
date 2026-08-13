use anyhow::{Result, bail};
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Debug, Deserialize)]
pub struct CommitMessage {
    /// Coherent purpose or project-level effect of the staged changes.
    pub intent: String,

    /// Important concrete changes that support the inferred intent.
    pub key_changes: Vec<String>,

    /// Nature of the change as a whole.
    pub change_kind: ChangeKind,

    /// Qualitative confidence in the inferred intent.
    pub confidence: Confidence,

    /// Final commit subject.
    pub subject: String,

    /// Additional information not already expressed by the subject.
    pub body: Option<String>,
}

/// High-level classification of the commit as a whole.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChangeKind {
    /// Introduces a new capability or feature.
    Feature,

    /// Intentionally changes existing behavior without primarily fixing a bug.
    BehaviorChange,

    /// Corrects unintended or erroneous behavior.
    BugFix,

    /// Restructures implementation without intentionally changing behavior or capabilities.
    Refactor,

    /// Primarily adds, removes, or changes tests.
    Tests,

    /// Primarily changes documentation.
    Documentation,

    /// Changes development, build, release, or repository tooling.
    Tooling,

    /// Primarily changes configuration.
    Configuration,

    /// Combines multiple change kinds without a single clearly dominant one.
    Mixed,
}

/// Qualitative confidence that the supplied repository evidence supports the inferred intent.
///
/// The variants are ordered from lowest to highest confidence.
#[derive(Debug, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "lowercase")]
pub enum Confidence {
    /// The inferred intent is weakly supported or substantially uncertain.
    Low,

    /// The inferred intent is reasonably supported but contains some uncertainty.
    Medium,

    /// The inferred intent is strongly supported by the supplied evidence.
    High,
}

impl CommitMessage {
    /// Return the normalized commit subject produced by the model.
    ///
    /// Leading and trailing whitespace is removed from each line, empty lines are discarded,
    /// and remaining lines are joined with spaces. Returns an error if the resulting subject is empty.
    pub fn normalized_subject(&self) -> Result<String> {
        let subject = self
            .subject
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join(" ");

        if subject.is_empty() {
            bail!("model returned an empty subject");
        }

        Ok(subject)
    }

    /// Return the normalized commit body, if the model provided a non-empty one.
    ///
    /// Leading and trailing whitespace is removed. A missing or whitespace-only body is returned as `None`.
    pub fn normalized_body(&self) -> Option<&str> {
        self.body
            .as_deref()
            .map(str::trim)
            .filter(|body| !body.is_empty())
    }

    /// Return the JSON schema expected for the model's structured response.
    ///
    /// The schema constrains the model to a [`CommitMessage`], describing its attributes meaning, types, etc..
    #[must_use]
    pub fn schema() -> Value {
        json!({
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 240,
                    "description": "The coherent purpose or project-level effect of the staged changes as a whole."
                },
                "key_changes": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 4,
                    "items": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 180,
                    },
                    "description": "The most important concrete changes that support the inferred intent. Focus on distinct, relevant changes rather than restating the diff."
                },
                "change_kind": {
                    "type": "string",
                    "enum": [
                        "feature",
                        "behavior_change",
                        "bug_fix",
                        "refactor",
                        "tests",
                        "documentation",
                        "tooling",
                        "configuration",
                        "mixed",
                    ],
                    "description": "The best high-level classification of the commit as a whole. \
                        Use feature for a new capability; \
                        behavior_change for an intentional change to existing behavior; \
                        bug_fix for correcting unintended behavior; \
                        refactor for internal restructuring without an intentional change in behavior or capabilities; \
                        tests for primarily test changes; documentation for primarily documentation changes; \
                        tooling for changes to development, build, release, or repository tooling rather than product functionality; \
                        configuration for primarily configuration changes; \
                        mixed when no single category clearly dominates."
                },
                "confidence": {
                    "type": "string",
                    "enum": [
                        "low",
                        "medium",
                        "high",
                    ],
                    "description": "Qualitative confidence that the supplied repository evidence supports the inferred intent."
                },
                "subject": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 100,
                    "description": "Imperative Git commit subject describing the coherent purpose of the change.",
                },
                "body": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 700,
                    "description": "Optional commit body containing useful information not already conveyed by the subject.",
                }
            },
            "required": [
                "intent",
                "key_changes",
                "change_kind",
                "confidence",
                "subject",
            ],
            "additionalProperties": false
        })
    }
}
