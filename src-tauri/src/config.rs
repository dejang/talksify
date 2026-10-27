use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct Config {
    pub(crate) hotkey: String,
    pub(crate) audio_device: Option<String>,
    pub(crate) model_dir: PathBuf,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            hotkey: "Ctrl+Alt+Space".into(),
            audio_device: None,
            model_dir: PathBuf::from("models"),
        }
    }
}

impl Config {
    pub(crate) fn load(resource_dir: &std::path::Path) -> Self {
        let config_path =
            dirs::config_dir().map(|d| d.join("talksify").join("config.toml"));

        let mut cfg: Self = config_path
            .as_ref()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|s| toml::from_str(&s).ok())
            .unwrap_or_default();

        cfg.model_dir = resource_dir.join("models");
        cfg
    }
}
