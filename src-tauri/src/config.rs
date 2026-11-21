use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct Config {
    hotkey: String,
    audio_device: Option<String>,
    model_dir: PathBuf,
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
    pub(crate) fn load(resource_dir: &Path) -> Self {
        let config_path = dirs::config_dir().map(|d| d.join("talksify").join("config.toml"));

        let mut cfg: Self = config_path
            .as_ref()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|s| toml::from_str(&s).ok())
            .unwrap_or_default();

        cfg.model_dir = resource_dir.join("models");
        cfg
    }

    pub(crate) fn hotkey(&self) -> &str {
        &self.hotkey
    }

    pub(crate) fn audio_device(&self) -> Option<&str> {
        self.audio_device.as_deref()
    }

    pub(crate) fn model_dir(&self) -> &Path {
        &self.model_dir
    }
}
