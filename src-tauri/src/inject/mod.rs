#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "windows")]
mod windows;

#[cfg(target_os = "macos")]
use macos::MacosTextInjector as PlatformTextInjector;
#[cfg(target_os = "windows")]
use windows::WindowsTextInjector as PlatformTextInjector;

#[derive(Debug, thiserror::Error)]
pub(crate) enum InjectError {
    #[error("accessibility permission denied: {0}")]
    PermissionDenied(String),
    #[error("clipboard error: {0}")]
    Clipboard(String),
    #[error("input simulation error: {0}")]
    Input(String),
}

pub(crate) trait TextInject: Send + Sync {
    fn type_text(&self, text: &str) -> Result<(), InjectError>;
    fn request_accessibility(&self) -> bool;
}

pub(crate) struct TextInjector {
    inner: PlatformTextInjector,
}

impl TextInjector {
    pub(crate) fn new() -> Self {
        Self {
            inner: PlatformTextInjector::new(),
        }
    }

    pub(crate) fn type_text(&self, text: &str) -> Result<(), InjectError> {
        self.inner.type_text(text)
    }

    pub(crate) fn request_accessibility(&self) -> bool {
        self.inner.request_accessibility()
    }
}

impl Default for TextInjector {
    fn default() -> Self {
        Self::new()
    }
}
