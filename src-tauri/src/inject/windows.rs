use super::{InjectError, TextInject};

pub(crate) struct WindowsTextInjector;

impl WindowsTextInjector {
    pub(crate) fn new() -> Self {
        Self
    }
}

impl TextInject for WindowsTextInjector {
    fn type_text(&self, _text: &str) -> Result<(), InjectError> {
        Err(InjectError::Input(
            "Windows text injection is not implemented yet".into(),
        ))
    }

    fn request_accessibility(&self) -> bool {
        log::warn!("Windows text injection is not implemented yet");
        false
    }
}
