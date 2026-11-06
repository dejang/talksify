use super::{InjectError, TextInject};

pub(crate) struct LinuxTextInjector;

impl LinuxTextInjector {
    pub(crate) fn new() -> Self {
        Self
    }
}

impl TextInject for LinuxTextInjector {
    fn type_text(&self, _text: &str) -> Result<(), InjectError> {
        // Returns an error instead of panicking: this runs on the hotkey path.
        Err(InjectError::Input(
            "Linux text injection is not implemented yet".into(),
        ))
    }

    fn request_accessibility(&self) -> bool {
        true
    }
}
