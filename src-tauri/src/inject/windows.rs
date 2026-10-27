use super::{InjectError, TextInject};

pub(crate) struct WindowsTextInjector;

impl WindowsTextInjector {
    pub(crate) fn new() -> Self {
        Self
    }
}

impl TextInject for WindowsTextInjector {
    fn type_text(&self, _text: &str) -> Result<(), InjectError> {
        todo!("Windows text injection is not implemented yet")
    }

    fn request_accessibility(&self) -> bool {
        todo!("Windows accessibility check is not implemented yet")
    }
}
