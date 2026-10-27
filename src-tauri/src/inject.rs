use enigo::{Enigo, Keyboard, Settings};

#[derive(Debug, thiserror::Error)]
pub(crate) enum InjectError {
    #[error("accessibility permission denied: {0}")]
    PermissionDenied(String),
    #[error("enigo error: {0}")]
    Enigo(String),
}

pub(crate) struct TextInjector {
    settings: Settings,
}

impl TextInjector {
    pub(crate) fn new() -> Self {
        Self {
            settings: Settings::default(),
        }
    }

    pub(crate) fn type_text(&self, text: &str) -> Result<(), InjectError> {
        // macOS drops CGEvents from untrusted processes without an error,
        // and enigo still reports success, so check trust up front.
        if !Self::is_trusted() {
            return Err(InjectError::PermissionDenied(
                "process is not trusted; grant it in System Settings > Privacy & Security > Accessibility".into(),
            ));
        }
        log::trace!("[inject] typing: {text:?}");
        let mut enigo =
            Enigo::new(&self.settings).map_err(|e| InjectError::PermissionDenied(e.to_string()))?;
        enigo
            .text(text)
            .map_err(|e| InjectError::Enigo(e.to_string()))?;
        log::info!("[inject] done");
        Ok(())
    }

    /// Whether the app may post input events; on macOS, prompts for Accessibility if not.
    pub(crate) fn request_accessibility() -> bool {
        #[cfg(target_os = "macos")]
        let trusted = {
            use core_foundation::base::TCFType;
            use core_foundation::boolean::CFBoolean;
            use core_foundation::dictionary::CFDictionary;
            use core_foundation::string::CFString;

            unsafe extern "C" {
                fn AXIsProcessTrustedWithOptions(options: *const std::ffi::c_void) -> bool;
                static kAXTrustedCheckOptionPrompt: *const std::ffi::c_void;
            }

            let key: CFString =
                unsafe { CFString::wrap_under_get_rule(kAXTrustedCheckOptionPrompt.cast()) };
            let value = CFBoolean::true_value();
            let opts = CFDictionary::from_CFType_pairs(&[(key, value)]);
            unsafe { AXIsProcessTrustedWithOptions(opts.as_concrete_TypeRef().cast()) }
        };
        #[cfg(not(target_os = "macos"))]
        let trusted = true;

        if !trusted {
            log::warn!("accessibility permission not granted — prompt shown to user");
        }
        trusted
    }

    fn is_trusted() -> bool {
        #[cfg(target_os = "macos")]
        let trusted = {
            unsafe extern "C" {
                fn AXIsProcessTrusted() -> bool;
            }
            unsafe { AXIsProcessTrusted() }
        };
        #[cfg(not(target_os = "macos"))]
        let trusted = true;
        trusted
    }
}

impl Default for TextInjector {
    fn default() -> Self {
        Self::new()
    }
}
