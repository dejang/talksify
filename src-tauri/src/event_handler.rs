use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, PhysicalPosition};

use crate::inject::TextInjector;
use talksify_core::Transcriber;

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
enum DictationState {
    Idle,
    Recording,
    Processing,
}

pub(crate) struct AppState {
    transcriber: Mutex<Transcriber>,
    injector: TextInjector,
    state: Mutex<DictationState>,
    /// Gap in logical pixels between the overlay and the bottom of the usable screen area.
    overlay_margin: f64,
    save_recordings: AtomicBool,
}

impl AppState {
    pub(crate) fn new(transcriber: Transcriber, injector: TextInjector) -> Self {
        Self {
            transcriber: Mutex::new(transcriber),
            injector,
            state: Mutex::new(DictationState::Idle),
            overlay_margin: 24.0,
            save_recordings: AtomicBool::new(false),
        }
    }

    pub(crate) fn is_recording(&self) -> bool {
        self.transcriber.lock().is_ok_and(|t| t.is_transcribing())
    }

    pub(crate) fn is_busy(&self) -> bool {
        self.state
            .lock()
            .is_ok_and(|s| !matches!(*s, DictationState::Idle))
    }

    pub(crate) fn toggle_save_recordings(&self) -> bool {
        !self.save_recordings.fetch_xor(true, Ordering::Relaxed)
    }

    pub(crate) fn start_dictation(&self, app: &AppHandle) -> Result<(), String> {
        let save_audio_to = self
            .save_recordings
            .load(Ordering::Relaxed)
            .then(|| {
                let Some(base) = dirs::data_local_dir() else {
                    log::error!("[recording] data_local_dir unavailable");
                    return None;
                };
                let name = format!("{}.wav", chrono::Local::now().format("%Y-%m-%d_%H-%M-%S"));
                Some(base.join("talksify").join("recordings").join(name))
            })
            .flatten();
        let mut transcriber = self
            .transcriber
            .lock()
            .map_err(|e| format!("lock poisoned: {e}"))?;
        transcriber
            .start_transcribing(save_audio_to)
            .map_err(|e| e.to_string())?;
        drop(transcriber);
        self.set_state(app, DictationState::Recording);
        Ok(())
    }

    pub(crate) fn stop_dictation(&self, app: &AppHandle) -> Result<(), String> {
        self.set_state(app, DictationState::Processing);
        let transcript = {
            let mut transcriber = self
                .transcriber
                .lock()
                .map_err(|e| format!("lock poisoned: {e}"))?;
            transcriber
                .stop_transcribing()
                .map_err(|e| e.to_string())?
                .cleaned
        };
        if transcript.is_empty() {
            log::info!("[event] empty transcript, skipping inject");
        } else if let Err(e) = self.injector.type_text(&transcript) {
            log::error!("[event] inject failed: {e}");
        }
        self.set_state(app, DictationState::Idle);
        Ok(())
    }

    fn set_state(&self, app: &AppHandle, new: DictationState) {
        if let Ok(mut s) = self.state.lock() {
            *s = new;
        }
        let _ = app.emit("dictation-state", new);

        let visible = !matches!(new, DictationState::Idle);
        if let Some(window) = app.get_webview_window("main") {
            if new == DictationState::Recording {
                // Centres the window at the bottom of the screen the pointer is on, above the Dock.
                let placed: tauri::Result<()> = (|| {
                    let cursor = window.cursor_position()?;
                    let monitor = match window.monitor_from_point(cursor.x, cursor.y)? {
                        Some(monitor) => monitor,
                        None => match window.primary_monitor()? {
                            Some(monitor) => monitor,
                            None => return Ok(()),
                        },
                    };
                    let area = monitor.work_area();
                    let size = window.outer_size()?;
                    let gap = self.overlay_margin * monitor.scale_factor();
                    let free_width = f64::from(area.size.width) - f64::from(size.width);
                    let x = f64::from(area.position.x) + free_width / 2.0;
                    let y = f64::from(area.position.y) + f64::from(area.size.height)
                        - f64::from(size.height)
                        - gap;
                    window.set_position(PhysicalPosition::new(x, y))
                })();
                if let Err(e) = placed {
                    log::warn!("[overlay] could not position window: {e}");
                }
            }
            if visible {
                let _ = window.show();
            } else {
                let _ = window.hide();
            }
        }
    }
}
