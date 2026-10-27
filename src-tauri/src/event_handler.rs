use std::sync::Mutex;

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, PhysicalPosition, WebviewWindow};

use crate::inject::TextInjector;
use crate::transcriber::Transcriber;

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum DictationState {
    Idle,
    Recording,
    Processing,
}

pub(crate) struct AppState {
    pub(crate) transcriber: Mutex<Transcriber>,
    pub(crate) injector: TextInjector,
    pub(crate) state: Mutex<DictationState>,
    /// Gap in logical pixels between the overlay and the bottom of the usable screen area.
    pub(crate) overlay_margin: f64,
}

fn set_state(app: &AppHandle, state: &AppState, new: DictationState) {
    if let Ok(mut s) = state.state.lock() {
        *s = new;
    }
    let _ = app.emit("dictation-state", new);

    let visible = !matches!(new, DictationState::Idle);
    if let Some(window) = app.get_webview_window("main") {
        if new == DictationState::Recording
            && let Err(e) = place_at_bottom(&window, state.overlay_margin)
        {
            log::warn!("[overlay] could not position window: {e}");
        }
        if visible {
            let _ = window.show();
        } else {
            let _ = window.hide();
        }
    }
}

/// Centres the window at the bottom of the screen the pointer is on, above the Dock.
fn place_at_bottom(window: &WebviewWindow, margin: f64) -> tauri::Result<()> {
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
    let gap = margin * monitor.scale_factor();
    let free_width = f64::from(area.size.width) - f64::from(size.width);
    let x = f64::from(area.position.x) + free_width / 2.0;
    let y = f64::from(area.position.y) + f64::from(area.size.height) - f64::from(size.height) - gap;
    window.set_position(PhysicalPosition::new(x, y))
}

pub(crate) fn start_dictation(app: &AppHandle, state: &AppState) -> Result<(), String> {
    let mut transcriber = state
        .transcriber
        .lock()
        .map_err(|e| format!("lock poisoned: {e}"))?;
    transcriber
        .start_transcribing()
        .map_err(|e| e.to_string())?;
    drop(transcriber);
    set_state(app, state, DictationState::Recording);
    Ok(())
}

pub(crate) fn stop_dictation(app: &AppHandle, state: &AppState) -> Result<(), String> {
    set_state(app, state, DictationState::Processing);
    let transcript = {
        let mut transcriber = state
            .transcriber
            .lock()
            .map_err(|e| format!("lock poisoned: {e}"))?;
        transcriber.stop_transcribing().map_err(|e| e.to_string())?
    };
    if transcript.is_empty() {
        log::info!("[event] empty transcript, skipping inject");
    } else if let Err(e) = state.injector.type_text(&transcript) {
        log::error!("[event] inject failed: {e}");
    }
    set_state(app, state, DictationState::Idle);
    Ok(())
}
