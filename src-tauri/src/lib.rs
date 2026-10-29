mod audio_device;
mod config;
mod event_handler;
mod inject;
mod llm_postprocessor;
mod parakeet_stt;
mod raw_transcript;
mod stt;
mod text_format;
mod transcriber;
mod updater;

use std::sync::{Arc, Mutex};

use tauri::menu::{CheckMenuItemBuilder, MenuBuilder, MenuItemBuilder};
use tauri::tray::TrayIconBuilder;
use tauri::{Emitter, Manager};
use tauri_plugin_autostart::ManagerExt;
use tauri_plugin_global_shortcut::{GlobalShortcutExt, ShortcutState};

use config::Config;
use event_handler::AppState;
use inject::TextInjector;
use transcriber::Transcriber;
use updater::Updater;

fn is_recording(state: &AppState) -> bool {
    state.transcriber.lock().is_ok_and(|t| t.is_transcribing())
}

fn toggle_dictation(app: &tauri::AppHandle) {
    let state: tauri::State<'_, AppState> = app.state();
    if is_recording(state.inner()) {
        log::info!("[hotkey] stop dictation");
        let _ = event_handler::stop_dictation(app, state.inner());
    } else {
        log::info!("[hotkey] start dictation");
        let _ = event_handler::start_dictation(app, state.inner());
    }
}

/// Disabling unregisters the hotkey (finishing a recording in progress); enabling registers
/// it again. Returns the menu label for the next toggle.
fn toggle_hotkey(app: &tauri::AppHandle, hotkey: &str) -> Result<&'static str, String> {
    let shortcuts = app.global_shortcut();
    if shortcuts.is_registered(hotkey) {
        if is_recording(app.state::<AppState>().inner()) {
            toggle_dictation(app);
        }
        shortcuts.unregister(hotkey).map_err(|e| e.to_string())?;
        Ok("Enable Dictation")
    } else {
        shortcuts.register(hotkey).map_err(|e| e.to_string())?;
        Ok("Disable Dictation")
    }
}

fn toggle_autostart(app: &tauri::AppHandle) -> Result<(), tauri_plugin_autostart::Error> {
    let autostart = app.autolaunch();
    if autostart.is_enabled()? {
        autostart.disable()
    } else {
        autostart.enable()
    }
}

fn build_tray(app: &tauri::App, hotkey: String) -> tauri::Result<Arc<Updater>> {
    let dictation_item = MenuItemBuilder::with_id("dictation", "Disable Dictation").build(app)?;
    let autostart_item = CheckMenuItemBuilder::with_id("autostart", "Start at Login")
        .checked(app.autolaunch().is_enabled().unwrap_or(false))
        .build(app)?;
    let update_item = MenuItemBuilder::with_id("update", "Check for Updates…").build(app)?;
    let quit_item = MenuItemBuilder::with_id("quit", "Quit").build(app)?;
    let updater = Updater::new(update_item.clone());
    let menu_updater = Arc::clone(&updater);

    let menu = MenuBuilder::new(app)
        .item(&dictation_item)
        .item(&autostart_item)
        .separator()
        .item(&update_item)
        .separator()
        .item(&quit_item)
        .build()?;

    TrayIconBuilder::new()
        .icon(tauri::include_image!("icons/tray.png"))
        .icon_as_template(true)
        .menu(&menu)
        .tooltip("Talksify")
        .on_menu_event(move |app, event| match event.id().as_ref() {
            "dictation" => match toggle_hotkey(app, &hotkey) {
                Ok(label) => {
                    let _ = dictation_item.set_text(label);
                }
                Err(e) => log::error!("[tray] toggle dictation: {e}"),
            },
            "autostart" => {
                if let Err(e) = toggle_autostart(app) {
                    log::error!("[tray] toggle autostart: {e}");
                }
                let _ = autostart_item.set_checked(app.autolaunch().is_enabled().unwrap_or(false));
            }
            "update" => menu_updater.handle_menu_click(app.clone()),
            "quit" => app.exit(0),
            _ => {}
        })
        .build(app)?;
    Ok(updater)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let result = tauri::Builder::default()
        .plugin(
            tauri_plugin_log::Builder::new()
                .level(log::LevelFilter::Debug)
                .build(),
        )
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(|app, _shortcut, event| {
                    if event.state() == ShortcutState::Pressed {
                        toggle_dictation(app);
                    }
                })
                .build(),
        )
        .plugin(tauri_plugin_autostart::Builder::new().build())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .setup(|app| {
            #[cfg(target_os = "macos")]
            app.set_activation_policy(tauri::ActivationPolicy::Accessory);

            let injector = TextInjector::new();
            if !injector.request_accessibility() {
                log::warn!("waiting for user to grant accessibility permission");
            }

            let resource_dir = app
                .path()
                .resource_dir()
                .map_err(|e| format!("resource dir: {e}"))?;
            let config = Config::load(&resource_dir);

            let level_handle = app.handle().clone();
            let on_level = Arc::new(move |level: f32| {
                let _ = level_handle.emit("audio-level", level);
            });
            let transcriber = Transcriber::new(&config, on_level)
                .map_err(|e| format!("transcriber init: {e}"))?;

            let app_state = AppState {
                transcriber: Mutex::new(transcriber),
                injector,
                state: Mutex::new(event_handler::DictationState::Idle),
                overlay_margin: 24.0,
            };
            app.manage(app_state);

            log::info!("registering hotkey: {}", config.hotkey);
            app.global_shortcut()
                .register(config.hotkey.as_str())
                .map_err(|e| format!("hotkey register: {e}"))?;
            log::info!("hotkey registered ok");

            let updater = build_tray(app, config.hotkey).map_err(|e| format!("tray: {e}"))?;
            updater.spawn_startup_check(app.handle().clone());

            Ok(())
        })
        .run(tauri::generate_context!());

    if let Err(e) = result {
        log::error!("tauri failed: {e}");
    }
}
