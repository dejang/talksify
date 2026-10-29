use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use tauri::AppHandle;
use tauri::Manager;
use tauri::menu::MenuItem;
use tauri_plugin_dialog::{DialogExt, MessageDialogButtons, MessageDialogKind};
use tauri_plugin_updater::{Update, UpdaterExt};

use crate::event_handler::{AppState, DictationState};

enum PendingUpdate {
    None,
    Available(Box<Update>),
    ReadyToRestart,
}

struct BusyGuard<'a>(&'a AtomicBool);

impl<'a> BusyGuard<'a> {
    fn try_acquire(flag: &'a AtomicBool) -> Option<Self> {
        flag.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .ok()
            .map(|_| Self(flag))
    }
}

impl Drop for BusyGuard<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

pub(crate) struct Updater {
    menu_item: MenuItem<tauri::Wry>,
    pending: Mutex<PendingUpdate>,
    busy: AtomicBool,
    max_notes_chars: usize,
}

impl Updater {
    pub(crate) fn new(menu_item: MenuItem<tauri::Wry>) -> Arc<Self> {
        Arc::new(Self {
            menu_item,
            pending: Mutex::new(PendingUpdate::None),
            busy: AtomicBool::new(false),
            max_notes_chars: 400,
        })
    }

    pub(crate) fn spawn_startup_check(self: &Arc<Self>, app: AppHandle) {
        let updater = Arc::clone(self);
        tauri::async_runtime::spawn(async move {
            updater.check_silently(&app).await;
        });
    }

    pub(crate) fn handle_menu_click(self: &Arc<Self>, app: AppHandle) {
        let updater = Arc::clone(self);
        tauri::async_runtime::spawn(async move {
            updater.run_interactive_flow(&app).await;
        });
    }

    fn set_menu(&self, text: &str, enabled: bool) {
        let _ = self.menu_item.set_text(text);
        let _ = self.menu_item.set_enabled(enabled);
    }

    fn store_pending(&self, next: PendingUpdate) {
        if let Ok(mut guard) = self.pending.lock() {
            *guard = next;
        }
    }

    fn take_pending(&self) -> PendingUpdate {
        self.pending
            .lock()
            .map_or(PendingUpdate::None, |mut guard| {
                std::mem::replace(&mut *guard, PendingUpdate::None)
            })
    }

    async fn check_silently(&self, app: &AppHandle) {
        let Some(_guard) = BusyGuard::try_acquire(&self.busy) else {
            return;
        };
        let client = match app.updater() {
            Ok(client) => client,
            Err(e) => {
                log::debug!("[updater] unavailable: {e}");
                return;
            }
        };
        match client.check().await {
            Ok(Some(update)) => {
                log::info!("[updater] update available: v{}", update.version);
                self.set_menu(&available_label(&update.version), true);
                self.store_pending(PendingUpdate::Available(Box::new(update)));
            }
            Ok(None) => log::debug!("[updater] up to date"),
            Err(e) => log::debug!("[updater] startup check skipped: {e}"),
        }
    }

    async fn run_interactive_flow(&self, app: &AppHandle) {
        let Some(_guard) = BusyGuard::try_acquire(&self.busy) else {
            return;
        };
        match self.take_pending() {
            PendingUpdate::ReadyToRestart => app.restart(),
            PendingUpdate::Available(update) => self.prompt_and_install(app, update).await,
            PendingUpdate::None => {
                if let Some(update) = self.check_interactively(app).await {
                    self.prompt_and_install(app, update).await;
                }
            }
        }
    }

    async fn check_interactively(&self, app: &AppHandle) -> Option<Box<Update>> {
        self.set_menu("Checking for Updates…", false);
        let result = match app.updater() {
            Ok(client) => client.check().await,
            Err(e) => Err(e),
        };
        match result {
            Ok(Some(update)) => {
                self.set_menu(&available_label(&update.version), true);
                Some(Box::new(update))
            }
            Ok(None) => {
                self.set_menu("Check for Updates…", true);
                let current = app.package_info().version.to_string();
                let msg = format!("Talksify v{current} is currently the newest version available.");
                show_info(app, "You’re Up to Date", &msg, MessageDialogKind::Info).await;
                None
            }
            Err(e) => {
                log::warn!("[updater] check failed: {e}");
                self.set_menu("Check for Updates…", true);
                let msg = format!("Could not check for updates:\n{e}");
                show_info(app, "Update Check Failed", &msg, MessageDialogKind::Warning).await;
                None
            }
        }
    }

    async fn prompt_and_install(&self, app: &AppHandle, update: Box<Update>) {
        let prompt = confirm_prompt(
            &update.version,
            &update.current_version,
            update.body.as_deref(),
            self.max_notes_chars,
        );
        let confirmed = ask_confirm(
            app,
            "Update Available",
            &prompt,
            "Download & Install",
            "Later",
        )
        .await;
        if !confirmed {
            self.set_menu(&available_label(&update.version), true);
            self.store_pending(PendingUpdate::Available(update));
            return;
        }

        if let Err(e) = self.download_update(&update).await {
            log::error!("[updater] download/install failed: {e}");
            let version = update.version.clone();
            self.set_menu(&available_label(&version), true);
            self.store_pending(PendingUpdate::Available(update));
            let msg = format!("Could not install Talksify v{version}:\n{e}");
            show_info(app, "Update Failed", &msg, MessageDialogKind::Error).await;
            return;
        }

        self.finish_or_restart(app, &update.version).await;
    }

    async fn download_update(&self, update: &Update) -> Result<(), tauri_plugin_updater::Error> {
        self.set_menu("Downloading Update (0%)…", false);
        let mut downloaded: u64 = 0;
        let mut last_label = String::new();
        let on_chunk_item = self.menu_item.clone();
        let on_finish_item = self.menu_item.clone();

        update
            .download_and_install(
                move |chunk_len, content_len| {
                    downloaded = downloaded.saturating_add(chunk_len as u64);
                    let label = progress_label(downloaded, content_len);
                    if label != last_label {
                        let _ = on_chunk_item.set_text(&label);
                        last_label = label;
                    }
                },
                move || {
                    let _ = on_finish_item.set_text("Installing Update…");
                },
            )
            .await
    }

    async fn finish_or_restart(&self, app: &AppHandle, version: &str) {
        if !is_dictation_busy(app) {
            app.restart();
        }
        self.set_menu("Restart to Apply Update", true);
        self.store_pending(PendingUpdate::ReadyToRestart);
        let msg = format!(
            "Talksify v{version} is installed. Restart now to finish updating, \
             or choose Later if you are still dictating."
        );
        if ask_confirm(app, "Update Installed", &msg, "Restart Now", "Later").await {
            app.restart();
        }
    }
}

fn is_dictation_busy(app: &AppHandle) -> bool {
    let state = app.state::<AppState>();
    state
        .state
        .lock()
        .is_ok_and(|s| !matches!(*s, DictationState::Idle))
}

async fn show_info(app: &AppHandle, title: &str, message: &str, kind: MessageDialogKind) {
    let app = app.clone();
    let title = title.to_owned();
    let message = message.to_owned();
    let _ = tauri::async_runtime::spawn_blocking(move || {
        app.dialog()
            .message(message)
            .title(title)
            .kind(kind)
            .buttons(MessageDialogButtons::Ok)
            .blocking_show();
    })
    .await;
}

async fn ask_confirm(
    app: &AppHandle,
    title: &str,
    message: &str,
    ok_label: &str,
    cancel_label: &str,
) -> bool {
    let app = app.clone();
    let title = title.to_owned();
    let message = message.to_owned();
    let buttons = MessageDialogButtons::OkCancelCustom(ok_label.to_owned(), cancel_label.to_owned());
    tauri::async_runtime::spawn_blocking(move || {
        app.dialog()
            .message(message)
            .title(title)
            .kind(MessageDialogKind::Info)
            .buttons(buttons)
            .blocking_show()
    })
    .await
    .unwrap_or(false)
}

fn available_label(version: &str) -> String {
    format!("Update Available (v{version})…")
}

fn progress_label(downloaded: u64, total: Option<u64>) -> String {
    if let Some(total) = total.filter(|&t| t > 0) {
        let pct = ((u128::from(downloaded) * 100) / u128::from(total)).min(100);
        format!("Downloading Update ({pct}%)…")
    } else {
        let mb = downloaded / (1024 * 1024);
        format!("Downloading Update ({mb} MB)…")
    }
}

fn confirm_prompt(version: &str, current: &str, notes: Option<&str>, max_chars: usize) -> String {
    let trimmed = notes.map(str::trim).filter(|s| !s.is_empty());
    let notes_section = match trimmed {
        Some(body) => {
            let excerpt = truncate_chars(body, max_chars);
            format!("\n\nRelease notes:\n{excerpt}")
        }
        None => String::new(),
    };
    format!(
        "Talksify v{version} is available (you have v{current}).{notes_section}\n\n\
         Download and install the update (~1.2 GB)? Talksify will restart when finished."
    )
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_owned();
    }
    let kept: String = text.chars().take(max_chars).collect();
    format!("{kept}…")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_progress_with_and_without_content_length() {
        assert_eq!(
            progress_label(500, Some(1000)),
            "Downloading Update (50%)…"
        );
        assert_eq!(
            progress_label(1500, Some(1000)),
            "Downloading Update (100%)…"
        );
        assert_eq!(
            progress_label(50 * 1024 * 1024, None),
            "Downloading Update (50 MB)…"
        );
        assert_eq!(
            progress_label(10 * 1024 * 1024, Some(0)),
            "Downloading Update (10 MB)…"
        );
    }

    #[test]
    fn formats_confirm_prompt_and_truncates_long_notes() {
        let short = confirm_prompt("0.3.0", "0.2.1", Some("  Faster STT  "), 100);
        assert!(short.contains("Talksify v0.3.0 is available (you have v0.2.1)."));
        assert!(short.contains("Release notes:\nFaster STT"));

        let without_notes = confirm_prompt("0.3.0", "0.2.1", Some("   "), 100);
        assert!(!without_notes.contains("Release notes:"));

        let truncated = confirm_prompt("0.3.0", "0.2.1", Some("abcdefghij"), 5);
        assert!(truncated.contains("abcde…"));
    }
}
