use std::process::ExitCode;

fn main() -> ExitCode {
    match talksify_lib::run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            log::error!("tauri failed: {e}");
            ExitCode::FAILURE
        }
    }
}
