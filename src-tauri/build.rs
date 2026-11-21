use std::process::ExitCode;

fn main() -> ExitCode {
    match tauri_build::try_build(tauri_build::Attributes::new()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error found during tauri-build: {error:#}");
            ExitCode::FAILURE
        }
    }
}
