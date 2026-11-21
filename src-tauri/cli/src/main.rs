mod args;
mod content_score;
mod corpus;
mod dump;
mod file_source;
mod harness;
mod replay;
mod report;
mod score;

use std::process::ExitCode;

use args::Args;

fn main() -> ExitCode {
    // Show harness reports without enabling verbose model/pipeline logs.
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn,talksify_cli::harness=info"),
    )
    .target(env_logger::Target::Stderr)
    .init();

    let Some(args) = Args::parse(std::env::args_os().skip(1)) else {
        eprintln!(
            "usage:\n  talksify-cli <file.wav | folder> [--models <dir>] [--dump-raw <path.json>]\n  \
             talksify-cli --replay <path.json> [--models <dir>]"
        );
        return ExitCode::FAILURE;
    };
    args.execute()
}
