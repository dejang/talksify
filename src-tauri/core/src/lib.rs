//! Local transcription pipeline shared by the desktop app and CLI.

mod audio_device;
mod audio_source;
mod conventions;
mod llm_postprocessor;
mod parakeet_stt;
mod raw_transcript;
mod recording;
mod stt;
mod text_format;
mod transcriber;

pub use audio_device::{AudioDevice, AudioError, HeapCons, InputFormat};
pub use audio_source::AudioSource;
pub use raw_transcript::Transcript;
pub use transcriber::{Transcriber, TranscriberError};
