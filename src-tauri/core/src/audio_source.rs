use crate::{AudioError, HeapCons, InputFormat};

/// Delivers native interleaved f32 samples through a ring buffer.
/// Sources are created on the audio-consumer thread at each start.
pub trait AudioSource {
    /// Starts capture and returns the read end of the ring buffer.
    /// # Errors
    /// Returns an error when the source cannot start.
    fn start_capture(&mut self) -> Result<HeapCons<f32>, AudioError>;

    /// Stops capture and releases its resources.
    /// # Errors
    /// Returns an error when the source cannot stop.
    fn stop_capture(&mut self) -> Result<(), AudioError>;

    /// Returns the native sample rate and channel count.
    fn input_format(&self) -> Option<InputFormat>;
}
