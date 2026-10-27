pub(crate) trait Stt: Send {
    fn push_audio(&mut self, audio: &[f32]);
    fn subscribe(&mut self, on_transcribe: Box<dyn Fn(String) + Send>);
    fn flush(&mut self);
}
