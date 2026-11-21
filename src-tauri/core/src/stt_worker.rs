use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};

use crate::audio_device::HeapCons;
use crate::stt::Stt;

/// Control messages sent from [`crate::Transcriber`] to the warm STT thread.
pub(crate) enum SttCommand {
    /// Audio for the recording that is starting, replacing the previous recording's buffer.
    Audio(HeapCons<f32>),
    Flush(Sender<()>),
    Shutdown,
}

/// Drives an [`Stt`] engine on the warm worker thread owned by [`crate::Transcriber`].
pub(crate) struct SttWorker<S: Stt> {
    stt: S,
    rx: Receiver<SttCommand>,
    shutdown: Arc<AtomicBool>,
    audio: Option<HeapCons<f32>>,
}

impl<S: Stt> SttWorker<S> {
    pub(crate) fn new(stt: S, rx: Receiver<SttCommand>, shutdown: Arc<AtomicBool>) -> Self {
        Self {
            stt,
            rx,
            shutdown,
            audio: None,
        }
    }

    pub(crate) fn run(mut self) {
        while !self.shutdown.load(Ordering::Relaxed) {
            // While recording, wake every 10 ms to pick up new audio; otherwise wait for a command.
            let command = if self.audio.is_some() {
                self.rx.recv_timeout(std::time::Duration::from_millis(10))
            } else {
                self.rx.recv().map_err(RecvTimeoutError::from)
            };
            if let Some(audio_rx) = &mut self.audio
                && let Ok(chunk) = audio_rx.read_chunk(audio_rx.slots())
            {
                let (first, second) = chunk.as_slices();
                self.stt.push_audio(first);
                self.stt.push_audio(second);
                chunk.commit_all();
            }
            match command {
                Ok(SttCommand::Audio(audio_rx)) => {
                    self.audio = Some(audio_rx);
                }
                Ok(SttCommand::Flush(done_tx)) => {
                    self.stt.flush();
                    self.audio = None;
                    let _ = done_tx.send(());
                }
                Err(RecvTimeoutError::Timeout) => {}
                Ok(SttCommand::Shutdown) | Err(RecvTimeoutError::Disconnected) => {
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::sync::mpsc::channel;

    use super::*;

    struct MockStt {
        pushed: Arc<Mutex<Vec<f32>>>,
        flushes: Arc<Mutex<usize>>,
    }

    impl Stt for MockStt {
        fn push_audio(&mut self, samples: &[f32]) {
            self.pushed.lock().unwrap().extend_from_slice(samples);
        }

        fn flush(&mut self) {
            *self.flushes.lock().unwrap() += 1;
        }

        fn subscribe(&mut self, _cb: Box<dyn Fn(String) + Send>) {}
    }

    #[test]
    fn drains_audio_flushes_and_shuts_down_cleanly() {
        let pushed = Arc::new(Mutex::new(Vec::new()));
        let flushes = Arc::new(Mutex::new(0));
        let stt = MockStt {
            pushed: Arc::clone(&pushed),
            flushes: Arc::clone(&flushes),
        };
        let (tx, rx) = channel();
        let shutdown = Arc::new(AtomicBool::new(false));
        let samples = [0.25f32, -0.5, 0.75];
        let (mut audio_tx, audio_rx) = rtrb::RingBuffer::new(8);
        for sample in samples {
            audio_tx.push(sample).unwrap();
        }
        drop(audio_tx);

        let (done_tx, done_rx) = channel();
        tx.send(SttCommand::Audio(audio_rx)).unwrap();
        tx.send(SttCommand::Flush(done_tx)).unwrap();
        tx.send(SttCommand::Shutdown).unwrap();

        SttWorker::new(stt, rx, shutdown).run();

        assert!(done_rx.try_recv().is_ok());
        assert_eq!(*pushed.lock().unwrap(), samples);
        assert_eq!(*flushes.lock().unwrap(), 1);
    }
}
