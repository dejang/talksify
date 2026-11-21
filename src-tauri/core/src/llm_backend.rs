use std::sync::Arc;

use mistralrs::blocking::BlockingModel;
use mistralrs::{Choice, GgufModelBuilder, RequestBuilder, TextMessageRole, TextMessages};

use crate::gguf_path::LocalGguf;

/// A normalized completion returned by an LLM chat backend.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Completion {
    pub(crate) text: String,
    pub(crate) finish_reason: String,
}

impl Completion {
    /// Extracts completion text and finish reason from the first chat response choice.
    pub(crate) fn from_choice(choice: Option<Choice>) -> Result<Self, String> {
        let (raw_text, finish_reason) = choice
            .and_then(|choice| Some((choice.message.content?, choice.finish_reason)))
            .ok_or_else(|| "LLM inference returned no completion content".to_string())?;
        let text = raw_text
            .trim()
            .lines()
            .map(str::trim_end)
            .collect::<Vec<_>>()
            .join("\n");
        Ok(Self {
            text,
            finish_reason,
        })
    }

    /// Rejects truncated or abnormal completion reasons for strict evaluation callers.
    pub(crate) fn validate_strict(&self) -> Result<(), String> {
        match self.finish_reason.as_str() {
            "stop" => Ok(()),
            "length" => Err("LLM inference truncated at length limit".to_string()),
            other => Err(format!(
                "LLM inference ended with unexpected finish_reason {other:?}"
            )),
        }
    }

    /// Strips outer `<transcript>` / `</transcript>` framing tags and surrounding whitespace.
    pub(crate) fn stripped_text(&self) -> &str {
        let result = self.text.trim();
        let result = result.strip_prefix("<transcript>").unwrap_or(result).trim();
        result
            .strip_suffix("</transcript>")
            .unwrap_or(result)
            .trim()
    }
}

/// Chat completion engine backing [`crate::llm_postprocessor::LlmPostProcessor`].
pub(crate) trait LlmBackend: Send + Sync {
    fn complete(&self, transcript: &str, prompt: &str) -> Result<Completion, String>;
}

/// Local GGUF chat model executed through `mistralrs`.
pub(crate) struct MistralBackend {
    model: BlockingModel,
}

impl MistralBackend {
    pub(crate) fn load_and_warm(gguf: &LocalGguf) -> Result<Self, String> {
        log::info!("loading LLM from {}/{}", gguf.dir, gguf.file);
        let start = std::time::Instant::now();
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| format!("runtime: {e}"))?;

        let builder = GgufModelBuilder::new(gguf.dir.clone(), vec![gguf.file.clone()])
            .without_paged_attn()
            .with_max_num_seqs(1)
            .with_prefix_cache_n(None);

        let model = rt
            .block_on(builder.build())
            .map_err(|e| format!("build: {e}"))?;
        let model = BlockingModel::new(model, Arc::new(rt));

        log::info!("[llm] model loaded in {:?}, warming...", start.elapsed());
        let warm_start = std::time::Instant::now();
        let warmup = RequestBuilder::from(
            TextMessages::new()
                .enable_thinking(false)
                .add_message(TextMessageRole::User, "hi"),
        )
        .set_deterministic_sampler()
        .set_sampler_max_len(8);

        model
            .send_chat_request(warmup)
            .map_err(|e| format!("warmup failed: {e}"))?;
        log::info!("[llm] warmup in {:?}", warm_start.elapsed());
        log::info!("[llm] ready (total {:?})", start.elapsed());
        Ok(Self { model })
    }
}

impl LlmBackend for MistralBackend {
    fn complete(&self, transcript: &str, prompt: &str) -> Result<Completion, String> {
        // Set the deterministic sampler before the token cap: it resets sampling parameters.
        // The cap stops a model that never ends.
        let request = RequestBuilder::from(
            TextMessages::new()
                .enable_thinking(false)
                .add_message(TextMessageRole::System, prompt)
                .add_message(
                    TextMessageRole::User,
                    format!("<transcript>\n{transcript}\n</transcript>"),
                ),
        )
        .set_deterministic_sampler()
        .set_sampler_max_len(transcript.split_whitespace().count() * 2 + 32);
        let response = self
            .model
            .send_chat_request(request)
            .map_err(|e| format!("LLM inference failed: {e}"))?;
        Completion::from_choice(response.choices.into_iter().next())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_choice(content: Option<&str>, finish_reason: &str) -> Choice {
        Choice {
            finish_reason: finish_reason.into(),
            stop_sequence: None,
            index: 0,
            message: mistralrs::ResponseMessage {
                content: content.map(str::to_owned),
                role: "assistant".into(),
                tool_calls: None,
                reasoning_content: None,
            },
            logprobs: None,
        }
    }

    #[test]
    fn completion_parsing_strict_validation_and_tag_stripping() {
        type ChoiceCase = (
            Option<(Option<&'static str>, &'static str)>,
            Result<(&'static str, &'static str), &'static str>,
            Option<Result<(), &'static str>>,
        );
        let choice_cases: &[ChoiceCase] = &[
            (None, Err("no completion content"), None),
            (Some((None, "stop")), Err("no completion content"), None),
            (
                Some((Some("  hello  \nworld  "), "stop")),
                Ok(("hello\nworld", "stop")),
                Some(Ok(())),
            ),
            (
                Some((Some("Elementarsch"), "length")),
                Ok(("Elementarsch", "length")),
                Some(Err("length")),
            ),
            (
                Some((Some("Partial"), "error")),
                Ok(("Partial", "error")),
                Some(Err("finish_reason")),
            ),
        ];
        for (input, expected_parse, expected_strict) in choice_cases {
            let choice = input.map(|(content, reason)| make_choice(content, reason));
            match (Completion::from_choice(choice), expected_parse) {
                (Ok(completion), Ok((text, reason))) => {
                    assert_eq!(completion.text, *text);
                    assert_eq!(completion.finish_reason, *reason);
                    match (completion.validate_strict(), expected_strict.unwrap()) {
                        (Ok(()), Ok(())) => {}
                        (Err(err), Err(substr)) => {
                            assert!(err.contains(substr), "expected {substr:?}, got {err:?}");
                        }
                        (got, want) => panic!("validate_strict mismatch: got {got:?}, want {want:?}"),
                    }
                }
                (Err(err), Err(substr)) => {
                    assert!(err.contains(substr), "expected {substr:?}, got {err:?}");
                }
                (got, want) => panic!("from_choice mismatch: got {got:?}, want {want:?}"),
            }
        }

        for (raw, expected) in [
            ("Clean text.", "Clean text."),
            (
                " \n<transcript>\nClean text.\n</transcript>\n ",
                "Clean text.",
            ),
            ("<transcript>Clean text.", "Clean text."),
            ("Clean text.</transcript>", "Clean text."),
            (
                "Keep <transcript>literal</transcript> text.",
                "Keep <transcript>literal</transcript> text.",
            ),
            ("<transcript>\n \n</transcript>", ""),
        ] {
            let completion = Completion {
                text: raw.into(),
                finish_reason: "stop".into(),
            };
            assert_eq!(completion.stripped_text(), expected, "{raw:?}");
        }
    }
}
