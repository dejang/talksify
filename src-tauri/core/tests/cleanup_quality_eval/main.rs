mod cases;
mod prose_matcher;

#[cfg(test)]
mod eval {
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    use talksify_core::TextCleaner;

    use crate::cases::suite::QualitySuite;
    use crate::prose_matcher::matcher::ProseMatcher;

    trait EvalCleaner {
        fn try_clean(&self, raw: &str) -> Result<String, String>;
    }

    impl EvalCleaner for TextCleaner {
        fn try_clean(&self, raw: &str) -> Result<String, String> {
            Self::try_clean(self, raw)
        }
    }

    struct CategoryScore {
        name: &'static str,
        passed: usize,
        total: usize,
    }

    #[derive(Default)]
    struct EvalRun {
        scores: Vec<CategoryScore>,
        failures: Vec<String>,
        inference_errors: Vec<String>,
    }

    impl EvalRun {
        fn execute(cleaner: &impl EvalCleaner, suite: &QualitySuite) -> Self {
            let mut matcher = ProseMatcher::default();
            let mut run = Self::default();
            for case in suite.cases() {
                let input = case.input_text();
                let start = Instant::now();
                match cleaner.try_clean(&input) {
                    Ok(output) => {
                        let elapsed = start.elapsed();
                        let failure = matcher.check(case, &output);
                        let status = if failure.is_none() { "PASS" } else { "FAIL" };
                        println!(
                            "[{status}] {}/{}\nIN:  {input}\nOUT: {output}\n({elapsed:?})\n",
                            case.category, case.label
                        );
                        run.record(case.category, failure.is_none());
                        if let Some(detail) = failure {
                            run.failures.push(detail);
                        }
                    }
                    Err(error) => {
                        let elapsed = start.elapsed();
                        println!(
                            "[ERROR] {}/{}\nIN:  {input}\nERR: {error}\n({elapsed:?})\n",
                            case.category, case.label
                        );
                        run.record(case.category, false);
                        let detail = format!(
                            "[{} / {}] inference error: {error}",
                            case.category, case.label
                        );
                        run.inference_errors.push(detail.clone());
                        run.failures.push(detail);
                    }
                }
            }
            run
        }

        fn record(&mut self, category: &'static str, passed: bool) {
            if let Some(entry) = self.scores.iter_mut().find(|entry| entry.name == category) {
                entry.total += 1;
                if passed {
                    entry.passed += 1;
                }
            } else {
                self.scores.push(CategoryScore {
                    name: category,
                    passed: usize::from(passed),
                    total: 1,
                });
            }
        }

        fn print_summary(&self) {
            println!("=== Cleanup Quality Eval Summary ===");
            let mut total_passed = 0;
            let mut total_cases = 0;
            for entry in &self.scores {
                total_passed += entry.passed;
                total_cases += entry.total;
                #[expect(clippy::cast_precision_loss, reason = "small eval counts fit f64")]
                let pct = entry.passed as f64 / entry.total as f64 * 100.0;
                println!(
                    "  {:<26} {:>2}/{:<2}  ({pct:>5.1}%)",
                    entry.name, entry.passed, entry.total
                );
            }
            #[expect(clippy::cast_precision_loss, reason = "small eval counts fit f64")]
            let pct = total_passed as f64 / total_cases.max(1) as f64 * 100.0;
            println!(
                "  {:<26} {:>2}/{:<2}  ({pct:>5.1}%)",
                "TOTAL", total_passed, total_cases
            );
            if !self.inference_errors.is_empty() {
                println!("  inference errors: {}", self.inference_errors.len());
            }
        }
    }

    struct StubCleaner {
        response: Result<String, String>,
    }

    impl EvalCleaner for StubCleaner {
        fn try_clean(&self, _raw: &str) -> Result<String, String> {
            self.response.clone()
        }
    }

    #[test]
    fn quality_eval_fails_explicitly_on_inference_error_and_unusable_output() {
        let suite = QualitySuite::standard().only("yes-no-question");
        assert_eq!(suite.cases().len(), 1);

        for (error_msg, expected_substr) in [
            ("LLM inference failed: injected error", "injected error"),
            (
                "LLM inference returned no usable transcript text",
                "no usable transcript text",
            ),
            ("LLM inference truncated at length limit", "length"),
        ] {
            let cleaner = StubCleaner {
                response: Err(error_msg.into()),
            };
            let run = EvalRun::execute(&cleaner, &suite);
            assert_eq!(run.scores.len(), 1);
            assert_eq!(run.scores[0].passed, 0);
            assert_eq!(run.scores[0].total, 1);
            assert_eq!(run.inference_errors.len(), 1);
            assert!(
                run.inference_errors[0].contains("yes-no-question")
                    && run.inference_errors[0].contains(expected_substr),
                "unexpected inference error: {:?}",
                run.inference_errors
            );
            assert_eq!(run.failures, run.inference_errors);
        }

        let ok_cleaner = StubCleaner {
            response: Ok("Did the nightly backup finish?".into()),
        };
        let ok_run = EvalRun::execute(&ok_cleaner, &suite);
        assert_eq!(ok_run.scores[0].passed, 1);
        assert_eq!(ok_run.inference_errors, Vec::<String>::new());
        assert_eq!(ok_run.failures, Vec::<String>::new());
    }

    #[test]
    #[ignore = "loads the bundled LLM; run with --ignored --nocapture"]
    fn cleanup_quality_eval() {
        let models = std::env::var_os("LLM_EVAL_MODELS").map_or_else(
            || Path::new(env!("CARGO_MANIFEST_DIR")).join("../models"),
            PathBuf::from,
        );
        let cleaner = TextCleaner::try_new(&models)
            .unwrap_or_else(|e| panic!("LLM failed to load from {}: {e}", models.display()));
        assert!(
            cleaner.is_ready(),
            "LLM failed to load from {}",
            models.display()
        );

        let suite = QualitySuite::standard();
        let run = EvalRun::execute(&cleaner, &suite);
        run.print_summary();

        assert!(
            run.inference_errors.is_empty(),
            "{} of {} cases hit inference errors ({} total failed):\n{}",
            run.inference_errors.len(),
            suite.cases().len(),
            run.failures.len(),
            run.failures.join("\n")
        );
        assert!(
            run.failures.is_empty(),
            "{} of {} cases failed:\n{}",
            run.failures.len(),
            suite.cases().len(),
            run.failures.join("\n")
        );
    }
}
