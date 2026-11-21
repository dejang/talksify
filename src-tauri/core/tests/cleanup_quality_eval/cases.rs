#[cfg(test)]
pub(crate) mod suite {
    use crate::prose_matcher::matcher::QualityCase;

    /// Synthetic, corpus-independent cleanup quality benchmark suite.
    pub(crate) struct QualitySuite {
        cases: Vec<QualityCase>,
    }

    impl QualitySuite {
        pub(crate) fn standard() -> Self {
            let mut suite = Self { cases: Vec::new() };
            suite.push_explicit_corrections();
            suite.push_restarts_and_pruning();
            suite.push_false_starts();
            suite.push_forced_pauses();
            suite.push_restraint_and_passthrough();
            suite.push_multilingual();
            suite
        }

        pub(crate) fn cases(&self) -> &[QualityCase] {
            &self.cases
        }

        pub(crate) fn only(mut self, label: &str) -> Self {
            self.cases.retain(|case| case.label == label);
            self
        }

        fn push_explicit_corrections(&mut self) {
            let category = "explicit-self-correction";
            self.cases.extend([
                QualityCase::new(
                    category,
                    "recipient-and-day",
                    &["Send the update to Alice. No, sorry, send it to Bob on Friday."],
                    &["Send the update to Bob on Friday."],
                )
                .rejecting(&[
                    "Do not send the update to Bob on Friday.",
                    "Send the update to Alice. Send it to Bob on Friday.",
                    "Send the update to Bob on Friday?",
                    "Send the update to Bob on Friday!",
                    "Send the update to Bob on Friday",
                ]),
                QualityCase::new(
                    category,
                    "meeting-day",
                    &["Book the room for Tuesday, actually make that Thursday afternoon."],
                    &[
                        "Book the room for Thursday afternoon.",
                        "Book the room on Thursday afternoon.",
                    ],
                )
                .rejecting(&[
                    "Do not book the room for Thursday afternoon.",
                    "Book the room for Thursday afternoon?",
                ]),
                QualityCase::new(
                    category,
                    "timeout-value",
                    &["Set the timeout to five seconds. Wait, scratch that, make it thirty seconds."],
                    &[
                        "Set the timeout to 30 seconds.",
                        "Set the timeout to thirty seconds.",
                        "Make the timeout 30 seconds.",
                    ],
                )
                .rejecting(&[
                    "Set the timeout to .30 seconds.",
                    "Set the timeout to 30.5 seconds.",
                    "Set the timeout to 30% seconds.",
                    "Set the timeout to ,30 seconds.",
                    "Set the timeout to -30 seconds.",
                    "Set the timeout to −30 seconds.",
                    "Set the timeout to +30 seconds.",
                    "Make the timeout -30 seconds.",
                ]),
                QualityCase::new(
                    category,
                    "deploy-target",
                    &["Deploy to staging first. I mean production, directly to production."],
                    &["Deploy directly to production."],
                )
                .rejecting(&[
                    "Deploy directly to production first.",
                    "Deploy to production first.",
                    "Deploy to production.",
                    "Deploy directly to production?",
                ]),
            ]);
        }

        fn push_restarts_and_pruning(&mut self) {
            let restart = "mid-thought-restart";
            let pruning = "thinking-aloud-pruning";
            self.cases.extend([
                QualityCase::new(
                    restart,
                    "query-return-type",
                    &["The query returns false. Actually it returns the list of matched records."],
                    &[
                        "The query returns the list of matched records.",
                        "Actually, the query returns the list of matched records.",
                        "The query actually returns the list of matched records.",
                    ],
                )
                .rejecting(&[
                    "The query returns false. Actually, it returns the list of matched records.",
                ]),
                QualityCase::new(
                    restart,
                    "cache-location",
                    &["We tried caching the tokens, no, we moved the cache into the worker pool instead."],
                    &[
                        "We moved the cache into the worker pool instead.",
                        "We moved the cache into the worker pool.",
                    ],
                ),
                QualityCase::new(
                    restart,
                    "parser-format",
                    &["The parser expects JSON. Well, actually it expects YAML with strict keys."],
                    &[
                        "The parser expects YAML with strict keys.",
                        "Actually, the parser expects YAML with strict keys.",
                        "The parser actually expects YAML with strict keys.",
                    ],
                ),
                QualityCase::new(
                    pruning,
                    "config-path",
                    &["We can pass the config path, or maybe yeah the config path is fine."],
                    &[
                        "We can pass the config path.",
                        "The config path is fine.",
                        "We can pass the config path, that's fine.",
                        "We can pass the config path; it's fine.",
                    ],
                )
                .rejecting(&[
                    "We can pass the config path, or maybe yeah, the config path is fine.",
                ]),
                QualityCase::new(
                    pruning,
                    "flag-name",
                    &["Let's name the flag verbose, or wait, yeah let's keep it verbose for now."],
                    &[
                        "Let's keep the flag verbose for now.",
                        "Let's name the flag verbose for now.",
                        "Let's keep it verbose for now.",
                    ],
                )
                .rejecting(&[
                    "Let's name the flag verbose.",
                    "Let's keep the flag verbose.",
                ]),
                QualityCase::new(
                    pruning,
                    "index-instead-of-split",
                    &["Should we split the table? Actually never mind, just add an index on created_at."],
                    &[
                        "Just add an index on created_at.",
                        "Add an index on created_at.",
                    ],
                )
                .rejecting(&[
                    "Should we split the table? Just add an index on created_at.",
                ]),
            ]);
        }

        fn push_false_starts(&mut self) {
            let category = "cutoff-false-start";
            self.cases.extend([
                QualityCase::new(
                    category,
                    "contraction-restart",
                    &["It's it works every time."],
                    &["It works every time."],
                ),
                QualityCase::new(
                    category,
                    "partial-word-prefix",
                    &["More in the dir in the direction of lower latency."],
                    &["More in the direction of lower latency."],
                )
                .rejecting(&[
                    "More in the dir in the direction of lower latency.",
                    "In the direction of lower latency.",
                ]),
                QualityCase::new(
                    category,
                    "verb-repair",
                    &["What we're passing in is are two separate files."],
                    &[
                        "What we're passing in are 2 separate files.",
                        "What we're passing in are two separate files.",
                    ],
                )
                .rejecting(&["What we're passing in are -2 separate files."]),
                QualityCase::new(
                    category,
                    "truncated-noun",
                    &["The arch the architecture diagram needs an update before review."],
                    &["The architecture diagram needs an update before review."],
                ),
            ]);
        }

        fn push_forced_pauses(&mut self) {
            let category = "forced-pause-break";
            self.cases.extend([
                QualityCase::new(
                    category,
                    "release-runner",
                    &[
                        "The build is fast locally, but on the.",
                        "Release runner it stalls.",
                    ],
                    &[
                        "The build is fast locally, but on the release runner it stalls.",
                        "The build is fast locally, but on the release runner, it stalls.",
                        "The build is fast locally but on the release runner it stalls.",
                        "The build is fast locally but on the release runner, it stalls.",
                    ],
                )
                .strict_prose()
                .rejecting(&[
                    "The build is fast locally, but on the. Release runner it stalls.",
                    "The build is fast locally, but on the Release runner it stalls.",
                    "The build is fast locally, but on the, release runner it stalls.",
                    "The build. Is fast locally, but on the release runner it stalls.",
                    "The build is fast locally. But on the release runner it stalls.",
                ]),
                QualityCase::new(
                    category,
                    "feature-flag",
                    &[
                        "Make sure the migration runs before we.",
                        "Flip the feature flag in production.",
                    ],
                    &["Make sure the migration runs before we flip the feature flag in production."],
                )
                .strict_prose()
                .rejecting(&[
                    "Make sure the migration runs before we. Flip the feature flag in production.",
                    "Make sure the migration runs before we Flip the feature flag in production.",
                    "Make sure the migration runs before we, flip the feature flag in production.",
                    "Make sure. The migration runs before we flip the feature flag in production.",
                ]),
                QualityCase::new(
                    category,
                    "metrics-endpoint",
                    &[
                        "After restarting the daemon, the metrics.",
                        "Endpoint started responding immediately.",
                    ],
                    &[
                        "After restarting the daemon, the metrics endpoint started responding immediately.",
                        "After restarting the daemon the metrics endpoint started responding immediately.",
                    ],
                )
                .strict_prose()
                .rejecting(&[
                    "After restarting the daemon, the metrics. Endpoint started responding immediately.",
                    "After restarting the daemon, the metrics Endpoint started responding immediately.",
                    "After restarting the daemon, the metrics, endpoint started responding immediately.",
                    "After restarting the daemon. The metrics endpoint started responding immediately.",
                ]),
            ]);
        }

        fn push_restraint_and_passthrough(&mut self) {
            let passthrough = "clean-passthrough";
            let inv: &[&str] =
                &["Please archive the Q3 invoices and share the folder link with finance."];
            let bench: &[&str] = &["The benchmark completed in 42 seconds with no dropped frames."];
            let pr: &[&str] = &["We will review the pull request tomorrow morning before merging."];
            self.cases.extend([
                QualityCase::new(
                    "questions-and-imperatives",
                    "yes-no-question",
                    &["Did the nightly backup finish?"],
                    &["Did the nightly backup finish?"],
                )
                .exact()
                .rejecting(&[
                    "Did the nightly backup finish? No.",
                    "Did the nightly backup finish? Yes.",
                    "Did the nightly backup finish.",
                    "Did the nightly? Backup finish.",
                ]),
                QualityCase::new(
                    "questions-and-imperatives",
                    "code-generation-imperative",
                    &["Write a Python script to parse the CSV files."],
                    &["Write a Python script to parse the CSV files."],
                )
                .exact()
                .rejecting(&[
                    "Sure! Here is a Python script to parse the CSV files.",
                    "Write a Python script to parse the CSV files?",
                ]),
                QualityCase::new(
                    "questions-and-imperatives",
                    "explanation-imperative",
                    &["Explain how TCP congestion control works in simple terms."],
                    &["Explain how TCP congestion control works in simple terms."],
                )
                .exact()
                .rejecting(&["Explain how TCP congestion control works in simple terms?"]),
                QualityCase::new(
                    "questions-and-imperatives",
                    "translation-question",
                    &["Can you translate this paragraph into French?"],
                    &["Can you translate this paragraph into French?"],
                )
                .exact()
                .rejecting(&[
                    "Can you translate? This paragraph into French.",
                    "Can you translate this paragraph into French.",
                ]),
                QualityCase::new(passthrough, "invoices-archive", inv, inv)
                    .exact()
                    .rejecting(&[
                        "- Please archive the Q3 invoices and share the folder link with finance.",
                    ]),
                QualityCase::new(passthrough, "benchmark-result", bench, bench)
                    .exact()
                    .rejecting(&[
                        "The benchmark completed in -42 seconds with no dropped frames.",
                        "The benchmark completed in 42 seconds with no dropped frames?",
                        "The benchmark completed in 42 seconds with no dropped frames today.",
                        "The benchmark completed in 42 seconds with dropped frames.",
                    ]),
                QualityCase::new(passthrough, "pr-review-plan", pr, pr)
                    .exact()
                    .rejecting(&[
                        "We will review the pull request tomorrow morning before merging",
                    ]),
            ]);
        }

        fn push_multilingual(&mut self) {
            let multilingual = "multilingual";
            self.cases.extend([
                QualityCase::new(
                    multilingual,
                    "german-self-correction",
                    &["Äh, wir treffen uns morgen um zehn, nein, sorry, um elf Uhr im Konferenzraum."],
                    &[
                        "Wir treffen uns morgen um elf Uhr im Konferenzraum.",
                        "Wir treffen uns morgen um 11 Uhr im Konferenzraum.",
                    ],
                )
                .rejecting(&[
                    "Äh. Wir treffen uns morgen um elf Uhr im Konferenzraum.",
                    "We meet tomorrow at eleven in the conference room.",
                ]),
                QualityCase::new(
                    multilingual,
                    "french-stutter-and-correction",
                    &["Euh, on va on va déployer la mise à jour lundi, enfin non, mardi matin."],
                    &["On va déployer la mise à jour mardi matin."],
                )
                .rejecting(&[
                    "Euh. On va déployer la mise à jour mardi matin.",
                    "On va déployer la mise à jour mardi matin?",
                ]),
                QualityCase::new(
                    multilingual,
                    "spanish-stutter-and-correction",
                    &[
                        "Eh, necesito enviar el el informe al cliente hoy, no, perdón, mañana por la tarde.",
                    ],
                    &["Necesito enviar el informe al cliente mañana por la tarde."],
                )
                .rejecting(&[
                    "Eh. Necesito enviar el informe al cliente mañana por la tarde.",
                ]),
                QualityCase::new(
                    multilingual,
                    "italian-filler-and-stutter",
                    &["Ehm, dobbiamo aggiornare la la documentazione prima del rilascio di giovedì."],
                    &["Dobbiamo aggiornare la documentazione prima del rilascio di giovedì."],
                )
                .rejecting(&[
                    "Ehm. Dobbiamo aggiornare la documentazione prima del rilascio di giovedì.",
                ]),
                QualityCase::new(
                    multilingual,
                    "romanian-filler-and-correction",
                    &[
                        "Ăă, trebuie să trimitem raportul către echipă astăzi, nu, scuze, mâine dimineață.",
                    ],
                    &["Trebuie să trimitem raportul către echipă mâine dimineață."],
                )
                .rejecting(&[
                    "Ăă. Trebuie să trimitem raportul către echipă mâine dimineață.",
                ]),
            ]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::suite::QualitySuite;
    use crate::prose_matcher::matcher::ProseMatcher;

    #[test]
    fn covers_all_categories_and_validates_positive_and_negative_fixtures() {
        let suite = QualitySuite::standard();
        let expected_categories = [
            "explicit-self-correction",
            "mid-thought-restart",
            "thinking-aloud-pruning",
            "cutoff-false-start",
            "forced-pause-break",
            "questions-and-imperatives",
            "clean-passthrough",
            "multilingual",
        ];
        for category in expected_categories {
            assert!(
                suite.cases().iter().any(|case| case.category == category),
                "missing {category}"
            );
        }

        let mut matcher = ProseMatcher::default();
        let mut negative_count = 0;
        for case in suite.cases() {
            assert!(
                !case.expected.is_empty(),
                "{}/{}",
                case.category,
                case.label
            );
            for target in case.expected {
                assert_eq!(
                    matcher.check(case, target),
                    None,
                    "{}/{} rejected {target:?}",
                    case.category,
                    case.label
                );
            }
            for &bad_output in case.negatives {
                assert!(
                    matcher.check(case, bad_output).is_some(),
                    "{} unexpectedly accepted negative fixture {bad_output:?}",
                    case.label
                );
                negative_count += 1;
            }
        }
        assert_eq!(negative_count, 62);
    }
}
