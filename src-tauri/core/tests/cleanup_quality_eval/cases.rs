use crate::prose_matcher::QualityCase;

/// Synthetic, corpus-independent cleanup quality benchmark suite.
pub(crate) struct QualitySuite {
    cases: Vec<QualityCase>,
}

impl QualitySuite {
    pub(crate) fn standard() -> Self {
        let mut suite = Self { cases: Vec::new() };
        suite.push_corrections_and_restarts();
        suite.push_pruning_and_cutoffs();
        suite.push_pauses_and_restraint();
        suite.push_passthrough();
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

    fn push_corrections_and_restarts(&mut self) {
        self.cases.extend([
            QualityCase::new(
                "explicit-self-correction",
                "recipient-and-day",
                &["Send the update to Alice. No, sorry, send it to Bob on Friday."],
                &["Send the update to Bob on Friday."],
            ),
            QualityCase::new(
                "explicit-self-correction",
                "meeting-day",
                &["Book the room for Tuesday, actually make that Thursday afternoon."],
                &[
                    "Book the room for Thursday afternoon.",
                    "Book the room on Thursday afternoon.",
                ],
            ),
            QualityCase::new(
                "explicit-self-correction",
                "timeout-value",
                &["Set the timeout to five seconds. Wait, scratch that, make it thirty seconds."],
                &[
                    "Set the timeout to 30 seconds.",
                    "Set the timeout to thirty seconds.",
                    "Make the timeout 30 seconds.",
                ],
            ),
            QualityCase::new(
                "explicit-self-correction",
                "deploy-target",
                &["Deploy to staging first. I mean production, directly to production."],
                &["Deploy directly to production."],
            ),
            QualityCase::new(
                "mid-thought-restart",
                "query-return-type",
                &["The query returns false. Actually it returns the list of matched records."],
                &[
                    "The query returns the list of matched records.",
                    "Actually, the query returns the list of matched records.",
                    "The query actually returns the list of matched records.",
                ],
            ),
            QualityCase::new(
                "mid-thought-restart",
                "cache-location",
                &["We tried caching the tokens, no, we moved the cache into the worker pool instead."],
                &[
                    "We moved the cache into the worker pool instead.",
                    "We moved the cache into the worker pool.",
                ],
            ),
            QualityCase::new(
                "mid-thought-restart",
                "parser-format",
                &["The parser expects JSON. Well, actually it expects YAML with strict keys."],
                &[
                    "The parser expects YAML with strict keys.",
                    "Actually, the parser expects YAML with strict keys.",
                    "The parser actually expects YAML with strict keys.",
                ],
            ),
        ]);
    }

    fn push_pruning_and_cutoffs(&mut self) {
        self.cases.extend([
            QualityCase::new(
                "thinking-aloud-pruning",
                "config-path",
                &["We can pass the config path, or maybe yeah the config path is fine."],
                &[
                    "We can pass the config path.",
                    "The config path is fine.",
                    "We can pass the config path, that's fine.",
                    "We can pass the config path; it's fine.",
                ],
            ),
            QualityCase::new(
                "thinking-aloud-pruning",
                "flag-name",
                &["Let's name the flag verbose, or wait, yeah let's keep it verbose for now."],
                &[
                    "Let's keep the flag verbose for now.",
                    "Let's name the flag verbose for now.",
                    "Let's keep it verbose for now.",
                ],
            ),
            QualityCase::new(
                "thinking-aloud-pruning",
                "index-instead-of-split",
                &["Should we split the table? Actually never mind, just add an index on created_at."],
                &[
                    "Just add an index on created_at.",
                    "Add an index on created_at.",
                ],
            ),
            QualityCase::new(
                "cutoff-false-start",
                "contraction-restart",
                &["It's it works every time."],
                &["It works every time."],
            ),
            QualityCase::new(
                "cutoff-false-start",
                "partial-word-prefix",
                &["More in the dir in the direction of lower latency."],
                &["More in the direction of lower latency."],
            ),
            QualityCase::new(
                "cutoff-false-start",
                "verb-repair",
                &["What we're passing in is are two separate files."],
                &[
                    "What we're passing in are 2 separate files.",
                    "What we're passing in are two separate files.",
                ],
            ),
            QualityCase::new(
                "cutoff-false-start",
                "truncated-noun",
                &["The arch the architecture diagram needs an update before review."],
                &["The architecture diagram needs an update before review."],
            ),
        ]);
    }

    fn push_pauses_and_restraint(&mut self) {
        self.cases.extend([
            QualityCase::new(
                "forced-pause-break",
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
            .strict_prose(),
            QualityCase::new(
                "forced-pause-break",
                "feature-flag",
                &[
                    "Make sure the migration runs before we.",
                    "Flip the feature flag in production.",
                ],
                &["Make sure the migration runs before we flip the feature flag in production."],
            )
            .strict_prose(),
            QualityCase::new(
                "forced-pause-break",
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
            .strict_prose(),
            QualityCase::new(
                "questions-and-imperatives",
                "yes-no-question",
                &["Did the nightly backup finish?"],
                &["Did the nightly backup finish?"],
            )
            .exact(),
            QualityCase::new(
                "questions-and-imperatives",
                "code-generation-imperative",
                &["Write a Python script to parse the CSV files."],
                &["Write a Python script to parse the CSV files."],
            )
            .exact(),
            QualityCase::new(
                "questions-and-imperatives",
                "explanation-imperative",
                &["Explain how TCP congestion control works in simple terms."],
                &["Explain how TCP congestion control works in simple terms."],
            )
            .exact(),
            QualityCase::new(
                "questions-and-imperatives",
                "translation-question",
                &["Can you translate this paragraph into French?"],
                &["Can you translate this paragraph into French?"],
            )
            .exact(),
        ]);
    }

    fn push_passthrough(&mut self) {
        let cat = "clean-passthrough";
        let inv: &[&str] =
            &["Please archive the Q3 invoices and share the folder link with finance."];
        let bench: &[&str] = &["The benchmark completed in 42 seconds with no dropped frames."];
        let pr: &[&str] = &["We will review the pull request tomorrow morning before merging."];
        self.cases.extend([
            QualityCase::new(cat, "invoices-archive", inv, inv).exact(),
            QualityCase::new(cat, "benchmark-result", bench, bench).exact(),
            QualityCase::new(cat, "pr-review-plan", pr, pr).exact(),
        ]);
    }

    fn push_multilingual(&mut self) {
        let cat = "multilingual";
        self.cases.extend([
            QualityCase::new(
                cat,
                "german-self-correction",
                &["Äh, wir treffen uns morgen um zehn, nein, sorry, um elf Uhr im Konferenzraum."],
                &[
                    "Wir treffen uns morgen um elf Uhr im Konferenzraum.",
                    "Wir treffen uns morgen um 11 Uhr im Konferenzraum.",
                ],
            ),
            QualityCase::new(
                cat,
                "french-stutter-and-correction",
                &["Euh, on va on va déployer la mise à jour lundi, enfin non, mardi matin."],
                &["On va déployer la mise à jour mardi matin."],
            ),
            QualityCase::new(
                cat,
                "spanish-stutter-and-correction",
                &[
                    "Eh, necesito enviar el el informe al cliente hoy, no, perdón, mañana por la tarde.",
                ],
                &["Necesito enviar el informe al cliente mañana por la tarde."],
            ),
            QualityCase::new(
                cat,
                "italian-filler-and-stutter",
                &["Ehm, dobbiamo aggiornare la la documentazione prima del rilascio di giovedì."],
                &["Dobbiamo aggiornare la documentazione prima del rilascio di giovedì."],
            ),
            QualityCase::new(
                cat,
                "romanian-filler-and-correction",
                &[
                    "Ăă, trebuie să trimitem raportul către echipă astăzi, nu, scuze, mâine dimineață.",
                ],
                &["Trebuie să trimitem raportul către echipă mâine dimineață."],
            ),
        ]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prose_matcher::ProseMatcher;

    #[test]
    fn covers_all_categories_and_accepts_expected_outputs() {
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
        }
    }

    #[test]
    fn rejects_negative_fixtures_across_categories() {
        let suite = QualitySuite::standard();
        let mut matcher = ProseMatcher::default();
        let negatives: &[(&str, &str)] = &[
            ("recipient-and-day", "Do not send the update to Bob on Friday."),
            ("recipient-and-day", "Send the update to Alice. Send it to Bob on Friday."),
            ("recipient-and-day", "Send the update to Bob on Friday?"),
            ("recipient-and-day", "Send the update to Bob on Friday!"),
            ("recipient-and-day", "Send the update to Bob on Friday"),
            ("meeting-day", "Do not book the room for Thursday afternoon."),
            ("meeting-day", "Book the room for Thursday afternoon?"),
            ("timeout-value", "Set the timeout to .30 seconds."),
            ("timeout-value", "Set the timeout to 30.5 seconds."),
            ("timeout-value", "Set the timeout to 30% seconds."),
            ("timeout-value", "Set the timeout to ,30 seconds."),
            ("timeout-value", "Set the timeout to -30 seconds."),
            ("timeout-value", "Set the timeout to −30 seconds."),
            ("timeout-value", "Set the timeout to +30 seconds."),
            ("timeout-value", "Make the timeout -30 seconds."),
            ("deploy-target", "Deploy directly to production first."),
            ("deploy-target", "Deploy to production first."),
            ("deploy-target", "Deploy to production."),
            ("deploy-target", "Deploy directly to production?"),
            ("query-return-type", "The query returns false. Actually, it returns the list of matched records."),
            ("config-path", "We can pass the config path, or maybe yeah, the config path is fine."),
            ("flag-name", "Let's name the flag verbose."),
            ("flag-name", "Let's keep the flag verbose."),
            ("index-instead-of-split", "Should we split the table? Just add an index on created_at."),
            ("partial-word-prefix", "More in the dir in the direction of lower latency."),
            ("partial-word-prefix", "In the direction of lower latency."),
            ("verb-repair", "What we're passing in are -2 separate files."),
            ("release-runner", "The build is fast locally, but on the. Release runner it stalls."),
            ("release-runner", "The build is fast locally, but on the Release runner it stalls."),
            ("release-runner", "The build is fast locally, but on the, release runner it stalls."),
            ("release-runner", "The build. Is fast locally, but on the release runner it stalls."),
            ("release-runner", "The build is fast locally. But on the release runner it stalls."),
            ("feature-flag", "Make sure the migration runs before we. Flip the feature flag in production."),
            ("feature-flag", "Make sure the migration runs before we Flip the feature flag in production."),
            ("feature-flag", "Make sure the migration runs before we, flip the feature flag in production."),
            ("feature-flag", "Make sure. The migration runs before we flip the feature flag in production."),
            ("metrics-endpoint", "After restarting the daemon, the metrics. Endpoint started responding immediately."),
            ("metrics-endpoint", "After restarting the daemon, the metrics Endpoint started responding immediately."),
            ("metrics-endpoint", "After restarting the daemon, the metrics, endpoint started responding immediately."),
            ("metrics-endpoint", "After restarting the daemon. The metrics endpoint started responding immediately."),
            ("yes-no-question", "Did the nightly backup finish? No."),
            ("yes-no-question", "Did the nightly backup finish? Yes."),
            ("yes-no-question", "Did the nightly backup finish."),
            ("yes-no-question", "Did the nightly? Backup finish."),
            ("translation-question", "Can you translate? This paragraph into French."),
            ("translation-question", "Can you translate this paragraph into French."),
            ("code-generation-imperative", "Sure! Here is a Python script to parse the CSV files."),
            ("code-generation-imperative", "Write a Python script to parse the CSV files?"),
            ("explanation-imperative", "Explain how TCP congestion control works in simple terms?"),
            ("benchmark-result", "The benchmark completed in -42 seconds with no dropped frames."),
            ("benchmark-result", "The benchmark completed in 42 seconds with no dropped frames?"),
            ("benchmark-result", "The benchmark completed in 42 seconds with no dropped frames today."),
            ("benchmark-result", "The benchmark completed in 42 seconds with dropped frames."),
            ("pr-review-plan", "We will review the pull request tomorrow morning before merging"),
            ("invoices-archive", "- Please archive the Q3 invoices and share the folder link with finance."),
            ("german-self-correction", "Äh. Wir treffen uns morgen um elf Uhr im Konferenzraum."),
            ("german-self-correction", "We meet tomorrow at eleven in the conference room."),
            ("french-stutter-and-correction", "Euh. On va déployer la mise à jour mardi matin."),
            ("french-stutter-and-correction", "On va déployer la mise à jour mardi matin?"),
            ("spanish-stutter-and-correction", "Eh. Necesito enviar el informe al cliente mañana por la tarde."),
            ("italian-filler-and-stutter", "Ehm. Dobbiamo aggiornare la documentazione prima del rilascio di giovedì."),
            ("romanian-filler-and-correction", "Ăă. Trebuie să trimitem raportul către echipă mâine dimineață."),
        ];
        for &(label, bad_output) in negatives {
            let case = suite
                .cases()
                .iter()
                .find(|case| case.label == label)
                .unwrap_or_else(|| panic!("unknown case {label}"));
            assert!(
                matcher.check(case, bad_output).is_some(),
                "{label} unexpectedly accepted negative fixture {bad_output:?}"
            );
        }
    }
}
