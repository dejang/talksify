"""Targeted English synthetic gap-fill generator addressing all 5 F-406 failure modes.

All prompts and templates are 100% original and engineered directly from the 5 architectural
failure categories (never derived from held-out recordings or eval cases):
1. Scoped self-corrections (keeping the NEW replacement & preserving surrounding sentences)
   + quoted-negation / false-positive "scratch that" hard negative traps
2. Cut-off word stems, contraction restarts, and verb/auxiliary false starts
3. Parakeet + Silero VAD mid-sentence ``. Capital`` pause breaks & stray mid-sentence capitals
4. Thinking-aloud / mid-thought course changes, conversational voice preservation, & colon lead-ins
5. Clean passthroughs, short negations, questions, and imperatives (never answer/follow/drop)
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from conventions import apply_talksify_conventions
from qa_filters import is_retraction_category


@dataclass(frozen=True)
class SyntheticRow:
    raw: str
    clean: str
    lang: str
    source: str
    category: str
    is_retraction: bool = False


_CONTEXT_OPENERS: tuple[str, ...] = (
    "Yesterday we reviewed the telemetry dashboard with the platform team and spotted two latency spikes during peak traffic.",
    "Before the sprint review starts, I want to make sure everyone on the backend group has the updated migration checklist.",
    "After running the benchmark overnight on the staging cluster, we collected enough traces to pinpoint the bottleneck.",
    "During the customer onboarding call this morning, the account manager walked through all three deployment milestones.",
    "When we audited the billing export job last week, we noticed several orders were queued behind the end-of-month batch.",
    "So far the new caching layer has cut database read load noticeably across every regional replica.",
    "Last Tuesday I met with the infrastructure leads to map out the hardware refresh for the second quarter.",
    "Earlier today the support desk forwarded a detailed bug report covering the desktop sync workflow.",
    "While testing the offline mode on a fresh laptop, we verified that local state recovers cleanly after a reboot.",
    "For the upcoming release candidate, QA asked us to freeze schema changes until the regression suite finishes.",
    "In our weekly design sync, the mobile engineers raised a few questions about token refresh timing.",
    "Right after lunch we paired on the ingestion worker and traced the stall back to a locked connection pool.",
)

_CONTEXT_CLOSERS: tuple[str, ...] = (
    "Once that piece lands, we can cut a beta build and ask early testers for feedback.",
    "Let's keep the rollout gradual so we can monitor error budgets at each percentage step.",
    "Hopefully the whole transition stays invisible to end users while simplifying our maintenance burden.",
    "Please drop a quick note in the engineering channel as soon as the pipeline turns green.",
    "I think that gives us a clean fallback path if anything unexpected shows up in production.",
    "Afterwards we should archive the temporary scripts and update the onboarding guide.",
    "That way the on-call rotation won't have to page anyone over the weekend.",
    "We can revisit the secondary metrics next month once the baseline settles.",
)

_CORRECTION_CUES: tuple[str, ...] = (
    "no, sorry,",
    "no, I'm sorry,",
    "wait, no,",
    "actually no,",
    "no wait,",
    "scratch that,",
    "sorry, I meant",
    "wait, I mean",
    "or rather,",
    "never mind, make it",
)

_PERIOD_CORRECTION_CUES: tuple[str, ...] = (
    "No, I'm sorry,",
    "No, sorry,",
    "Wait, no,",
    "Actually no,",
    "Sorry, I meant",
    "Actually, I mean",
)

_NOUN_REPLACEMENTS: tuple[tuple[str, str, str], ...] = (
    ("we should rename the {old} in the sidebar", "tab", "panel"),
    ("we need to update the {old} before shipping", "label", "heading"),
    ("the client asked to replace the {old} on the summary page", "chart", "table"),
    ("let's move the {old} to the upstairs studio", "orientation", "workshop"),
    ("we assigned the {old} to the analytics squad", "ticket", "epic"),
    ("they stored the {old} in the encrypted vault", "token", "certificate"),
    ("please highlight the {old} in the release notes", "fix", "feature"),
    ("we should benchmark the {old} under heavy load", "parser", "indexer"),
    ("I forgot to mention the {old} during standup", "deadline", "milestone"),
    ("we can export the {old} as a compressed archive", "logs", "snapshots"),
    ("remember to double check the {old} in staging", "webhook", "endpoint"),
    ("we picked the wrong {old} for the comparison", "metric", "baseline"),
    ("the designer adjusted the {old} for dark mode", "border", "background"),
    ("we need to migrate the {old} to the new cluster", "queue", "scheduler"),
    ("can we verify the {old} with legal first", "clause", "schedule"),
    ("they referenced the wrong {old} in the handbook", "policy", "procedure"),
    ("we should swap out the {old} in the starter kit", "template", "example"),
    ("please reserve the {old} for the design review", "auditorium", "boardroom"),
)

_SCHEDULE_REPLACEMENTS: tuple[tuple[str, str, str], ...] = (
    ("push the storefront launch to {old}", "October", "November"),
    ("book the train ticket for {old}", "Wednesday morning", "Friday evening"),
    ("raise the retry cap to {old}", "3 attempts", "6 attempts"),
    ("bump the worker limit to {old}", "4 threads", "8 threads"),
    ("assign the hardware order to {old}", "facilities", "operations"),
    ("start the database compaction at {old}", "2 a.m.", "4 a.m."),
    ("cap the batch size at {old}", "50 records", "200 records"),
    ("route traffic to {old}", "Frankfurt", "Dublin"),
    ("keep the retention window at {old}", "14 days", "30 days"),
    ("allocate {old} for the audio worker", "8 gigabytes", "16 gigabytes"),
)

_QUOTED_NEGATION_TEMPLATES: tuple[tuple[str, str], ...] = (
    (
        "I showed the prototype to {person} and {pronoun} looked at the screen saying, no, that's definitely not it, so we went back to the drawing board.",
        'I showed the prototype to {person} and {pronoun} looked at the screen saying, "No, that\'s definitely not it," so we went back to the drawing board.',
    ),
    (
        "During the review {person} laughed and said scratch that, we can do way better than a static banner.",
        'During the review {person} laughed and said, "Scratch that, we can do way better than a static banner."',
    ),
    (
        "When the vendor quoted the renewal price, {person} immediately replied no, sorry, that is completely out of budget for this year.",
        'When the vendor quoted the renewal price, {person} immediately replied, "No, sorry, that is completely out of budget for this year."',
    ),
    (
        "In the postmortem notes someone wrote never mind and closed the thread without explaining why the job failed.",
        'In the postmortem notes someone wrote "never mind" and closed the thread without explaining why the job failed.',
    ),
    (
        "Every time we suggest adding another modal, {person} keeps saying no, that's not what users asked for.",
        'Every time we suggest adding another modal, {person} keeps saying, "No, that\'s not what users asked for."',
    ),
    (
        "He told the support team forget it, don't bother escalating unless the error happens twice in a row.",
        'He told the support team, "Forget it, don\'t bother escalating unless the error happens twice in a row."',
    ),
    (
        "I remember telling {person} no, wait, let's keep the legacy flag around for one more cycle just in case.",
        'I remember telling {person}, "No, wait, let\'s keep the legacy flag around for one more cycle just in case."',
    ),
    (
        "The customer literally wrote scratch that in the reply email because they had attached last month's spreadsheet by mistake.",
        'The customer literally wrote "scratch that" in the reply email because they had attached last month\'s spreadsheet by mistake.',
    ),
    (
        "Don't delete the archive folder even if the cleanup banner says ignore this warning.",
        'Don\'t delete the archive folder even if the cleanup banner says "ignore this warning."',
    ),
    (
        "When {person} saw the initial draft, {pronoun} shook their head saying actually no, this misses the main point entirely.",
        'When {person} saw the initial draft, {pronoun} shook their head saying, "Actually no, this misses the main point entirely."',
    ),
)

_PEOPLE: tuple[tuple[str, str], ...] = (
    ("Elena", "she"),
    ("David", "he"),
    ("Priya", "she"),
    ("Tomas", "he"),
    ("Nadia", "she"),
    ("Daniel", "he"),
    ("Sofia", "she"),
    ("Lukas", "he"),
)

_CUTOFF_PAIRS: tuple[tuple[str, str], ...] = (
    ("in the dir in the direction", "in the direction"),
    ("the arch the architecture", "the architecture"),
    ("for the conf for the configuration", "for the configuration"),
    ("on the dep on the deployment", "on the deployment"),
    ("with the imp with the implementation", "with the implementation"),
    ("to the auth to the authentication", "to the authentication"),
    ("from the doc from the documentation", "from the documentation"),
    ("into the pip into the pipeline", "into the pipeline"),
    ("about the perf about the performance", "about the performance"),
    ("of the sch of the scheduler", "of the scheduler"),
    ("under the env under the environment", "under the environment"),
    ("across the net across the network", "across the network"),
    ("through the ser through the serialization", "through the serialization"),
    ("during the mig during the migration", "during the migration"),
    ("against the reg against the regression", "against the regression"),
    ("around the sync around the synchronization", "around the synchronization"),
)

_CONTRACTION_FALSE_STARTS: tuple[tuple[str, str], ...] = (
    ("It's it runs much faster now.", "It runs much faster now."),
    ("It's it loads reliably on Linux and macOS.", "It loads reliably on Linux and macOS."),
    ("We're we have enough headroom for the launch.", "We have enough headroom for the launch."),
    ("That's that seems like a cleaner abstraction.", "That seems like a cleaner abstraction."),
    ("There's there are three edge cases left to cover.", "There are three edge cases left to cover."),
    ("What we pass is are two items in the payload.", "What we pass are two items in the payload."),
    ("Is it are they planning to join the call?", "Are they planning to join the call?"),
    ("Does it do they support streaming uploads?", "Do they support streaming uploads?"),
    ("Was it were we supposed to tag the release today?", "Were we supposed to tag the release today?"),
    ("The result matches e the is completely deterministic.", "The result is completely deterministic."),
    ("He's he wants to review the diff before we merge.", "He wants to review the diff before we merge."),
    ("She's she already tested the installer on Windows.", "She already tested the installer on Windows."),
    ("I'm I think we should keep the prompt concise.", "I think we should keep the prompt concise."),
    ("You're you can toggle that option in preferences.", "You can toggle that option in preferences."),
    ("They're they need to rotate the signing keys first.", "They need to rotate the signing keys first."),
    ("The metrics is are both trending in the right direction.", "The metrics are both trending in the right direction."),
)

_VAD_PAUSE_TEMPLATES: tuple[tuple[str, str], ...] = (
    (
        "The packaging script compiles smoothly on laptops, while on the. Build server it hits a missing linker flag.",
        "The packaging script compiles smoothly on laptops, while on the build server it hits a missing linker flag.",
    ),
    (
        "Before we. Ship the update to everyone, let's verify the rollback script.",
        "Before we ship the update to everyone, let's verify the rollback script.",
    ),
    (
        "We delayed the migration. Because. The staging replica fell behind by ten minutes.",
        "We delayed the migration because the staging replica fell behind by 10 minutes.",
    ),
    (
        "The audio buffer fills quickly. And. Then the worker flushes it to disk.",
        "The audio buffer fills quickly, and then the worker flushes it to disk.",
    ),
    (
        "Hopefully we can Work towards a simpler setup without. Breaking existing shortcuts.",
        "Hopefully we can work towards a simpler setup without breaking existing shortcuts.",
    ),
    (
        "I checked the config file in. The root directory and Everything looked fine.",
        "I checked the config file in the root directory and everything looked fine.",
    ),
    (
        "We should Make sure the installer cleans up temporary files. When. The user cancels.",
        "We should make sure the installer cleans up temporary files when the user cancels.",
    ),
    (
        "The overlay stays hidden until. You press the hotkey to Start dictating.",
        "The overlay stays hidden until you press the hotkey to start dictating.",
    ),
    (
        "In order to. Keep memory usage under one gigabyte, we quantize the weights.",
        "In order to keep memory usage under one gigabyte, we quantize the weights.",
    ),
    (
        "They wanted to. Compare latency across. Both desktop platforms before deciding.",
        "They wanted to compare latency across both desktop platforms before deciding.",
    ),
    (
        "If the network drops during. Upload, the client Should retry automatically.",
        "If the network drops during upload, the client should retry automatically.",
    ),
    (
        "So the whole point of. Running locally is that. Nothing ever leaves your machine.",
        "So the whole point of running locally is that nothing ever leaves your machine.",
    ),
)

_THINKING_ALOUD_PAIRS: tuple[tuple[str, str], ...] = (
    (
        "So I was thinking we could maybe use a modal, well actually no, I think a small floating bar at the bottom is way less intrusive.",
        "So I think a small floating bar at the bottom is way less intrusive.",
    ),
    (
        "Um, the problem is this. Every time the daemon restarts, it drops the open socket and the client has to reconnect manually.",
        "The problem is this: every time the daemon restarts, it drops the open socket and the client has to reconnect manually.",
    ),
    (
        "So here's what I think. We keep the existing file format for now, and we add a version header so future migrations are painless.",
        "So here's what I think: we keep the existing file format for now, and we add a version header so future migrations are painless.",
    ),
    (
        "Uh, the plan is. First, we benchmark the baseline. Second, we fine-tune the adapter. Third, we validate on held-out clips.",
        "The plan is: First, we benchmark the baseline. Second, we fine-tune the adapter. Third, we validate on held-out clips.",
    ),
    (
        "There are a few things we need to pack. Chargers, adapters, headphones, and a spare cable.",
        "There are a few things we need to pack: chargers, adapters, headphones, and a spare cable.",
    ),
    (
        "Maybe we can also fix the following. Speed up startup time, trim unused assets, and log exact error codes when loading fails.",
        "Maybe we can also fix the following: speed up startup time, trim unused assets, and log exact error codes when loading fails.",
    ),
    (
        "I started to write a custom parser, wait, I mean, we already depend on serde, so let's just derive it automatically.",
        "We already depend on serde, so let's just derive it automatically.",
    ),
    (
        "Well, so really what I'm trying to say is, like, if we cache the prompt prefix, maybe the response drops below half a second.",
        "So really what I'm trying to say is, if we cache the prompt prefix, maybe the response drops below half a second.",
    ),
    (
        "Wait, let me rephrase that. Instead of polling every second, we should trigger the refresh only when the file changes on disk.",
        "Instead of polling every second, we should trigger the refresh only when the file changes on disk.",
    ),
    (
        "Actually I don't think we should block the release, I think we should ship the core fix now and follow up next week.",
        "Actually, I don't think we should block the release. I think we should ship the core fix now and follow up next week.",
    ),
    (
        "Maybe use two formats, well no, sticking with a single schema saves a lot of maintenance work.",
        "Sticking with a single schema saves a lot of maintenance work.",
    ),
    (
        "Maybe spawn a daemon, or wait, no, keeping the worker inside the main process avoids IPC overhead.",
        "Keeping the worker inside the main process avoids IPC overhead.",
    ),
)

_CLEAN_PASSTHROUGHS: tuple[str, ...] = (
    "I don't want that feature enabled by default.",
    "We don't need the extra confirmation dialog.",
    "What time does the deployment window open tomorrow?",
    "How do we rotate the API credentials without downtime?",
    "Could you check whether the macOS notarization ticket cleared before noon?",
    "Write a concise changelog entry covering the Linux audio fix.",
    "Export the audio waveform diagram as a crisp PNG for the handbook.",
    "Please inspect Cargo.toml and bump the patch version to 0.4.2.",
    "The acoustic calibration finished in 7 minutes with steady microphone gain.",
    "Make sure the export saves to summary.csv and keeps 100% of the rows.",
    "I still have to fix the login redirect, the slow filter, and the broken CSV export.",
    "So I really think this architecture is simple, fast, and easy to debug.",
    "Hopefully the new driver resolves the crackle we heard on Bluetooth headsets.",
    "Why did the telemetry collector drop packets after the firmware swap?",
    "Stop the worker, clear the stale lockfile, and restart the service.",
    "No, I don't agree with removing the safety check before we have metrics.",
)


def _add_english_fillers(text: str, rng: random.Random) -> str:
    """Inject realistic spoken fillers into an English sentence for training variety."""
    fillers = ("um,", "uh,", "uh", "um", "erm,")
    words = text.split()
    if len(words) < 5:
        return f"Um, {text[0].lower() + text[1:]}" if text[0].isupper() else f"um, {text}"
    pos = rng.randint(1, min(4, len(words) - 2))
    filler = rng.choice(fillers)
    words.insert(pos, filler)
    return " ".join(words)


def generate_english_gap_fill(seed: int = 407, target_count: int = 2400) -> list[SyntheticRow]:
    """Generate targeted English synthetic pairs covering all 5 F-406 failure modes.

    Deterministic given ``seed``. Returns up to ``target_count`` unique pairs spanning:
    - Scoped local self-corrections in short & long (100-220 word) multi-sentence dictations
    - Quoted-negation & false-positive "scratch that" hard negative traps
    - Cut-off word stems, contraction false starts, and auxiliary/verb restarts
    - Parakeet + Silero VAD mid-sentence ``. Capital`` pause breaks & stray mid-sentence capitals
    - Thinking-aloud / mid-thought course changes & colon lead-ins
    - Clean passthroughs, negations, questions, and imperatives
    """
    rng = random.Random(seed)
    rows: list[SyntheticRow] = []
    seen_raw: set[str] = set()

    def add(raw: str, clean: str, category: str) -> None:
        raw_s = " ".join(raw.split())
        clean_s = apply_talksify_conventions(" ".join(clean.split()))
        key = raw_s.lower()
        if key in seen_raw or not raw_s or not clean_s:
            return
        seen_raw.add(key)
        rows.append(
            SyntheticRow(
                raw=raw_s,
                clean=clean_s,
                lang="en",
                source="synthetic_en_gapfill",
                category=category,
                is_retraction=is_retraction_category(category, "synthetic_en_gapfill"),
            )
        )

    # 1. Scoped local self-corrections (multi-sentence & long dictations so the model learns
    #    never to wipe preceding sentences and always to keep the NEW replacement).
    n_scoped = max(300, int(target_count * 0.28))
    for _ in range(n_scoped * 2):
        if sum(1 for r in rows if r.category == "scoped_self_correction") >= n_scoped:
            break
        n_openers = rng.choice([0, 1, 1, 2, 3])
        openers = rng.sample(_CONTEXT_OPENERS, k=n_openers)
        closer = rng.choice(_CONTEXT_CLOSERS) if rng.random() < 0.7 else ""
        cue = rng.choice(_CORRECTION_CUES)
        use_noun = rng.random() < 0.65
        use_period_sep = rng.random() < 0.35
        if use_noun:
            tpl, old, new = rng.choice(_NOUN_REPLACEMENTS)
            if use_period_sep:
                pcue = rng.choice(_PERIOD_CORRECTION_CUES)
                mid_raw = f"Also, {tpl.format(old=old)}. {pcue} the {new}."
            else:
                mid_raw = f"Also, {tpl.format(old=old)}, {cue} {new}."
            mid_clean = f"Also, {tpl.format(old=new)}."
        else:
            tpl, old, new = rng.choice(_SCHEDULE_REPLACEMENTS)
            lead = rng.choice(["We should", "Let's", "I think we need to", "Can we"])
            punct = "?" if lead == "Can we" else "."
            if use_period_sep and punct == ".":
                pcue = rng.choice(_PERIOD_CORRECTION_CUES)
                mid_raw = f"{lead} {tpl.format(old=old)}. {pcue} {new}."
            else:
                mid_raw = f"{lead} {tpl.format(old=old)}, {cue} {new}{punct}"
            mid_clean = f"{lead} {tpl.format(old=new)}{punct}"

        if rng.random() < 0.45:
            mid_raw = _add_english_fillers(mid_raw, rng)

        raw_parts = [*openers, mid_raw, *([closer] if closer else [])]
        clean_parts = [*openers, mid_clean, *([closer] if closer else [])]
        add(" ".join(raw_parts), " ".join(clean_parts), "scoped_self_correction")

    # 2. Quoted-negation & false-positive "scratch that" hard negative traps (MUST NOT DELETE!).
    n_quoted = max(200, int(target_count * 0.16))
    for _ in range(n_quoted * 3):
        if sum(1 for r in rows if r.category == "quoted_negation_trap") >= n_quoted:
            break
        person, pronoun = rng.choice(_PEOPLE)
        raw_tpl, clean_tpl = rng.choice(_QUOTED_NEGATION_TEMPLATES)
        raw_mid = raw_tpl.format(person=person, pronoun=pronoun)
        clean_mid = clean_tpl.format(person=person, pronoun=pronoun)
        if rng.random() < 0.4:
            raw_mid = _add_english_fillers(raw_mid, rng)
        openers = rng.sample(_CONTEXT_OPENERS, k=rng.choice([0, 1, 2]))
        closer = rng.choice(_CONTEXT_CLOSERS) if rng.random() < 0.5 else ""
        raw_parts = [*openers, raw_mid, *([closer] if closer else [])]
        clean_parts = [*openers, clean_mid, *([closer] if closer else [])]
        add(" ".join(raw_parts), " ".join(clean_parts), "quoted_negation_trap")

    # 3. Cut-off word stems & contraction/verb false starts.
    n_cutoff = max(250, int(target_count * 0.18))
    cutoff_frames = (
        ("We are moving {frag} that the team agreed on.", "{clean}"),
        ("Take a look at {frag} before we freeze the branch.", "{clean}"),
        ("Everything depends on {frag} holding steady under load.", "{clean}"),
        ("I noticed a small issue {frag} during yesterday's test.", "{clean}"),
        ("Let's align {frag} so both services share the same schema.", "{clean}"),
    )
    for raw_c, clean_c in _CONTRACTION_FALSE_STARTS:
        add(raw_c, clean_c, "cutoff_false_start")
    for _ in range(n_cutoff * 3):
        if sum(1 for r in rows if r.category == "cutoff_false_start") >= n_cutoff:
            break
        if rng.random() < 0.6:
            frag_raw, frag_clean = rng.choice(_CUTOFF_PAIRS)
            frame, _ = rng.choice(cutoff_frames)
            raw_s = frame.format(frag=frag_raw)
            clean_s = frame.format(frag=frag_clean)
        else:
            raw_s, clean_s = rng.choice(_CONTRACTION_FALSE_STARTS)
        if rng.random() < 0.35:
            raw_s = _add_english_fillers(raw_s, rng)
        if rng.random() < 0.5:
            opener = rng.choice(_CONTEXT_OPENERS)
            raw_s = f"{opener} {raw_s}"
            clean_s = f"{opener} {clean_s}"
        if rng.random() < 0.35:
            closer = rng.choice(_CONTEXT_CLOSERS)
            raw_s = f"{raw_s} {closer}"
            clean_s = f"{clean_s} {closer}"
        add(raw_s, clean_s, "cutoff_false_start")

    # 4. Parakeet + Silero VAD mid-sentence `. Capital` pause breaks & stray capitals.
    n_vad = max(200, int(target_count * 0.14))
    for raw_v, clean_v in _VAD_PAUSE_TEMPLATES:
        add(raw_v, clean_v, "vad_pause_break")
    for _ in range(n_vad * 3):
        if sum(1 for r in rows if r.category == "vad_pause_break") >= n_vad:
            break
        raw_v, clean_v = rng.choice(_VAD_PAUSE_TEMPLATES)
        if rng.random() < 0.4:
            raw_v = _add_english_fillers(raw_v, rng)
        openers = rng.sample(_CONTEXT_OPENERS, k=rng.choice([0, 1]))
        closer = rng.choice(_CONTEXT_CLOSERS) if rng.random() < 0.5 else ""
        raw_parts = [*openers, raw_v, *([closer] if closer else [])]
        clean_parts = [*openers, clean_v, *([closer] if closer else [])]
        add(" ".join(raw_parts), " ".join(clean_parts), "vad_pause_break")

    # 5. Thinking-aloud / mid-thought course changes & colon lead-ins.
    n_think = max(180, int(target_count * 0.12))
    for raw_t, clean_t in _THINKING_ALOUD_PAIRS:
        add(raw_t, clean_t, "thinking_aloud")
    for _ in range(n_think * 3):
        if sum(1 for r in rows if r.category == "thinking_aloud") >= n_think:
            break
        raw_t, clean_t = rng.choice(_THINKING_ALOUD_PAIRS)
        openers = rng.sample(_CONTEXT_OPENERS, k=rng.choice([0, 1, 2]))
        closer = rng.choice(_CONTEXT_CLOSERS) if rng.random() < 0.5 else ""
        raw_parts = [*openers, raw_t, *([closer] if closer else [])]
        clean_parts = [*openers, clean_t, *([closer] if closer else [])]
        add(" ".join(raw_parts), " ".join(clean_parts), "thinking_aloud")

    # 6. Clean passthroughs, short negations, questions, and imperatives (strict raw == clean).
    n_pass = max(180, int(target_count * 0.12))
    for s in _CLEAN_PASSTHROUGHS:
        norm = apply_talksify_conventions(s)
        add(norm, norm, "clean_passthrough")
    for _ in range(n_pass * 3):
        if sum(1 for r in rows if r.category == "clean_passthrough") >= n_pass:
            break
        k = rng.choice([1, 2, 3])
        if k == 1:
            norm = apply_talksify_conventions(rng.choice(_CLEAN_PASSTHROUGHS))
            add(norm, norm, "clean_passthrough")
        else:
            parts = [rng.choice(_CONTEXT_OPENERS), rng.choice(_CLEAN_PASSTHROUGHS)]
            if k == 3:
                parts.append(rng.choice(_CONTEXT_CLOSERS))
            joined = apply_talksify_conventions(" ".join(parts))
            add(joined, joined, "clean_passthrough")

    rng.shuffle(rows)
    return rows[:target_count]
