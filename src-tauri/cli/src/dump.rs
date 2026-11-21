use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tempfile::NamedTempFile;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct RawEntry {
    pub(crate) file: RecordingName,
    pub(crate) duration_secs: f64,
    pub(crate) expected: String,
    pub(crate) raw: String,
}

/// Recording filename stored in raw dumps and reports.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub(crate) struct RecordingName(String);

impl RecordingName {
    /// Extracts a path's filename component into a [`RecordingName`].
    pub(crate) fn from_path(path: &Path) -> Self {
        Self(
            path.file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned(),
        )
    }

    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }

    /// Renders this recording name as a single-line, unambiguous quoted string.
    pub(crate) fn quoted(&self) -> String {
        format!("{:?}", self.0)
    }
}

/// Validated collection of raw-transcript entries for JSON dump and fast text-only replay.
#[derive(Debug, Default, Clone)]
pub(crate) struct RawDump {
    entries: Vec<RawEntry>,
}

impl RawDump {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn read(path: &Path) -> Result<Self, String> {
        let json =
            fs::read_to_string(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        let entries: Vec<RawEntry> = serde_json::from_str(&json)
            .map_err(|e| format!("cannot parse {}: {e}", path.display()))?;
        let dump = Self { entries };
        dump.validate(path)?;
        Ok(dump)
    }

    pub(crate) fn push(&mut self, entry: RawEntry) {
        self.entries.push(entry);
    }

    pub(crate) fn entries(&self) -> &[RawEntry] {
        &self.entries
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn validate(&self, path: &Path) -> Result<(), String> {
        for (index, entry) in self.entries.iter().enumerate() {
            if !entry.duration_secs.is_finite() || entry.duration_secs < 0.0 {
                return Err(format!(
                    "invalid duration_secs ({}) for entry {index} ({:?}) in {}",
                    entry.duration_secs,
                    entry.file.as_str(),
                    path.display()
                ));
            }
        }
        Ok(())
    }

    fn to_json_pretty(&self, path: &Path) -> Result<String, String> {
        self.validate(path)?;
        serde_json::to_string_pretty(&self.entries)
            .map_err(|e| format!("cannot serialize {}: {e}", path.display()))
    }
}

/// Staging file that accumulates a [`RawDump`] for atomic no-clobber publication.
pub(crate) struct StagedDump {
    destination: PathBuf,
    staging: NamedTempFile,
    dump: RawDump,
}

impl StagedDump {
    pub(crate) fn prepare(path: &Path) -> Result<Self, String> {
        let bytes = path.as_os_str().as_encoded_bytes();
        let trailing_sep = bytes
            .last()
            .is_some_and(|&byte| std::path::is_separator(char::from(byte)));
        let trailing_cur_dir = bytes == b"."
            || (bytes.len() >= 2
                && bytes[bytes.len() - 1] == b'.'
                && std::path::is_separator(char::from(bytes[bytes.len() - 2])));
        if trailing_sep || trailing_cur_dir || path.file_name().is_none() {
            return Err(format!(
                "cannot write {}: invalid destination path",
                path.display()
            ));
        }
        match fs::symlink_metadata(path) {
            Ok(_) => {
                return Err(format!(
                    "cannot write {}: destination already exists",
                    path.display()
                ));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(format!("cannot inspect {}: {error}", path.display())),
        }

        let parent = path
            .parent()
            .filter(|dir| !dir.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let staging = NamedTempFile::new_in(parent)
            .map_err(|e| format!("cannot create staging file for {}: {e}", path.display()))?;
        Ok(Self {
            destination: path.to_path_buf(),
            staging,
            dump: RawDump::new(),
        })
    }

    pub(crate) fn destination(&self) -> &Path {
        &self.destination
    }

    pub(crate) fn push(&mut self, entry: RawEntry) {
        self.dump.push(entry);
    }

    pub(crate) fn publish(self) -> Result<usize, String> {
        let Self {
            destination,
            mut staging,
            dump,
        } = self;
        let json = dump.to_json_pretty(&destination)?;
        staging
            .write_all(json.as_bytes())
            .and_then(|()| staging.as_file().sync_all())
            .map_err(|e| format!("cannot write {}: {e}", destination.display()))?;
        staging.persist_noclobber(&destination).map_err(|err| {
            if err.error.kind() == std::io::ErrorKind::AlreadyExists {
                format!(
                    "cannot write {}: destination already exists",
                    destination.display()
                )
            } else {
                format!("cannot publish {}: {}", destination.display(), err.error)
            }
        })?;
        Ok(dump.len())
    }
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;

    #[test]
    fn raw_dump_round_trips_and_rejects_negative_durations() {
        let dir = TempDir::new().unwrap();
        let dump_path = dir.path().join("raw.json");
        let entries = vec![
            RawEntry {
                file: RecordingName::from_path(Path::new("short.wav")),
                duration_secs: 12.5,
                expected: "Hello world.".into(),
                raw: "um hello world.".into(),
            },
            RawEntry {
                file: RecordingName::from_path(Path::new("long.wav")),
                duration_secs: 75.25,
                expected: "12 files in .wav.\nSecond line.".into(),
                raw: "twelve files in dot wav.\nsecond line.".into(),
            },
        ];

        let mut staged = StagedDump::prepare(&dump_path).unwrap();
        for entry in &entries {
            staged.push(entry.clone());
        }
        assert_eq!(staged.publish().unwrap(), entries.len());
        let loaded = RawDump::read(&dump_path).unwrap();
        assert_eq!(loaded.len(), entries.len());
        for (actual, expected) in loaded.entries().iter().zip(&entries) {
            assert_eq!(actual.file, expected.file);
            assert!((actual.duration_secs - expected.duration_secs).abs() < 1e-9);
            assert_eq!(actual.expected, expected.expected);
            assert_eq!(actual.raw, expected.raw);
        }

        for (name, duration) in [("neg_long.json", -90.0), ("neg_small.json", -0.01)] {
            let bad_path = dir.path().join(name);
            let json = format!(
                r#"[{{"file":"bad.wav","duration_secs":{duration},"expected":"Hello.","raw":"hello."}}]"#
            );
            fs::write(&bad_path, json).unwrap();
            assert!(
                RawDump::read(&bad_path).is_err(),
                "expected invalid duration {duration} to fail"
            );
        }
    }

    #[test]
    fn staged_dump_publishes_safely_without_clobbering() {
        let dir = TempDir::new().unwrap();
        let sample = RawEntry {
            file: RecordingName::from_path(Path::new("sample.wav")),
            duration_secs: 4.5,
            expected: "Hello world.".into(),
            raw: "hello world.".into(),
        };

        // Normal and 250-byte destinations both publish atomically with owner-only permissions.
        let long_name = format!("{}.json", "a".repeat(245));
        for filename in ["raw.json", long_name.as_str()] {
            let sub = TempDir::new_in(dir.path()).unwrap();
            let dest = sub.path().join(filename);
            let mut staged = StagedDump::prepare(&dest).unwrap();

            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let mut staged_count = 0;
                for entry in fs::read_dir(sub.path()).unwrap() {
                    staged_count += 1;
                    let mode = fs::symlink_metadata(entry.unwrap().path())
                        .unwrap()
                        .permissions()
                        .mode()
                        & 0o777;
                    assert_eq!(mode, 0o600, "staging file must be owner-only");
                }
                assert_eq!(staged_count, 1);
            }

            staged.push(sample.clone());
            assert_eq!(staged.publish().unwrap(), 1);
            assert_eq!(RawDump::read(&dest).unwrap().len(), 1);

            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let mode = fs::symlink_metadata(&dest).unwrap().permissions().mode() & 0o777;
                assert_eq!(mode, 0o600, "published dump must stay owner-only");
            }
        }

        // Pre-existing and concurrently-created destinations fail without clobbering or leaking temps.
        let existing = dir.path().join("existing.json");
        fs::write(&existing, "keep").unwrap();
        assert!(StagedDump::prepare(&existing).is_err());
        assert_eq!(fs::read_to_string(&existing).unwrap(), "keep");

        let raced_dir = TempDir::new_in(dir.path()).unwrap();
        let raced = raced_dir.path().join("raced.json");
        let mut staged_raced = StagedDump::prepare(&raced).unwrap();
        fs::write(&raced, "concurrent").unwrap();
        staged_raced.push(sample);
        assert!(staged_raced.publish().is_err());
        assert_eq!(fs::read_to_string(&raced).unwrap(), "concurrent");
        let remaining: Vec<_> = fs::read_dir(raced_dir.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(remaining, vec!["raced.json"]);
    }
}
