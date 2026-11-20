//! Reading Unity `.meta` files.
//!
//! A `.meta` file is YAML, but everything the asset database needs (format version, GUID, folder
//! flag and importer kind) lives on top-level lines, so a line scanner is enough and avoids
//! parsing megabytes of sprite-sheet data. Importer *settings* are intentionally not interpreted
//! yet; they stay on disk untouched.

use crate::guid::{Guid, GuidError};

/// What the asset database learns from one `.meta` file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetaData {
    /// `fileFormatVersion` (Unity currently writes 2).
    pub format_version: u32,
    /// The asset's stable identity.
    pub guid: Guid,
    /// `folderAsset: yes`.
    pub folder_asset: bool,
    /// Name of the importer block, e.g. `TextureImporter`, `DefaultImporter`, `MonoImporter`.
    pub importer: Option<String>,
}

/// Why a `.meta` file could not be understood.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MetaError {
    /// The file is not valid UTF-8.
    #[error("not valid UTF-8 text")]
    NotUtf8,
    /// No `fileFormatVersion:` line.
    #[error("missing `fileFormatVersion`")]
    MissingFormatVersion,
    /// `fileFormatVersion` is not a number.
    #[error("`fileFormatVersion` is not a number: `{0}`")]
    BadFormatVersion(String),
    /// No `guid:` line.
    #[error("missing `guid`")]
    MissingGuid,
    /// `guid:` is malformed.
    #[error("invalid `guid`: {0}")]
    BadGuid(#[from] GuidError),
}

impl MetaData {
    /// Parse the bytes of a `.meta` file.
    pub fn parse(bytes: &[u8]) -> Result<Self, MetaError> {
        let text = std::str::from_utf8(bytes).map_err(|_| MetaError::NotUtf8)?;
        let text = text.strip_prefix('\u{feff}').unwrap_or(text);

        let mut format_version = None;
        let mut guid = None;
        let mut folder_asset = false;
        let mut importer = None;

        for line in text.lines() {
            // Only top-level keys are interesting; nested settings are indented.
            if line.is_empty() || line.starts_with([' ', '\t', '#', '-']) {
                continue;
            }
            let Some((key, value)) = line.split_once(':') else {
                continue;
            };
            let value = value.trim();
            match key {
                "fileFormatVersion" => {
                    format_version = Some(
                        value
                            .parse::<u32>()
                            .map_err(|_| MetaError::BadFormatVersion(value.to_owned()))?,
                    );
                }
                "guid" => guid = Some(value.parse::<Guid>()?),
                "folderAsset" => folder_asset = value == "yes",
                _ if value.is_empty() && key.ends_with("Importer") && importer.is_none() => {
                    importer = Some(key.to_owned());
                }
                _ => {}
            }
        }

        Ok(Self {
            format_version: format_version.ok_or(MetaError::MissingFormatVersion)?,
            guid: guid.ok_or(MetaError::MissingGuid)?,
            folder_asset,
            importer,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Built line by line: a `\` string continuation would strip the indentation that makes the
    /// second `guid:` a nested key.
    fn texture_meta() -> String {
        [
            "fileFormatVersion: 2",
            "guid: 8f1c2d3e4a5b60718293a4b5c6d7e8f9",
            "TextureImporter:",
            "  internalIDToNameTable: []",
            "  serializedVersion: 12",
            "  guid: ffffffffffffffffffffffffffffffff",
            "  mipmaps:",
            "    mipMapMode: 0",
            "  userData: ",
            "",
        ]
        .join("\r\n")
    }

    #[test]
    fn parses_texture_meta_with_crlf_and_ignores_nested_keys() {
        let m = MetaData::parse(texture_meta().as_bytes()).unwrap();
        assert_eq!(m.format_version, 2);
        assert_eq!(m.guid.to_string(), "8f1c2d3e4a5b60718293a4b5c6d7e8f9");
        assert!(!m.folder_asset);
        assert_eq!(m.importer.as_deref(), Some("TextureImporter"));
    }

    #[test]
    fn parses_folder_meta() {
        let text = "fileFormatVersion: 2\nguid: 00000000000000000000000000000abc\nfolderAsset: yes\nDefaultImporter:\n  externalObjects: {}\n  userData: \n";
        let m = MetaData::parse(text.as_bytes()).unwrap();
        assert!(m.folder_asset);
        assert_eq!(m.importer.as_deref(), Some("DefaultImporter"));
    }

    #[test]
    fn accepts_utf8_bom() {
        let mut bytes = vec![0xEF, 0xBB, 0xBF];
        bytes.extend_from_slice(b"fileFormatVersion: 2\nguid: 00000000000000000000000000000abc\n");
        assert!(MetaData::parse(&bytes).is_ok());
    }

    #[test]
    fn importer_is_optional() {
        let m = MetaData::parse(b"fileFormatVersion: 2\nguid: 00000000000000000000000000000abc\n")
            .unwrap();
        assert_eq!(m.importer, None);
    }

    #[test]
    fn reports_each_kind_of_damage() {
        assert_eq!(
            MetaData::parse(&[0xff, 0xfe]).unwrap_err(),
            MetaError::NotUtf8
        );
        assert_eq!(
            MetaData::parse(b"guid: 00000000000000000000000000000abc\n").unwrap_err(),
            MetaError::MissingFormatVersion
        );
        assert_eq!(
            MetaData::parse(b"fileFormatVersion: two\n").unwrap_err(),
            MetaError::BadFormatVersion("two".to_owned())
        );
        assert_eq!(
            MetaData::parse(b"fileFormatVersion: 2\n").unwrap_err(),
            MetaError::MissingGuid
        );
        assert!(matches!(
            MetaData::parse(b"fileFormatVersion: 2\nguid: nope\n").unwrap_err(),
            MetaError::BadGuid(GuidError::Length(4))
        ));
    }

    #[test]
    fn garbage_never_panics() {
        for input in [&b""[..], b":", b"::::", b"guid", b"\n\n\n", b"\xef\xbb\xbf"] {
            let _ = MetaData::parse(input);
        }
    }
}
