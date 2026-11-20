//! Streaming reader that splits a UnityYAML file into documents.
//!
//! A Unity scene is a `%YAML` preamble followed by documents, each introduced by a header line:
//!
//! ```text
//! --- !u!1 &1234567890
//! --- !u!4 &55 stripped
//! ```
//!
//! (`!u!<class id>`, `&<file id>`, optional `stripped`.) The reader hands out one header at a
//! time and only collects the body of documents the caller asks for, so a 300 MB scene can be
//! walked while holding a few kilobytes.

use std::io::{self, BufRead};

const HEADER_PREFIX: &[u8] = b"--- !u!";
const UTF8_BOM: &[u8] = b"\xEF\xBB\xBF";

/// The header line of one document.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    /// Unity class ID (`!u!<id>`).
    pub class_id: u32,
    /// File ID of the object (`&<id>`); unique within the file, may be negative.
    pub file_id: i64,
    /// `stripped`: a placeholder for an object that lives in a prefab, with only a few fields.
    pub stripped: bool,
}

/// Errors from reading UnityYAML.
#[derive(Debug, thiserror::Error)]
pub enum YamlError {
    /// The underlying reader failed.
    #[error("read error: {0}")]
    Io(#[from] io::Error),
    /// A line starts like a document header but cannot be parsed.
    #[error("line {line}: malformed document header {text:?}")]
    BadHeader {
        /// 1-based line number.
        line: u64,
        /// The offending line (truncated).
        text: String,
    },
}

/// Walks the documents of a UnityYAML stream.
#[derive(Debug)]
pub struct DocumentReader<R> {
    reader: R,
    line: Vec<u8>,
    /// A header already read while collecting the previous body.
    pending: Option<Header>,
    line_no: u64,
    bytes: u64,
}

impl<R: BufRead> DocumentReader<R> {
    /// Wrap a buffered reader positioned at the start of the file.
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            line: Vec::with_capacity(256),
            pending: None,
            line_no: 0,
            bytes: 0,
        }
    }

    /// Whether the stream starts with the `%YAML` directive of text-serialized Unity assets
    /// (after an optional UTF-8 BOM). Binary-serialized scenes fail this. Call before reading.
    pub fn looks_like_unity_yaml(&mut self) -> io::Result<bool> {
        let head = self.reader.fill_buf()?;
        let head = head.strip_prefix(UTF8_BOM).unwrap_or(head);
        Ok(head.starts_with(b"%YAML"))
    }

    /// Bytes consumed so far.
    pub fn bytes_read(&self) -> u64 {
        self.bytes
    }

    /// Advance to the next document and return its header. Any unread body of the previous
    /// document is skipped without being stored.
    pub fn next_header(&mut self) -> Result<Option<Header>, YamlError> {
        if let Some(header) = self.pending.take() {
            return Ok(Some(header));
        }
        while self.read_line()? {
            if is_header(&self.line) {
                return Ok(Some(parse_header(&self.line, self.line_no)?));
            }
        }
        Ok(None)
    }

    /// Read the body of the document whose header was just returned, replacing `body`'s
    /// contents. Stops before the next header (which the next `next_header` call returns).
    pub fn read_body(&mut self, body: &mut String) -> Result<(), YamlError> {
        body.clear();
        while self.read_line()? {
            if is_header(&self.line) {
                self.pending = Some(parse_header(&self.line, self.line_no)?);
                return Ok(());
            }
            match std::str::from_utf8(&self.line) {
                Ok(text) => body.push_str(text),
                Err(_) => body.push_str(&String::from_utf8_lossy(&self.line)),
            }
        }
        Ok(())
    }

    fn read_line(&mut self) -> io::Result<bool> {
        self.line.clear();
        let n = self.reader.read_until(b'\n', &mut self.line)?;
        if n == 0 {
            return Ok(false);
        }
        self.bytes += n as u64;
        self.line_no += 1;
        Ok(true)
    }
}

fn is_header(line: &[u8]) -> bool {
    line.starts_with(HEADER_PREFIX)
}

fn parse_header(line: &[u8], line_no: u64) -> Result<Header, YamlError> {
    let bad = || YamlError::BadHeader {
        line: line_no,
        text: String::from_utf8_lossy(&line[..line.len().min(80)])
            .trim_end()
            .to_owned(),
    };
    let text = std::str::from_utf8(line).map_err(|_| bad())?.trim_end();
    let rest = text.strip_prefix("--- !u!").ok_or_else(bad)?;
    let mut parts = rest.split_ascii_whitespace();
    let class_id = parts.next().and_then(|c| c.parse().ok()).ok_or_else(bad)?;
    let file_id = parts
        .next()
        .and_then(|f| f.strip_prefix('&'))
        .and_then(|f| f.parse().ok())
        .ok_or_else(bad)?;
    let stripped = match parts.next() {
        None => false,
        Some("stripped") => true,
        Some(_) => return Err(bad()),
    };
    Ok(Header {
        class_id,
        file_id,
        stripped,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    const SAMPLE: &str = "%YAML 1.1\n%TAG !u! tag:unity3d.com,2011:\n--- !u!1 &10\nGameObject:\n  m_Name: A\n--- !u!4 &-20 stripped\nTransform:\n  m_Father: {fileID: 0}\n--- !u!29 &30\nOcclusionCullingSettings:\n  m_ObjectHideFlags: 0\n";

    fn reader(text: &str) -> DocumentReader<Cursor<Vec<u8>>> {
        DocumentReader::new(Cursor::new(text.as_bytes().to_vec()))
    }

    #[test]
    fn walks_headers_and_bodies() {
        let mut docs = reader(SAMPLE);
        assert!(docs.looks_like_unity_yaml().unwrap());
        let mut body = String::new();

        let h = docs.next_header().unwrap().unwrap();
        assert_eq!(
            h,
            Header {
                class_id: 1,
                file_id: 10,
                stripped: false
            }
        );
        docs.read_body(&mut body).unwrap();
        assert_eq!(body, "GameObject:\n  m_Name: A\n");

        let h = docs.next_header().unwrap().unwrap();
        assert_eq!(h.file_id, -20);
        assert!(h.stripped);
        // Skip this body without reading it.
        let h = docs.next_header().unwrap().unwrap();
        assert_eq!(h.class_id, 29);
        docs.read_body(&mut body).unwrap();
        assert!(body.starts_with("OcclusionCullingSettings:"));
        assert!(docs.next_header().unwrap().is_none());
        assert_eq!(docs.bytes_read(), SAMPLE.len() as u64);
    }

    #[test]
    fn handles_crlf_and_missing_final_newline() {
        let text = "%YAML 1.1\r\n--- !u!1 &1\r\nGameObject:\r\n  m_Name: A";
        let mut docs = reader(text);
        let mut body = String::new();
        assert_eq!(docs.next_header().unwrap().unwrap().file_id, 1);
        docs.read_body(&mut body).unwrap();
        assert_eq!(body, "GameObject:\r\n  m_Name: A");
        assert!(docs.next_header().unwrap().is_none());
    }

    #[test]
    fn rejects_binary_and_accepts_bom() {
        assert!(!reader("Unity binary\0\0").looks_like_unity_yaml().unwrap());
        assert!(
            reader("\u{feff}%YAML 1.1\n")
                .looks_like_unity_yaml()
                .unwrap()
        );
    }

    #[test]
    fn malformed_header_is_an_error_with_a_line_number() {
        let mut docs = reader("%YAML 1.1\n--- !u!x &1\n");
        match docs.next_header() {
            Err(YamlError::BadHeader { line, .. }) => assert_eq!(line, 2),
            other => panic!("expected BadHeader, got {other:?}"),
        }
        let mut docs = reader("--- !u!1 &1 weird\n");
        assert!(docs.next_header().is_err());
        let mut docs = reader("--- !u!1\n");
        assert!(docs.next_header().is_err());
    }

    #[test]
    fn empty_input_has_no_documents() {
        assert!(reader("").next_header().unwrap().is_none());
        assert!(reader("%YAML 1.1\n").next_header().unwrap().is_none());
    }

    #[test]
    fn invalid_utf8_is_replaced_not_fatal() {
        let mut bytes = b"--- !u!1 &1\nGameObject:\n  m_Name: ".to_vec();
        bytes.extend_from_slice(&[0xFF, 0xFE, b'\n']);
        let mut docs = DocumentReader::new(Cursor::new(bytes));
        let mut body = String::new();
        docs.next_header().unwrap();
        docs.read_body(&mut body).unwrap();
        assert!(body.contains('\u{fffd}'));
    }
}
