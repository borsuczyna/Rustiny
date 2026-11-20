//! Reading the top-level fields of one document body without building a YAML tree.
//!
//! Unity writes every object as `TypeName:` followed by fields indented by two spaces. Sequence
//! items sit at the same indent (`  - {fileID: 3}`) and deeper content is further indented:
//!
//! ```text
//! GameObject:
//!   m_Component:
//!   - component: {fileID: 4}
//!   m_Name: Player
//! ```
//!
//! [`fields`] yields each top-level key with its inline value and the lines that belong to it,
//! borrowing from the body, so reading a document allocates nothing.

/// One top-level field of a document.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Field<'a> {
    /// The key, e.g. `m_Name`.
    pub key: &'a str,
    /// The inline value after `key:`, trimmed (empty when the value is a block).
    pub value: &'a str,
    block: &'a str,
}

impl<'a> Field<'a> {
    /// The raw lines following the key line that belong to this field.
    pub fn block(&self) -> &'a str {
        self.block
    }

    /// Items of a block sequence (`  - ...`) at the top-level indent, with the dash removed.
    pub fn items(&self) -> impl Iterator<Item = &'a str> + use<'a> {
        self.block
            .lines()
            .filter_map(|line| line.strip_prefix("  - "))
            .map(str::trim)
    }
}

/// Iterator over the top-level fields of a document body (see [`fields`]).
#[derive(Debug, Clone)]
pub struct Fields<'a> {
    rest: &'a str,
}

/// The top-level fields of a document body. The first line (`TypeName:`) is skipped.
pub fn fields(body: &str) -> Fields<'_> {
    Fields {
        rest: split_line(body).1,
    }
}

/// The type name on the first line of a body (`GameObject` for `GameObject:`).
pub fn type_name(body: &str) -> &str {
    split_line(body).0.trim_end_matches(['\r', ':', ' '])
}

impl<'a> Iterator for Fields<'a> {
    type Item = Field<'a>;

    fn next(&mut self) -> Option<Field<'a>> {
        loop {
            if self.rest.is_empty() {
                return None;
            }
            let (line, after) = split_line(self.rest);
            if !is_key_line(line) {
                self.rest = after;
                continue;
            }

            // The block runs until the next top-level key line.
            let mut offset = 0;
            let mut cursor = after;
            while !cursor.is_empty() {
                let (next_line, next_cursor) = split_line(cursor);
                if is_key_line(next_line) {
                    break;
                }
                offset += cursor.len() - next_cursor.len();
                cursor = next_cursor;
            }
            let block = &after[..offset];
            self.rest = &after[offset..];

            let line = line.trim_end_matches('\r');
            let Some((key, value)) = line[2..].split_once(':') else {
                continue;
            };
            return Some(Field {
                key,
                value: value.trim(),
                block,
            });
        }
    }
}

/// `(line without its '\n', everything after it)`.
fn split_line(text: &str) -> (&str, &str) {
    match text.find('\n') {
        Some(i) => (&text[..i], &text[i + 1..]),
        None => (text, ""),
    }
}

/// A line is a top-level key line when it is indented by exactly two spaces and does not start
/// a sequence item, a comment or deeper content.
fn is_key_line(line: &str) -> bool {
    let b = line.as_bytes();
    b.len() > 2
        && b[0] == b' '
        && b[1] == b' '
        && !matches!(b[2], b' ' | b'-' | b'#' | b'\r' | b'\t')
}

#[cfg(test)]
mod tests {
    use super::*;

    const GAME_OBJECT: &str = "GameObject:\n  m_ObjectHideFlags: 0\n  serializedVersion: 6\n  m_Component:\n  - component: {fileID: 4}\n  - component: {fileID: 33}\n  m_Layer: 0\n  m_Name: Player\n  m_IsActive: 1\n";

    #[test]
    fn splits_top_level_fields() {
        assert_eq!(type_name(GAME_OBJECT), "GameObject");
        let all: Vec<_> = fields(GAME_OBJECT).map(|f| (f.key, f.value)).collect();
        assert_eq!(
            all,
            vec![
                ("m_ObjectHideFlags", "0"),
                ("serializedVersion", "6"),
                ("m_Component", ""),
                ("m_Layer", "0"),
                ("m_Name", "Player"),
                ("m_IsActive", "1"),
            ]
        );
    }

    #[test]
    fn sequence_items_belong_to_their_key() {
        let component = fields(GAME_OBJECT)
            .find(|f| f.key == "m_Component")
            .unwrap();
        let items: Vec<_> = component.items().collect();
        assert_eq!(
            items,
            vec!["component: {fileID: 4}", "component: {fileID: 33}"]
        );
    }

    #[test]
    fn nested_content_stays_inside_the_block() {
        let body = "PrefabInstance:\n  m_Modification:\n    m_TransformParent: {fileID: 7}\n    m_Modifications:\n    - target: {fileID: 1}\n      propertyPath: m_Name\n      value: Bob\n  m_SourcePrefab: {fileID: 1, guid: abc, type: 3}\n";
        let all: Vec<_> = fields(body).collect();
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].key, "m_Modification");
        assert!(all[0].block().contains("value: Bob"));
        assert_eq!(
            all[0].items().count(),
            0,
            "deeper dashes are not top-level items"
        );
        assert_eq!(all[1].key, "m_SourcePrefab");
    }

    #[test]
    fn empty_flow_sequences_and_crlf() {
        let body = "Transform:\r\n  m_Children: []\r\n  m_Father: {fileID: 0}\r\n";
        let all: Vec<_> = fields(body).collect();
        assert_eq!(all[0].value, "[]");
        assert_eq!(all[1].value, "{fileID: 0}");
        assert_eq!(type_name(body), "Transform");
    }

    #[test]
    fn colon_in_value_and_missing_trailing_newline() {
        let body = "GameObject:\n  m_Name: 'a: b'\n  m_Layer: 5";
        let all: Vec<_> = fields(body).collect();
        assert_eq!(all[0].value, "'a: b'");
        assert_eq!(all[1].value, "5");
    }

    #[test]
    fn empty_body() {
        assert_eq!(fields("").count(), 0);
        assert_eq!(fields("GameObject:\n").count(), 0);
    }
}
