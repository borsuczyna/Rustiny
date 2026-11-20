//! Parsing of the small value shapes Unity writes: quoted strings, `{fileID: N}` references
//! and `{x: 0, y: 1, z: 2}` vectors.

use std::borrow::Cow;

/// Value of `key` inside a flow mapping such as `{fileID: 5, guid: abc, type: 3}`.
/// The returned slice is trimmed and ends at the next `,` or `}`.
pub fn flow_value<'a>(map: &'a str, key: &str) -> Option<&'a str> {
    let mut from = 0;
    while let Some(found) = map[from..].find(key) {
        let at = from + found;
        let after_key = at + key.len();
        let starts_a_key = at == 0 || matches!(map.as_bytes()[at - 1], b'{' | b' ' | b',');
        if starts_a_key && map[after_key..].starts_with(':') {
            let value = &map[after_key + 1..];
            let end = value.find([',', '}']).unwrap_or(value.len());
            return Some(value[..end].trim());
        }
        from = at + 1;
    }
    None
}

/// The `fileID` inside a reference such as `{fileID: 11}` or `component: {fileID: 11}`.
pub fn file_id_of(text: &str) -> Option<i64> {
    flow_value(text, "fileID")?.parse().ok()
}

/// `{x: 1, y: 2, z: 3}` as three floats.
pub fn parse_vec3(text: &str) -> Option<[f32; 3]> {
    Some([
        flow_value(text, "x")?.parse().ok()?,
        flow_value(text, "y")?.parse().ok()?,
        flow_value(text, "z")?.parse().ok()?,
    ])
}

/// `{x: 0, y: 0, z: 0, w: 1}` as four floats (a quaternion).
pub fn parse_vec4(text: &str) -> Option<[f32; 4]> {
    Some([
        flow_value(text, "x")?.parse().ok()?,
        flow_value(text, "y")?.parse().ok()?,
        flow_value(text, "z")?.parse().ok()?,
        flow_value(text, "w")?.parse().ok()?,
    ])
}

/// A YAML scalar as text: plain scalars are trimmed; `'single'` and `"double"` quoted ones are
/// unescaped (Unity writes non-ASCII characters in double quotes as `\xXX`, `\uXXXX`, `\UXXXXXXXX`).
pub fn unquote(value: &str) -> Cow<'_, str> {
    let value = value.trim();
    if let Some(inner) = value.strip_prefix('\'').and_then(|v| v.strip_suffix('\'')) {
        return if inner.contains("''") {
            Cow::Owned(inner.replace("''", "'"))
        } else {
            Cow::Borrowed(inner)
        };
    }
    if let Some(inner) = value.strip_prefix('"').and_then(|v| v.strip_suffix('"')) {
        return if inner.contains('\\') {
            Cow::Owned(unescape_double_quoted(inner))
        } else {
            Cow::Borrowed(inner)
        };
    }
    Cow::Borrowed(value)
}

fn unescape_double_quoted(inner: &str) -> String {
    let mut out = String::with_capacity(inner.len());
    let mut chars = inner.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some('r') => out.push('\r'),
            Some('0') => out.push('\0'),
            Some('"') => out.push('"'),
            Some('\\') => out.push('\\'),
            Some('/') => out.push('/'),
            Some(' ') => out.push(' '),
            Some('x') => push_code_point(&mut out, &mut chars, 2),
            Some('u') => push_code_point(&mut out, &mut chars, 4),
            Some('U') => push_code_point(&mut out, &mut chars, 8),
            Some(other) => {
                out.push('\\');
                out.push(other);
            }
            None => out.push('\\'),
        }
    }
    out
}

fn push_code_point(out: &mut String, chars: &mut std::str::Chars<'_>, digits: usize) {
    let hex: String = chars.clone().take(digits).collect();
    let parsed = if hex.len() == digits {
        u32::from_str_radix(&hex, 16).ok()
    } else {
        None
    };
    match parsed {
        Some(code) => {
            for _ in 0..digits {
                chars.next();
            }
            out.push(char::from_u32(code).unwrap_or('\u{fffd}'));
        }
        None => out.push('\u{fffd}'),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flow_values_and_references() {
        assert_eq!(
            flow_value("{fileID: 5, guid: ab, type: 3}", "guid"),
            Some("ab")
        );
        assert_eq!(
            flow_value("{fileID: 5, guid: ab, type: 3}", "type"),
            Some("3")
        );
        assert_eq!(flow_value("{x: 1}", "y"), None);
        // `max` must not satisfy a lookup for `x`.
        assert_eq!(flow_value("{max: 9}", "x"), None);
        assert_eq!(file_id_of("{fileID: 11400000}"), Some(11_400_000));
        assert_eq!(file_id_of("component: {fileID: -42}"), Some(-42));
        assert_eq!(file_id_of("4: {fileID: 7}"), Some(7));
        assert_eq!(file_id_of("{fileID: 0}"), Some(0));
        assert_eq!(file_id_of("{guid: ab}"), None);
        assert_eq!(file_id_of("{fileID: nope}"), None);
    }

    #[test]
    fn vectors() {
        assert_eq!(
            parse_vec3("{x: 1, y: -2.5, z: 1e-05}"),
            Some([1.0, -2.5, 1e-5])
        );
        assert_eq!(
            parse_vec4("{x: 0, y: 0, z: 0, w: 1}"),
            Some([0.0, 0.0, 0.0, 1.0])
        );
        assert_eq!(parse_vec3("{x: 1, y: 2}"), None);
        assert_eq!(parse_vec3("{x: a, y: 2, z: 3}"), None);
        let inf = parse_vec3("{x: Infinity, y: 0, z: 0}").unwrap();
        assert!(inf[0].is_infinite());
    }

    #[test]
    fn unquoting() {
        assert_eq!(unquote("  Player "), "Player");
        assert_eq!(unquote("'it''s'"), "it's");
        assert_eq!(unquote("'a: b'"), "a: b");
        assert_eq!(unquote(r#""a\"b\\c""#), "a\"b\\c");
        assert_eq!(unquote(r#""Zaż\xF3ł""#), "Zażół");
        assert_eq!(unquote(r#""\U0001F600""#), "😀");
        assert_eq!(unquote(r#""bad \uZZ""#), "bad \u{fffd}ZZ");
        assert_eq!(unquote(""), "");
        assert_eq!(unquote("''"), "");
    }
}
