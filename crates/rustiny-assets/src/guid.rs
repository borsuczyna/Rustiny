//! Unity asset GUIDs: 128-bit identifiers written as 32 hexadecimal digits in `.meta` files.

use std::fmt;
use std::str::FromStr;

/// A Unity asset GUID. The all-zero value is not a valid asset identity and cannot be parsed.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Guid(u128);

/// Why a GUID string was rejected.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum GuidError {
    /// Not exactly 32 characters.
    #[error("a GUID has 32 hexadecimal digits, found {0} characters")]
    Length(usize),
    /// Contains a character that is not a hexadecimal digit.
    #[error("GUID contains a non-hexadecimal character")]
    NotHex,
    /// All zeros, which Unity treats as "no asset".
    #[error("the all-zero GUID is not a valid asset identity")]
    Zero,
}

impl Guid {
    /// The raw 128-bit value.
    pub fn as_u128(self) -> u128 {
        self.0
    }
}

impl FromStr for Guid {
    type Err = GuidError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.len() != 32 {
            return Err(GuidError::Length(s.chars().count()));
        }
        // `from_str_radix` accepts a leading `+`; require pure hex digits ourselves.
        if !s.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(GuidError::NotHex);
        }
        let value = u128::from_str_radix(s, 16).map_err(|_| GuidError::NotHex)?;
        if value == 0 {
            return Err(GuidError::Zero);
        }
        Ok(Self(value))
    }
}

impl fmt::Display for Guid {
    /// Lower-case, 32 digits, exactly as Unity writes it.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:032x}", self.0)
    }
}

impl fmt::Debug for Guid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Guid({self})")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_lowercase_text() {
        let text = "0123456789abcdef0123456789abcdef";
        let g: Guid = text.parse().unwrap();
        assert_eq!(g.to_string(), text);
    }

    #[test]
    fn uppercase_is_accepted_and_normalised() {
        let g: Guid = "0123456789ABCDEF0123456789ABCDEF".parse().unwrap();
        assert_eq!(g.to_string(), "0123456789abcdef0123456789abcdef");
    }

    #[test]
    fn leading_zeros_are_preserved() {
        let g: Guid = "00000000000000000000000000000001".parse().unwrap();
        assert_eq!(g.to_string(), "00000000000000000000000000000001");
    }

    #[test]
    fn rejects_bad_input() {
        assert_eq!("".parse::<Guid>(), Err(GuidError::Length(0)));
        assert_eq!("abc".parse::<Guid>(), Err(GuidError::Length(3)));
        assert_eq!(
            "0123456789abcdef0123456789abcdeg".parse::<Guid>(),
            Err(GuidError::NotHex)
        );
        assert_eq!(
            "+123456789abcdef0123456789abcdef".parse::<Guid>(),
            Err(GuidError::NotHex)
        );
        assert_eq!(
            "00000000000000000000000000000000".parse::<Guid>(),
            Err(GuidError::Zero)
        );
    }
}
