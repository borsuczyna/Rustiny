//! Reading the `m_Modification` block of a `PrefabInstance`.

use super::types::{AssetRef, Modification, Overrides, Part};
use crate::unity_yaml::{file_id_of, unquote};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Section {
    Other,
    Modifications,
    Removed,
}

/// The parent transform (0 = none) and the overrides in an `m_Modification` block. Overrides of
/// properties that do not affect drawing (script fields, euler-angle hints, ...) are dropped.
pub fn parse_modification(block: &str) -> (i64, Overrides) {
    let mut parent = 0;
    let mut out = Overrides::default();
    let mut section = Section::Other;
    let mut target: Option<i64> = None;
    let mut path: Option<String> = None;
    let mut value = String::new();
    let mut reference: Option<AssetRef> = None;

    let flush = |target: &mut Option<i64>,
                 path: &mut Option<String>,
                 value: &mut String,
                 reference: &mut Option<AssetRef>,
                 out: &mut Overrides| {
        if let (Some(t), Some(p)) = (target.take(), path.take())
            && Part::of_property(&p).is_some()
        {
            out.mods.push(Modification {
                target: t,
                path: p,
                value: std::mem::take(value),
                reference: reference.take(),
            });
        }
        value.clear();
        *reference = None;
    };

    for line in block.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("m_TransformParent:") {
            parent = file_id_of(rest).unwrap_or(0);
        } else if t.starts_with("m_Modifications:") {
            section = Section::Modifications;
        } else if t.starts_with("m_RemovedComponents:") || t.starts_with("m_RemovedGameObjects:") {
            flush(&mut target, &mut path, &mut value, &mut reference, &mut out);
            section = Section::Removed;
        } else if t.starts_with("m_AddedGameObjects:")
            || t.starts_with("m_AddedComponents:")
            || t.starts_with("serializedVersion:")
            || t.starts_with("m_SourcePrefab:")
        {
            flush(&mut target, &mut path, &mut value, &mut reference, &mut out);
            section = Section::Other;
        } else {
            match section {
                Section::Modifications => {
                    if let Some(rest) = t.strip_prefix("- target:") {
                        flush(&mut target, &mut path, &mut value, &mut reference, &mut out);
                        target = file_id_of(rest);
                    } else if let Some(rest) = t.strip_prefix("propertyPath:") {
                        path = Some(unquote(rest).into_owned());
                    } else if let Some(rest) = t.strip_prefix("value:") {
                        value = unquote(rest).into_owned();
                    } else if let Some(rest) = t.strip_prefix("objectReference:") {
                        reference = AssetRef::parse(rest);
                    }
                }
                Section::Removed => {
                    if t.starts_with("- ")
                        && let Some(id) = file_id_of(t)
                    {
                        out.removed.push(id);
                    }
                }
                Section::Other => {}
            }
        }
    }
    flush(&mut target, &mut path, &mut value, &mut reference, &mut out);
    (parent, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const BLOCK: &str = "\
    serializedVersion: 3
    m_TransformParent: {fileID: 77}
    m_Modifications:
    - target: {fileID: 11, guid: aaaa, type: 3}
      propertyPath: m_LocalPosition.x
      value: 1.5
      objectReference: {fileID: 0}
    - target: {fileID: 12, guid: aaaa, type: 3}
      propertyPath: m_LocalEulerAnglesHint.x
      value: 90
      objectReference: {fileID: 0}
    - target: {fileID: 13, guid: aaaa, type: 3}
      propertyPath: 'm_Materials.Array.data[2]'
      value: 
      objectReference: {fileID: 2100000, guid: bbbb, type: 2}
    - target: {fileID: 14, guid: aaaa, type: 3}
      propertyPath: m_Name
      value: Tree
      objectReference: {fileID: 0}
    m_RemovedComponents:
    - {fileID: 9, guid: aaaa, type: 3}
    m_RemovedGameObjects: []
    m_AddedGameObjects: []
    m_AddedComponents: []
";

    #[test]
    fn relevant_overrides_references_and_removals_are_read() {
        let (parent, o) = parse_modification(BLOCK);
        assert_eq!(parent, 77);
        assert_eq!(o.mods.len(), 3, "the euler hint is dropped");
        assert_eq!(o.mods[0].path, "m_LocalPosition.x");
        assert_eq!(o.mods[0].value, "1.5");
        assert_eq!(o.mods[1].path, "m_Materials.Array.data[2]");
        assert_eq!(
            o.mods[1].reference,
            Some(AssetRef {
                file_id: 2_100_000,
                guid: "bbbb".into()
            })
        );
        assert_eq!(o.mods[2].value, "Tree");
        assert_eq!(o.removed, vec![9]);
    }

    #[test]
    fn a_null_reference_is_none() {
        assert_eq!(AssetRef::parse("{fileID: 0}"), None);
        assert_eq!(AssetRef::parse("{fileID: 5}"), None, "no guid: same file");
    }
}
