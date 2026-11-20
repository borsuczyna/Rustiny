//! The `fileID` Unity gives the sub-assets of an imported model.
//!
//! Since Unity 2019.1 the IDs are not stored in the `.meta` file: they are the 64-bit xxHash of
//! the text `Type:<native type>-><name>0`, read as a signed number. A mesh called `chassis` is
//! `xxh64("Type:Mesh->chassis0")`. The ID depends only on the type and the name, so two model
//! files with a node of the same name give the same ID. Checked against the real project: the
//! mesh of `chassis` is -4256789094099829062 in `sizuro.fbx` and in `chassis 1.fbx`.

const P1: u64 = 11_400_714_785_074_694_791;
const P2: u64 = 14_029_467_366_897_019_727;
const P3: u64 = 1_609_587_929_392_839_161;
const P4: u64 = 9_650_029_242_287_828_579;
const P5: u64 = 2_870_177_450_012_600_261;

fn round(acc: u64, input: u64) -> u64 {
    acc.wrapping_add(input.wrapping_mul(P2))
        .rotate_left(31)
        .wrapping_mul(P1)
}

fn merge(acc: u64, value: u64) -> u64 {
    (acc ^ round(0, value)).wrapping_mul(P1).wrapping_add(P4)
}

fn read_u64(bytes: &[u8]) -> u64 {
    let mut word = [0u8; 8];
    word.copy_from_slice(&bytes[..8]);
    u64::from_le_bytes(word)
}

fn read_u32(bytes: &[u8]) -> u32 {
    let mut word = [0u8; 4];
    word.copy_from_slice(&bytes[..4]);
    u32::from_le_bytes(word)
}

/// xxHash64 of `data` with seed 0.
pub fn xxh64(data: &[u8]) -> u64 {
    let mut rest = data;
    let mut hash = if data.len() >= 32 {
        let mut lanes = [P1.wrapping_add(P2), P2, 0, 0u64.wrapping_sub(P1)];
        while rest.len() >= 32 {
            for (lane, chunk) in lanes.iter_mut().zip(rest.as_chunks::<8>().0) {
                *lane = round(*lane, u64::from_le_bytes(*chunk));
            }
            rest = &rest[32..];
        }
        let mut hash = lanes[0]
            .rotate_left(1)
            .wrapping_add(lanes[1].rotate_left(7))
            .wrapping_add(lanes[2].rotate_left(12))
            .wrapping_add(lanes[3].rotate_left(18));
        for lane in lanes {
            hash = merge(hash, lane);
        }
        hash
    } else {
        P5
    };
    hash = hash.wrapping_add(data.len() as u64);
    while rest.len() >= 8 {
        hash ^= round(0, read_u64(rest));
        hash = hash.rotate_left(27).wrapping_mul(P1).wrapping_add(P4);
        rest = &rest[8..];
    }
    if rest.len() >= 4 {
        hash ^= u64::from(read_u32(rest)).wrapping_mul(P1);
        hash = hash.rotate_left(23).wrapping_mul(P2).wrapping_add(P3);
        rest = &rest[4..];
    }
    for &byte in rest {
        hash ^= u64::from(byte).wrapping_mul(P5);
        hash = hash.rotate_left(11).wrapping_mul(P1);
    }
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(P2);
    hash ^= hash >> 29;
    hash = hash.wrapping_mul(P3);
    hash ^= hash >> 32;
    hash
}

/// The `fileID` of the `Mesh` Unity creates for the model node called `name`.
pub fn mesh_file_id(name: &str) -> i64 {
    let text = format!("Type:Mesh->{name}0");
    xxh64(text.as_bytes()) as i64
}

/// The root of the paths of a model file's game objects.
pub const ROOT_PATH: &str = "//RootNode/root";

/// The `fileID` of one object of a model file used as a prefab. `kind` is the native type
/// (`GameObject`, `Transform`, `MeshFilter`, `MeshRenderer`) and `path` the node's path under
/// [`ROOT_PATH`] (`//RootNode/root/house/door`). A game object is named by that path; a
/// component by the path of its game object plus the component's type (`.../door/Transform`).
pub fn object_id(kind: &str, node_path: &str) -> i64 {
    let text = if kind == "GameObject" {
        format!("Type:GameObject->{node_path}0")
    } else {
        format!("Type:{kind}->{node_path}/{kind}0")
    };
    xxh64(text.as_bytes()) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xxh64_matches_the_reference_values() {
        assert_eq!(xxh64(b""), 0xEF46_DB37_51D8_E999);
        assert_eq!(xxh64(b"abc"), 0x44BC_2CF5_AD77_0999);
        // Longer than one 32-byte stripe, and with every kind of tail.
        assert_eq!(
            xxh64(b"Nobody inspects the spammish repetition"),
            0xFBCE_A83C_8A37_8BF1
        );
    }

    #[test]
    fn mesh_ids_match_the_ones_in_the_real_project() {
        // Taken from `m_Mesh` references in Summer Hideout's scene, whose FBX files have a node
        // of that name.
        assert_eq!(mesh_file_id("chassis"), -4_256_789_094_099_829_062);
        assert_ne!(mesh_file_id("chassis.001"), mesh_file_id("chassis"));
    }

    #[test]
    fn object_ids_match_the_override_targets_in_the_real_project() {
        // Targets of prefab-instance overrides of `house-01.fbx` and of single-node model files.
        assert_eq!(object_id("GameObject", ROOT_PATH), 919_132_149_155_446_097);
        assert_eq!(
            object_id("Transform", ROOT_PATH),
            -8_679_921_383_154_817_045
        );
        assert_eq!(
            object_id("MeshRenderer", ROOT_PATH),
            -7_511_558_181_221_131_132
        );
        let child = format!("{ROOT_PATH}/bus-sign");
        assert_eq!(object_id("Transform", &child), -471_608_532_724_919_658);
        let plants = format!("{ROOT_PATH}/daisy-plants-02");
        assert_eq!(object_id("GameObject", &plants), 1_048_325_429_367_875_318);
    }
}
