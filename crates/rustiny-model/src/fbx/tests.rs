//! FBX import tests on files built in memory (no real asset needed).

use super::*;
use flate2::Compression;
use flate2::write::ZlibEncoder;
use std::io::Write;

/// A property to serialise.
enum P {
    I(i64),
    D(f64),
    S(&'static str),
    Ints(Vec<i32>),
    Doubles(Vec<f64>),
    /// A double array stored zlib-compressed.
    Packed(Vec<f64>),
}

struct N {
    name: &'static str,
    props: Vec<P>,
    children: Vec<N>,
}

fn n(name: &'static str, props: Vec<P>, children: Vec<N>) -> N {
    N {
        name,
        props,
        children,
    }
}

fn p70(entries: Vec<(&'static str, Vec<f64>)>) -> N {
    let children = entries
        .into_iter()
        .map(|(name, values)| {
            let mut props = vec![P::S(name), P::S("Vector"), P::S(""), P::S("A")];
            props.extend(values.into_iter().map(P::D));
            n("P", props, vec![])
        })
        .collect();
    n("Properties70", vec![], children)
}

fn write_prop(out: &mut Vec<u8>, prop: &P) {
    match prop {
        P::I(v) => {
            out.push(b'L');
            out.extend(v.to_le_bytes());
        }
        P::D(v) => {
            out.push(b'D');
            out.extend(v.to_le_bytes());
        }
        P::S(s) => {
            out.push(b'S');
            out.extend((s.len() as u32).to_le_bytes());
            out.extend(s.as_bytes());
        }
        P::Ints(values) => {
            out.push(b'i');
            out.extend((values.len() as u32).to_le_bytes());
            out.extend(0u32.to_le_bytes());
            out.extend((values.len() as u32 * 4).to_le_bytes());
            for v in values {
                out.extend(v.to_le_bytes());
            }
        }
        P::Doubles(values) => {
            out.push(b'd');
            out.extend((values.len() as u32).to_le_bytes());
            out.extend(0u32.to_le_bytes());
            out.extend((values.len() as u32 * 8).to_le_bytes());
            for v in values {
                out.extend(v.to_le_bytes());
            }
        }
        P::Packed(values) => {
            let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            let mut enc = ZlibEncoder::new(Vec::new(), Compression::default());
            enc.write_all(&raw).unwrap();
            let packed = enc.finish().unwrap();
            out.push(b'd');
            out.extend((values.len() as u32).to_le_bytes());
            out.extend(1u32.to_le_bytes());
            out.extend((packed.len() as u32).to_le_bytes());
            out.extend(packed);
        }
    }
}

/// Version 7400 record (32-bit fields).
fn write_node(out: &mut Vec<u8>, node: &N) {
    let start = out.len();
    out.extend([0u8; 12]); // end offset, property count, property bytes: patched below
    out.push(node.name.len() as u8);
    out.extend(node.name.as_bytes());
    let props_start = out.len();
    for p in &node.props {
        write_prop(out, p);
    }
    let props_len = out.len() - props_start;
    for child in &node.children {
        write_node(out, child);
    }
    if !node.children.is_empty() {
        out.extend([0u8; 13]);
    }
    let end = out.len() as u32;
    out[start..start + 4].copy_from_slice(&end.to_le_bytes());
    out[start + 4..start + 8].copy_from_slice(&(node.props.len() as u32).to_le_bytes());
    out[start + 8..start + 12].copy_from_slice(&(props_len as u32).to_le_bytes());
}

fn file(nodes: &[N]) -> Vec<u8> {
    let mut out = b"Kaydara FBX Binary  \0".to_vec();
    out.extend([0x1A, 0x00]);
    out.extend(7400u32.to_le_bytes());
    for node in nodes {
        write_node(&mut out, node);
    }
    out.extend([0u8; 13]);
    out
}

/// A unit cube (side 100) as six quads with per-corner normals.
fn cube_geometry(id: i64) -> N {
    let h = 50.0;
    let corners = [
        [-h, -h, -h],
        [h, -h, -h],
        [h, h, -h],
        [-h, h, -h],
        [-h, -h, h],
        [h, -h, h],
        [h, h, h],
        [-h, h, h],
    ];
    // Quads counter-clockwise seen from outside (right-handed), last index bit-inverted.
    let quads: [([i32; 4], [f64; 3]); 6] = [
        ([0, 3, 2, 1], [0.0, 0.0, -1.0]),
        ([4, 5, 6, 7], [0.0, 0.0, 1.0]),
        ([0, 1, 5, 4], [0.0, -1.0, 0.0]),
        ([3, 7, 6, 2], [0.0, 1.0, 0.0]),
        ([0, 4, 7, 3], [-1.0, 0.0, 0.0]),
        ([1, 2, 6, 5], [1.0, 0.0, 0.0]),
    ];
    let mut indices = Vec::new();
    let mut normals = Vec::new();
    for (quad, normal) in quads {
        indices.extend([quad[0], quad[1], quad[2], !quad[3]]);
        for _ in 0..4 {
            normals.extend(normal);
        }
    }
    n(
        "Geometry",
        vec![P::I(id), P::S("Cube\0\u{1}Geometry"), P::S("Mesh")],
        vec![
            n("Vertices", vec![P::Doubles(corners.concat())], vec![]),
            n("PolygonVertexIndex", vec![P::Ints(indices)], vec![]),
            n(
                "LayerElementNormal",
                vec![P::I(0)],
                vec![
                    n(
                        "MappingInformationType",
                        vec![P::S("ByPolygonVertex")],
                        vec![],
                    ),
                    n("ReferenceInformationType", vec![P::S("Direct")], vec![]),
                    n("Normals", vec![P::Packed(normals)], vec![]),
                ],
            ),
        ],
    )
}

fn model(id: i64, name: &'static str, entries: Vec<(&'static str, Vec<f64>)>) -> N {
    n(
        "Model",
        vec![P::I(id), P::S(name), P::S("Mesh")],
        vec![p70(entries)],
    )
}

fn connection(child: i64, parent: i64) -> N {
    n("C", vec![P::S("OO"), P::I(child), P::I(parent)], vec![])
}

fn settings(entries: Vec<(&'static str, Vec<f64>)>) -> N {
    n("GlobalSettings", vec![], vec![p70(entries)])
}

fn cube_file(
    global: Vec<(&'static str, Vec<f64>)>,
    local: Vec<(&'static str, Vec<f64>)>,
) -> Vec<u8> {
    file(&[
        settings(global),
        n(
            "Objects",
            vec![],
            vec![cube_geometry(10), model(20, "Cube\0\u{1}Model", local)],
        ),
        n(
            "Connections",
            vec![],
            vec![connection(10, 20), connection(20, 0)],
        ),
    ])
}

fn default_axes() -> Vec<(&'static str, Vec<f64>)> {
    vec![
        ("UpAxis", vec![1.0]),
        ("UpAxisSign", vec![1.0]),
        ("FrontAxis", vec![2.0]),
        ("FrontAxisSign", vec![1.0]),
        ("CoordAxis", vec![0.0]),
        ("CoordAxisSign", vec![1.0]),
        ("UnitScaleFactor", vec![1.0]),
    ]
}

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-4
}

#[test]
fn a_cube_is_converted_to_metres_and_unity_axes() {
    // 100 cm cube moved 100 cm along +X in a Y-up centimetre file.
    let bytes = cube_file(
        default_axes(),
        vec![("Lcl Translation", vec![100.0, 0.0, 0.0])],
    );
    let mesh = import(&bytes, ImportSettings::default()).unwrap();
    assert!(mesh.is_valid());
    assert_eq!(mesh.triangle_count(), 12);
    assert_eq!(mesh.positions.len(), 24, "flat normals keep faces separate");
    let b = mesh.bounds().unwrap();
    // Unity negates X: the cube spans x in [-1.5, -0.5] metres.
    assert!(close(b.min[0], -1.5) && close(b.max[0], -0.5), "{b:?}");
    assert!(close(b.min[1], -0.5) && close(b.max[1], 0.5), "{b:?}");
    assert!(close(b.min[2], -0.5) && close(b.max[2], 0.5), "{b:?}");
}

#[test]
fn normals_are_unit_and_point_away_from_the_centre() {
    let bytes = cube_file(default_axes(), vec![]);
    let mesh = import(&bytes, ImportSettings::default()).unwrap();
    for (p, n) in mesh.positions.iter().zip(&mesh.normals) {
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        assert!(close(len, 1.0), "normal {n:?} is not unit");
        let outward = p[0] * n[0] + p[1] * n[1] + p[2] * n[2];
        assert!(
            outward > 0.4,
            "vertex {p:?} has normal {n:?} pointing inwards"
        );
    }
}

#[test]
fn the_unit_setting_and_global_scale_change_the_size() {
    let bytes = cube_file(default_axes(), vec![]);
    let metres = import(
        &bytes,
        ImportSettings {
            global_scale: 2.0,
            use_file_scale: true,
        },
    )
    .unwrap();
    assert!(close(metres.bounds().unwrap().max[1], 1.0));
    // Without the file unit the numbers are taken as they are.
    let raw = import(
        &bytes,
        ImportSettings {
            global_scale: 1.0,
            use_file_scale: false,
        },
    )
    .unwrap();
    assert!(close(raw.bounds().unwrap().max[1], 50.0));
    // A file in metres (unit factor 100) keeps its numbers.
    let mut axes = default_axes();
    axes[6] = ("UnitScaleFactor", vec![100.0]);
    let in_metres = import(&cube_file(axes, vec![]), ImportSettings::default()).unwrap();
    assert!(close(in_metres.bounds().unwrap().max[1], 50.0));
}

#[test]
fn z_up_files_are_turned_upright() {
    // Z-up, Y front (negative), X right: a point 100 cm up in the file is 1 m up in Unity.
    let axes = vec![
        ("UpAxis", vec![2.0]),
        ("UpAxisSign", vec![1.0]),
        ("FrontAxis", vec![1.0]),
        ("FrontAxisSign", vec![-1.0]),
        ("CoordAxis", vec![0.0]),
        ("CoordAxisSign", vec![1.0]),
        ("UnitScaleFactor", vec![1.0]),
    ];
    let bytes = cube_file(axes, vec![("Lcl Translation", vec![0.0, 0.0, 100.0])]);
    let b = import(&bytes, ImportSettings::default())
        .unwrap()
        .bounds()
        .unwrap();
    assert!(close(b.min[1], 0.5) && close(b.max[1], 1.5), "{b:?}");
}

#[test]
fn rotation_and_parent_transforms_compose_down_the_hierarchy() {
    // Parent turned 90 degrees about Z; the cube sits 100 cm along the parent's X.
    let bytes = file(&[
        settings(default_axes()),
        n(
            "Objects",
            vec![],
            vec![
                cube_geometry(10),
                model(
                    20,
                    "Cube\0\u{1}Model",
                    vec![("Lcl Translation", vec![100.0, 0.0, 0.0])],
                ),
                model(
                    30,
                    "Pivot\0\u{1}Model",
                    vec![("Lcl Rotation", vec![0.0, 0.0, 90.0])],
                ),
            ],
        ),
        n(
            "Connections",
            vec![],
            vec![connection(10, 20), connection(20, 30), connection(30, 0)],
        ),
    ]);
    let b = import(&bytes, ImportSettings::default())
        .unwrap()
        .bounds()
        .unwrap();
    // File space: the cube ends up along +Y (x -> y); Unity keeps Y.
    assert!(close(b.min[1], 0.5) && close(b.max[1], 1.5), "{b:?}");
    assert!(close(b.min[0], -0.5) && close(b.max[0], 0.5), "{b:?}");
}

#[test]
fn polygons_without_normals_get_outward_flat_normals() {
    let mut cube = cube_geometry(10);
    cube.children.retain(|c| c.name != "LayerElementNormal");
    let bytes = file(&[
        settings(default_axes()),
        n(
            "Objects",
            vec![],
            vec![cube, model(20, "Cube\0\u{1}Model", vec![])],
        ),
        n("Connections", vec![], vec![connection(10, 20)]),
    ]);
    let mesh = import(&bytes, ImportSettings::default()).unwrap();
    for (p, n) in mesh.positions.iter().zip(&mesh.normals) {
        assert!(p[0] * n[0] + p[1] * n[1] + p[2] * n[2] > 0.4, "{p:?} {n:?}");
    }
}

#[test]
fn material_slots_become_sections() {
    let mut cube = cube_geometry(10);
    cube.children.push(n(
        "LayerElementMaterial",
        vec![P::I(0)],
        vec![
            n("MappingInformationType", vec![P::S("ByPolygon")], vec![]),
            n("Materials", vec![P::Ints(vec![0, 0, 1, 1, 1, 1])], vec![]),
        ],
    ));
    let material =
        |id: i64, name: &'static str| n("Material", vec![P::I(id), P::S(name), P::S("")], vec![]);
    let bytes = file(&[
        settings(default_axes()),
        n(
            "Objects",
            vec![],
            vec![
                cube,
                model(20, "Cube\0\u{1}Model", vec![]),
                material(41, "Wall\0\u{1}Material"),
                material(42, "Roof\0\u{1}Material"),
            ],
        ),
        n(
            "Connections",
            vec![],
            vec![connection(10, 20), connection(41, 20), connection(42, 20)],
        ),
    ]);
    let mesh = import(&bytes, ImportSettings::default()).unwrap();
    assert_eq!(mesh.materials, ["Wall", "Roof"]);
    assert_eq!(mesh.sections.len(), 2);
    assert_eq!(mesh.sections[0].material, Some(0));
    assert_eq!(
        mesh.sections[0].index_count, 12,
        "two quads = four triangles"
    );
    assert_eq!(mesh.sections[1].material, Some(1));
    assert_eq!(mesh.sections[1].first_index, 12);
    assert_eq!(mesh.sections[1].index_count, 24);
    assert!(mesh.is_valid());
}

#[test]
fn settings_are_read_from_meta_text() {
    let meta = "fileFormatVersion: 2
ModelImporter:
  meshes:
    globalScale: 0.01
    useFileUnits: 0
";
    assert_eq!(
        ImportSettings::from_meta(meta),
        ImportSettings {
            global_scale: 0.01,
            use_file_scale: false
        }
    );
    let defaults = "ModelImporter:
  meshes:
    globalScale: 1
    useFileUnits: 1
";
    assert_eq!(
        ImportSettings::from_meta(defaults),
        ImportSettings::default()
    );
    // Nonsense falls back to the defaults instead of poisoning the import.
    for bad in [
        "globalScale: nan",
        "globalScale: -3",
        "globalScale: 0",
        "globalScale: abc",
        "",
    ] {
        assert_eq!(
            ImportSettings::from_meta(bad),
            ImportSettings::default(),
            "{bad}"
        );
    }
}

#[test]
fn compressed_arrays_round_trip() {
    let (_, nodes) = parse(&file(&[n(
        "Data",
        vec![P::Packed(vec![1.5, -2.0, 3.25])],
        vec![],
    )]))
    .unwrap();
    assert_eq!(nodes[0].f64_array().unwrap(), [1.5, -2.0, 3.25]);
}

#[test]
fn bad_input_is_an_error_not_a_panic() {
    assert_eq!(parse(b"hello").unwrap_err(), FbxError::NotBinaryFbx);
    assert_eq!(
        parse(b"Kaydara FBX Ascii  ; not binary at all......").unwrap_err(),
        FbxError::NotBinaryFbx
    );
    let good = cube_file(default_axes(), vec![]);
    // Every truncation either parses what is there or reports an error; none may panic.
    for cut in (0..good.len()).step_by(7) {
        let _ = import(&good[..cut], ImportSettings::default());
    }
    // A header-only file has no mesh.
    let empty = file(&[]);
    assert_eq!(
        import(&empty, ImportSettings::default()).unwrap_err(),
        FbxError::NoMesh
    );
    // An array that claims to hold a billion doubles is refused before allocating.
    let mut huge = b"Kaydara FBX Binary  \0".to_vec();
    huge.extend([0x1A, 0x00]);
    huge.extend(7400u32.to_le_bytes());
    let mut record = Vec::new();
    record.extend([0u8; 12]);
    record.push(1);
    record.push(b'X');
    record.push(b'd');
    record.extend(u32::MAX.to_le_bytes());
    record.extend(0u32.to_le_bytes());
    record.extend(0u32.to_le_bytes());
    let end = (huge.len() + record.len()) as u32;
    record[0..4].copy_from_slice(&end.to_le_bytes());
    record[4..8].copy_from_slice(&1u32.to_le_bytes());
    huge.extend(record);
    assert_eq!(parse(&huge).unwrap_err(), FbxError::ArrayTooLarge);
}

/// Real files: `RUSTINY_BENCH_FBX=<file.fbx> cargo test -p rustiny-model -- --ignored --nocapture`.
#[test]
#[ignore = "needs RUSTINY_BENCH_FBX"]
fn import_a_real_file() {
    let Some(path) = std::env::var_os("RUSTINY_BENCH_FBX") else {
        eprintln!("RUSTINY_BENCH_FBX is not set; nothing imported");
        return;
    };
    let bytes = std::fs::read(path).unwrap();
    let start = std::time::Instant::now();
    let mesh = import(&bytes, ImportSettings::default()).unwrap();
    eprintln!(
        "{} bytes -> {} vertices, {} triangles, {} sections, bounds {:?}, {:.2} ms",
        bytes.len(),
        mesh.positions.len(),
        mesh.triangle_count(),
        mesh.sections.len(),
        mesh.bounds(),
        start.elapsed().as_secs_f64() * 1000.0
    );
    assert!(mesh.is_valid());
}

/// A file with two cubes: "Wheel" 100 cm to the right of "Body", both under a parent that is
/// turned and scaled (which a mesh sub-asset must not contain).
fn two_cube_file() -> Vec<u8> {
    file(&[
        settings(default_axes()),
        n(
            "Objects",
            vec![],
            vec![
                cube_geometry(10),
                cube_geometry(11),
                model(
                    20,
                    "Wheel\0\u{1}Model",
                    vec![("Lcl Translation", vec![100.0, 0.0, 0.0])],
                ),
                model(21, "Body\0\u{1}Model", vec![]),
                model(
                    30,
                    "Root\0\u{1}Model",
                    vec![
                        ("Lcl Rotation", vec![0.0, 0.0, 90.0]),
                        ("Lcl Scaling", vec![3.0, 3.0, 3.0]),
                    ],
                ),
            ],
        ),
        n(
            "Connections",
            vec![],
            vec![
                connection(10, 20),
                connection(11, 21),
                connection(20, 30),
                connection(21, 30),
                connection(30, 0),
            ],
        ),
    ])
}

#[test]
fn one_mesh_of_a_file_keeps_the_space_of_its_own_node() {
    let bytes = two_cube_file();
    // "Wheel" is 100 cm to the right of "Body" and both sit under a turned, scaled parent; the
    // mesh sub-asset has none of that.
    for name in ["Wheel", "Body"] {
        let id = mesh_file_id(name);
        let mesh = import_part(&bytes, ImportSettings::default(), Part::Mesh(id)).unwrap();
        assert_eq!(mesh.triangle_count(), 12, "{name}");
        let b = mesh.bounds().unwrap();
        for axis in 0..3 {
            assert!(
                close(b.min[axis], -0.5) && close(b.max[axis], 0.5),
                "{name}: {b:?}"
            );
        }
    }
    // The whole file, by contrast, has both cubes in place.
    let whole = import(&bytes, ImportSettings::default()).unwrap();
    assert_eq!(whole.triangle_count(), 24);
    assert!(whole.bounds().unwrap().max[1] > 3.0);
}

#[test]
fn a_mesh_that_cannot_be_told_apart_is_not_guessed() {
    // Two meshes and an ID that is neither: an error, so the caller can fall back.
    let result = import_part(
        &two_cube_file(),
        ImportSettings::default(),
        Part::Mesh(mesh_file_id("Visual")),
    );
    assert_eq!(result.unwrap_err(), FbxError::NoMesh);
    // A file with a single mesh needs no match.
    let single = cube_file(
        default_axes(),
        vec![("Lcl Translation", vec![100.0, 0.0, 0.0])],
    );
    let mesh = import_part(&single, ImportSettings::default(), Part::Mesh(1)).unwrap();
    assert!(close(mesh.bounds().unwrap().max[0], 0.5));
}

/// A point moved by a node's local transform (scale, then rotation, then translation).
fn place(node: &FbxNode, p: [f32; 3]) -> [f32; 3] {
    let [x, y, z, w] = node.rotation;
    let s = [
        p[0] * node.scale[0],
        p[1] * node.scale[1],
        p[2] * node.scale[2],
    ];
    // v + 2w(q x v) + 2 q x (q x v)
    let q = [x, y, z];
    let cross = |a: [f32; 3], b: [f32; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let t = cross(q, s).map(|v| v * 2.0);
    let u = cross(q, t);
    [
        s[0] + w * t[0] + u[0] + node.position[0],
        s[1] + w * t[1] + u[1] + node.position[1],
        s[2] + w * t[2] + u[2] + node.position[2],
    ]
}

#[test]
fn node_transforms_put_each_mesh_where_the_flattened_model_has_it() {
    let bytes = two_cube_file();
    let list = nodes(&bytes, ImportSettings::default()).unwrap();
    assert_eq!(
        list.iter().map(|n| n.name.as_str()).collect::<Vec<_>>(),
        ["Root", "Wheel", "Body"]
    );
    assert_eq!(list[0].parent, None);
    assert_eq!((list[1].parent, list[2].parent), (Some(0), Some(0)));
    assert!(!list[0].has_mesh && list[1].has_mesh && list[2].has_mesh);
    // Every vertex of the two part meshes, moved down the node chain, is a vertex of the
    // flattened file (same set of positions, whatever the order).
    let whole = import(&bytes, ImportSettings::default()).unwrap();
    let mut expected: Vec<[i32; 3]> = whole
        .positions
        .iter()
        .map(|p| p.map(|v| (v * 1000.0).round() as i32))
        .collect();
    let mut got: Vec<[i32; 3]> = Vec::new();
    for (index, name) in [(1, "Wheel"), (2, "Body")] {
        let mesh = import_part(
            &bytes,
            ImportSettings::default(),
            Part::Mesh(mesh_file_id(name)),
        )
        .unwrap();
        for p in &mesh.positions {
            let in_root = place(&list[index], *p);
            let world = place(&list[0], in_root);
            got.push(world.map(|v| (v * 1000.0).round() as i32));
        }
    }
    expected.sort_unstable();
    expected.dedup();
    got.sort_unstable();
    got.dedup();
    assert_eq!(got, expected);
}
