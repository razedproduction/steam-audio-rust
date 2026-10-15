//! Parity tests for stage 6.4a against golden data from `tools/golden/src/serialization.cpp`.
//!
//! Each golden case is a whole FlatBuffer written by upstream (FlatBuffers 1.12). The port must write the same bytes
//! (`EXACT`) and read them back into objects equal to the ones it serialized, which also covers cross-loading
//! (risk R4): both implementations read the same bytes. Malformed data must give an error, never a panic.

mod common;

use std::sync::Arc;

use common::{assign_materials, committed_sub_scene, draw_transform, load_obj, static_mesh};
use parity::{Lcg, Tolerance, golden};
use steam_audio::Error;
use steam_audio::geometry::Aabb;
use steam_audio::math::Vector3;
use steam_audio::scene::{InstancedMesh, Scene, StaticMesh};
use steam_audio::simulation::EnergyField;

#[track_caller]
fn check_bytes(name: &str, actual: &[u8]) {
    let expected = golden::<u8>(name);
    if let Err(m) = parity::compare(actual, &expected.data, Tolerance::EXACT) {
        panic!("{name}: {m}");
    }
}

fn assert_same_static_mesh(actual: &StaticMesh, expected: &StaticMesh) {
    assert_eq!(actual.mesh(), expected.mesh());
    assert_eq!(actual.bvh(), expected.bvh());
    assert_eq!(actual.material_indices(), expected.material_indices());
    assert_eq!(actual.materials(), expected.materials());
}

// ---------------------------------------------------------------------------------------------------------------------
// Energy fields
// ---------------------------------------------------------------------------------------------------------------------

const FIELD_SIZES: [(usize, usize); 4] = [(1, 7), (4, 13), (9, 1), (16, 30)];

fn energy_fields() -> Vec<EnergyField> {
    let mut lcg = Lcg::new(211);
    let mut fields = Vec::new();
    for (i, &(num_channels, num_bins)) in FIELD_SIZES.iter().enumerate() {
        let mut field = EnergyField::with_size(num_channels, num_bins);
        for x in field.data_mut() {
            *x = lcg.next_signed();
        }
        if i == FIELD_SIZES.len() - 1 {
            let data = field.data_mut();
            data[0] = f32::NAN;
            data[1] = f32::INFINITY;
            data[2] = f32::NEG_INFINITY;
            data[3] = -0.0;
        }
        fields.push(field);
    }
    fields
}

#[test]
fn energy_field() {
    let fields = energy_fields();
    for (i, field) in fields.iter().enumerate() {
        let name = format!("serialization/energy_field_{i}");
        check_bytes(&name, &field.serialize());

        let read = EnergyField::deserialize(&golden::<u8>(&name).data).expect("valid energy field");
        assert_eq!(
            (read.num_channels(), read.num_bins()),
            (field.num_channels(), field.num_bins())
        );
        let bits = |f: &EnergyField| f.data().iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&read), bits(field), "{name}");
    }

    let empty = EnergyField::new(0.0, 1);
    check_bytes("serialization/energy_field_empty", &empty.serialize());
    assert!(matches!(
        EnergyField::deserialize(&empty.serialize()),
        Err(Error::MalformedSerializedData(_))
    ));

    let sizes: Vec<i32> = fields
        .iter()
        .chain([&empty])
        .map(|f| f.serialized_size() as i32)
        .collect();
    assert_eq!(
        sizes,
        golden::<i32>("serialization/energy_field_sizes").data
    );
}

// ---------------------------------------------------------------------------------------------------------------------
// Static meshes and scenes
// ---------------------------------------------------------------------------------------------------------------------

#[test]
fn static_meshes() {
    let mut lcg = Lcg::new(223);
    for name in ["smallbox", "boxroom", "simplescene", "sphere", "teapot"] {
        let mut obj = load_obj(name);
        if name == "simplescene" {
            assign_materials(&mut lcg, &mut obj);
        }
        let mesh = static_mesh(&obj);

        let golden_name = format!("serialization/static_mesh_{name}");
        check_bytes(&golden_name, &mesh.serialize());

        let read =
            StaticMesh::deserialize(&golden::<u8>(&golden_name).data).expect("valid static mesh");
        assert_same_static_mesh(&read, &mesh);
    }
}

#[test]
fn scene() {
    let mut lcg = Lcg::new(227);
    let boxroom = load_obj("boxroom");
    let mut simplescene = load_obj("simplescene");
    assign_materials(&mut lcg, &mut simplescene);
    let smallbox = load_obj("smallbox");
    let sphere = load_obj("sphere");

    let sub_scene: Arc<Scene> = committed_sub_scene(&smallbox);

    let mut scene = Scene::new();
    let boxroom_id = scene.insert_static_mesh(static_mesh(&boxroom));
    let simplescene_id = scene.insert_static_mesh(static_mesh(&simplescene));
    let smallbox_id = scene.insert_static_mesh(static_mesh(&smallbox));
    scene.add_static_mesh(boxroom_id);
    scene.add_static_mesh(smallbox_id);
    scene.add_static_mesh(simplescene_id);
    scene.add_static_mesh(smallbox_id);

    let bounds = Aabb::new(Vector3::new(-1.0, -1.0, -1.0), Vector3::new(1.0, 1.0, 1.0));
    let transform = draw_transform(&mut lcg, &bounds);
    let instance = scene.insert_instanced_mesh(InstancedMesh::new(sub_scene, transform));
    scene.add_instanced_mesh(instance);
    scene.commit();

    let sphere_id = scene.insert_static_mesh(static_mesh(&sphere));
    scene.add_static_mesh(sphere_id);

    let bytes = scene.serialize();
    check_bytes("serialization/scene", &bytes);

    let read = Scene::deserialize(&golden::<u8>("serialization/scene").data).expect("valid scene");
    assert_eq!(read.num_static_meshes(), 4);
    assert_eq!(read.num_instanced_meshes(), 0);
    assert_eq!(read.version(), 0);
    assert_eq!(read.serialize(), bytes);

    // Committing a deserialized scene changes nothing: its meshes are already committed.
    let mut committed = read.clone();
    committed.commit();
    assert_eq!(committed.version(), 0);
    assert_eq!(committed.serialize(), bytes);
}

// ---------------------------------------------------------------------------------------------------------------------
// Malformed data
// ---------------------------------------------------------------------------------------------------------------------

#[test]
fn truncated_data_is_an_error() {
    let mesh = golden::<u8>("serialization/static_mesh_smallbox").data;
    for len in 0..mesh.len() {
        assert!(
            StaticMesh::deserialize(&mesh[..len]).is_err(),
            "static mesh truncated to {len} bytes"
        );
    }

    let field = golden::<u8>("serialization/energy_field_0").data;
    for len in 0..field.len() {
        assert!(
            EnergyField::deserialize(&field[..len]).is_err(),
            "energy field truncated to {len} bytes"
        );
    }
}

#[test]
fn corrupted_data_never_panics() {
    let scene = golden::<u8>("serialization/scene").data;
    let mut lcg = Lcg::new(229);
    let mut num_ok = 0;
    for _ in 0..512 {
        let mut data = scene.clone();
        let num_changes = 1 + (lcg.next_u32() % 4) as usize;
        for _ in 0..num_changes {
            let position = lcg.next_u32() as usize % data.len();
            let value = lcg.next_u32() as u8;
            data[position] = value;
        }
        if Scene::deserialize(&data).is_ok() {
            num_ok += 1;
        }
    }
    // Most corruptions hit vertex coordinates or materials, which stay valid.
    assert!(num_ok > 0);
}

#[test]
fn invalid_indices_are_errors() {
    let obj = load_obj("smallbox");
    let mut data = static_mesh(&obj).serialize();

    // Replace the first triangle's first index with 8, past the last vertex.
    let first = obj.triangles[0].indices;
    let pattern: Vec<u8> = first.iter().flat_map(|i| i.to_le_bytes()).collect();
    let position = data
        .windows(12)
        .position(|w| w == pattern.as_slice())
        .expect("first triangle in the buffer");
    data[position..position + 4].copy_from_slice(&8i32.to_le_bytes());
    assert!(matches!(
        StaticMesh::deserialize(&data),
        Err(Error::MalformedSerializedData(_))
    ));
}
