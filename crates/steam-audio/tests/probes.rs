//! Parity tests for stage 6.4b against golden data from `tools/golden/src/probes.cpp`: probe trees, probe generation,
//! probe neighbourhoods, baked reflections and their lookups, and probe batch serialization.

mod common;

use std::sync::Arc;

use common::{
    committed_sub_scene, draw_point, draw_transform, draw_vector, draw01, load_obj, mesh_bounds,
};
use parity::{Lcg, Tolerance, golden};
use steam_audio::Error;
use steam_audio::baking::{
    BakedData, BakedDataIdentifier, BakedDataType, BakedDataVariation, BakedReflectionsData,
};
use steam_audio::geometry::{Aabb, Sphere};
use steam_audio::math::{Matrix4, Quaternion, Vector3};
use steam_audio::probes::{
    Probe, ProbeBatch, ProbeGeneration, ProbeManager, ProbeNeighborhood, ProbeTree, generate_probes,
};
use steam_audio::scene::Scene;
use steam_audio::simulation::baked_reflection::{
    find_unique_probe_batches, lookup_energy_field, lookup_reverb,
};
use steam_audio::simulation::{EnergyField, Reverb};

#[track_caller]
fn check<T: parity::Element + Into<f64> + Copy>(name: &str, actual: &[T], tol: Tolerance) {
    let expected = golden::<T>(&format!("probes/{name}"));
    if let Err(m) = parity::compare(actual, &expected.data, tol) {
        panic!("{name}: {m}");
    }
}

fn indices(entries: &[Option<usize>]) -> impl Iterator<Item = i32> + '_ {
    entries.iter().map(|e| e.map_or(-1, |i| i as i32))
}

/// The generator's `boxTransform`: the box `bounds`, optionally rotated by an Lcg quaternion about its center.
fn box_transform(lcg: &mut Lcg, bounds: &Aabb, rotate: bool) -> Matrix4 {
    let mut rotation = [[0.0f32; 3]; 3];
    for (i, row) in rotation.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    if rotate {
        let x = lcg.next_signed();
        let y = lcg.next_signed();
        let z = lcg.next_signed();
        let w = lcg.next_signed();
        let mut q = Quaternion::new(x, y, z, w);
        q.normalize();
        let r = q.to_rotation_matrix();
        for (i, row) in rotation.iter_mut().enumerate() {
            for (j, value) in row.iter_mut().enumerate() {
                *value = r[(i, j)];
            }
        }
    }

    let extents = bounds.extents();
    let center = bounds.center();
    let mut m = Matrix4::ZERO;
    for i in 0..3 {
        for j in 0..3 {
            m[(i, j)] = rotation[i][j] * extents[j];
        }
        m[(i, 3)] = center[i];
    }
    m[(3, 3)] = 1.0;
    m
}

fn probe_values(probes: &[Probe]) -> Vec<f32> {
    probes
        .iter()
        .flat_map(|p| {
            let c = p.influence.center;
            [c.x, c.y, c.z, p.influence.radius]
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------------------------------
// Probe trees
// ---------------------------------------------------------------------------------------------------------------------

const TREE_SIZES: [usize; 5] = [1, 2, 7, 37, 130];

#[test]
fn probe_trees() {
    let mut lcg = Lcg::new(233);

    for (k, &num_probes) in TREE_SIZES.iter().enumerate() {
        let mut probes = vec![Probe::default(); num_probes];
        for i in 0..num_probes {
            if i % 5 == 4 {
                probes[i].influence.center = probes[i - 3].influence.center;
            } else {
                let v = draw_vector(&mut lcg);
                probes[i].influence.center = v * 10.0;
            }
            let r = draw01(&mut lcg);
            probes[i].influence.radius = 0.5 + 3.0 * r;
        }

        let tree = ProbeTree::new(&probes);
        assert_eq!(tree.nodes().len(), 2 * num_probes - 1);

        let mut values = Vec::new();
        let mut data = Vec::new();
        for node in tree.nodes() {
            let b = node.bounds();
            values.extend([b.min.x, b.min.y, b.min.z, b.max.x, b.max.y, b.max.z]);
            values.push(node.split_coordinate());
            data.push(node.packed_data());
        }
        check(&format!("tree_{k}_nodes"), &values, Tolerance::EXACT);
        check(&format!("tree_{k}_data"), &data, Tolerance::EXACT);

        let mut queries8 = Vec::new();
        let mut queries3 = Vec::new();
        for _ in 0..64 {
            let v = draw_vector(&mut lcg);
            let point = v * 12.0;

            let mut found = [None; 8];
            tree.influencing_probes(point, &probes, &mut found);
            queries8.extend(indices(&found));

            let mut found = [None; 3];
            tree.influencing_probes(point, &probes, &mut found);
            queries3.extend(indices(&found));
        }
        check(&format!("tree_{k}_queries8"), &queries8, Tolerance::EXACT);
        check(&format!("tree_{k}_queries3"), &queries3, Tolerance::EXACT);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Probe generation
// ---------------------------------------------------------------------------------------------------------------------

#[test]
fn probe_generation() {
    let mut lcg = Lcg::new(239);

    let mut empty_scene = Scene::new();
    empty_scene.commit();

    let unit_box = Aabb::new(Vector3::new(-1.0, -1.0, -1.0), Vector3::new(1.0, 1.0, 1.0));
    let mut centroids = Vec::new();
    for _ in 0..3 {
        let transform = draw_transform(&mut lcg, &unit_box);
        let mut probes = Vec::new();
        generate_probes(
            &empty_scene,
            &transform,
            ProbeGeneration::Centroid,
            &mut probes,
        );
        assert_eq!(probes.len(), 1);
        centroids.extend(probe_values(&probes));
    }
    check("generator_centroid", &centroids, Tolerance::F32);

    let cases = [
        ("boxroom", "boxroom", false, 1.3, 1.5),
        ("rotated", "simplescene", true, 2.1, 0.7),
    ];
    for (name, mesh, rotate, spacing, height) in cases {
        let obj = load_obj(mesh);
        let scene = committed_sub_scene(&obj);
        let transform = box_transform(&mut lcg, &mesh_bounds(&obj), rotate);

        let mut probes = Vec::new();
        generate_probes(
            &*scene,
            &transform,
            ProbeGeneration::UniformFloor { spacing, height },
            &mut probes,
        );
        let name = format!("generator_floor_{name}");
        let expected = golden::<f32>(&format!("probes/{name}"));
        assert_eq!(probes.len(), expected.shape[0], "{name}");
        check(&name, &probe_values(&probes), Tolerance::F32);
    }
}

#[test]
fn degenerate_floor_box_leaves_probes_alone() {
    let scene = Scene::new();
    let mut probes = vec![Probe::default(); 3];
    generate_probes(
        &scene,
        &Matrix4::ZERO,
        ProbeGeneration::UniformFloor {
            spacing: 1.0,
            height: 1.0,
        },
        &mut probes,
    );
    assert_eq!(probes.len(), 3);
}

// ---------------------------------------------------------------------------------------------------------------------
// Probe batches, neighbourhoods, baked reflections
// ---------------------------------------------------------------------------------------------------------------------

const FIELD_CHANNELS: usize = 4;
const FIELD_BINS: usize = 9;
const NUM_BANDS: usize = 3;

fn identifier(variation: BakedDataVariation, center: Vector3, radius: f32) -> BakedDataIdentifier {
    BakedDataIdentifier {
        variation,
        data_type: BakedDataType::Reflections,
        endpoint_influence: Sphere::new(center, radius),
    }
}

/// The generator's `addLayer`.
fn add_layer(
    lcg: &mut Lcg,
    batch: &mut ProbeBatch,
    identifier: BakedDataIdentifier,
    has_convolution: bool,
    has_parametric: bool,
) {
    let mut data = BakedReflectionsData::new(
        identifier,
        batch.num_probes(),
        has_convolution,
        has_parametric,
    );

    for i in 0..batch.num_probes() {
        if i % 4 == 3 {
            continue;
        }

        if has_convolution {
            if i % 4 == 2 {
                data.set_energy_field(i, None);
            } else {
                let mut field = EnergyField::with_size(FIELD_CHANNELS, FIELD_BINS);
                for x in field.data_mut() {
                    *x = draw01(lcg);
                }
                data.set_energy_field(i, Some(field));
            }
        }

        if has_parametric {
            let mut reverb = Reverb::default();
            for time in &mut reverb.reverb_times {
                let x = draw01(lcg);
                *time = 0.1 + 2.0 * x;
            }
            data.set_reverb(i, reverb);
        }
    }

    batch.add_data(identifier, BakedData::Reflections(data));
}

fn reflections_mut<'a>(
    batch: &'a mut ProbeBatch,
    identifier: &BakedDataIdentifier,
) -> &'a mut BakedReflectionsData {
    match batch.data_mut(identifier).expect("layer exists") {
        BakedData::Reflections(data) => data,
        _ => unreachable!(),
    }
}

fn append_sizes(sizes: &mut Vec<i32>, batch: &ProbeBatch) {
    sizes.extend(
        batch
            .data_layers()
            .map(|(_, data)| data.serialized_size() as i32),
    );
}

/// Checks the batch's bytes and that they deserialize into a batch with the same probes, which serializes to the same
/// bytes.
#[track_caller]
fn check_batch(name: &str, batch: &ProbeBatch) {
    let bytes = batch.serialize();
    check(name, &bytes, Tolerance::EXACT);

    let read = ProbeBatch::deserialize(&golden::<u8>(&format!("probes/{name}")).data)
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_eq!(read.probes(), batch.probes(), "{name}");
    assert_eq!(read.serialize(), bytes, "{name}");
}

#[test]
fn batches_neighborhoods_and_lookups() {
    let mut lcg = Lcg::new(241);

    let boxroom = load_obj("boxroom");
    let bounds = mesh_bounds(&boxroom);
    let scene = committed_sub_scene(&boxroom);

    let mut b0 = ProbeBatch::new();
    {
        let transform = box_transform(&mut lcg, &bounds, false);
        let mut probes = Vec::new();
        generate_probes(
            &*scene,
            &transform,
            ProbeGeneration::UniformFloor {
                spacing: 4.0,
                height: 1.5,
            },
            &mut probes,
        );
        assert_eq!(probes.len(), 25);
        b0.add_probes(&probes);
    }

    let mut b1 = ProbeBatch::new();
    for _ in 0..25 {
        let center = draw_point(&mut lcg, &bounds);
        let r = draw01(&mut lcg);
        b1.add_probe(Sphere::new(center, 1.0 + 3.0 * r));
    }

    let reverb_id = identifier(BakedDataVariation::Reverb, Vector3::ZERO, 0.0);
    let source_a = draw_point(&mut lcg, &bounds);
    let source_b = Vector3::new(source_a.x, source_a.y + 0.5, source_a.z);
    let source_a_id = identifier(BakedDataVariation::StaticSource, source_a, 5.0);
    let source_b_id = identifier(BakedDataVariation::StaticSource, source_b, 5.0);
    let listener_id = identifier(
        BakedDataVariation::StaticListener,
        Vector3::new(1.0, 2.0, 3.0),
        4.0,
    );

    add_layer(&mut lcg, &mut b0, reverb_id, true, true);
    add_layer(&mut lcg, &mut b0, listener_id, false, true);
    add_layer(&mut lcg, &mut b1, source_b_id, true, false);
    add_layer(&mut lcg, &mut b1, reverb_id, true, true);
    add_layer(&mut lcg, &mut b1, source_a_id, true, false);

    b0.commit();
    b1.commit();

    let b0 = Arc::new(b0);
    let b1 = Arc::new(b1);
    let mut manager = ProbeManager::new();
    manager.add_probe_batch(b0.clone());
    manager.add_probe_batch(b1.clone());
    manager.add_probe_batch(b0.clone());
    manager.commit();
    let batches = manager.probe_batches();

    let lookup_ids = [reverb_id, source_a_id, listener_id];
    const NUM_POINTS: usize = 48;
    const OUT_CHANNELS: usize = 9;
    const OUT_BINS: usize = 7;

    let golden_unique = golden::<i32>("probes/lookup_unique");

    let mut influencing_batches = Vec::new();
    let mut influencing_probes = Vec::new();
    let mut visible_batches = Vec::new();
    let mut visible_probes = Vec::new();
    let mut weights = Vec::new();
    let mut nearest = Vec::new();
    let mut fields: [Vec<f32>; 3] = Default::default();
    let mut reverbs: [Vec<f32>; 3] = Default::default();

    let mut neighborhood = ProbeNeighborhood::default();
    let mut unique = Vec::new();

    for p in 0..NUM_POINTS {
        let point = draw_point(&mut lcg, &bounds);

        manager.influencing_probes(point, &mut neighborhood);
        influencing_batches.extend(indices(&neighborhood.batches));
        influencing_probes.extend(indices(&neighborhood.probe_indices));

        neighborhood.check_occlusion(&*scene, point, batches);
        visible_batches.extend(indices(&neighborhood.batches));
        visible_probes.extend(indices(&neighborhood.probe_indices));

        neighborhood.calc_weights(point, batches);
        weights.extend_from_slice(&neighborhood.weights);

        nearest.push(neighborhood.find_nearest(point, batches) as i32);

        // Upstream's order of unique batches depends on their addresses: check the set, then add in its order.
        find_unique_probe_batches(&neighborhood, batches, &mut unique);
        let canonical = |b: usize| {
            batches
                .iter()
                .position(|x| Arc::ptr_eq(x, &batches[b]))
                .expect("batch in list")
        };
        let mut found: Vec<usize> = unique.iter().map(|&b| canonical(b)).collect();
        let upstream_order: Vec<usize> = golden_unique
            .row(p)
            .iter()
            .filter(|&&b| b >= 0)
            .map(|&b| b as usize)
            .collect();
        let mut expected = upstream_order.clone();
        found.sort_unstable();
        expected.sort_unstable();
        assert_eq!(found, expected, "unique batches at point {p}");

        for k in 0..3 {
            let mut field = EnergyField::with_size(OUT_CHANNELS, OUT_BINS);
            field.data_mut().fill(7.0);
            lookup_energy_field(
                &lookup_ids[k],
                &neighborhood,
                batches,
                &upstream_order,
                &mut field,
            );
            fields[k].extend_from_slice(field.data());

            let mut reverb = Reverb {
                reverb_times: [7.0; NUM_BANDS],
            };
            lookup_reverb(
                &lookup_ids[k],
                &neighborhood,
                batches,
                &upstream_order,
                &mut reverb,
            );
            reverbs[k].extend_from_slice(&reverb.reverb_times);
        }
    }

    // The cases are meant to cover: entries found, entries occluded, points without any visible probe (outputs left
    // alone), and lookups that found something.
    let count_valid = |probes: &[i32]| probes.iter().filter(|&&p| p >= 0).count();
    let num_found = count_valid(&influencing_probes);
    let num_visible = count_valid(&visible_probes);
    assert!(
        num_visible > 0 && num_visible < num_found,
        "{num_visible} of {num_found}"
    );
    let entries_per_point = influencing_probes.len() / NUM_POINTS;
    let num_blind = visible_probes
        .chunks(entries_per_point)
        .filter(|c| count_valid(c) == 0)
        .count();
    assert!(num_blind > 0 && num_blind < NUM_POINTS, "{num_blind}");
    for k in 0..3 {
        assert!(fields[k].iter().any(|&x| x != 0.0 && x != 7.0) || k == 2);
        assert!(reverbs[k].iter().any(|&x| x != 0.0 && x != 7.0) || k == 1);
    }

    check(
        "neighborhood_batches",
        &influencing_batches,
        Tolerance::EXACT,
    );
    check("neighborhood_probes", &influencing_probes, Tolerance::EXACT);
    check(
        "neighborhood_visible_batches",
        &visible_batches,
        Tolerance::EXACT,
    );
    check(
        "neighborhood_visible_probes",
        &visible_probes,
        Tolerance::EXACT,
    );
    check("neighborhood_weights", &weights, Tolerance::F32);
    check("neighborhood_nearest", &nearest, Tolerance::EXACT);
    for k in 0..3 {
        check(&format!("lookup_fields_{k}"), &fields[k], Tolerance::F32);
        check(&format!("lookup_reverbs_{k}"), &reverbs[k], Tolerance::F32);
    }

    // Serialization, and changes to B1.
    drop(manager);
    let b0 = Arc::try_unwrap(b0).expect("sole owner");
    let mut b1 = Arc::try_unwrap(b1).expect("sole owner");

    let mut sizes = Vec::new();
    check_batch("batch0", &b0);
    append_sizes(&mut sizes, &b0);

    let mut serialize_b1 = |step: usize, b1: &ProbeBatch| {
        check_batch(&format!("batch1_{step}"), b1);
        append_sizes(&mut sizes, b1);
    };

    serialize_b1(0, &b1);

    b1.update_probe_position(4, draw_point(&mut lcg, &bounds));
    b1.remove_probe(5);
    let center = draw_point(&mut lcg, &bounds);
    b1.add_probe(Sphere::new(center, 2.0));
    serialize_b1(1, &b1);

    b1.update_endpoint(&source_a_id, &Sphere::new(source_a, 2.0));
    serialize_b1(2, &b1);
    // Some baked probes lost their field: they are written as null offsets (to the end of the buffer).
    let layer = |name: &str| {
        let read = ProbeBatch::deserialize(&golden::<u8>(&format!("probes/{name}")).data).unwrap();
        match read.data(&source_a_id) {
            Some(BakedData::Reflections(data)) => data.clone(),
            _ => unreachable!(),
        }
    };
    let (before, after) = (layer("batch1_1"), layer("batch1_2"));
    let dropped = (0..after.num_probes())
        .filter(|&i| {
            before.energy_field(i).is_some()
                && after.energy_field(i).is_none()
                && !after.needs_update(i)
        })
        .count();
    assert!(dropped > 0);

    b1.update_endpoint(&source_a_id, &Sphere::new(source_a, 8.0));
    serialize_b1(3, &b1);

    b1.update_endpoint(&source_a_id, &Sphere::new(source_b, 5.0));
    serialize_b1(4, &b1);

    {
        let data = reflections_mut(&mut b1, &reverb_id);
        data.set_has_convolution(false);
        data.set_has_parametric(false);
        data.set_has_parametric(true);
    }
    serialize_b1(5, &b1);

    b1.remove_data(&source_b_id);
    {
        let data = reflections_mut(&mut b1, &source_a_id);
        data.set_energy_field(0, None);
        let mut field = EnergyField::with_size(1, 3);
        for x in field.data_mut() {
            *x = draw01(&mut lcg);
        }
        data.set_energy_field(1, Some(field));
    }
    serialize_b1(6, &b1);

    check("batch_sizes", &sizes, Tolerance::EXACT);
}

// ---------------------------------------------------------------------------------------------------------------------
// Malformed data
// ---------------------------------------------------------------------------------------------------------------------

#[test]
fn truncated_batches_are_errors() {
    let data = golden::<u8>("probes/batch1_0").data;
    for len in 0..data.len() {
        assert!(
            ProbeBatch::deserialize(&data[..len]).is_err(),
            "batch truncated to {len} bytes"
        );
    }
}

#[test]
fn corrupted_batches_never_panic() {
    let batch = golden::<u8>("probes/batch1_2").data;
    let mut lcg = Lcg::new(251);
    let mut num_ok = 0;
    for _ in 0..512 {
        let mut data = batch.clone();
        let num_changes = 1 + (lcg.next_u32() % 4) as usize;
        for _ in 0..num_changes {
            let position = lcg.next_u32() as usize % data.len();
            let value = lcg.next_u32() as u8;
            data[position] = value;
        }
        if let Ok(read) = ProbeBatch::deserialize(&data) {
            num_ok += 1;
            // Whatever loaded must serialize again.
            let _ = read.serialize();
        }
    }
    // Most corruptions hit energies, which stay valid.
    assert!(num_ok > 0);
}

#[test]
fn unsupported_and_empty_batches_are_errors() {
    let mut batch = ProbeBatch::new();
    assert!(matches!(
        ProbeBatch::deserialize(&batch.serialize()),
        Err(Error::MalformedSerializedData(_))
    ));

    batch.add_probe(Sphere::new(Vector3::ZERO, 1.0));
    let pathing = BakedDataIdentifier {
        variation: BakedDataVariation::Dynamic,
        data_type: BakedDataType::Pathing,
        endpoint_influence: Sphere::default(),
    };
    batch.add_data(
        pathing,
        BakedData::Reflections(BakedReflectionsData::new(pathing, 1, false, true)),
    );
    assert!(matches!(
        ProbeBatch::deserialize(&batch.serialize()),
        Err(Error::MalformedSerializedData(_))
    ));
}
