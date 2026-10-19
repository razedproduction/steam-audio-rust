//! Parity tests for stage 7a against golden data from `tools/golden/src/pathing.cpp`: visibility graphs, path bakes
//! (serialized batches), baked path lookups and probe path reconstruction, and A* searches.

mod common;

use std::cell::Cell;

use common::{committed_sub_scene, draw_point, load_obj, mesh_bounds};
use parity::{Lcg, Tolerance, golden};
use steam_audio::baking::{BakedData, BakedDataIdentifier, BakedDataType, BakedDataVariation};
use steam_audio::geometry::{Aabb, Sphere};
use steam_audio::math::{Matrix4, Vector3};
use steam_audio::pathing::{
    BakedPathData, PathBakeParams, PathBaker, PathFinder, ProbePath, ProbeVisibilityGraph,
    ProbeVisibilityTester, ShortestPathParams, SoundPath, VisibilityParams,
};
use steam_audio::probes::{Probe, ProbeBatch, ProbeGeneration, generate_probes};
use steam_audio::scene::Scene;
use steam_audio::thread_pool::ThreadPool;

#[track_caller]
fn check<T: parity::Element + Into<f64> + Copy>(name: &str, actual: &[T], tol: Tolerance) {
    let expected = golden::<T>(&format!("pathing/{name}"));
    if let Err(m) = parity::compare(actual, &expected.data, tol) {
        panic!("{name}: {m}");
    }
}

/// The generator's `boxTransform`.
fn box_transform(bounds: &Aabb) -> Matrix4 {
    let extents = bounds.extents();
    let center = bounds.center();
    let mut m = Matrix4::ZERO;
    for i in 0..3 {
        m[(i, i)] = extents[i];
        m[(i, 3)] = center[i];
    }
    m[(3, 3)] = 1.0;
    m
}

fn identifier() -> BakedDataIdentifier {
    BakedDataIdentifier {
        variation: BakedDataVariation::Dynamic,
        data_type: BakedDataType::Pathing,
        endpoint_influence: Sphere::default(),
    }
}

fn pathing_data(batch: &ProbeBatch) -> &BakedPathData {
    match batch.data(&identifier()).expect("pathing layer") {
        BakedData::Pathing(data) => data,
        _ => unreachable!(),
    }
}

fn append_probe_path(out: &mut Vec<i32>, path: &ProbePath) {
    let index = |i: Option<usize>| i.map_or(-1, |i| i as i32);
    out.extend([
        i32::from(path.valid),
        index(path.start),
        index(path.end),
        path.nodes.len() as i32,
    ]);
    out.extend(path.nodes.iter().map(|&i| i as i32));
}

fn append_sound_path(out: &mut Vec<i32>, path: &SoundPath) {
    out.extend([
        i32::from(path.first_probe),
        i32::from(path.last_probe),
        i32::from(path.probe_after_first),
        i32::from(path.probe_before_last),
        i32::from(path.direct),
    ]);
}

fn append_vector(out: &mut Vec<f32>, v: Vector3) {
    out.extend([v.x, v.y, v.z]);
}

/// The generator's `BakeCase`.
struct BakeCase {
    spacing: f32,
    height: f32,
    num_samples: usize,
    radius: f32,
    threshold: f32,
    vis_range: f32,
    vis_range_real_time: f32,
    path_range: f32,
    asymmetric_vis_range: bool,
    prune_vis_graph: bool,
}

const BAKES: [BakeCase; 2] = [
    BakeCase {
        spacing: 4.5,
        height: 1.5,
        num_samples: 1,
        radius: 0.0,
        threshold: 0.5,
        vis_range: 1000.0,
        vis_range_real_time: 1000.0,
        path_range: 1000.0,
        asymmetric_vis_range: false,
        prune_vis_graph: false,
    },
    BakeCase {
        spacing: 5.5,
        height: 1.0,
        num_samples: 3,
        radius: 0.8,
        threshold: 0.3,
        vis_range: 15.0,
        vis_range_real_time: 9.0,
        path_range: 40.0,
        asymmetric_vis_range: true,
        prune_vis_graph: true,
    },
];

const NUM_QUERIES: usize = 48;

impl BakeCase {
    fn params(&self, down: Vector3) -> PathBakeParams {
        PathBakeParams {
            num_samples: self.num_samples,
            radius: self.radius,
            threshold: self.threshold,
            vis_range: self.vis_range,
            vis_range_real_time: self.vis_range_real_time,
            path_range: self.path_range,
            asymmetric_vis_range: self.asymmetric_vis_range,
            down,
            prune_vis_graph: self.prune_vis_graph,
        }
    }
}

/// Checks the lookups of every pair of probes in `data` against the golden data of bake `prefix`, drawing a source
/// point per start probe from `lcg`.
fn check_lookups(
    prefix: &str,
    lcg: &mut Lcg,
    bounds: &Aabb,
    probes: &[Probe],
    data: &BakedPathData,
) {
    let n = probes.len();
    let mut lookups = Vec::new();
    let mut probe_paths = Vec::new();
    let mut values = Vec::new();
    for s in 0..n {
        let source = draw_point(lcg, bounds);
        for e in 0..n {
            let (path, probe_path) = data.lookup_probe_path(s, e);
            assert_eq!(data.lookup_shortest_path(s, e), path);

            append_sound_path(&mut lookups, &path);
            lookups.push(i32::from(path.is_valid()));
            append_probe_path(&mut probe_paths, &probe_path);

            values.extend([path.distance_internal, path.deviation_internal]);
            if path.is_valid() {
                values.push(path.distance(probes, s, e));
                values.push(path.deviation(probes, s, e));
                append_vector(&mut values, path.to_virtual_source(probes, s, e));
                values.push(path.distance_from(probes, source, e));
                append_vector(&mut values, path.to_virtual_source_from(probes, source, e));
            } else {
                values.extend([0.0; 9]);
            }
        }
    }
    check(&format!("{prefix}_lookups"), &lookups, Tolerance::EXACT);
    check(
        &format!("{prefix}_probe_paths"),
        &probe_paths,
        Tolerance::EXACT,
    );
    check(&format!("{prefix}_lookup_values"), &values, Tolerance::F32);
}

/// Bakes every case in turn, checking graphs, batches and lookups; returns the last batch and its "down".
fn bake_all(lcg: &mut Lcg, scene: &Scene, bounds: &Aabb) -> (ProbeBatch, Vector3) {
    let pool = ThreadPool::new(1);
    let mut last = None;

    for (k, case) in BAKES.iter().enumerate() {
        let prefix = format!("bake{k}");

        let mut probes = Vec::new();
        generate_probes(
            scene,
            &box_transform(bounds),
            ProbeGeneration::UniformFloor {
                spacing: case.spacing,
                height: case.height,
            },
            &mut probes,
        );
        let mut batch = ProbeBatch::new();
        batch.add_probes(&probes);
        let n = batch.num_probes();
        let probe_values: Vec<f32> = probes
            .iter()
            .flat_map(|p| {
                let c = p.influence.center;
                [c.x, c.y, c.z, p.influence.radius]
            })
            .collect();
        check(&format!("{prefix}_probes"), &probe_values, Tolerance::F32);

        let mut down = Vector3::new(0.0, -1.0, 0.0);
        if case.asymmetric_vis_range {
            let x = lcg.next_signed();
            let z = lcg.next_signed();
            down = Vector3::new(0.3 * x, -1.0, 0.3 * z).normalize_or_zero();
        }
        let params = case.params(down);

        // The graph the bake starts from.
        let tester = ProbeVisibilityTester::new(case.num_samples, case.asymmetric_vis_range, down);
        let graph = ProbeVisibilityGraph::new(
            scene,
            batch.probes(),
            &tester,
            VisibilityParams {
                radius: case.radius,
                threshold: case.threshold,
                vis_range: case.vis_range,
            },
            &pool,
        );
        assert_eq!(graph.num_probes(), n);
        let mut rows = Vec::new();
        let mut indices = Vec::new();
        let mut costs = Vec::new();
        for i in 0..n {
            rows.push(indices.len() as i32);
            for entry in graph.adjacent(i) {
                indices.push(entry.index as i32);
                costs.push(entry.cost);
            }
        }
        rows.push(indices.len() as i32);
        check(&format!("{prefix}_graph_rows"), &rows, Tolerance::EXACT);
        check(
            &format!("{prefix}_graph_indices"),
            &indices,
            Tolerance::EXACT,
        );
        check(&format!("{prefix}_graph_costs"), &costs, Tolerance::F32);

        let baker = PathBaker::new();
        let mut reports = Vec::new();
        baker.bake(scene, &identifier(), &params, &pool, &mut batch, |p| {
            reports.push(p)
        });
        // The last reports are the probes indexed, then 1.
        let tail: Vec<f32> = (1..=n)
            .map(|i| (i * n) as f32 / (n * n) as f32)
            .chain([1.0])
            .collect();
        assert!(reports.ends_with(&tail), "{prefix}: progress {reports:?}");

        // Byte for byte on Windows. Elsewhere libm's `atan2f` (deviations) may differ by an ulp; the lookups below
        // compare the paths with tolerances.
        let name = format!("{prefix}_batch");
        let golden_bytes = golden::<u8>(&format!("pathing/{name}")).data;
        if cfg!(windows) {
            check(&name, &batch.serialize(), Tolerance::EXACT);
        }

        let data = pathing_data(&batch);
        assert!(!data.needs_update());
        check(
            &format!("{prefix}_size"),
            &[data.serialized_size() as i32],
            Tolerance::EXACT,
        );
        if !case.prune_vis_graph {
            assert_eq!(data.vis_graph(), &graph, "{prefix}");
        }

        // The golden batch loads into the same graph (and on Windows the same data), and writes the same bytes.
        let read = ProbeBatch::deserialize(&golden_bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(read.probes(), batch.probes(), "{name}");
        assert_eq!(pathing_data(&read).vis_graph(), data.vis_graph(), "{name}");
        if cfg!(windows) {
            assert_eq!(pathing_data(&read), data, "{name}");
        }
        assert_eq!(read.serialize(), golden_bytes, "{name}");

        check_lookups(&prefix, lcg, bounds, batch.probes(), data);

        last = Some((batch, down));
    }

    last.expect("at least one bake")
}

#[test]
fn bakes_lookups_and_searches() {
    let mut lcg = Lcg::new(257);

    let obj = load_obj("simplescene");
    let bounds = mesh_bounds(&obj);
    let scene = committed_sub_scene(&obj);

    let (batch, down) = bake_all(&mut lcg, &scene, &bounds);

    // A* on the last bake's pruned graph.
    let case = &BAKES[1];
    let n = batch.num_probes();
    let data = pathing_data(&batch);
    let tester = ProbeVisibilityTester::new(case.num_samples, case.asymmetric_vis_range, down);
    let finder = PathFinder::new(n, 1);

    let mut paths = Vec::new();
    let mut values = Vec::new();
    for q in 0..NUM_QUERIES {
        let (mut start, mut end) = (0, 0);
        if q > 0 {
            start = (lcg.next_u32() % n as u32) as usize;
            end = (lcg.next_u32() % n as u32) as usize;
        }

        for flags in 0..4 {
            let params = ShortestPathParams {
                radius: case.radius,
                threshold: case.threshold,
                simplify_paths: flags & 1 != 0,
                real_time_vis: flags & 2 != 0,
            };
            let probe_path = finder.find_shortest_path(
                &*scene,
                batch.probes(),
                data.vis_graph(),
                &tester,
                start,
                end,
                &params,
                0,
            );
            append_probe_path(&mut paths, &probe_path);

            let sound_path = SoundPath::from_probe_path(&probe_path, batch.probes());
            append_sound_path(&mut paths, &sound_path);
            values.extend([sound_path.distance_internal, sound_path.deviation_internal]);
        }
    }
    check("astar_paths", &paths, Tolerance::EXACT);
    check("astar_values", &values, Tolerance::F32);
}

#[test]
fn bakes_on_several_threads_match_one() {
    let obj = load_obj("simplescene");
    let bounds = mesh_bounds(&obj);
    let scene = committed_sub_scene(&obj);

    for (k, case) in BAKES.iter().enumerate() {
        let mut probes = Vec::new();
        generate_probes(
            &*scene,
            &box_transform(&bounds),
            ProbeGeneration::UniformFloor {
                spacing: case.spacing,
                height: case.height,
            },
            &mut probes,
        );
        let mut batch = ProbeBatch::new();
        batch.add_probes(&probes);

        // The down vector does not matter here: both bakes use the same one.
        let params = case.params(Vector3::new(0.0, -1.0, 0.0));
        let mut bytes = Vec::new();
        for num_threads in [1, 3] {
            PathBaker::new().bake(
                &*scene,
                &identifier(),
                &params,
                &ThreadPool::new(num_threads),
                &mut batch,
                |_| {},
            );
            bytes.push(batch.serialize());
        }
        assert_eq!(bytes[0], bytes[1], "bake {k}");
    }
}

#[test]
fn cancelled_bake_leaves_no_layer() {
    let obj = load_obj("simplescene");
    let bounds = mesh_bounds(&obj);
    let scene = committed_sub_scene(&obj);

    let mut probes = Vec::new();
    generate_probes(
        &*scene,
        &box_transform(&bounds),
        ProbeGeneration::UniformFloor {
            spacing: 6.0,
            height: 1.5,
        },
        &mut probes,
    );
    let mut batch = ProbeBatch::new();
    batch.add_probes(&probes);

    let params = BAKES[0].params(Vector3::new(0.0, -1.0, 0.0));
    let pool = ThreadPool::new(1);
    let baker = PathBaker::new();
    baker.bake(&*scene, &identifier(), &params, &pool, &mut batch, |_| {});
    assert!(batch.has_data(&identifier()));

    // Cancelled in each phase: after the first report of the graph, of the paths, of the indexing.
    for cancel_at in [1, batch.num_probes() + 3, 2 * batch.num_probes() + 4] {
        let reports = Cell::new(0);
        baker.bake(&*scene, &identifier(), &params, &pool, &mut batch, |_| {
            reports.set(reports.get() + 1);
            if reports.get() == cancel_at {
                baker.cancel();
            }
        });
        assert!(!batch.has_data(&identifier()), "cancelled at {cancel_at}");
    }

    // The flag is cleared: the next bake completes. Cancelling with no bake running does nothing.
    baker.cancel();
    baker.bake(&*scene, &identifier(), &params, &pool, &mut batch, |_| {});
    assert!(batch.has_data(&identifier()));
}

#[test]
fn probe_changes_make_paths_need_baking() {
    let golden_batch = golden::<u8>("pathing/bake1_batch").data;
    let changes: [fn(&mut ProbeBatch); 3] = [
        |b| b.update_probe_position(1, Vector3::new(1.0, 2.0, 3.0)),
        |b| b.add_probe(Sphere::new(Vector3::ZERO, 1.0)),
        |b| b.remove_probe(0),
    ];
    for change in changes {
        let mut batch = ProbeBatch::deserialize(&golden_batch).unwrap();
        assert!(!pathing_data(&batch).needs_update());
        change(&mut batch);
        assert!(pathing_data(&batch).needs_update());
    }
}

#[test]
fn truncated_batches_are_errors() {
    let data = golden::<u8>("pathing/bake1_batch").data;
    for len in 0..data.len() {
        assert!(
            ProbeBatch::deserialize(&data[..len]).is_err(),
            "batch truncated to {len} bytes"
        );
    }
}

#[test]
fn corrupted_batches_never_panic() {
    let batch = golden::<u8>("pathing/bake1_batch").data;
    let mut lcg = Lcg::new(263);
    let mut num_ok = 0;
    for _ in 0..256 {
        let mut data = batch.clone();
        let num_changes = 1 + (lcg.next_u32() % 4) as usize;
        for _ in 0..num_changes {
            let position = lcg.next_u32() as usize % data.len();
            let value = lcg.next_u32() as u8;
            data[position] = value;
        }
        let Ok(read) = ProbeBatch::deserialize(&data) else {
            continue;
        };
        num_ok += 1;
        let _ = read.serialize();
        if let Some(BakedData::Pathing(paths)) = read.data(&identifier()) {
            let n = read.num_probes();
            for pair in (0..n * n).step_by(7) {
                let _ = paths.lookup_probe_path(pair / n, pair % n);
            }
        }
    }
    assert!(num_ok > 0);
}
