//! Scenes, static meshes and instanced meshes (upstream `api_scene.cpp`, `scene_factory.cpp`, `custom_scene.cpp`).
//!
//! A default scene is the core's [`Scene`] behind a read-write lock: ray tracing (probe generation, baking) reads it,
//! changes write it. Meshes belong to the scene they were created in and are referred to by id; their handles keep the
//! scene alive. A custom scene calls the caller's ray tracing callbacks, as upstream's `CustomScene` does.
//!
//! An instanced mesh shows its sub-scene as it was when the instance was created, committed (upstream commits the
//! sub-scene again at every commit of the parent, so later changes to it would show too; `DEVIATIONS.md`).

use std::ffi::{CStr, c_void};
use std::sync::{Arc, Mutex, RwLock};

use steam_audio::geometry::Ray;
use steam_audio::scene::{
    Hit, InstancedMesh, InstancedMeshId, Material, RayTracer, Scene, StaticMesh, StaticMeshId,
    Triangle,
};

use crate::context::context_arc;
use crate::handle::{Handle, lock, object, read, retain_release, write};
use crate::serialized_object::{store_serialized_bytes, with_serialized_bytes};
use crate::types::*;
use crate::{c_path, from_ipl_matrix, from_ipl_vector, guard, to_ipl_vector};

// --------------------------------------------------------------------------------------------------------------------
// Scene objects
// --------------------------------------------------------------------------------------------------------------------

/// What an `IPLScene` holds: Embree and Radeon Rays scenes are out of scope.
pub(crate) enum SceneObject {
    Default(DefaultScene),
    Custom(CustomScene),
}

impl SceneObject {
    /// Runs `f` with the scene's ray tracer (a default scene is locked for reading meanwhile).
    pub(crate) fn with_ray_tracer<R>(&self, f: impl FnOnce(&dyn RayTracer) -> R) -> R {
        match self {
            Self::Default(scene) => f(&*read(&scene.scene)),
            Self::Custom(scene) => f(scene),
        }
    }

    fn as_default(&self) -> Option<&DefaultScene> {
        match self {
            Self::Default(scene) => Some(scene),
            Self::Custom(_) => None,
        }
    }
}

pub(crate) struct DefaultScene {
    scene: Arc<RwLock<Scene>>,
    /// A committed copy of the scene for instancing it, made when first needed after a change.
    snapshot: Mutex<Option<Arc<Scene>>>,
}

impl DefaultScene {
    fn new(scene: Scene) -> Self {
        Self {
            scene: Arc::new(RwLock::new(scene)),
            snapshot: Mutex::new(None),
        }
    }

    /// Changes the scene.
    fn update<R>(&self, f: impl FnOnce(&mut Scene) -> R) -> R {
        let result = f(&mut write(&self.scene));
        // After releasing the scene, so that `snapshot` (which locks the snapshot, then the scene) cannot deadlock.
        *lock(&self.snapshot) = None;
        result
    }

    /// The scene with its pending changes committed, as an instanced mesh shows it.
    fn snapshot(&self) -> Arc<Scene> {
        let mut snapshot = lock(&self.snapshot);
        Arc::clone(snapshot.get_or_insert_with(|| {
            let mut scene = read(&self.scene).clone();
            scene.commit();
            Arc::new(scene)
        }))
    }
}

pub(crate) type SceneHandle = Handle<SceneObject>;

/// The scene behind a handle.
///
/// # Safety
///
/// `scene` is null or a live scene.
pub(crate) unsafe fn scene_arc<'a>(scene: IPLScene) -> Option<&'a Arc<SceneObject>> {
    // SAFETY: guaranteed by the caller.
    unsafe { object::<SceneObject, _>(scene) }
}

// --------------------------------------------------------------------------------------------------------------------
// Custom scenes
// --------------------------------------------------------------------------------------------------------------------

/// A scene that calls the caller's ray tracer (upstream `CustomScene`).
pub(crate) struct CustomScene {
    closest_hit: IPLClosestHitCallback,
    any_hit: IPLAnyHitCallback,
    batched_closest_hit: IPLBatchedClosestHitCallback,
    batched_any_hit: IPLBatchedAnyHitCallback,
    user_data: *mut c_void,
}

// SAFETY: the callbacks and their user data are the caller's, who must allow calls from the threads the library
// simulates on, as upstream (which calls them from its simulation threads) requires.
unsafe impl Send for CustomScene {}
// SAFETY: as for `Send`; the scene itself holds no state that the calls change.
unsafe impl Sync for CustomScene {}

fn to_ipl_ray(ray: &Ray) -> IPLRay {
    IPLRay {
        origin: to_ipl_vector(ray.origin),
        direction: to_ipl_vector(ray.direction),
    }
}

fn from_ipl_material(m: &IPLMaterial) -> Material {
    Material {
        absorption: m.absorption,
        scattering: m.scattering,
        transmission: m.transmission,
    }
}

fn to_ipl_material(m: &Material) -> IPLMaterial {
    IPLMaterial {
        absorption: m.absorption,
        scattering: m.scattering,
        transmission: m.transmission,
    }
}

fn index_to_ipl(index: Option<usize>) -> IPLint32 {
    index.map_or(-1, |i| i32::try_from(i).unwrap_or(-1))
}

/// `hit` as the callbacks see it, with its material copied into `material` (null if it has none).
fn to_ipl_hit(hit: &Hit, material: &mut IPLMaterial) -> IPLHit {
    let material = match &hit.material {
        Some(m) => {
            *material = to_ipl_material(m);
            material as *mut IPLMaterial
        }
        None => std::ptr::null_mut(),
    };
    IPLHit {
        distance: hit.distance,
        triangleIndex: index_to_ipl(hit.triangle_index),
        objectIndex: index_to_ipl(hit.object_index),
        materialIndex: index_to_ipl(hit.material_index),
        normal: to_ipl_vector(hit.normal),
        material,
    }
}

/// A hit written by a callback; negative indices are missing, and the material is copied.
///
/// # Safety
///
/// `hit.material` is null or points to a material.
unsafe fn from_ipl_hit(hit: &IPLHit) -> Hit {
    Hit {
        distance: hit.distance,
        triangle_index: usize::try_from(hit.triangleIndex).ok(),
        object_index: usize::try_from(hit.objectIndex).ok(),
        material_index: usize::try_from(hit.materialIndex).ok(),
        normal: from_ipl_vector(hit.normal),
        // SAFETY: guaranteed by the caller.
        material: unsafe { hit.material.as_ref() }.map(from_ipl_material),
    }
}

fn check_batch(rays: usize, min_distances: usize, max_distances: usize, outputs: usize) {
    assert!(
        rays == min_distances && rays == max_distances && rays == outputs,
        "batch lengths differ"
    );
}

impl RayTracer for CustomScene {
    /// Calls the closest-hit callback with a hit that reports nothing (no callback: nothing is hit; upstream calls
    /// null).
    fn closest_hit(&self, ray: &Ray, min_distance: f32, max_distance: f32) -> Hit {
        let Some(callback) = self.closest_hit else {
            return Hit::default();
        };
        let ray = to_ipl_ray(ray);
        let mut material = IPLMaterial::default();
        let mut hit = to_ipl_hit(&Hit::default(), &mut material);
        // SAFETY: valid pointers; the callback's contract is the caller's.
        unsafe { callback(&ray, min_distance, max_distance, &mut hit, self.user_data) };
        // SAFETY: the callback leaves the material null or points it at a material, as `phonon.h` requires.
        unsafe { from_ipl_hit(&hit) }
    }

    /// Calls the any-hit callback (no callback: nothing is hit).
    fn any_hit(&self, ray: &Ray, min_distance: f32, max_distance: f32) -> bool {
        let Some(callback) = self.any_hit else {
            return false;
        };
        let ray = to_ipl_ray(ray);
        let mut occluded = 0u8;
        // SAFETY: valid pointers; the callback's contract is the caller's.
        unsafe {
            callback(
                &ray,
                min_distance,
                max_distance,
                &mut occluded,
                self.user_data,
            )
        };
        occluded != 0
    }

    /// Calls the batched closest-hit callback with the hits as they are, or the closest-hit callback for each ray.
    fn closest_hits(
        &self,
        rays: &[Ray],
        min_distances: &[f32],
        max_distances: &[f32],
        hits: &mut [Hit],
    ) {
        check_batch(
            rays.len(),
            min_distances.len(),
            max_distances.len(),
            hits.len(),
        );
        let Some(callback) = self.batched_closest_hit else {
            for (i, hit) in hits.iter_mut().enumerate() {
                *hit = self.closest_hit(&rays[i], min_distances[i], max_distances[i]);
            }
            return;
        };

        let ipl_rays: Vec<IPLRay> = rays.iter().map(to_ipl_ray).collect();
        let mut materials = vec![IPLMaterial::default(); hits.len()];
        let mut ipl_hits: Vec<IPLHit> = hits
            .iter()
            .zip(&mut materials)
            .map(|(hit, material)| to_ipl_hit(hit, material))
            .collect();
        let n = i32::try_from(rays.len()).expect("a ray batch fits an int");
        // SAFETY: valid arrays of `n` elements; the callback's contract is the caller's.
        unsafe {
            callback(
                n,
                ipl_rays.as_ptr(),
                min_distances.as_ptr(),
                max_distances.as_ptr(),
                ipl_hits.as_mut_ptr(),
                self.user_data,
            );
        }
        for (hit, ipl_hit) in hits.iter_mut().zip(&ipl_hits) {
            // SAFETY: the callback leaves each material null or points it at a material.
            *hit = unsafe { from_ipl_hit(ipl_hit) };
        }
    }

    /// Calls the batched any-hit callback with the flags as they are, or the any-hit callback for each ray (a ray
    /// whose maximum distance is negative then counts as occluded, as upstream).
    fn any_hits(
        &self,
        rays: &[Ray],
        min_distances: &[f32],
        max_distances: &[f32],
        occluded: &mut [bool],
    ) {
        check_batch(
            rays.len(),
            min_distances.len(),
            max_distances.len(),
            occluded.len(),
        );
        let Some(callback) = self.batched_any_hit else {
            for (i, occluded) in occluded.iter_mut().enumerate() {
                *occluded = if max_distances[i] >= 0.0 {
                    self.any_hit(&rays[i], min_distances[i], max_distances[i])
                } else {
                    true
                };
            }
            return;
        };

        let ipl_rays: Vec<IPLRay> = rays.iter().map(to_ipl_ray).collect();
        let mut flags: Vec<u8> = occluded.iter().map(|&o| u8::from(o)).collect();
        let n = i32::try_from(rays.len()).expect("a ray batch fits an int");
        // SAFETY: valid arrays of `n` elements; the callback's contract is the caller's.
        unsafe {
            callback(
                n,
                ipl_rays.as_ptr(),
                min_distances.as_ptr(),
                max_distances.as_ptr(),
                flags.as_mut_ptr(),
                self.user_data,
            );
        }
        for (occluded, &flag) in occluded.iter_mut().zip(&flags) {
            *occluded = flag != 0;
        }
    }
}

// --------------------------------------------------------------------------------------------------------------------
// Scenes
// --------------------------------------------------------------------------------------------------------------------

/// Creates a scene: the built-in ray tracer (`IPL_SCENETYPE_DEFAULT`) or the caller's callbacks
/// (`IPL_SCENETYPE_CUSTOM`). Embree, Radeon Rays and unknown types give `IPL_STATUS_INITIALIZATION`, as upstream built
/// without them.
///
/// # Safety
///
/// `context` is null or a live context, and the other pointers are null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplSceneCreate(
    context: IPLContext,
    settings: *mut IPLSceneSettings,
    scene: *mut IPLScene,
) -> IPLerror {
    guard(IPL_STATUS_FAILURE, || {
        // SAFETY: guaranteed by the caller.
        let Some(context) = (unsafe { context_arc(context) }) else {
            return IPL_STATUS_FAILURE;
        };
        // SAFETY: guaranteed by the caller.
        let (Some(settings), Some(out)) = (unsafe { settings.as_ref() }, unsafe { scene.as_mut() })
        else {
            return IPL_STATUS_FAILURE;
        };

        let value = match settings.r#type {
            IPL_SCENETYPE_DEFAULT => SceneObject::Default(DefaultScene::new(Scene::new())),
            IPL_SCENETYPE_CUSTOM => SceneObject::Custom(CustomScene {
                closest_hit: settings.closestHitCallback,
                any_hit: settings.anyHitCallback,
                batched_closest_hit: settings.batchedClosestHitCallback,
                batched_any_hit: settings.batchedAnyHitCallback,
                user_data: settings.userData,
            }),
            _ => return IPL_STATUS_INITIALIZATION,
        };
        *out = Handle::into_raw(value, Some(context));
        IPL_STATUS_SUCCESS
    })
}

retain_release!(IPLScene, SceneObject, iplSceneRetain, iplSceneRelease);

/// Loads a default scene from a serialized object (every static mesh added and committed). The progress callback is
/// never called, as upstream.
///
/// Returns `IPL_STATUS_INITIALIZATION` for a type other than `IPL_SCENETYPE_DEFAULT` (upstream cannot load custom
/// scenes and is built without the others), and `IPL_STATUS_FAILURE` for data that is not a serialized scene (upstream
/// does not check it).
///
/// # Safety
///
/// `context` is null or a live context, `serializedObject` is null or a live serialized object, and the other pointers
/// are null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplSceneLoad(
    context: IPLContext,
    settings: *mut IPLSceneSettings,
    serializedObject: IPLSerializedObject,
    _progressCallback: IPLProgressCallback,
    _progressCallbackUserData: *mut c_void,
    scene: *mut IPLScene,
) -> IPLerror {
    guard(IPL_STATUS_FAILURE, || {
        // SAFETY: guaranteed by the caller.
        let Some(context) = (unsafe { context_arc(context) }) else {
            return IPL_STATUS_FAILURE;
        };
        // SAFETY: guaranteed by the caller.
        let (Some(settings), Some(out)) = (unsafe { settings.as_ref() }, unsafe { scene.as_mut() })
        else {
            return IPL_STATUS_FAILURE;
        };
        if serializedObject.is_null() {
            return IPL_STATUS_FAILURE;
        }
        if settings.r#type != IPL_SCENETYPE_DEFAULT {
            return IPL_STATUS_INITIALIZATION;
        }

        // SAFETY: guaranteed by the caller.
        let loaded = unsafe { with_serialized_bytes(serializedObject, Scene::deserialize) };
        match loaded {
            Some(Ok(value)) => {
                let value = SceneObject::Default(DefaultScene::new(value));
                *out = Handle::into_raw(value, Some(context));
                IPL_STATUS_SUCCESS
            }
            _ => IPL_STATUS_FAILURE,
        }
    })
}

/// Saves a default scene's committed static meshes into a serialized object (nothing for a custom scene).
///
/// # Safety
///
/// `scene` is null or a live scene, and `serializedObject` null or a live serialized object.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplSceneSave(scene: IPLScene, serializedObject: IPLSerializedObject) {
    guard((), || {
        // SAFETY: guaranteed by the caller.
        let Some(scene) = (unsafe { scene_arc(scene) }) else {
            return;
        };
        if let Some(scene) = scene.as_default() {
            let bytes = read(&scene.scene).serialize();
            // SAFETY: guaranteed by the caller.
            unsafe { store_serialized_bytes(serializedObject, bytes) };
        }
    })
}

/// Writes a default scene's committed geometry as an OBJ file and an `.mtl` file next to it (nothing for a custom
/// scene; files that cannot be written are skipped, where upstream crashes).
///
/// # Safety
///
/// `scene` is null or a live scene, and `fileBaseName` null or a NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplSceneSaveOBJ(scene: IPLScene, fileBaseName: IPLstring) {
    guard((), || {
        // SAFETY: guaranteed by the caller.
        let Some(scene) = (unsafe { scene_arc(scene) }) else {
            return;
        };
        if fileBaseName.is_null() {
            return;
        }
        // SAFETY: guaranteed by the caller.
        let name = unsafe { CStr::from_ptr(fileBaseName) };
        if let (Some(scene), Some(path)) = (scene.as_default(), c_path(name)) {
            // Upstream ignores write errors too (it does not check `fopen`).
            let _ = read(&scene.scene).dump_obj(&path);
        }
    })
}

/// Makes the changes to a scene's meshes visible to rays (nothing for a custom scene).
///
/// # Safety
///
/// `scene` is null or a live scene.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplSceneCommit(scene: IPLScene) {
    guard((), || {
        // SAFETY: guaranteed by the caller.
        if let Some(scene) = unsafe { scene_arc(scene) }.and_then(|s| s.as_default()) {
            scene.update(Scene::commit);
        }
    })
}

// --------------------------------------------------------------------------------------------------------------------
// Static meshes
// --------------------------------------------------------------------------------------------------------------------

/// A static mesh and the scene that stores it; `None` for a mesh of a custom scene (upstream's handle then holds null,
/// and every function ignores it).
pub(crate) struct StaticMeshObject(Option<(Arc<SceneObject>, StaticMeshId)>);

impl Drop for StaticMeshObject {
    /// The last reference is gone: the scene keeps the mesh while it is added, as upstream's lists keep it alive.
    fn drop(&mut self) {
        if let Some((scene, id)) = &self.0 {
            if let Some(scene) = scene.as_default() {
                scene.update(|s| s.forget_static_mesh(*id));
            }
        }
    }
}

/// The parent scene's handle (for its context) and object.
///
/// # Safety
///
/// `scene` is null or a live scene.
unsafe fn parent<'a>(scene: IPLScene) -> Option<&'a SceneHandle> {
    // SAFETY: guaranteed by the caller.
    unsafe { SceneHandle::from_raw(scene) }
}

/// Builds a static mesh from the settings; `None` if a pointer is null or an index is out of range (upstream reads out
/// of bounds).
///
/// # Safety
///
/// The settings' pointers are null or valid for their counts.
unsafe fn static_mesh(settings: &IPLStaticMeshSettings) -> Option<StaticMesh> {
    let num_vertices = usize::try_from(settings.numVertices).ok()?;
    let num_triangles = usize::try_from(settings.numTriangles).ok()?;
    let num_materials = usize::try_from(settings.numMaterials).ok()?;
    if settings.vertices.is_null()
        || settings.triangles.is_null()
        || settings.materialIndices.is_null()
        || settings.materials.is_null()
    {
        return None;
    }

    // SAFETY: guaranteed by the caller.
    let (vertices, triangles, material_indices, materials) = unsafe {
        (
            std::slice::from_raw_parts(settings.vertices, num_vertices),
            std::slice::from_raw_parts(settings.triangles, num_triangles),
            std::slice::from_raw_parts(settings.materialIndices, num_triangles),
            std::slice::from_raw_parts(settings.materials, num_materials),
        )
    };

    let index = |i: i32, n: usize| u32::try_from(i).ok().filter(|&i| (i as usize) < n);
    let triangles = triangles
        .iter()
        .map(|t| {
            Some(Triangle::new(
                index(t.indices[0], num_vertices)?,
                index(t.indices[1], num_vertices)?,
                index(t.indices[2], num_vertices)?,
            ))
        })
        .collect::<Option<Vec<_>>>()?;
    let material_indices = material_indices
        .iter()
        .map(|&i| index(i, num_materials))
        .collect::<Option<Vec<_>>>()?;
    let vertices: Vec<_> = vertices.iter().copied().map(from_ipl_vector).collect();
    let materials: Vec<_> = materials.iter().map(from_ipl_material).collect();

    Some(StaticMesh::new(
        &vertices,
        &triangles,
        &material_indices,
        &materials,
    ))
}

/// Creates a static mesh in a scene (not yet added to it). Returns `IPL_STATUS_FAILURE` if a count is not positive, a
/// pointer is null, or a vertex or material index is out of range. In a custom scene the mesh is an empty object, as
/// upstream.
///
/// # Safety
///
/// `scene` is null or a live scene, `settings` and `staticMesh` are null or valid, and the settings' arrays hold their
/// counts.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplStaticMeshCreate(
    scene: IPLScene,
    settings: *mut IPLStaticMeshSettings,
    staticMesh: *mut IPLStaticMesh,
) -> IPLerror {
    guard(IPL_STATUS_FAILURE, || {
        // SAFETY: guaranteed by the caller.
        let Some(parent) = (unsafe { parent(scene) }) else {
            return IPL_STATUS_FAILURE;
        };
        // SAFETY: guaranteed by the caller.
        let (Some(settings), Some(out)) =
            (unsafe { settings.as_ref() }, unsafe { staticMesh.as_mut() })
        else {
            return IPL_STATUS_FAILURE;
        };
        if settings.numVertices <= 0 || settings.numTriangles <= 0 || settings.numMaterials <= 0 {
            return IPL_STATUS_FAILURE;
        }

        let scene = parent.get();
        let value = match scene.as_default() {
            Some(default) => {
                // SAFETY: guaranteed by the caller.
                let Some(mesh) = (unsafe { static_mesh(settings) }) else {
                    return IPL_STATUS_FAILURE;
                };
                let id = default.update(|s| s.insert_static_mesh(mesh));
                StaticMeshObject(Some((Arc::clone(scene), id)))
            }
            None => StaticMeshObject(None),
        };
        *out = Handle::into_raw(value, parent.context().cloned());
        IPL_STATUS_SUCCESS
    })
}

retain_release!(
    IPLStaticMesh,
    StaticMeshObject,
    iplStaticMeshRetain,
    iplStaticMeshRelease
);

/// Loads a static mesh into a scene (not yet added to it). The progress callback is never called, as upstream. Returns
/// `IPL_STATUS_FAILURE` for data that is not a serialized static mesh; in a custom scene the mesh is an empty object.
///
/// # Safety
///
/// `scene` is null or a live scene, `serializedObject` null or a live serialized object, and `staticMesh` null or
/// valid.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplStaticMeshLoad(
    scene: IPLScene,
    serializedObject: IPLSerializedObject,
    _progressCallback: IPLProgressCallback,
    _progressCallbackUserData: *mut c_void,
    staticMesh: *mut IPLStaticMesh,
) -> IPLerror {
    guard(IPL_STATUS_FAILURE, || {
        // SAFETY: guaranteed by the caller.
        let Some(parent) = (unsafe { parent(scene) }) else {
            return IPL_STATUS_FAILURE;
        };
        // SAFETY: guaranteed by the caller.
        let Some(out) = (unsafe { staticMesh.as_mut() }) else {
            return IPL_STATUS_FAILURE;
        };
        if serializedObject.is_null() {
            return IPL_STATUS_FAILURE;
        }

        let scene = parent.get();
        let value = match scene.as_default() {
            Some(default) => {
                // SAFETY: guaranteed by the caller.
                let loaded =
                    unsafe { with_serialized_bytes(serializedObject, StaticMesh::deserialize) };
                let Some(Ok(mesh)) = loaded else {
                    return IPL_STATUS_FAILURE;
                };
                let id = default.update(|s| s.insert_static_mesh(mesh));
                StaticMeshObject(Some((Arc::clone(scene), id)))
            }
            None => StaticMeshObject(None),
        };
        *out = Handle::into_raw(value, parent.context().cloned());
        IPL_STATUS_SUCCESS
    })
}

/// The mesh behind a handle and its scene, if it belongs to a default scene.
///
/// # Safety
///
/// `mesh` is null or a live static mesh.
unsafe fn static_mesh_ref<'a>(
    mesh: IPLStaticMesh,
) -> Option<(&'a DefaultScene, &'a Arc<SceneObject>, StaticMeshId)> {
    // SAFETY: guaranteed by the caller.
    let object = unsafe { object::<StaticMeshObject, _>(mesh) }?;
    let (scene, id) = object.0.as_ref()?;
    Some((scene.as_default()?, scene, *id))
}

/// The mesh, if `scene` is the scene it was created in (upstream allows any scene; `phonon.h` requires this one).
///
/// # Safety
///
/// `mesh` is null or a live static mesh, and `scene` null or a live scene.
unsafe fn static_mesh_in<'a>(
    mesh: IPLStaticMesh,
    scene: IPLScene,
) -> Option<(&'a DefaultScene, StaticMeshId)> {
    // SAFETY: guaranteed by the caller.
    let (default, owner, id) = unsafe { static_mesh_ref(mesh) }?;
    // SAFETY: guaranteed by the caller.
    let scene = unsafe { scene_arc(scene) }?;
    Arc::ptr_eq(owner, scene).then_some((default, id))
}

/// Saves a static mesh into a serialized object (nothing for a mesh of a custom scene).
///
/// # Safety
///
/// `staticMesh` is null or a live static mesh, and `serializedObject` null or a live serialized object.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplStaticMeshSave(
    staticMesh: IPLStaticMesh,
    serializedObject: IPLSerializedObject,
) {
    guard((), || {
        // SAFETY: guaranteed by the caller.
        let Some((default, _, id)) = (unsafe { static_mesh_ref(staticMesh) }) else {
            return;
        };
        let bytes = read(&default.scene)
            .static_mesh(id)
            .map(StaticMesh::serialize);
        if let Some(bytes) = bytes {
            // SAFETY: guaranteed by the caller.
            unsafe { store_serialized_bytes(serializedObject, bytes) };
        }
    })
}

/// Stages adding a static mesh to the scene it was created in (nothing for another scene).
///
/// # Safety
///
/// `staticMesh` is null or a live static mesh, and `scene` null or a live scene.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplStaticMeshAdd(staticMesh: IPLStaticMesh, scene: IPLScene) {
    guard((), || {
        // SAFETY: guaranteed by the caller.
        if let Some((default, id)) = unsafe { static_mesh_in(staticMesh, scene) } {
            default.update(|s| s.add_static_mesh(id));
        }
    })
}

/// Stages removing a static mesh from the scene it was created in (nothing for another scene).
///
/// # Safety
///
/// `staticMesh` is null or a live static mesh, and `scene` null or a live scene.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplStaticMeshRemove(staticMesh: IPLStaticMesh, scene: IPLScene) {
    guard((), || {
        // SAFETY: guaranteed by the caller.
        if let Some((default, id)) = unsafe { static_mesh_in(staticMesh, scene) } {
            default.update(|s| s.remove_static_mesh(id));
        }
    })
}

/// Stages replacing material `index` of a committed static mesh, applied at the next commit. Does nothing for an index
/// out of range (upstream writes out of bounds) or a mesh not committed in `scene`.
///
/// # Safety
///
/// `staticMesh` is null or a live static mesh, `scene` null or a live scene, and `newMaterial` null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplStaticMeshSetMaterial(
    staticMesh: IPLStaticMesh,
    scene: IPLScene,
    newMaterial: *mut IPLMaterial,
    index: IPLint32,
) {
    guard((), || {
        // SAFETY: guaranteed by the caller.
        let (Some((default, id)), Some(material)) =
            (unsafe { static_mesh_in(staticMesh, scene) }, unsafe {
                newMaterial.as_ref()
            })
        else {
            return;
        };
        let material = from_ipl_material(material);
        default.update(|s| {
            let num_materials = s.static_mesh(id).map_or(0, StaticMesh::num_materials);
            if let Some(index) = usize::try_from(index).ok().filter(|&i| i < num_materials) {
                s.set_static_mesh_material(id, material, index);
            }
        });
    })
}

// --------------------------------------------------------------------------------------------------------------------
// Instanced meshes
// --------------------------------------------------------------------------------------------------------------------

/// An instanced mesh and the scene that stores it; `None` in a custom scene.
pub(crate) struct InstancedMeshObject(Option<(Arc<SceneObject>, InstancedMeshId)>);

impl Drop for InstancedMeshObject {
    fn drop(&mut self) {
        if let Some((scene, id)) = &self.0 {
            if let Some(scene) = scene.as_default() {
                scene.update(|s| s.forget_instanced_mesh(*id));
            }
        }
    }
}

/// Creates an instanced mesh: `settings.subScene` placed into `scene` with `settings.transform` (not yet added). The
/// instance shows the sub-scene as it is now, with its pending changes committed. Returns `IPL_STATUS_FAILURE` for a
/// null sub-scene, or a custom sub-scene in a default scene (upstream casts it to a default scene); in a custom scene
/// the mesh is an empty object, as upstream.
///
/// # Safety
///
/// `scene` and `settings.subScene` are null or live scenes, and the other pointers are null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplInstancedMeshCreate(
    scene: IPLScene,
    settings: *mut IPLInstancedMeshSettings,
    instancedMesh: *mut IPLInstancedMesh,
) -> IPLerror {
    guard(IPL_STATUS_FAILURE, || {
        // SAFETY: guaranteed by the caller.
        let Some(parent) = (unsafe { parent(scene) }) else {
            return IPL_STATUS_FAILURE;
        };
        // SAFETY: guaranteed by the caller.
        let (Some(settings), Some(out)) = (unsafe { settings.as_ref() }, unsafe {
            instancedMesh.as_mut()
        }) else {
            return IPL_STATUS_FAILURE;
        };
        // SAFETY: guaranteed by the caller.
        let Some(sub_scene) = (unsafe { scene_arc(settings.subScene) }) else {
            return IPL_STATUS_FAILURE;
        };

        let scene = parent.get();
        let value = match scene.as_default() {
            Some(default) => {
                let Some(sub_scene) = sub_scene.as_default() else {
                    return IPL_STATUS_FAILURE;
                };
                let mesh =
                    InstancedMesh::new(sub_scene.snapshot(), from_ipl_matrix(&settings.transform));
                let id = default.update(|s| s.insert_instanced_mesh(mesh));
                InstancedMeshObject(Some((Arc::clone(scene), id)))
            }
            None => InstancedMeshObject(None),
        };
        *out = Handle::into_raw(value, parent.context().cloned());
        IPL_STATUS_SUCCESS
    })
}

retain_release!(
    IPLInstancedMesh,
    InstancedMeshObject,
    iplInstancedMeshRetain,
    iplInstancedMeshRelease
);

/// The mesh, if `scene` is the scene it was created in.
///
/// # Safety
///
/// `mesh` is null or a live instanced mesh, and `scene` null or a live scene.
unsafe fn instanced_mesh_in<'a>(
    mesh: IPLInstancedMesh,
    scene: IPLScene,
) -> Option<(&'a DefaultScene, InstancedMeshId)> {
    // SAFETY: guaranteed by the caller.
    let object = unsafe { object::<InstancedMeshObject, _>(mesh) }?;
    let (owner, id) = object.0.as_ref()?;
    // SAFETY: guaranteed by the caller.
    let scene = unsafe { scene_arc(scene) }?;
    Arc::ptr_eq(owner, scene).then_some((owner.as_default()?, *id))
}

/// Stages adding an instanced mesh to the scene it was created in (nothing for another scene).
///
/// # Safety
///
/// `instancedMesh` is null or a live instanced mesh, and `scene` null or a live scene.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplInstancedMeshAdd(
    instancedMesh: IPLInstancedMesh,
    scene: IPLScene,
) {
    guard((), || {
        // SAFETY: guaranteed by the caller.
        if let Some((default, id)) = unsafe { instanced_mesh_in(instancedMesh, scene) } {
            default.update(|s| s.add_instanced_mesh(id));
        }
    })
}

/// Stages removing an instanced mesh from the scene it was created in (nothing for another scene).
///
/// # Safety
///
/// `instancedMesh` is null or a live instanced mesh, and `scene` null or a live scene.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplInstancedMeshRemove(
    instancedMesh: IPLInstancedMesh,
    scene: IPLScene,
) {
    guard((), || {
        // SAFETY: guaranteed by the caller.
        if let Some((default, id)) = unsafe { instanced_mesh_in(instancedMesh, scene) } {
            default.update(|s| s.remove_instanced_mesh(id));
        }
    })
}

/// Moves an instanced mesh (row-major transform). It takes effect at once; a changed transform bumps the version of
/// the scene the mesh was created in at its next commit. `scene` must not be null but is otherwise not used, as
/// upstream.
///
/// # Safety
///
/// `instancedMesh` is null or a live instanced mesh, and `scene` null or a live scene.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn iplInstancedMeshUpdateTransform(
    instancedMesh: IPLInstancedMesh,
    scene: IPLScene,
    transform: IPLMatrix4x4,
) {
    guard((), || {
        if scene.is_null() {
            return;
        }
        // SAFETY: guaranteed by the caller.
        let Some(object) = (unsafe { object::<InstancedMeshObject, _>(instancedMesh) }) else {
            return;
        };
        if let Some((owner, id)) = &object.0 {
            if let Some(default) = owner.as_default() {
                let transform = from_ipl_matrix(&transform);
                default.update(|s| s.update_instanced_mesh_transform(*id, transform));
            }
        }
    })
}
