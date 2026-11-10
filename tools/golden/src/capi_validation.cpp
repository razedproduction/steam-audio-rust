// Golden data for stage 8d: the C API's validation layer (api_validation_layer.cpp) and device functions
// (api_embree_device, api_opencl_device, api_radeonrays_device, api_tan_device), and the input of upstream's
// sample_binaural (core/src/samples), which tools/golden/run_sample.py runs against both libraries.
//
// Everything is called through the ipl* functions. A validating context logs a warning for each invalid argument or
// result; the log callback records them, each prefixed with its log level as a digit, and every section of calls
// writes what it recorded as bytes (<section>.npy). crates/phonon-capi/tests/capi_validation.rs makes the same calls.
//
// The values are chosen by hand rather than drawn from Lcg: each is invalid in one way, and the calls only use invalid
// values that upstream survives (its validated factories skip the null checks of the unvalidated ones, so a null
// output pointer or null settings would crash). Statuses are not compared: upstream's validated device factories
// report success with an empty device, the port reports failure (DEVIATIONS.md). Upstream logs through a global log
// that the last context created sets; every context here uses the same callback.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <malloc.h>
#include <stdexcept>
#include <string>
#include <vector>

#include <context.h>
#include <phonon.h>

#include "generators.h"
#include "npy.h"

namespace {

using namespace golden;

std::string gLog;

void IPLCALL record(IPLLogLevel level, const char* message)
{
    gLog += static_cast<char>('0' + level);
    gLog += message;
}

void flush(const std::filesystem::path& out, const char* section)
{
    std::vector<uint8_t> bytes(gLog.begin(), gLog.end());
    writeNpy(out / (std::string(section) + ".npy"), bytes);
    gLog.clear();
}

void check(IPLerror status, const char* what)
{
    if (status != IPL_STATUS_SUCCESS)
        throw std::runtime_error(std::string(what) + " failed with status " + std::to_string(status));
}

// Upstream frees blocks with _aligned_free when no free callback is given.
void* IPLCALL alignedAllocate(IPLsize size, IPLsize alignment)
{
    return _aligned_malloc(size, alignment);
}

float floatFromBits(uint32_t bits)
{
    float x;
    std::memcpy(&x, &bits, sizeof(x));
    return x;
}

const float kNaN = floatFromBits(0x7fc00000u);           // nan
const float kNegNaN = floatFromBits(0xffc00000u);        // -nan(ind)
const float kNegNaNPayload = floatFromBits(0xffc00001u); // -nan
const float kInf = std::numeric_limits<float>::infinity();

IPLContext makeContext(IPLuint32 version, IPLContextFlags flags)
{
    IPLContextSettings settings{};
    settings.version = version;
    settings.logCallback = record;
    settings.flags = flags;
    IPLContext context = nullptr;
    check(iplContextCreate(&settings, &context), "iplContextCreate");
    ipl::Context::sSIMDLevel = ipl::SIMDLevel::AVX;
    return context;
}

IPLCoordinateSpace3 identity()
{
    IPLCoordinateSpace3 c{};
    c.right = {1, 0, 0};
    c.up = {0, 1, 0};
    c.ahead = {0, 0, -1};
    c.origin = {0, 0, 0};
    return c;
}

IPLMatrix4x4 identityMatrix()
{
    IPLMatrix4x4 m{};
    for (auto i = 0; i < 4; ++i)
        m.elements[i][i] = 1.0f;
    return m;
}

float IPLCALL nanDistance(IPLfloat32, void*)
{
    return kNaN;
}

float IPLCALL nanBand1(IPLfloat32, IPLint32 band, void*)
{
    return band == 1 ? kNaN : 0.5f;
}

// A buffer of `channels × samples` samples, `value` everywhere.
struct Buffer
{
    std::vector<float> samples;
    std::vector<float*> channels;
    IPLAudioBuffer buffer{};

    Buffer(int numChannels, int numSamples, float value = 0.25f)
        : samples(static_cast<size_t>(numChannels) * numSamples, value)
        , channels(numChannels)
    {
        for (auto i = 0; i < numChannels; ++i)
            channels[i] = &samples[static_cast<size_t>(i) * numSamples];
        buffer = IPLAudioBuffer{numChannels, numSamples, channels.data()};
    }

    IPLAudioBuffer* operator&() { return &buffer; }
};

// --------------------------------------------------------------------------------------------------------------------
// Context, model functions
// --------------------------------------------------------------------------------------------------------------------

void contextCases(IPLContext v)
{
    // Relative direction: NaNs of three kinds, infinities.
    iplCalculateRelativeDirection(v, IPLVector3{kNaN, kNegNaN, kNegNaNPayload}, IPLVector3{kInf, -kInf, 0.0f},
                                  IPLVector3{0, 0, -1}, IPLVector3{0, 1, 0});

    // Directivity: formatting of finite floats (the listener is straight ahead, so the result stays finite).
    const float weights[] = {1.5f, -1e-9f, 3e38f, 1.0000001f, -0.0f, 123456.789f, 0.5f, -7.0f};
    for (auto w : weights)
    {
        IPLDirectivity model{};
        model.dipoleWeight = w;
        model.dipolePower = 1.0f;
        iplDirectivityCalculate(v, identity(), IPLVector3{0, 0, -1}, &model);
    }
    const float powers[] = {-123.456789f, -2.5e-7f, 2.0f};
    for (auto p : powers)
    {
        IPLDirectivity model{};
        model.dipoleWeight = 0.5f;
        model.dipolePower = p;
        iplDirectivityCalculate(v, identity(), IPLVector3{0, 0, -1}, &model);
    }
    iplDirectivityCalculate(v, identity(), IPLVector3{kNaN, 0, 0}, nullptr);

    // Distance attenuation.
    iplDistanceAttenuationCalculate(v, IPLVector3{kInf, 0, 0}, IPLVector3{0, 0, 0}, nullptr);
    {
        IPLDistanceAttenuationModel model{};
        model.type = static_cast<IPLDistanceAttenuationModelType>(7);
        iplDistanceAttenuationCalculate(v, IPLVector3{1, 0, 0}, IPLVector3{0, 0, 0}, &model);
    }
    {
        IPLDistanceAttenuationModel model{};
        model.type = IPL_DISTANCEATTENUATIONTYPE_INVERSEDISTANCE;
        model.minDistance = -1.0f;
        iplDistanceAttenuationCalculate(v, IPLVector3{3, 0, 0}, IPLVector3{0, 0, 0}, &model);
    }
    {
        IPLDistanceAttenuationModel model{};
        model.type = IPL_DISTANCEATTENUATIONTYPE_CALLBACK;
        model.dirty = static_cast<IPLbool>(5);
        iplDistanceAttenuationCalculate(v, IPLVector3{2, 0, 0}, IPLVector3{0, 0, 0}, &model);
    }
    {
        IPLDistanceAttenuationModel model{};
        model.type = IPL_DISTANCEATTENUATIONTYPE_CALLBACK;
        model.callback = nanDistance;
        iplDistanceAttenuationCalculate(v, IPLVector3{2, 0, 0}, IPLVector3{0, 0, 0}, &model);
    }

    // Air absorption.
    iplAirAbsorptionCalculate(v, IPLVector3{1, 0, 0}, IPLVector3{0, 0, 0}, nullptr, nullptr);
    {
        IPLAirAbsorptionModel model{};
        model.type = IPL_AIRABSORPTIONTYPE_EXPONENTIAL;
        model.coefficients[0] = 1.0f;
        model.coefficients[1] = kInf;
        model.coefficients[2] = 2.0f;
        float out[3] = {};
        iplAirAbsorptionCalculate(v, IPLVector3{1, 0, 0}, IPLVector3{0, 0, 0}, &model, out);
    }
    {
        IPLAirAbsorptionModel model{};
        model.type = static_cast<IPLAirAbsorptionModelType>(9);
        model.dirty = static_cast<IPLbool>(-1);
        float out[3] = {};
        iplAirAbsorptionCalculate(v, IPLVector3{1, 0, 0}, IPLVector3{0, 0, 0}, &model, out);
    }
    {
        IPLAirAbsorptionModel model{};
        model.type = IPL_AIRABSORPTIONTYPE_CALLBACK;
        model.callback = nanBand1;
        float out[3] = {};
        iplAirAbsorptionCalculate(v, IPLVector3{1, 0, 0}, IPLVector3{0, 0, 0}, &model, out);
    }
}

// --------------------------------------------------------------------------------------------------------------------
// Audio buffers
// --------------------------------------------------------------------------------------------------------------------

void audioBufferCases(IPLContext v)
{
    IPLAudioBuffer allocated{};
    iplAudioBufferAllocate(v, 0, -3, &allocated);
    iplAudioBufferAllocate(v, 2, 5, nullptr);
    check(iplAudioBufferAllocate(v, 2, 37, &allocated), "iplAudioBufferAllocate");
    iplAudioBufferFree(v, nullptr);
    iplAudioBufferFree(v, &allocated);

    Buffer a(2, 37);
    a.channels[1][36] = kNaN;
    std::vector<float> interleaved(2 * 37, 0.0f);
    iplAudioBufferInterleave(v, &a, interleaved.data());
    iplAudioBufferInterleave(v, nullptr, nullptr);

    Buffer b(2, 37);
    interleaved.assign(2 * 37, 0.5f);
    interleaved[10] = kInf;
    iplAudioBufferDeinterleave(v, interleaved.data(), &b);
    iplAudioBufferDeinterleave(v, nullptr, &b);

    // Mismatched shapes (backed by enough memory for upstream, which processes them).
    Buffer c(2, 37);
    Buffer d(2, 37);
    IPLAudioBuffer oneChannel{1, 37, c.channels.data()};
    iplAudioBufferMix(v, &oneChannel, &d);
    IPLAudioBuffer shorter{2, 30, c.channels.data()};
    iplAudioBufferMix(v, &shorter, &d);
    iplAudioBufferMix(v, &c, &d);

    Buffer mono(1, 37);
    iplAudioBufferDownmix(v, &a, &mono);
    iplAudioBufferDownmix(v, &c, &d);
    iplAudioBufferDownmix(v, &shorter, &mono);

    Buffer e(4, 37);
    Buffer f(4, 37);
    Buffer g(9, 37);
    iplAudioBufferConvertAmbisonics(v, static_cast<IPLAmbisonicsType>(5), static_cast<IPLAmbisonicsType>(-1), &e, &f);
    iplAudioBufferConvertAmbisonics(v, IPL_AMBISONICSTYPE_N3D, IPL_AMBISONICSTYPE_SN3D, &e, &g);
    iplAudioBufferConvertAmbisonics(v, IPL_AMBISONICSTYPE_N3D, IPL_AMBISONICSTYPE_FUMA, &e, &f);
}

// --------------------------------------------------------------------------------------------------------------------
// Devices
// --------------------------------------------------------------------------------------------------------------------

void deviceCases(IPLContext v)
{
    IPLEmbreeDevice embree = nullptr;
    iplEmbreeDeviceCreate(v, nullptr, &embree);
    iplEmbreeDeviceRelease(&embree);

    IPLOpenCLDeviceSettings settings{};
    settings.type = static_cast<IPLOpenCLDeviceType>(4);
    settings.numCUsToReserve = -1;
    settings.fractionCUsForIRUpdate = 1.5f;
    settings.requiresTAN = static_cast<IPLbool>(3);
    IPLOpenCLDeviceList list = nullptr;
    iplOpenCLDeviceListCreate(v, &settings, &list);

    // Upstream's validated factory made an empty list; the port's test passes a dangling non-null handle.
    IPLOpenCLDevice device = nullptr;
    iplOpenCLDeviceCreate(v, list, 2, &device);
    iplOpenCLDeviceRelease(&device);
    iplOpenCLDeviceListRelease(&list);

    iplOpenCLDeviceCreateFromExisting(v, nullptr, nullptr, &device);
    iplOpenCLDeviceRelease(&device);
}

std::vector<int32_t> deviceStatuses(IPLContext plain)
{
    std::vector<int32_t> statuses;
    IPLEmbreeDevice embree = nullptr;
    IPLEmbreeDeviceSettings embreeSettings{};
    statuses.push_back(iplEmbreeDeviceCreate(plain, &embreeSettings, &embree));
    statuses.push_back(embree == nullptr);

    IPLOpenCLDeviceSettings settings{};
    IPLOpenCLDeviceList list = nullptr;
    statuses.push_back(iplOpenCLDeviceListCreate(plain, &settings, &list));
    statuses.push_back(list == nullptr);
    statuses.push_back(iplOpenCLDeviceListGetNumDevices(nullptr));

    IPLOpenCLDevice device = nullptr;
    statuses.push_back(iplOpenCLDeviceCreate(plain, nullptr, 0, &device));
    statuses.push_back(iplOpenCLDeviceCreateFromExisting(plain, nullptr, nullptr, &device));
    statuses.push_back(device == nullptr);

    IPLRadeonRaysDevice rr = nullptr;
    IPLRadeonRaysDeviceSettings rrSettings{};
    statuses.push_back(iplRadeonRaysDeviceCreate(nullptr, &rrSettings, &rr));
    IPLTrueAudioNextDevice tan = nullptr;
    IPLTrueAudioNextDeviceSettings tanSettings{};
    statuses.push_back(iplTrueAudioNextDeviceCreate(nullptr, &tanSettings, &tan));
    statuses.push_back(rr == nullptr && tan == nullptr);
    return statuses;
}

// --------------------------------------------------------------------------------------------------------------------
// HRTFs and effects
// --------------------------------------------------------------------------------------------------------------------

void effectCases(IPLContext v)
{
    IPLAudioSettings audio{48000, 256};
    const int f = audio.frameSize;

    IPLHRTFSettings hrtfSettings{};
    hrtfSettings.type = IPL_HRTFTYPE_DEFAULT;
    hrtfSettings.volume = -1.0f;
    hrtfSettings.normType = static_cast<IPLHRTFNormType>(5);
    IPLHRTF badHrtf = nullptr;
    iplHRTFCreate(v, &audio, &hrtfSettings, &badHrtf);
    iplHRTFRelease(&badHrtf);

    hrtfSettings.volume = 1.0f;
    hrtfSettings.normType = IPL_HRTFNORMTYPE_NONE;
    IPLHRTF hrtf = nullptr;
    check(iplHRTFCreate(v, &audio, &hrtfSettings, &hrtf), "iplHRTFCreate");

    Buffer mono(1, f);
    Buffer stereo(2, f);
    Buffer ambi(4, f);
    Buffer surround(6, f);

    // Panning.
    {
        IPLPanningEffectSettings settings{};
        settings.speakerLayout.type = IPL_SPEAKERLAYOUTTYPE_STEREO;
        IPLPanningEffect effect = nullptr;
        check(iplPanningEffectCreate(v, &audio, &settings, &effect), "iplPanningEffectCreate");
        IPLPanningEffectParams params{IPLVector3{1, 0, 0}};
        iplPanningEffectApply(effect, &params, &mono, &stereo);
        iplPanningEffectRelease(&effect);
    }

    // Binaural: a spatial blend above 1 (an unknown interpolation crashes upstream).
    {
        IPLBinauralEffectSettings settings{hrtf};
        IPLBinauralEffect effect = nullptr;
        check(iplBinauralEffectCreate(v, &audio, &settings, &effect), "iplBinauralEffectCreate");
        IPLBinauralEffectParams params{};
        params.direction = IPLVector3{1, 0, 0};
        params.interpolation = IPL_HRTFINTERPOLATION_NEAREST;
        params.hrtf = hrtf;
        params.spatialBlend = 1.5f;
        iplBinauralEffectApply(effect, &params, &mono, &stereo);
        iplBinauralEffectRelease(&effect);
    }

    // Virtual surround.
    {
        IPLVirtualSurroundEffectSettings settings{};
        settings.speakerLayout.type = IPL_SPEAKERLAYOUTTYPE_SURROUND_5_1;
        settings.hrtf = hrtf;
        IPLVirtualSurroundEffect effect = nullptr;
        check(iplVirtualSurroundEffectCreate(v, &audio, &settings, &effect), "iplVirtualSurroundEffectCreate");
        IPLVirtualSurroundEffectParams params{hrtf};
        iplVirtualSurroundEffectApply(effect, &params, &surround, &stereo);
        iplVirtualSurroundEffectRelease(&effect);
    }

    // Ambisonics encode, panning, binaural, rotation with a negative order; decode without an HRTF.
    {
        IPLAmbisonicsEncodeEffectSettings settings{1};
        IPLAmbisonicsEncodeEffect effect = nullptr;
        check(iplAmbisonicsEncodeEffectCreate(v, &audio, &settings, &effect), "iplAmbisonicsEncodeEffectCreate");
        IPLAmbisonicsEncodeEffectParams params{IPLVector3{0, 1, 0}, -1};
        iplAmbisonicsEncodeEffectApply(effect, &params, &mono, &ambi);
        iplAmbisonicsEncodeEffectRelease(&effect);
    }
    {
        IPLAmbisonicsPanningEffectSettings settings{};
        settings.speakerLayout.type = IPL_SPEAKERLAYOUTTYPE_STEREO;
        settings.maxOrder = 1;
        IPLAmbisonicsPanningEffect effect = nullptr;
        check(iplAmbisonicsPanningEffectCreate(v, &audio, &settings, &effect), "iplAmbisonicsPanningEffectCreate");
        IPLAmbisonicsPanningEffectParams params{-2};
        iplAmbisonicsPanningEffectApply(effect, &params, &ambi, &stereo);
        iplAmbisonicsPanningEffectRelease(&effect);
    }
    {
        IPLAmbisonicsBinauralEffectSettings settings{hrtf, 1};
        IPLAmbisonicsBinauralEffect effect = nullptr;
        check(iplAmbisonicsBinauralEffectCreate(v, &audio, &settings, &effect),
              "iplAmbisonicsBinauralEffectCreate");
        IPLAmbisonicsBinauralEffectParams params{hrtf, -1};
        iplAmbisonicsBinauralEffectApply(effect, &params, &ambi, &stereo);
        iplAmbisonicsBinauralEffectRelease(&effect);
    }
    {
        IPLAmbisonicsRotationEffectSettings settings{1};
        IPLAmbisonicsRotationEffect effect = nullptr;
        check(iplAmbisonicsRotationEffectCreate(v, &audio, &settings, &effect),
              "iplAmbisonicsRotationEffectCreate");
        IPLAmbisonicsRotationEffectParams params{identity(), -1};
        Buffer rotated(4, f);
        iplAmbisonicsRotationEffectApply(effect, &params, &ambi, &rotated);
        iplAmbisonicsRotationEffectRelease(&effect);
    }
    {
        IPLAmbisonicsDecodeEffectSettings settings{};
        settings.speakerLayout.type = IPL_SPEAKERLAYOUTTYPE_STEREO;
        settings.hrtf = hrtf;
        settings.maxOrder = 1;
        IPLAmbisonicsDecodeEffect effect = nullptr;
        check(iplAmbisonicsDecodeEffectCreate(v, &audio, &settings, &effect), "iplAmbisonicsDecodeEffectCreate");
        IPLAmbisonicsDecodeEffectParams params{};
        params.order = 1;
        params.hrtf = nullptr;
        params.orientation = identity();
        params.binaural = IPL_FALSE;
        iplAmbisonicsDecodeEffectApply(effect, &params, &ambi, &stereo);
        iplAmbisonicsDecodeEffectRelease(&effect);
    }

    // Direct effect: every parameter out of range (but finite).
    {
        IPLDirectEffectSettings settings{1};
        IPLDirectEffect effect = nullptr;
        check(iplDirectEffectCreate(v, &audio, &settings, &effect), "iplDirectEffectCreate");
        IPLDirectEffectParams params{};
        params.flags = static_cast<IPLDirectEffectFlags>(64 | IPL_DIRECTEFFECTFLAGS_APPLYDISTANCEATTENUATION);
        params.transmissionType = static_cast<IPLTransmissionType>(5);
        params.distanceAttenuation = 1.5f;
        params.airAbsorption[0] = -0.1f;
        params.airAbsorption[1] = 0.5f;
        params.airAbsorption[2] = 2.0f;
        params.directivity = -1.0f;
        params.occlusion = 2.0f;
        params.transmission[0] = 0.5f;
        params.transmission[1] = 1.25f;
        params.transmission[2] = -3.0f;
        Buffer out(1, f);
        iplDirectEffectApply(effect, &params, &mono, &out);
        iplDirectEffectRelease(&effect);
    }

    // Reflection effect: a parametric effect created with empty sizes, applied with hybrid and TrueAudio Next
    // parameters (it applies its own type).
    {
        IPLReflectionEffectSettings settings{IPL_REFLECTIONEFFECTTYPE_PARAMETRIC, 0, 0};
        IPLReflectionEffect effect = nullptr;
        check(iplReflectionEffectCreate(v, &audio, &settings, &effect), "iplReflectionEffectCreate");
        IPLReflectionEffectParams params{};
        params.type = IPL_REFLECTIONEFFECTTYPE_HYBRID;
        params.ir = nullptr;
        params.numChannels = 0;
        params.irSize = 0;
        params.reverbTimes[0] = params.reverbTimes[1] = params.reverbTimes[2] = 1.0f;
        params.eq[0] = 0.0f;
        params.eq[1] = 0.5f;
        params.eq[2] = 1.5f;
        params.delay = 0;
        Buffer out(1, f);
        iplReflectionEffectApply(effect, &params, &mono, &out, nullptr);
        params.type = IPL_REFLECTIONEFFECTTYPE_TAN;
        params.tanDevice = nullptr;
        params.tanSlot = -1;
        iplReflectionEffectApply(effect, &params, &mono, &out, nullptr);
        iplReflectionEffectRelease(&effect);
    }

    // Reflection mixer of a type that needs no channels.
    {
        IPLReflectionEffectSettings settings{IPL_REFLECTIONEFFECTTYPE_PARAMETRIC, 0, -1};
        IPLReflectionMixer mixer = nullptr;
        check(iplReflectionMixerCreate(v, &audio, &settings, &mixer), "iplReflectionMixerCreate");
        IPLReflectionEffectParams params{};
        params.type = IPL_REFLECTIONEFFECTTYPE_CONVOLUTION;
        params.numChannels = 0;
        params.irSize = 0;
        iplReflectionMixerApply(mixer, &params, &ambi);
        iplReflectionMixerRelease(&mixer);
    }

    // Path effect: EQ coefficients out of range.
    {
        IPLPathEffectSettings settings{};
        settings.maxOrder = 1;
        settings.spatialize = IPL_FALSE;
        settings.hrtf = hrtf;
        IPLPathEffect effect = nullptr;
        check(iplPathEffectCreate(v, &audio, &settings, &effect), "iplPathEffectCreate");
        float sh[4] = {0.5f, 0.1f, 0.2f, 0.3f};
        IPLPathEffectParams params{};
        params.eqCoeffs[0] = 0.0f;
        params.eqCoeffs[1] = 0.5f;
        params.eqCoeffs[2] = 1.5f;
        params.shCoeffs = sh;
        params.order = 1;
        params.binaural = IPL_FALSE;
        params.hrtf = hrtf;
        params.listener = identity();
        Buffer out(4, f);
        iplPathEffectApply(effect, &params, &mono, &out);
        iplPathEffectRelease(&effect);
    }

    iplHRTFRelease(&hrtf);
}

// --------------------------------------------------------------------------------------------------------------------
// Scenes and meshes
// --------------------------------------------------------------------------------------------------------------------

IPLStaticMesh makeQuad(IPLScene scene, const IPLMaterial& material)
{
    IPLVector3 vertices[4] = {{-1, 0, -1}, {1, 0, -1}, {1, 0, 1}, {-1, 0, 1}};
    IPLTriangle triangles[2] = {{{0, 2, 1}}, {{0, 3, 2}}};
    IPLint32 materialIndices[2] = {0, 0};
    IPLMaterial materials[1] = {material};
    IPLStaticMeshSettings settings{};
    settings.numVertices = 4;
    settings.numTriangles = 2;
    settings.numMaterials = 1;
    settings.vertices = vertices;
    settings.triangles = triangles;
    settings.materialIndices = materialIndices;
    settings.materials = materials;
    IPLStaticMesh mesh = nullptr;
    check(iplStaticMeshCreate(scene, &settings, &mesh), "iplStaticMeshCreate");
    return mesh;
}

void sceneCases(IPLContext v)
{
    {
        IPLSceneSettings settings{};
        settings.type = IPL_SCENETYPE_CUSTOM;
        IPLScene custom = nullptr;
        iplSceneCreate(v, &settings, &custom);
        iplSceneRelease(&custom);
        settings.type = static_cast<IPLSceneType>(7);
        IPLScene unknown = nullptr;
        iplSceneCreate(v, &settings, &unknown);
        iplSceneRelease(&unknown);
    }

    IPLSceneSettings settings{};
    settings.type = IPL_SCENETYPE_DEFAULT;
    IPLScene scene = nullptr;
    check(iplSceneCreate(v, &settings, &scene), "iplSceneCreate");

    IPLMaterial bad{};
    bad.absorption[0] = 0.1f;
    bad.absorption[1] = 1.5f;
    bad.absorption[2] = 0.2f;
    bad.scattering = -0.5f;
    bad.transmission[0] = 0.1f;
    bad.transmission[1] = 0.2f;
    bad.transmission[2] = 2.0f;
    auto mesh = makeQuad(scene, bad);

    iplStaticMeshAdd(mesh, nullptr);
    iplStaticMeshRemove(mesh, nullptr);
    iplStaticMeshSave(mesh, nullptr);
    iplStaticMeshAdd(mesh, scene);
    iplSceneCommit(scene);
    iplSceneSave(scene, nullptr);
    iplSceneSaveOBJ(scene, nullptr);

    IPLScene sub = nullptr;
    check(iplSceneCreate(v, &settings, &sub), "iplSceneCreate");
    IPLMaterial good{{0.1f, 0.2f, 0.3f}, 0.05f, {0.1f, 0.05f, 0.03f}};
    auto subMesh = makeQuad(sub, good);
    iplStaticMeshAdd(subMesh, sub);
    iplSceneCommit(sub);

    IPLInstancedMeshSettings instanceSettings{};
    instanceSettings.subScene = sub;
    instanceSettings.transform = identityMatrix();
    instanceSettings.transform.elements[1][2] = kNaN;
    IPLInstancedMesh instance = nullptr;
    check(iplInstancedMeshCreate(scene, &instanceSettings, &instance), "iplInstancedMeshCreate");
    auto transform = identityMatrix();
    transform.elements[3][0] = -kInf;
    iplInstancedMeshUpdateTransform(instance, nullptr, transform);
    iplInstancedMeshUpdateTransform(instance, scene, identityMatrix());
    iplInstancedMeshAdd(instance, nullptr);
    iplInstancedMeshRemove(instance, nullptr);

    // Saving and loading, with valid arguments.
    IPLSerializedObjectSettings serializedSettings{};
    IPLSerializedObject serialized = nullptr;
    check(iplSerializedObjectCreate(v, &serializedSettings, &serialized), "iplSerializedObjectCreate");
    iplSceneSave(scene, serialized);
    IPLScene loaded = nullptr;
    check(iplSceneLoad(v, &settings, serialized, nullptr, nullptr, &loaded), "iplSceneLoad");
    iplSceneRelease(&loaded);
    iplSerializedObjectRelease(&serialized);

    serializedSettings.data = nullptr;
    serializedSettings.size = 10;
    iplSerializedObjectCreate(v, &serializedSettings, &serialized);
    iplSerializedObjectRelease(&serialized);

    iplInstancedMeshRelease(&instance);
    iplStaticMeshRelease(&subMesh);
    iplStaticMeshRelease(&mesh);
    iplSceneRelease(&sub);
    iplSceneRelease(&scene);
}

// --------------------------------------------------------------------------------------------------------------------
// Probes and baking
// --------------------------------------------------------------------------------------------------------------------

void probeCases(IPLContext v)
{
    IPLSceneSettings sceneSettings{};
    sceneSettings.type = IPL_SCENETYPE_DEFAULT;
    IPLScene scene = nullptr;
    check(iplSceneCreate(v, &sceneSettings, &scene), "iplSceneCreate");
    IPLMaterial good{{0.1f, 0.2f, 0.3f}, 0.05f, {0.1f, 0.05f, 0.03f}};
    auto floor = makeQuad(scene, good);
    iplStaticMeshAdd(floor, scene);
    iplSceneCommit(scene);

    IPLProbeArray array = nullptr;
    check(iplProbeArrayCreate(v, &array), "iplProbeArrayCreate");
    IPLProbeGenerationParams params{};
    params.type = static_cast<IPLProbeGenerationType>(5);
    params.spacing = -1.0f;
    params.height = 0.0f;
    params.transform = identityMatrix();
    params.transform.elements[0][3] = kNaN;
    iplProbeArrayGenerateProbes(array, nullptr, &params);

    params.type = IPL_PROBEGENERATIONTYPE_UNIFORMFLOOR;
    params.spacing = 0.5f;
    params.height = 0.0f;
    params.transform = identityMatrix();
    params.transform.elements[0][0] = 2.0f;
    params.transform.elements[1][1] = 2.0f;
    params.transform.elements[2][2] = 2.0f;
    iplProbeArrayGenerateProbes(array, scene, &params);
    iplProbeArrayGetProbe(array, -1);
    iplProbeArrayGetProbe(array, 1000);
    iplProbeArrayGetProbe(array, 0);

    IPLProbeBatch batch = nullptr;
    check(iplProbeBatchCreate(v, &batch), "iplProbeBatchCreate");
    iplProbeBatchAddProbeArray(batch, nullptr);
    iplProbeBatchRemoveProbe(batch, 5);
    iplProbeBatchAddProbeArray(batch, array);
    iplProbeBatchCommit(batch);
    iplProbeBatchRemoveProbe(batch, -2);

    IPLBakedDataIdentifier unknown{};
    unknown.type = static_cast<IPLBakedDataType>(5);
    unknown.variation = static_cast<IPLBakedDataVariation>(9);
    unknown.endpointInfluence.center = IPLVector3{0, kInf, 0};
    unknown.endpointInfluence.radius = -2.0f;
    iplProbeBatchRemoveData(batch, &unknown);
    iplProbeBatchGetDataSize(batch, &unknown);

    IPLBakedDataIdentifier reverb{};
    reverb.type = IPL_BAKEDDATATYPE_REFLECTIONS;
    reverb.variation = IPL_BAKEDDATAVARIATION_REVERB;
    iplProbeBatchGetEnergyField(batch, &reverb, 1000, nullptr);
    iplProbeBatchGetReverb(batch, &reverb, -1, nullptr);
    iplProbeBatchSave(batch, nullptr);

    IPLSerializedObjectSettings serializedSettings{};
    IPLSerializedObject serialized = nullptr;
    check(iplSerializedObjectCreate(v, &serializedSettings, &serialized), "iplSerializedObjectCreate");
    iplProbeBatchSave(batch, serialized);
    IPLProbeBatch loaded = nullptr;
    check(iplProbeBatchLoad(v, serialized, &loaded), "iplProbeBatchLoad");
    iplProbeBatchRelease(&loaded);
    iplSerializedObjectRelease(&serialized);

    // A batch with an invalid probe, never committed.
    IPLProbeBatch other = nullptr;
    check(iplProbeBatchCreate(v, &other), "iplProbeBatchCreate");
    iplProbeBatchAddProbe(other, IPLSphere{IPLVector3{kNaN, 0, 0}, -1.0f});
    iplProbeBatchRelease(&other);

    // Bakes with every parameter invalid and no scene (upstream then returns before baking).
    {
        IPLReflectionsBakeParams bake{};
        bake.sceneType = static_cast<IPLSceneType>(9);
        bake.identifier.type = static_cast<IPLBakedDataType>(7);
        bake.identifier.variation = static_cast<IPLBakedDataVariation>(5);
        bake.identifier.endpointInfluence.center = IPLVector3{kInf, 0, 0};
        bake.bakeFlags = static_cast<IPLReflectionsBakeFlags>(4);
        bake.numRays = 0;
        bake.numDiffuseSamples = -1;
        bake.numBounces = 0;
        bake.simulatedDuration = 0.0f;
        bake.savedDuration = 1.0f;
        bake.order = 0;
        bake.numThreads = 0;
        bake.irradianceMinDistance = 0.0f;
        iplReflectionsBakerBake(v, &bake, nullptr, nullptr);
        bake.sceneType = IPL_SCENETYPE_CUSTOM;
        bake.rayBatchSize = 0;
        iplReflectionsBakerBake(v, &bake, nullptr, nullptr);
        bake.sceneType = IPL_SCENETYPE_RADEONRAYS;
        bake.bakeBatchSize = 0;
        iplReflectionsBakerBake(v, &bake, nullptr, nullptr);
    }
    {
        IPLPathBakeParams bake{};
        bake.identifier.type = IPL_BAKEDDATATYPE_PATHING;
        bake.identifier.variation = IPL_BAKEDDATAVARIATION_DYNAMIC;
        bake.numSamples = 0;
        bake.radius = -1.0f;
        bake.threshold = 0.0f;
        bake.visRange = 0.0f;
        bake.pathRange = -5.0f;
        bake.numThreads = -1;
        iplPathBakerBake(v, &bake, nullptr, nullptr);
    }

    iplProbeBatchRelease(&batch);
    iplProbeArrayRelease(&array);
    iplStaticMeshRelease(&floor);
    iplSceneRelease(&scene);
}

// --------------------------------------------------------------------------------------------------------------------
// Simulator and sources
// --------------------------------------------------------------------------------------------------------------------

IPLSimulationSettings directSettings()
{
    IPLSimulationSettings settings{};
    settings.flags = IPL_SIMULATIONFLAGS_DIRECT;
    settings.sceneType = IPL_SCENETYPE_DEFAULT;
    settings.reflectionType = IPL_REFLECTIONEFFECTTYPE_CONVOLUTION;
    settings.maxNumOcclusionSamples = 4;
    settings.maxNumRays = 1;
    settings.numDiffuseSamples = 1;
    settings.maxDuration = 1.0f;
    settings.maxOrder = 0;
    settings.maxNumSources = 2;
    settings.numThreads = 1;
    settings.rayBatchSize = 1;
    settings.numVisSamples = 1;
    settings.samplingRate = 48000;
    settings.frameSize = 256;
    return settings;
}

void simulationCases(IPLContext v)
{
    {
        auto settings = directSettings();
        settings.flags = static_cast<IPLSimulationFlags>(IPL_SIMULATIONFLAGS_DIRECT | 16);
        settings.sceneType = IPL_SCENETYPE_CUSTOM;
        settings.reflectionType = IPL_REFLECTIONEFFECTTYPE_TAN;
        settings.maxNumRays = 0;
        settings.numDiffuseSamples = 0;
        settings.maxDuration = 0.0f;
        settings.rayBatchSize = 0;
        settings.numVisSamples = 0;
        IPLSimulator bad = nullptr;
        iplSimulatorCreate(v, &settings, &bad);
        iplSimulatorRelease(&bad);
        settings.flags = IPL_SIMULATIONFLAGS_DIRECT;
        settings.sceneType = IPL_SCENETYPE_RADEONRAYS;
        settings.reflectionType = static_cast<IPLReflectionEffectType>(-1);
        iplSimulatorCreate(v, &settings, &bad);
        iplSimulatorRelease(&bad);
    }

    auto settings = directSettings();
    IPLSimulator simulator = nullptr;
    check(iplSimulatorCreate(v, &settings, &simulator), "iplSimulatorCreate");
    iplSimulatorSetScene(simulator, nullptr);
    iplSimulatorAddProbeBatch(simulator, nullptr);
    iplSimulatorRemoveProbeBatch(simulator, nullptr);

    IPLSimulationSharedInputs shared{};
    shared.listener = identity();
    shared.listener.up.y = kNaN;
    shared.numRays = 0;
    shared.numBounces = 0;
    shared.duration = 0.0f;
    shared.order = -1;
    shared.irradianceMinDistance = 0.0f;
    iplSimulatorSetSharedInputs(simulator,
                                static_cast<IPLSimulationFlags>(IPL_SIMULATIONFLAGS_REFLECTIONS | 32), &shared);
    iplSimulatorSetSharedInputs(simulator, IPL_SIMULATIONFLAGS_DIRECT, nullptr);
    shared.listener = identity();
    iplSimulatorSetSharedInputs(simulator, IPL_SIMULATIONFLAGS_DIRECT, &shared);

    IPLSourceSettings sourceSettings{static_cast<IPLSimulationFlags>(9)};
    IPLSource source = nullptr;
    check(iplSourceCreate(simulator, &sourceSettings, &source), "iplSourceCreate");
    iplSourceAdd(source, nullptr);
    iplSourceRemove(source, nullptr);

    // Every input invalid, with flags that select no simulation (so upstream stores nothing).
    IPLDeviationModel deviation{};
    deviation.type = static_cast<IPLDeviationModelType>(2);
    IPLSimulationInputs inputs{};
    inputs.flags = static_cast<IPLSimulationFlags>(IPL_SIMULATIONFLAGS_DIRECT | IPL_SIMULATIONFLAGS_REFLECTIONS |
                                                   IPL_SIMULATIONFLAGS_PATHING);
    inputs.directFlags = static_cast<IPLDirectSimulationFlags>(32);
    inputs.source = identity();
    inputs.source.origin.z = kNaN;
    inputs.distanceAttenuationModel.type = IPL_DISTANCEATTENUATIONTYPE_CALLBACK;
    inputs.distanceAttenuationModel.dirty = static_cast<IPLbool>(2);
    inputs.airAbsorptionModel.type = static_cast<IPLAirAbsorptionModelType>(4);
    inputs.directivity.dipoleWeight = 2.0f;
    inputs.directivity.dipolePower = -1.0f;
    inputs.occlusionType = static_cast<IPLOcclusionType>(3);
    inputs.occlusionRadius = 0.0f;
    inputs.numOcclusionSamples = 0;
    inputs.reverbScale[0] = 0.0f;
    inputs.reverbScale[1] = 1.0f;
    inputs.reverbScale[2] = -1.0f;
    inputs.hybridReverbTransitionTime = 0.0f;
    inputs.hybridReverbOverlapPercent = 0.0f;
    inputs.baked = IPL_TRUE;
    inputs.bakedDataIdentifier.type = static_cast<IPLBakedDataType>(3);
    inputs.bakedDataIdentifier.variation = IPL_BAKEDDATAVARIATION_DYNAMIC;
    inputs.bakedDataIdentifier.endpointInfluence.radius = -1.0f;
    inputs.pathingProbes = nullptr;
    inputs.visRadius = -1.0f;
    inputs.visThreshold = -0.5f;
    inputs.visRange = -2.0f;
    inputs.pathingOrder = -1;
    inputs.enableValidation = static_cast<IPLbool>(2);
    inputs.findAlternatePaths = static_cast<IPLbool>(3);
    inputs.numTransmissionRays = 0;
    inputs.deviationModel = &deviation;
    iplSourceSetInputs(source, static_cast<IPLSimulationFlags>(64), &inputs);
    iplSourceSetInputs(source, IPL_SIMULATIONFLAGS_DIRECT, nullptr);

    IPLSimulationOutputs outputs{};
    iplSourceGetOutputs(source, IPL_SIMULATIONFLAGS_DIRECT, &outputs);
    iplSourceGetOutputs(source, static_cast<IPLSimulationFlags>(IPL_SIMULATIONFLAGS_DIRECT | 8), nullptr);
    outputs.direct.flags = static_cast<IPLDirectEffectFlags>(128);
    outputs.direct.transmissionType = static_cast<IPLTransmissionType>(-1);
    iplSourceGetOutputs(source, IPL_SIMULATIONFLAGS_DIRECT, &outputs);

    iplSourceRelease(&source);
    iplSimulatorRelease(&simulator);
}

// --------------------------------------------------------------------------------------------------------------------
// Energy fields, impulse responses, reconstructors
// --------------------------------------------------------------------------------------------------------------------

void reconstructorCases(IPLContext v)
{
    {
        IPLEnergyFieldSettings settings{0.0f, 0};
        IPLEnergyField field = nullptr;
        iplEnergyFieldCreate(v, &settings, &field);
        iplEnergyFieldRelease(&field);
        IPLImpulseResponseSettings irSettings{0.0f, 0, -1};
        IPLImpulseResponse ir = nullptr;
        iplImpulseResponseCreate(v, &irSettings, &ir);
        iplImpulseResponseRelease(&ir);
    }

    IPLReconstructorSettings settings{1.0f, 1, 24000};
    IPLReconstructor reconstructor = nullptr;
    check(iplReconstructorCreate(v, &settings, &reconstructor), "iplReconstructorCreate");

    IPLReconstructorSharedInputs shared{0.0f, -1};
    iplReconstructorReconstruct(reconstructor, -1, nullptr, &shared, nullptr);
    IPLReconstructorInputs inputs[2] = {};
    IPLReconstructorOutputs outputs[2] = {};
    shared = IPLReconstructorSharedInputs{0.5f, 1};
    iplReconstructorReconstruct(reconstructor, 2, inputs, &shared, outputs);
    iplReconstructorReconstruct(reconstructor, 1, inputs, nullptr, outputs);

    iplReconstructorRelease(&reconstructor);
}

void generate(const std::filesystem::path& out)
{
    // A context without validation first, so that upstream's global log has the callback before the validated
    // context's settings are checked.
    auto plain = makeContext(STEAMAUDIO_VERSION, static_cast<IPLContextFlags>(0));

    {
        IPLContextSettings settings{};
        settings.version = STEAMAUDIO_VERSION;
        settings.logCallback = record;
        settings.allocateCallback = alignedAllocate;
        settings.simdLevel = static_cast<IPLSIMDLevel>(7);
        settings.flags = IPL_CONTEXTFLAGS_VALIDATION;
        IPLContext bad = nullptr;
        check(iplContextCreate(&settings, &bad), "iplContextCreate");
        iplContextRelease(&bad);
    }
    {
        // A caller of API 4.4 cannot ask for validation.
        auto old = makeContext((STEAMAUDIO_VERSION_MAJOR << 16) | (4 << 8), IPL_CONTEXTFLAGS_VALIDATION);
        iplCalculateRelativeDirection(old, IPLVector3{kNaN, 0, 0}, IPLVector3{}, IPLVector3{0, 0, -1},
                                      IPLVector3{0, 1, 0});
        iplContextRelease(&old);
    }

    auto v = makeContext(STEAMAUDIO_VERSION, IPL_CONTEXTFLAGS_VALIDATION);
    contextCases(v);
    flush(out, "context");

    audioBufferCases(v);
    flush(out, "audio_buffer");

    deviceCases(v);
    flush(out, "devices");
    writeNpy(out / "device_statuses.npy", deviceStatuses(plain));

    effectCases(v);
    flush(out, "effects");

    sceneCases(v);
    flush(out, "scene");

    probeCases(v);
    flush(out, "probes");

    simulationCases(v);
    flush(out, "simulation");

    reconstructorCases(v);
    flush(out, "reconstructor");

    iplContextRelease(&v);
    iplContextRelease(&plain);
}

const Register reg("capi_validation", generate);

// --------------------------------------------------------------------------------------------------------------------
// sample_binaural input
// --------------------------------------------------------------------------------------------------------------------

// The raw input of core/src/samples/sample_binaural.cpp: 12 frames of 1024 samples and a partial frame, which the
// sample drops. tools/golden/run_sample.py writes it as inputaudio.raw and runs the sample.
void generateSampleInput(const std::filesystem::path& out)
{
    Lcg lcg(0x8d);
    std::vector<float> input(12 * 1024 + 300);
    for (auto& x : input)
    {
        auto value = lcg.nextSigned();
        x = value;
    }
    writeNpy(out / "input.npy", input);
}

const Register sampleReg("capi_sample", generateSampleInput);

}
