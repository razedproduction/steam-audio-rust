# Golden data generator target. Not a standalone project: included into the reference build by hook.cmake,
# in the scope of core/CMakeLists.txt, so source paths are made absolute.

set(GOLDEN_DIR ${CMAKE_CURRENT_LIST_DIR})

add_executable(golden
    ${GOLDEN_DIR}/src/main.cpp
    ${GOLDEN_DIR}/src/fft.cpp
    ${GOLDEN_DIR}/src/geometry.cpp
    ${GOLDEN_DIR}/src/arrays.cpp
    ${GOLDEN_DIR}/src/audio_buffer.cpp
    ${GOLDEN_DIR}/src/sh.cpp
    ${GOLDEN_DIR}/src/dsp.cpp
    ${GOLDEN_DIR}/src/hrtf.cpp
    ${GOLDEN_DIR}/src/convolution.cpp
    ${GOLDEN_DIR}/src/spatialization.cpp
    ${GOLDEN_DIR}/src/ambisonics.cpp
    ${GOLDEN_DIR}/src/sofa.cpp
    ${GOLDEN_DIR}/src/sofa_easy.cpp
    ${GOLDEN_DIR}/src/sofa_hrtf.cpp
    ${GOLDEN_DIR}/src/direct.cpp
    ${GOLDEN_DIR}/src/scene.cpp
    ${GOLDEN_DIR}/src/direct_simulator.cpp
    ${GOLDEN_DIR}/src/reconstructor.cpp
    ${GOLDEN_DIR}/src/reverb.cpp
    ${GOLDEN_DIR}/src/reflection_simulator.cpp
    ${GOLDEN_DIR}/src/serialization.cpp
    ${GOLDEN_DIR}/src/probes.cpp
    ${GOLDEN_DIR}/src/simulation.cpp
    ${GOLDEN_DIR}/src/pathing.cpp
    ${GOLDEN_DIR}/src/path_simulation.cpp
    ${GOLDEN_DIR}/src/capi.cpp
    ${GOLDEN_DIR}/src/capi_baking.cpp
    ${GOLDEN_DIR}/src/capi_simulation.cpp
    ${GOLDEN_DIR}/src/capi_validation.cpp
)

target_link_libraries(golden PRIVATE core hrtf)
target_compile_definitions(golden PRIVATE STEAMAUDIO_BUILDING_CORE)
target_include_directories(golden PRIVATE ${GOLDEN_DIR}/src)

# Upstream builds as C++14; the generator only needs C++17 for std::filesystem.
set_target_properties(golden PROPERTIES CXX_STANDARD 17)

# Upstream's sample_binaural (core/src/samples), unchanged, built twice: against upstream's library, as upstream's own
# samples/CMakeLists.txt builds it, and against the port's `phonon` library (only phonon.h and the import library),
# when GOLDEN_PORT_PHONON_DIR names the Cargo output directory holding phonon.dll and phonon.dll.lib. run_sample.py
# runs either one.
set(SAMPLE_BINAURAL_SOURCE ${CMAKE_SOURCE_DIR}/src/samples/sample_binaural.cpp)

add_executable(sample_binaural_upstream ${SAMPLE_BINAURAL_SOURCE})
target_link_libraries(sample_binaural_upstream PRIVATE core hrtf)
target_compile_definitions(sample_binaural_upstream PRIVATE STEAMAUDIO_BUILDING_CORE)

set(GOLDEN_PORT_PHONON_DIR "" CACHE PATH "Directory with the port's phonon.dll and phonon.dll.lib")
if (GOLDEN_PORT_PHONON_DIR)
    add_executable(sample_binaural_port ${SAMPLE_BINAURAL_SOURCE})
    target_include_directories(sample_binaural_port PRIVATE ${CMAKE_SOURCE_DIR}/src/core ${CMAKE_BINARY_DIR}/src/core)
    target_link_libraries(sample_binaural_port PRIVATE ${GOLDEN_PORT_PHONON_DIR}/phonon.dll.lib)
    add_custom_command(TARGET sample_binaural_port POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${GOLDEN_PORT_PHONON_DIR}/phonon.dll
                $<TARGET_FILE_DIR:sample_binaural_port>)
endif()
