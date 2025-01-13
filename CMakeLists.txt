cmake_minimum_required(VERSION 3.25)

# For std::filesystem in onnx optimizer
# Must be a cache variable and be set before project()
# Reference: https://cmake.org/cmake/help/latest/variable/CMAKE_OSX_DEPLOYMENT_TARGET.html
# It can be a normal variable if policy CMP0126 is set to NEW.
set(CMAKE_OSX_DEPLOYMENT_TARGET 10.15 CACHE STRING "Minimum OS X deployment version")

project(faster-rwkv CXX)

set(CMAKE_CXX_STANDARD 17)
include(FetchContent)

option(FR_BUILD_EXECUTABLE "" ON)

option(FR_ENABLE_CUDA "Enable CUDA" OFF)
# ONNX support is a WIP
option(FR_ENABLE_ONNX_EXPORTING "Enable ONNX exporting" OFF)
option(FR_ENABLE_ONNX "Enable ONNX" OFF)
if (FR_ENABLE_ONNX OR FR_ENABLE_ONNX_EXPORTING OR (FR_ENABLE_NCNN AND WIN32))
    option(FR_BUILD_PROTOBUF "Build protobuf (needed by onnx or ncnn on windows)" ON)
else()
    option(FR_BUILD_PROTOBUF "Build protobuf (needed by onnx or ncnn on windows)" OFF)
endif()
option(FR_ENABLE_NCNN "Enable NCNN" ON)
option(FR_ENABLE_QNN "Enable Qualcomm QNN backend" OFF)
option(FR_ENABLE_MTK "Enable MediaTek NeuronPilot backend" OFF)
option(FR_ENABLE_WEBRWKV "Enable WebRWKV backend" OFF)
option(FR_ENABLE_RWKVCPP "Enable rwkv.cpp backend" ON)
option(FR_BUILD_PYTHON "" OFF)
if (DEFINED ANDROID_NDK)
    option(FR_BUILD_JNI "" ON)
else()
    option(FR_BUILD_JNI "" OFF)
endif()
option(FR_ENABLE_TESTS "Enable the tests of faster-rwkv" OFF)
option(FR_MSVC_STATIC_RUNTIME "" OFF)
option(FR_BUILD_SHARED_LIB "Build faster-rwkv as a shared library" OFF)
option(FR_BUILD_STATIC_WRAPPER_LIB "Build faster-rwkv wrapper as a static library" OFF)

if(NOT DEFINED CMAKE_POSITION_INDEPENDENT_CODE)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

if (FR_ENABLE_CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES native)
    endif()

    enable_language(CUDA)

    set(cuda_kernel_srcs
        kernels/cuda/activations/silu.cu
        kernels/cuda/matmul.cpp
        kernels/cuda/cat.cu
        kernels/cuda/layer_norm.cu
        kernels/cuda/group_norm.cu
        kernels/cuda/cast_dtype.cu
        kernels/cuda/att.cu
        kernels/cuda/att_seq.cu
        kernels/cuda/ffn.cu
        kernels/cuda/ffn_seq.cu
        kernels/cuda/fill.cu
        kernels/cuda/flip.cu
        kernels/cuda/pad.cu
        kernels/cuda/repeat.cu
        kernels/cuda/slice.cu
        kernels/cuda/transpose.cu
        kernels/cuda/element_wise.cu
        kernels/cuda/allocator.cpp
    )
endif()

if (FR_ENABLE_NCNN)
    FetchContent_Declare(
        ncnn
        GIT_REPOSITORY https://github.com/daquexian/ncnn
        # This commit reduces peak memory usage
        # https://github.com/Tencent/ncnn/pull/4966
        GIT_TAG c0daa4fd
        SYSTEM
        )
    option(NCNN_BUILD_BENCHMARK "" OFF)
    option(NCNN_BUILD_TOOLS "" OFF)
    option(NCNN_BUILD_EXAMPLES "" OFF)
    option(NCNN_BUILD_TESTS "" OFF)
    option(NCNN_PIXEL "" OFF)
    option(NCNN_PIXEL_ROTATE "" OFF)
    option(NCNN_PIXEL_AFFINE "" OFF)
    option(NCNN_PIXEL_DRAWING "" OFF)
    option(NCNN_DISABLE_EXCEPTION "" OFF)
    if (FR_BUILD_PYTHON)
        # pybind11 requires RTTI
        option(NCNN_DISABLE_RTTI "" OFF)
    endif()
    if (NOT CMAKE_SYSTEM_NAME STREQUAL "Android" AND NOT CMAKE_CROSSCOMPILING AND NOT WIN32)
        option(NCNN_OPENMP "" OFF)
        option(NCNN_SIMPLEOMP "" ON)
    endif()
    # Termux
    if (CMAKE_SYSTEM_NAME STREQUAL "Android" AND NOT DEFINED ANDROID_NDK)
        option(NCNN_PLATFORM_API "" OFF)
    endif()
    include(disable_unused_ncnn_layers)
    FetchContent_MakeAvailable(ncnn)
    set(ncnn_deps ncnn)
    set(ncnn_kernel_srcs
        kernels/ncnn/init_model.cpp
        kernels/ncnn/model_forward.cpp
    )
endif()

if (FR_ENABLE_RWKVCPP AND ANDROID)
    set(rwkv_cpp_srcs
        kernels/rwkv.cpp/init_model.cpp
        kernels/rwkv.cpp/model_forward.cpp
    )
endif()

if (FR_ENABLE_QNN)
    set(qnn_kernel_srcs
        kernels/qnn/init_model.cpp
        kernels/qnn/model_forward.cpp
    )
endif()

if (FR_ENABLE_MTK)
    set(mtk_kernel_srcs
        kernels/mtk/init_model.cpp
        kernels/mtk/model_forward.cpp
    )
endif()

if (FR_BUILD_PROTOBUF)
    option(protobuf_BUILD_TESTS "Build tests" OFF)
    option(protobuf_MSVC_STATIC_RUNTIME "" ${FR_USE_MSVC_STATIC_RUNTIME})
    FetchContent_Declare(
        protobuf
        GIT_REPOSITORY https://github.com/protocolbuffers/protobuf
        GIT_TAG v3.20.1
        SYSTEM
        SOURCE_SUBDIR cmake
    )
    FetchContent_MakeAvailable(protobuf)
endif()

if (FR_ENABLE_ONNX OR FR_ENABLE_ONNX_EXPORTING)
    FetchContent_Declare(
        onnx
        GIT_REPOSITORY https://github.com/onnx/onnx
        GIT_TAG v1.14.0
        SYSTEM
    )
    option(ONNX_USE_MSVC_STATIC_RUNTIME "" ${FR_USE_MSVC_STATIC_RUNTIME})
    FetchContent_MakeAvailable(onnx)
    if (FR_ENABLE_ONNX_EXPORTING)
        set(onnx_kernel_srcs
            kernels/export-onnx/kernels.cpp
        )
    endif()
endif()
if (FR_ENABLE_ONNX)
    # WIP
    option(onnxruntime_USE_FULL_PROTOBUF "" ON)
    option(onnxruntime_BUILD_UNIT_TESTS "Build ONNXRuntime unit tests" OFF)
    option(onnxruntime_BUILD_SHARED_LIB "Build a shared library" ON)
    option(onnxruntime_DISABLE_RTTI "Disable RTTI" OFF)
    option(onnxruntime_DISABLE_EXCEPTIONS "Disable exception handling. Requires onnxruntime_MINIMAL_BUILD currently." OFF)
    FetchContent_Declare(
        onnxruntime
        GIT_REPOSITORY https://github.com/microsoft/onnxruntime
        GIT_TAG 5af6279440a0db698b0afbfc3de655400562831f
        SOURCE_SUBDIR cmake
        SYSTEM
    )
    FetchContent_MakeAvailable(onnxruntime)
    list(APPEND onnx_kernel_srcs
        kernels/onnx/init_model.cpp
        kernels/onnx/model_forward.cpp
    )
endif()

FetchContent_Declare(
        msgpack
        GIT_REPOSITORY https://github.com/msgpack/msgpack-c
        GIT_TAG cpp-6.1.0
        SYSTEM
        )
option(MSGPACK_USE_BOOST "" OFF)
FetchContent_MakeAvailable(msgpack)

set(INTERNAL_SRC
    utils.cpp
    model.cpp 
    tensor.cpp
    tokenizer.cpp
    sampler.cpp
    kernels/shape/shape_inference.cpp
    kernels/cpu/allocator.cpp
    kernels/cpu/fill.cpp
    kernels/cpu/cast_dtype.cpp
    kernels/cpu/repeat.cpp
    kernels/cpu/transpose.cpp
    kernels/cpu/slice.cpp
    kernels/cpu/flip.cpp
    kernels/cpu/pad.cpp
    kernels/default/gather_ops.cpp
    kernels/default/view_ops.cpp
    kernels/cpu/softmax.cpp
    kernels/default/att.cpp
    kernels/default/ffn.cpp
    kernels/default/init_model.cpp
    kernels/default/model_forward.cpp
    kernels/default/model_forward_seq.cpp
    kernels/export-ncnn/kernels.cpp
    ${cuda_kernel_srcs}
    ${ncnn_kernel_srcs}
    ${onnx_kernel_srcs}
    ${qnn_kernel_srcs}
    ${mtk_kernel_srcs}
    ${rwkv_cpp_srcs}
    )

add_library(faster_rwkv_internal ${INTERNAL_SRC})
target_link_libraries(faster_rwkv_internal PUBLIC msgpack-cxx)
target_link_libraries(faster_rwkv_internal PUBLIC ${ncnn_deps})
target_include_directories(faster_rwkv_internal PUBLIC ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

if (FR_ENABLE_QNN)
    target_compile_definitions(faster_rwkv_internal PUBLIC FR_ENABLE_QNN)
    target_include_directories(faster_rwkv_internal PUBLIC kernels/qnn/include)
    if (WIN32)
        find_library(SHLWAPI Shlwapi.lib) 
        target_link_directories(faster_rwkv_internal PUBLIC kernels/qnn/obj/aarch64-windows/)
        target_link_libraries(faster_rwkv_internal PUBLIC rwkv-qualcomm SHLWAPI)
    else()
        target_link_directories(faster_rwkv_internal PUBLIC kernels/qnn/obj/aarch64-android/)
        target_link_libraries(faster_rwkv_internal PUBLIC rwkv-qualcomm)
    endif()
endif()

if (FR_ENABLE_MTK)
    target_compile_definitions(faster_rwkv_internal PUBLIC FR_ENABLE_MTK)
    target_include_directories(faster_rwkv_internal PUBLIC kernels/mtk/include)
    target_link_directories(faster_rwkv_internal PUBLIC kernels/mtk/obj/aarch64-android/)
    target_link_libraries(faster_rwkv_internal PUBLIC rwkv_mtk)
endif()

if (FR_ENABLE_RWKVCPP AND ANDROID)
    target_compile_definitions(faster_rwkv_internal PUBLIC FR_ENABLE_RWKVCPP)
    target_link_directories(faster_rwkv_internal PUBLIC kernels/rwkv.cpp/obj/aarch64-android/)
    target_link_libraries(faster_rwkv_internal PUBLIC rwkv ggml)
endif()

if (DEFINED ANDROID_NDK)
    find_library(log-lib log)
    find_library(android-lib android)
    target_link_libraries(faster_rwkv_internal PUBLIC ${log-lib} ${android-lib})
endif()

if (FR_BUILD_STATIC_WRAPPER_LIB)
    add_library(faster_rwkv_static STATIC faster_rwkvd.cpp ${INTERNAL_SRC})
    if (FR_ENABLE_WEBRWKV)
        target_compile_definitions(faster_rwkv_static PUBLIC FR_ENABLE_WEBRWKV)
    endif()
    target_link_libraries(faster_rwkv_static PUBLIC msgpack-cxx)
    target_link_libraries(faster_rwkv_static PUBLIC ${ncnn_deps})
    target_include_directories(faster_rwkv_static PUBLIC ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
    
    include(GNUInstallDirs)
    install(TARGETS faster_rwkv_static EXPORT faster_rwkv_static
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
    install(FILES
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
endif()

if (FR_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(faster_rwkv_internal PUBLIC CUDA::cudart CUDA::cublas)
    target_compile_definitions(faster_rwkv_internal PUBLIC FR_ENABLE_CUDA)
endif()
if (FR_ENABLE_NCNN)
    target_compile_definitions(faster_rwkv_internal PUBLIC FR_ENABLE_NCNN)
endif()
if (FR_ENABLE_ONNX_EXPORTING)
    target_compile_definitions(faster_rwkv_internal PUBLIC FR_ENABLE_ONNX)
    target_link_libraries(faster_rwkv_internal PRIVATE onnx)
endif()
if (FR_ENABLE_ONNX)
    target_link_libraries(faster_rwkv_internal PRIVATE onnxruntime)
    target_include_directories(faster_rwkv_internal PRIVATE
        ${CMAKE_BINARY_DIR}/_deps/onnxruntime-src/include/onnxruntime/core/session/
        ${CMAKE_BINARY_DIR}/_deps/onnxruntime-src/include/
    )
endif()

if (DEFINED ANDROID_NDK)
    target_compile_definitions(faster_rwkv_internal PUBLIC FR_ENABLE_ANDROID_ASSET)
endif()

if (NOT WIN32)
    set(CMAKE_LINK_LIBRARY_USING_WHOLE_ARCHIVE "LINKER:--whole-archive" "<LINK_ITEM>" "LINKER:--no-whole-archive")
    set(CMAKE_LINK_LIBRARY_USING_WHOLE_ARCHIVE_SUPPORTED TRUE)
endif()

add_library(faster_rwkv INTERFACE)
target_link_libraries(faster_rwkv INTERFACE "$<LINK_LIBRARY:WHOLE_ARCHIVE,faster_rwkv_internal>")

if (FR_BUILD_SHARED_LIB)
    if (WIN32)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
    add_library(faster_rwkvd SHARED faster_rwkvd.cpp)
    target_include_directories(faster_rwkvd PUBLIC ${msgpack_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
    target_compile_definitions(faster_rwkvd PUBLIC MSGPACK_NO_BOOST)
    if (FR_ENABLE_WEBRWKV)
        if (NOT IOS)
            FetchContent_Declare(
                Corrosion
                GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
                GIT_TAG v0.5
            )
            FetchContent_MakeAvailable(Corrosion)

            FetchContent_Declare(
                web_rwkv_ffi
                GIT_REPOSITORY https://github.com/cryscan/web-rwkv-ffi.git
                GIT_TAG main
            )
            FetchContent_MakeAvailable(web_rwkv_ffi)

            corrosion_import_crate(MANIFEST_PATH ${web_rwkv_ffi_SOURCE_DIR}/Cargo.toml
                CRATE_TYPES staticlib)
        endif()

        target_compile_definitions(faster_rwkv_internal PUBLIC FR_ENABLE_WEBRWKV)
        if (WIN32)
            target_link_libraries(faster_rwkvd PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,faster_rwkv_internal>" web_rwkv_ffi 
                ws2_32 opengl32 d3d12 d3dcompiler userenv kernel32 user32 ntdll bcrypt propsys runtimeobject)
        endif()
        if (APPLE)
            set(WEBRWKV_EXTRA_LIBS "-framework QuartzCore -framework Metal -framework CoreGraphics -framework CoreFoundation -lobjc -liconv")
            target_link_libraries(faster_rwkvd PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,faster_rwkv_internal>" web_rwkv_ffi
                ${WEBRWKV_EXTRA_LIBS})
        endif()
        if (DEFINED ANDROID_NDK)
            target_link_libraries(faster_rwkvd PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,faster_rwkv_internal>" web_rwkv_ffi)
        endif()
    else()
        target_link_libraries(faster_rwkvd "$<LINK_LIBRARY:WHOLE_ARCHIVE,faster_rwkv_internal>")
    endif()
endif()

if (FR_BUILD_EXECUTABLE)
    option(BENCHMARK_ENABLE_TESTING "Enable testing of the benchmark library." OFF)
    option(BENCHMARK_ENABLE_GTEST_TESTS "Enable building the unit tests which depend on gtest" OFF)
    FetchContent_Declare(
            benchmark
            GIT_REPOSITORY https://github.com/google/benchmark
            GIT_TAG v1.8.2
            SYSTEM
            )
    FetchContent_MakeAvailable(benchmark)
    add_executable(bench_model bench_model.cpp)
    target_link_libraries(bench_model benchmark::benchmark faster_rwkv)
endif()

if (FR_ENABLE_ONNX_EXPORTING)
    add_subdirectory(export_onnx)
endif()

if (FR_BUILD_JNI)
    add_subdirectory(aar)
endif()

if (FR_BUILD_PYTHON)
    add_subdirectory(python)
endif()

if (FR_BUILD_EXECUTABLE)
    add_subdirectory(tools)
    add_subdirectory(examples)
endif()

if(FR_ENABLE_TESTS)
    add_subdirectory(tests)
endif()
