cmake_minimum_required(VERSION 3.16)
project(recc CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OPENCV_INCLUDE_DIR "" CACHE PATH "Directory containing opencv2/*.hpp")
set(OPENCV_LIB_DIR     "" CACHE PATH "Directory containing libopencv_core.so etc.")
set(CLIPPER_INCLUDE_DIR "" CACHE PATH "Directory containing polyclipping/clipper.hpp")
set(CLIPPER_LIB_DIR     "" CACHE PATH "Directory containing libpolyclipping.so")
set(ZLIB_LIB_DIR        "" CACHE PATH "Directory containing libz (needed by opencv_core's persistence/FileStorage code even without imgcodecs)")

set(RKNN_SDK_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party" CACHE PATH "Directory containing rknn_api.h")
set(RKNN_SDK_LIB_DIR    "" CACHE PATH "Directory containing librknnrt.so (board SDK only)")

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OPENCV_INCLUDE_DIR}
    ${CLIPPER_INCLUDE_DIR}
)
link_directories(${OPENCV_LIB_DIR} ${CLIPPER_LIB_DIR} ${ZLIB_LIB_DIR})

# ocr_core: the OCR logic (detector, recognizer, alarm detector, and
# shared helpers) as a static library, not a runnable program.
add_library(ocr_core STATIC
    pipeline/ocr/rknn_executor.cpp
    pipeline/ocr/preprocess.cpp
    pipeline/ocr/ppocr_det.cpp
    pipeline/ocr/ppocr_rec.cpp
    pipeline/ocr/ppocr_system.cpp
    pipeline/ocr/alarm_detector.cpp
    pipeline/ocr/text_correction.cpp
)

target_include_directories(ocr_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${RKNN_SDK_INCLUDE_DIR})
# imgcodecs (imread/imwrite/imdecode/imencode) is intentionally not linked:
# no code in this library reads or writes image files -- frames arrive as
# already-decoded cv::Mat objects from an earlier pipeline stage. Dropping it
# avoids needing libjpeg/libpng for the board build. zlib is still required,
# though: opencv_core's own persistence (FileStorage) code calls gz* functions
# internally regardless of whether imgcodecs is linked in.
#
# opencv_core and opencv_imgproc are wrapped in a linker group: imgproc's
# object files call back into core-internal helpers (softdouble, DFT,
# BufferArea, etc.) that our own code never calls directly, so a plain
# left-to-right link only pulls in the core.a members our code references
# and leaves those internal-to-imgproc needs unresolved. --start-group
# makes ld keep re-scanning both archives until nothing new resolves,
# instead of a single pass.
#
# dl comes right after the opencv group deliberately: opencv_core's plugin
# loader (dlopen/dlsym/dlclose/dladdr) is only referenced from inside
# opencv_core.a itself, never from our own code, so -ldl has to appear
# after opencv_core in the link line or the linker never sees a pending
# need for it to satisfy.
target_link_libraries(ocr_core PUBLIC
    -Wl,--start-group opencv_core opencv_imgproc -Wl,--end-group
    dl polyclipping z pthread
)

if(RKNN_SDK_LIB_DIR)
    target_link_directories(ocr_core PUBLIC ${RKNN_SDK_LIB_DIR})
    target_link_libraries(ocr_core PUBLIC rknnrt)
endif()

# Builds an executable that links ocr_core and loads raw .bgr888 images
# (benchmark.cpp fits this shape). No stb_image/JPEG-PNG decode anymore --
# this pipeline is raw BGR888 only (see pipeline/ocr/image_io.cpp).
function(add_ocr_tool name source)
    add_executable(${name} ${source} pipeline/ocr/image_io.cpp)
    target_link_libraries(${name} PRIVATE ocr_core)
    # librknnrt.so itself dynamically depends on the system's own
    # libstdc++.so.6 at runtime (independent of ours being statically
    # linked in) -- allow that to resolve at load time on the board rather
    # than demanding a matching .so be present at link time here.
    target_link_options(${name} PRIVATE -Wl,--allow-shlib-undefined)
endfunction()

# Goal-based automation: match an offline-drafted rule against real OCR
# output and turn it into a KVM action sequence. No OpenCV/RKNN
# dependency -- this logic never touches an image, only the text and
# coordinates OCR already produced, so it builds and runs on any machine
# and isn't gated behind BUILD_TOOLS like the RKNN-dependent targets below.
add_executable(automation_runner
    pipeline/automation/automation_runner.cpp
    pipeline/automation/rule_matcher.cpp
    pipeline/automation/kvm_sequence.cpp
    pipeline/automation/rule.cpp
    pipeline/automation/screens.cpp
    pipeline/automation/json_value.cpp
)
target_include_directories(automation_runner PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Builds the runnable benchmark program (and ocr_server below) once a real
# librknnrt.so is available (RKNN_SDK_LIB_DIR set).
option(BUILD_TOOLS "Build the benchmark tool and ocr_server" OFF)
if(BUILD_TOOLS)
    if(NOT RKNN_SDK_LIB_DIR)
        message(FATAL_ERROR "BUILD_TOOLS requires RKNN_SDK_LIB_DIR (needs the real librknnrt.so)")
    endif()
    add_ocr_tool(benchmark pipeline/ocr/benchmark.cpp)

    # Internal OCR API server: same OCR + alarm pipeline as `ocr`, kept
    # resident and served over a minimal HTTP interface instead of run once
    # against a single file. See the "Internal OCR API" section of the
    # top-level README.md for the request/response contract.
    add_executable(ocr_server
        api/ocr_server.cpp
        api/http_server.cpp
        api/json_write.cpp
    )
    target_link_libraries(ocr_server PRIVATE ocr_core)
    target_link_options(ocr_server PRIVATE -Wl,--allow-shlib-undefined)
endif()
