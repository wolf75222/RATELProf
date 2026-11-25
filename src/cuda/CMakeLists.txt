if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "The CUDA profiler requires Linux")
endif()

find_package(Threads REQUIRED)
find_package(Python3 3.8 REQUIRED COMPONENTS Interpreter)
find_program(RATELPROF_CLANG clang)
option(RATELPROF_CUDA_FULL_HOST_WRAPPERS "Generate host wrappers from the installed SDK headers" ON)

set(RATELPROF_CUPTI_ROOT "" CACHE PATH "Use a CUPTI package separate from the CUDA toolkit")
if(RATELPROF_CUPTI_ROOT)
    find_path(RATELPROF_CUPTI_OVERRIDE_INCLUDE cupti_activity.h PATHS "${RATELPROF_CUPTI_ROOT}"
        PATH_SUFFIXES include extras/CUPTI/include NO_DEFAULT_PATH REQUIRED)
    find_library(RATELPROF_CUPTI_OVERRIDE_LIBRARY cupti PATHS "${RATELPROF_CUPTI_ROOT}"
        PATH_SUFFIXES lib64 lib extras/CUPTI/lib64 NO_DEFAULT_PATH REQUIRED)
    add_library(ratelprof_cupti SHARED IMPORTED GLOBAL)
    set_target_properties(ratelprof_cupti PROPERTIES
        IMPORTED_LOCATION "${RATELPROF_CUPTI_OVERRIDE_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${RATELPROF_CUPTI_OVERRIDE_INCLUDE};${CUDAToolkit_INCLUDE_DIRS}")
    set(CUPTI_TARGET ratelprof_cupti)
    set(RATELPROF_CUPTI_META_DIR "${RATELPROF_CUPTI_OVERRIDE_INCLUDE}")
else()
    if(NOT TARGET CUDA::cupti)
        message(FATAL_ERROR "CUPTI is required. Set RATELPROF_CUPTI_ROOT for a separate package.")
    endif()
    get_target_property(CUPTI_INCLUDE_DIRS CUDA::cupti INTERFACE_INCLUDE_DIRECTORIES)
    find_path(RATELPROF_CUPTI_TOOLKIT_INCLUDE cupti_activity.h HINTS ${CUPTI_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIRS}
        "${CUDAToolkit_LIBRARY_ROOT}/extras/CUPTI/include" NO_DEFAULT_PATH REQUIRED)
    set(CUPTI_TARGET CUDA::cupti)
    set(RATELPROF_CUPTI_META_DIR "${RATELPROF_CUPTI_TOOLKIT_INCLUDE}")
endif()
set(RATELPROF_CUPTI_TARGET ${CUPTI_TARGET} PARENT_SCOPE)
set(RATELPROF_CUPTI_HEADERS "${RATELPROF_CUPTI_META_DIR}" PARENT_SCOPE)

set(CUPTI_ACTIVITY_RECORDS "${CMAKE_CURRENT_BINARY_DIR}/cupti_activity_records.inc")
set(CUPTI_ACTIVITY_HEADERS "${RATELPROF_CUPTI_META_DIR}/cupti_activity.h")
if(EXISTS "${RATELPROF_CUPTI_META_DIR}/cupti_activity_deprecated.h")
    list(APPEND CUPTI_ACTIVITY_HEADERS "${RATELPROF_CUPTI_META_DIR}/cupti_activity_deprecated.h")
endif()
add_custom_command(OUTPUT "${CUPTI_ACTIVITY_RECORDS}"
    COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/generate_activity_records.py"
        "${RATELPROF_CUPTI_META_DIR}" "${CUPTI_ACTIVITY_RECORDS}"
    DEPENDS generate_activity_records.py ${CUPTI_ACTIVITY_HEADERS}
    VERBATIM)

set(CUPTI_ARGUMENT_TABLE "${CMAKE_CURRENT_BINARY_DIR}/cupti_arguments.inc")
add_custom_command(OUTPUT "${CUPTI_ARGUMENT_TABLE}"
    COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/generate_cupti_arguments.py"
        "${RATELPROF_CUPTI_META_DIR}" "${CUPTI_ARGUMENT_TABLE}"
    DEPENDS generate_cupti_arguments.py
        "${RATELPROF_CUPTI_META_DIR}/cupti_driver_cbid.h"
        "${RATELPROF_CUPTI_META_DIR}/cupti_runtime_cbid.h"
        "${RATELPROF_CUPTI_META_DIR}/generated_cuda_meta.h"
        "${RATELPROF_CUPTI_META_DIR}/generated_cuda_runtime_api_meta.h"
    VERBATIM)

set(MSGPACK_DIR "${PROJECT_SOURCE_DIR}/src/common/memory_structure")
add_library(ratelprof_cuda SHARED
    collector.cpp
    cupti_arguments.cpp
    module_capture.cpp
    output_plugin.cpp
    "${CUPTI_ACTIVITY_RECORDS}"
    "${CUPTI_ARGUMENT_TABLE}"
    "${MSGPACK_DIR}/msgpack.c"
    "${MSGPACK_DIR}/msgpack_ext.c"
)
target_compile_features(ratelprof_cuda PRIVATE cxx_std_17)
target_compile_options(ratelprof_cuda PRIVATE -Wall -Wextra)
target_include_directories(ratelprof_cuda PRIVATE "${MSGPACK_DIR}" "${PROJECT_SOURCE_DIR}/src/tools"
    "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(ratelprof_cuda PRIVATE ${CUPTI_TARGET} Threads::Threads ${CMAKE_DL_LIBS})

function(add_host_trace library)
    add_library(${library} SHARED host_wrappers.cpp ${ARGN})
    target_compile_features(${library} PRIVATE cxx_std_17)
    target_compile_options(${library} PRIVATE -Wall -Wextra)
    target_link_libraries(${library} PRIVATE ratelprof_cuda ${CMAKE_DL_LIBS})
    set_target_properties(${library} PROPERTIES INSTALL_RPATH "$ORIGIN")
    install(TARGETS ${library} LIBRARY DESTINATION lib)
endfunction()

function(enable_host_signatures library domain prefix)
    if(NOT RATELPROF_CLANG OR NOT RATELPROF_CUDA_FULL_HOST_WRAPPERS)
        return()
    endif()

    set(arguments --clang "${RATELPROF_CLANG}" --domain "${domain}" --prefix "${prefix}")
    foreach(header IN LISTS ARGN)
        list(APPEND arguments --header "${header}")
    endforeach()
    get_target_property(includes ${library} INCLUDE_DIRECTORIES)
    if(includes)
        foreach(directory IN LISTS includes)
            list(APPEND arguments --include "${directory}")
        endforeach()
    endif()
    set(script "${CMAKE_CURRENT_SOURCE_DIR}/generate_host_wrappers.py")
    set(generated "${CMAKE_CURRENT_BINARY_DIR}/${library}_wrappers.inc")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${script}")
    execute_process(COMMAND "${Python3_EXECUTABLE}" "${script}" ${arguments} --output "${generated}"
        RESULT_VARIABLE status OUTPUT_VARIABLE summary ERROR_VARIABLE details OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(status EQUAL 0)
        target_compile_definitions(${library} PRIVATE RATELPROF_GENERATED_HOST_WRAPPERS="${generated}")
        message(STATUS "${summary}")
    else()
        message(WARNING "Using the hand-written ${domain} wrapper set. ${details}")
    endif()
endfunction()

add_host_trace(ratelprof_cuda_omp)
target_compile_definitions(ratelprof_cuda_omp PRIVATE RATELPROF_CUDA_OMP_ROUTINES)

option(RATELPROF_CUDA_CUBLAS "Build the optional cuBLAS wrappers" ON)
if(RATELPROF_CUDA_CUBLAS)
    find_path(RATELPROF_CUBLAS_INCLUDE_DIR cublas_v2.h HINTS ${CUDAToolkit_INCLUDE_DIRS} NO_DEFAULT_PATH)
    if(RATELPROF_CUBLAS_INCLUDE_DIR)
        add_host_trace(ratelprof_cuda_cublas)
        target_compile_definitions(ratelprof_cuda_cublas PRIVATE RATELPROF_CUDA_CUBLAS)
        target_include_directories(ratelprof_cuda_cublas PRIVATE ${RATELPROF_CUBLAS_INCLUDE_DIR})
        set(cublas_headers cublas_v2.h)
        if(EXISTS "${RATELPROF_CUBLAS_INCLUDE_DIR}/cublasLt.h")
            list(APPEND cublas_headers cublasLt.h)
        endif()
        enable_host_signatures(ratelprof_cuda_cublas Cublas cublas ${cublas_headers})
        target_compile_definitions(ratelprof_cuda PRIVATE RATELPROF_CUDA_CUBLAS)
        message(STATUS "CUDA backend: cuBLAS tracing enabled")
    endif()
endif()

option(RATELPROF_CUDA_NCCL "Build the optional NCCL wrappers" ON)
if(RATELPROF_CUDA_NCCL)
    find_path(RATELPROF_NCCL_INCLUDE_DIR nccl.h
        HINTS "${NCCL_ROOT}" ENV NCCL_ROOT PATH_SUFFIXES include)
    if(RATELPROF_NCCL_INCLUDE_DIR)
        add_host_trace(ratelprof_cuda_nccl)
        target_compile_definitions(ratelprof_cuda_nccl PRIVATE RATELPROF_CUDA_NCCL)
        target_include_directories(ratelprof_cuda_nccl PRIVATE
            ${RATELPROF_NCCL_INCLUDE_DIR} ${CUDAToolkit_INCLUDE_DIRS})
        enable_host_signatures(ratelprof_cuda_nccl Nccl nccl nccl.h)
        target_compile_definitions(ratelprof_cuda PRIVATE RATELPROF_CUDA_NCCL)
        message(STATUS "CUDA backend: NCCL tracing enabled")
    endif()
endif()

option(RATELPROF_CUDA_MPI "Build MPI interposition with the system MPI headers" ON)
if(RATELPROF_CUDA_MPI)
    find_package(MPI QUIET COMPONENTS C)
    if(MPI_C_FOUND)
        add_host_trace(ratelprof_cuda_mpi)
        target_compile_definitions(ratelprof_cuda_mpi PRIVATE RATELPROF_CUDA_MPI)
        target_include_directories(ratelprof_cuda_mpi PRIVATE ${MPI_C_INCLUDE_DIRS})
        enable_host_signatures(ratelprof_cuda_mpi Mpi MPI_ mpi.h)
        target_compile_definitions(ratelprof_cuda PRIVATE RATELPROF_CUDA_MPI)
        message(STATUS "CUDA backend: MPI tracing enabled")
    endif()
endif()

if(RATELPROF_CLANG)
    execute_process(COMMAND "${RATELPROF_CLANG}" -print-resource-dir
        OUTPUT_VARIABLE RATELPROF_CLANG_RESOURCE OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(EXISTS "${RATELPROF_CLANG_RESOURCE}/include/omp.h")
        set(omp_header "${RATELPROF_CLANG_RESOURCE}/include/omp.h")
        target_compile_definitions(ratelprof_cuda_omp PRIVATE RATELPROF_OMP_HEADER="${omp_header}")
        enable_host_signatures(ratelprof_cuda_omp OmpRoutine omp_ "${omp_header}")
    endif()
endif()
find_path(RATELPROF_OMPT_INCLUDE_DIR omp-tools.h HINTS "${RATELPROF_CLANG_RESOURCE}/include")
if(RATELPROF_OMPT_INCLUDE_DIR)
    include(CheckCXXSourceCompiles)
    check_cxx_source_compiles("
        #include \"${RATELPROF_OMPT_INCLUDE_DIR}/omp-tools.h\"
        int main() { return ompt_callback_target_emi; }
    " RATELPROF_OMPT_HAVE_EMI)
    if(RATELPROF_OMPT_HAVE_EMI)
        target_sources(ratelprof_cuda_omp PRIVATE ompt.cpp)
        target_compile_definitions(ratelprof_cuda_omp PRIVATE
            RATELPROF_OMPT_HEADER="${RATELPROF_OMPT_INCLUDE_DIR}/omp-tools.h")
        target_compile_definitions(ratelprof_cuda PRIVATE RATELPROF_CUDA_OMPT)
        message(STATUS "CUDA backend: OpenMP OMPT tracing enabled")
    endif()
endif()
# Preserve the toolkit's CUPTI lookup when installing outside the toolkit.
set_target_properties(ratelprof_cuda PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

install(TARGETS ratelprof_cuda LIBRARY DESTINATION lib)
install(FILES ratelprof_cuda_plugin.h DESTINATION include)
