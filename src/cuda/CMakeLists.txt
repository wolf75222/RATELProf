if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "The CUDA profiler requires Linux")
endif()

if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 13.2)
    message(FATAL_ERROR "Select CUDA 12.9, 13.0 or 13.1 with CUDAToolkit_ROOT")
endif()
find_package(Threads REQUIRED)

set(MSGPACK_DIR "${PROJECT_SOURCE_DIR}/src/common/memory_structure")
add_library(ratelprof_cuda SHARED
    collector.cpp
    "${MSGPACK_DIR}/msgpack.c"
    "${MSGPACK_DIR}/msgpack_ext.c"
)
target_compile_features(ratelprof_cuda PRIVATE cxx_std_17)
target_compile_options(ratelprof_cuda PRIVATE -Wall -Wextra)
target_include_directories(ratelprof_cuda PRIVATE "${MSGPACK_DIR}" "${PROJECT_SOURCE_DIR}/src/tools")
target_link_libraries(ratelprof_cuda PRIVATE CUDA::cupti Threads::Threads ${CMAKE_DL_LIBS})

function(add_host_trace library)
    add_library(${library} SHARED host_wrappers.cpp ${ARGN})
    target_compile_features(${library} PRIVATE cxx_std_17)
    target_compile_options(${library} PRIVATE -Wall -Wextra)
    target_link_libraries(${library} PRIVATE ratelprof_cuda ${CMAKE_DL_LIBS})
    set_target_properties(${library} PROPERTIES INSTALL_RPATH "$ORIGIN")
    install(TARGETS ${library} LIBRARY DESTINATION lib)
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
        target_compile_definitions(ratelprof_cuda PRIVATE RATELPROF_CUDA_MPI)
        message(STATUS "CUDA backend: MPI tracing enabled")
    endif()
endif()

find_program(RATELPROF_CLANG clang)
if(RATELPROF_CLANG)
    execute_process(COMMAND "${RATELPROF_CLANG}" -print-resource-dir
        OUTPUT_VARIABLE RATELPROF_CLANG_RESOURCE OUTPUT_STRIP_TRAILING_WHITESPACE)
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
