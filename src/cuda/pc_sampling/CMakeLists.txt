find_package(Threads REQUIRED)

function(add_pc_sampler target source)
    add_library(${target} SHARED "${source}" ../module_capture.cpp)
    target_compile_features(${target} PRIVATE cxx_std_17)
    target_compile_options(${target} PRIVATE -Wall -Wextra)
    target_link_libraries(${target} PRIVATE ${RATELPROF_CUPTI_TARGET} Threads::Threads ${CMAKE_DL_LIBS})
    set_target_properties(${target} PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
    install(TARGETS ${target} LIBRARY DESTINATION lib)
endfunction()

if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 11.8 AND
   EXISTS "${RATELPROF_CUPTI_HEADERS}/cupti_pcsampling.h")
    add_pc_sampler(ratelprof_cuda_pcsampling sampling.cpp)
endif()

if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 10.2 AND CUDAToolkit_VERSION VERSION_LESS 13.0)
    add_pc_sampler(ratelprof_cuda_pcsampling_legacy legacy.cpp)
    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 11.8 AND
       EXISTS "${RATELPROF_CUPTI_HEADERS}/cupti_pcsampling.h")
        target_compile_definitions(ratelprof_cuda_pcsampling_legacy PRIVATE RATELPROF_HAVE_CUBIN_CRC)
    endif()
endif()
