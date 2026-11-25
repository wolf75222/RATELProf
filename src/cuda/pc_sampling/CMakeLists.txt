find_package(Threads REQUIRED)

add_library(ratelprof_cuda_pcsampling SHARED sampling.cpp ../module_capture.cpp)
target_compile_features(ratelprof_cuda_pcsampling PRIVATE cxx_std_17)
target_compile_options(ratelprof_cuda_pcsampling PRIVATE -Wall -Wextra)
target_link_libraries(ratelprof_cuda_pcsampling PRIVATE CUDA::cupti Threads::Threads ${CMAKE_DL_LIBS})
set_target_properties(ratelprof_cuda_pcsampling PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

install(TARGETS ratelprof_cuda_pcsampling LIBRARY DESTINATION lib)
