#ifndef RATELPROF_CUDA_OUTPUT_PLUGIN_H
#define RATELPROF_CUDA_OUTPUT_PLUGIN_H

#include <cstdint>

void ratelprof_cuda_run_output_plugin(const char *report_path, std::int64_t rank,
                                    std::int64_t pid, std::int64_t exit_code);

#endif
