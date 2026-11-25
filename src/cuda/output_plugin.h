#ifndef RATELPROF_CUDA_OUTPUT_PLUGIN_H
#define RATELPROF_CUDA_OUTPUT_PLUGIN_H

#include "argument_values.h"
#include "ratelprof_cuda_plugin.h"

#include <cstdint>
#include <memory>
#include <string>

struct PluginActivity {
    ratelprof_cuda_activity_v1 activity{};
    std::string name;
    std::string annotation_domain;
    TraceArguments arguments;
};

class CudaActivityPlugin {
public:
    CudaActivityPlugin();
    ~CudaActivityPlugin();
    void open();
    bool captures_activity() const;
    void set_periodic_flush(bool (*flush)());

    void enqueue(PluginActivity activity);
    void count_dropped_activity();
    void drain();
    void close();
    void process_report(const char *path, std::int64_t rank,
                        std::int64_t pid, std::int64_t exit_code);

private:
    struct Instance;
    std::unique_ptr<Instance> instance_;
};

bool ratelprof_cuda_in_plugin_callback();

void ratelprof_cuda_run_output_plugin(const char *report_path, std::int64_t rank,
                                    std::int64_t pid, std::int64_t exit_code);

#endif
