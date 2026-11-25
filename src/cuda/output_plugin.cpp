#include "output_plugin.h"
#include "ratelprof_cuda_plugin.h"

#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

void ratelprof_cuda_run_output_plugin(const char *report_path, std::int64_t rank,
                                     std::int64_t pid, std::int64_t exit_code)
{
    const char *library = std::getenv("RATELPROF_CUDA_PLUGIN");
    if (!library || !*library || std::strcmp(library, "msgpack") == 0 ||
        std::strcmp(library, "stdout") == 0)
        return;

    void *handle = dlopen(library, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        std::fprintf(stderr, "RATELProf CUDA: cannot load output plugin %s: %s\n",
                     library, dlerror());
        return;
    }

    using ProcessReport = decltype(&ratelprof_cuda_plugin_process_report_v1);
    constexpr char symbol[] = "ratelprof_cuda_plugin_process_report_v1";
    auto process = reinterpret_cast<ProcessReport>(dlsym(handle, symbol));
    if (!process) {
        std::fprintf(stderr, "RATELProf CUDA: plugin %s does not export %s\n", library, symbol);
    } else {
        const ratelprof_cuda_report_v1 report{1, sizeof(ratelprof_cuda_report_v1),
                                            report_path, rank, pid, exit_code};
        try {
            const int status = process(&report);
            if (status)
                std::fprintf(stderr, "RATELProf CUDA: plugin %s returned %d\n", library, status);
        } catch (...) {
            std::fprintf(stderr, "RATELProf CUDA: plugin %s threw an exception\n", library);
        }
    }
    dlclose(handle);
}
