#ifndef RATELPROF_CUDA_PLUGIN_H
#define RATELPROF_CUDA_PLUGIN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* One invocation per captured process, after the report has been published.
 * Paths and the report descriptor are valid for the duration of the call.
 * mpi_rank is -1 when the process is outside an MPI launch.
 */
typedef struct ratelprof_cuda_report_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    const char *path;
    int64_t mpi_rank;
    int64_t process_id;
    int64_t exit_code;
} ratelprof_cuda_report_v1;

/* Export this symbol from a shared library passed to profile --plugin.
 * Return zero on success. A nonzero status is diagnosed and the report
 * remains available at report->path.
 */
int ratelprof_cuda_plugin_process_report_v1(const ratelprof_cuda_report_v1 *report);

#ifdef __cplusplus
}
#endif

#endif
