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

/* In an activity, UINT32_MAX denotes an unavailable CUDA identifier.
 * A copy endpoint equal to RATELPROF_CUDA_HOST_DEVICE is CPU memory.
 */
#define RATELPROF_CUDA_NO_ID UINT32_MAX
#define RATELPROF_CUDA_HOST_DEVICE (UINT32_MAX - 1U)

enum ratelprof_cuda_value_kind_v1 {
    RATELPROF_CUDA_VALUE_U64 = 1,
    RATELPROF_CUDA_VALUE_I64,
    RATELPROF_CUDA_VALUE_F64,
    RATELPROF_CUDA_VALUE_BOOL,
    RATELPROF_CUDA_VALUE_STRING,
    RATELPROF_CUDA_VALUE_DIM3
};

typedef struct ratelprof_cuda_argument_v1 {
    const char *name;
    const char *type;
    uint32_t value_kind;
    union {
        uint64_t u64;
        int64_t i64;
        double f64;
        uint8_t boolean;
        const char *string;
        uint64_t dim3[3];
    } value;
} ratelprof_cuda_argument_v1;

typedef struct ratelprof_cuda_activity_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint64_t sequence;
    int64_t mpi_rank;
    int64_t process_id;
    const char *domain;
    const char *name;
    const char *annotation_domain;
    uint64_t start_ns;
    uint64_t stop_ns;
    uint64_t submitted_ns;
    uint64_t experiment_start_epoch_ns;
    uint64_t thread_id;
    uint64_t byte_count;
    uint64_t graph_node;
    uint64_t caller_address;
    uint32_t cupti_correlation_id;
    uint32_t cupti_runtime_correlation_id;
    uint32_t device_id;
    uint32_t context_id;
    uint32_t stream_id;
    uint32_t source_device_id;
    uint32_t destination_device_id;
    uint32_t source_memory_kind;
    uint32_t destination_memory_kind;
    uint32_t cuda_event_id;
    uint32_t registers_per_thread;
    uint32_t local_memory_bytes;
    uint32_t shared_memory_bytes;
    uint32_t memset_value;
    uint64_t cuda_event_sync_id;
    int32_t grid[3];
    int32_t block[3];
    const ratelprof_cuda_argument_v1 *arguments;
    uint64_t argument_count;
} ratelprof_cuda_activity_v1;

/* Optional entry point enabled by profile --plugin PATH --plugin-live.
 *
 * One worker invokes the function in collection order, which may differ from
 * timestamp order. Host calls and GPU activities arrive as CUPTI delivers them,
 * including the final activity flush. The post-report callback, if exported,
 * runs after the activity worker drains.
 *
 * Pointers are valid for this call only. Times are relative to capture start.
 * Sequence numbers start at 1 and link to plugin_sequence in the report. Gaps
 * can occur if the plugin queue fills. Other identifiers use CUDA ordinals
 * and CUPTI IDs, with unavailable fields set to zero or RATELPROF_CUDA_NO_ID.
 * CUDA activity callbacks should avoid issuing CUDA calls to keep the
 * application trace independent of the plugin's work.
 *
 * Return zero on success. Nonzero results are diagnosed.
 */
int ratelprof_cuda_plugin_process_activity_v1(const ratelprof_cuda_activity_v1 *activity);

#ifdef __cplusplus
}
#endif

#endif
