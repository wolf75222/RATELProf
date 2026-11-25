#include "host_trace.h"

#include RATELPROF_OMPT_HEADER

#include <cstdlib>
#include <cstring>

namespace {

void parallel_begin(ompt_data_t *, const ompt_frame_t *, ompt_data_t *parallel, unsigned int,
                    int, const void *)
{
    parallel->value = host_begin(Ompt);
}

void parallel_end(ompt_data_t *parallel, ompt_data_t *, int, const void *caller)
{
    host_end(Ompt, "OpenMP parallel region", parallel->value, caller);
}

const char *target_name(ompt_target_t kind)
{
    switch (kind) {
    case ompt_target_enter_data:
    case ompt_target_enter_data_nowait: return "OpenMP target enter data";
    case ompt_target_exit_data:
    case ompt_target_exit_data_nowait: return "OpenMP target exit data";
    case ompt_target_update:
    case ompt_target_update_nowait: return "OpenMP target update";
    default: return "OpenMP target";
    }
}

void target(ompt_target_t kind, ompt_scope_endpoint_t endpoint, int, ompt_data_t *,
            ompt_data_t *, ompt_data_t *data, const void *caller)
{
    if (endpoint == ompt_scope_begin)
        data->value = host_begin(Ompt);
    else if (endpoint == ompt_scope_end)
        host_end(Ompt, target_name(kind), data->value, caller);
}

const char *data_name(ompt_target_data_op_t op)
{
    switch (op) {
    case ompt_target_data_alloc:
    case ompt_target_data_alloc_async: return "OpenMP target allocation";
    case ompt_target_data_transfer_to_device:
    case ompt_target_data_transfer_to_device_async: return "OpenMP transfer to device";
    case ompt_target_data_transfer_from_device:
    case ompt_target_data_transfer_from_device_async: return "OpenMP transfer from device";
    case ompt_target_data_delete:
    case ompt_target_data_delete_async: return "OpenMP target deallocation";
    default: return "OpenMP target data operation";
    }
}

void target_data(ompt_scope_endpoint_t endpoint, ompt_data_t *, ompt_data_t *, ompt_id_t *op_id,
                 ompt_target_data_op_t op, void *, int, void *, int, size_t, const void *caller)
{
    if (endpoint == ompt_scope_begin)
        *op_id = host_begin(OmpTarget);
    else if (endpoint == ompt_scope_end)
        host_end(OmpTarget, data_name(op), *op_id, caller);
}

void target_submit(ompt_scope_endpoint_t endpoint, ompt_data_t *, ompt_id_t *op_id, unsigned int)
{
    if (endpoint == ompt_scope_begin)
        *op_id = host_begin(OmpTarget);
    else if (endpoint == ompt_scope_end)
        host_end(OmpTarget, "OpenMP target submit", *op_id, nullptr);
}

int initialize(ompt_function_lookup_t lookup, int, ompt_data_t *)
{
    const auto set = reinterpret_cast<ompt_set_callback_t>(lookup("ompt_set_callback"));
    if (!set)
        return 0;
    set(ompt_callback_parallel_begin, reinterpret_cast<ompt_callback_t>(parallel_begin));
    set(ompt_callback_parallel_end, reinterpret_cast<ompt_callback_t>(parallel_end));
    set(ompt_callback_target_emi, reinterpret_cast<ompt_callback_t>(target));
    set(ompt_callback_target_data_op_emi, reinterpret_cast<ompt_callback_t>(target_data));
    set(ompt_callback_target_submit_emi, reinterpret_cast<ompt_callback_t>(target_submit));
    return 1;
}

bool requested(const char *name)
{
    const char *value = std::getenv(name);
    return value && std::strcmp(value, "1") == 0;
}

} // namespace

extern "C" ompt_start_tool_result_t *ompt_start_tool(unsigned int, const char *)
{
    static ompt_start_tool_result_t tool{initialize, nullptr, {0}};
    if (requested("RATELPROF_DOMAIN_OMP_REGION") || requested("RATELPROF_DOMAIN_OMP_TGT_RTL"))
        return &tool;
    return nullptr;
}
