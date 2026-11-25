#include "host_trace.h"

#include RATELPROF_OMPT_HEADER

#include <cstdlib>
#include <cstring>
#include <utility>

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

void target(ompt_target_t kind, ompt_scope_endpoint_t endpoint, int device, ompt_data_t *,
            ompt_data_t *, ompt_data_t *data, const void *caller)
{
    if (endpoint == ompt_scope_begin)
        data->value = host_begin(Ompt);
    else if (endpoint == ompt_scope_end)
        host_end(Ompt, target_name(kind), data->value, caller, int64_t{0}, false,
                 {{"device", "int", trace_argument_value(device)}});
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
                 ompt_target_data_op_t op, void *source, int source_device,
                 void *destination, int destination_device, size_t bytes, const void *caller)
{
    if (endpoint == ompt_scope_begin)
        *op_id = host_begin(OmpTarget);
    else if (endpoint == ompt_scope_end)
        host_end(OmpTarget, data_name(op), *op_id, caller, int64_t{0}, false, {
            {"source", "void *", trace_argument_value(source)},
            {"source_device", "int", trace_argument_value(source_device)},
            {"destination", "void *", trace_argument_value(destination)},
            {"destination_device", "int", trace_argument_value(destination_device)},
            {"bytes", "size_t", trace_argument_value(bytes)}
        });
}

void target_submit(ompt_scope_endpoint_t endpoint, ompt_data_t *, ompt_id_t *op_id, unsigned int teams)
{
    if (endpoint == ompt_scope_begin)
        *op_id = host_begin(OmpTarget);
    else if (endpoint == ompt_scope_end)
        host_end(OmpTarget, "OpenMP target submit", *op_id, nullptr, int64_t{0}, false,
                 {{"requested_teams", "unsigned int", trace_argument_value(teams)}});
}

void trace_point(const char *name, const void *caller, TraceArguments args)
{
    host_end(OmpTarget, name, host_begin(OmpTarget), caller, int64_t{0}, false,
             std::move(args), true);
}

void device_initialize(int device, const char *type, ompt_device_t *,
                       ompt_function_lookup_t, const char *)
{
    trace_point("OpenMP device initialize", nullptr, {
        {"device", "int", trace_argument_value(device)},
        {"type", "string", std::string(type ? type : "")}
    });
}

void device_finalize(int device)
{
    trace_point("OpenMP device finalize", nullptr,
                {{"device", "int", trace_argument_value(device)}});
}

void device_load(int device, const char *filename, int64_t offset, void *, size_t bytes,
                 void *, void *, uint64_t module)
{
    trace_point("OpenMP device module load", nullptr, {
        {"device", "int", trace_argument_value(device)},
        {"filename", "string", std::string(filename ? filename : "")},
        {"file_offset", "int64_t", trace_argument_value(offset)},
        {"bytes", "size_t", trace_argument_value(bytes)},
        {"module_id", "uint64_t", trace_argument_value(module)}
    });
}

void device_unload(int device, uint64_t module)
{
    trace_point("OpenMP device module unload", nullptr, {
        {"device", "int", trace_argument_value(device)},
        {"module_id", "uint64_t", trace_argument_value(module)}
    });
}

void target_map(ompt_data_t *, unsigned int items, void **, void **, size_t *,
                unsigned int *, const void *caller)
{
    trace_point("OpenMP target map", caller,
                {{"items", "unsigned int", trace_argument_value(items)}});
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
    set(ompt_callback_target_map_emi, reinterpret_cast<ompt_callback_t>(target_map));
    set(ompt_callback_device_initialize, reinterpret_cast<ompt_callback_t>(device_initialize));
    set(ompt_callback_device_finalize, reinterpret_cast<ompt_callback_t>(device_finalize));
    set(ompt_callback_device_load, reinterpret_cast<ompt_callback_t>(device_load));
    set(ompt_callback_device_unload, reinterpret_cast<ompt_callback_t>(device_unload));
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
