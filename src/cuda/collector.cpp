#include <cupti.h>
#include <cupti_driver_cbid.h>
#include <cupti_runtime_cbid.h>
#include <dlfcn.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

extern "C" {
#include "msgpack.h"
}
#include "version.h"
#include "host_trace.h"

namespace {

#if CUDA_VERSION >= 13010
using KernelRecord = CUpti_ActivityKernel11;
using ContextRecord = CUpti_ActivityContext4;
#elif CUDA_VERSION >= 13000
using KernelRecord = CUpti_ActivityKernel10;
using ContextRecord = CUpti_ActivityContext3;
#else
using KernelRecord = CUpti_ActivityKernel9;
using ContextRecord = CUpti_ActivityContext3;
#endif

enum Domain { Runtime, Driver, Kernel, Copy, Memset, Sync, Nvtx, MpiHost,
              OmpRoutineHost, OmpTargetHost, OmptHost, CublasHost, NcclHost, DomainCount };
constexpr const char *domains[] = {
    "CUDA_RUNTIME", "CUDA_DRIVER", "KERNEL", "COPY", "CUDA_MEMSET", "CUDA_SYNC", "NVTX",
    "MPI", "OMP_ROUTINE", "OMP_TGT_RTL", "OMP_REGION", "CUBLAS", "NCCL"
};
constexpr Domain host_domains[] = {MpiHost, OmpRoutineHost, OmpTargetHost, OmptHost, CublasHost, NcclHost};
constexpr uint32_t unknown = std::numeric_limits<uint32_t>::max();
constexpr uint64_t host = uint64_t{unknown} + 1;
constexpr uint64_t no_device = host + 1;
constexpr uint64_t correlation_source = 1ULL << 32;
constexpr size_t activity_buffer_size = 256 * 1024;

struct Event {
    Domain domain;
    uint64_t start, stop;
    uint32_t correlation = 0;
    uint32_t runtime_correlation = 0;
    uint32_t device = unknown, context = unknown, stream = unknown;
    uint32_t pid = 0, tid = 0;
    uint64_t size = 0;
    uint64_t submitted = 0;
    uint64_t src = no_device, dst = no_device;
    uint32_t src_kind = 0, dst_kind = 0;
    uint32_t local = 0, shared = 0, registers = 0, value = 0;
    uint32_t result = 0, cuda_event = unknown;
    uint64_t graph_node = 0;
    uintptr_t caller = 0;
    int64_t host_result = 0;
    bool has_host_result = false;
    std::array<int32_t, 3> grid{}, block{};
    std::string name, marker_domain;

    Event(Domain kind, uint64_t begin, uint64_t end) : domain(kind), start(begin), stop(end) {}
};

struct Device {
    std::string name, uuid, arch;
    uint32_t sms = 0, warp = 0;
    uint64_t memory = 0;
};

struct Argument {
    const char *name;
    const char *type;
    uint64_t value;
};

struct Trace {
    std::mutex mutex;
    std::array<std::vector<Event>, DomainCount> events;
    std::map<uint32_t, Device> devices;
    std::unordered_map<uint32_t, uint32_t> contexts;
    std::unordered_map<uint32_t, Event> markers;
    std::unordered_map<uint32_t, uint64_t> early_marker_ends;
    std::unordered_map<uint64_t, std::vector<Argument>> arguments;
    std::atomic<uint64_t> dropped{0};
    std::atomic<bool> accepting_host{true};
    std::array<bool, DomainCount> enabled_domains{};
    std::vector<CUpti_ActivityKind> enabled;
    CUpti_SubscriberHandle subscriber = nullptr;
    std::vector<std::string> argv;
    std::string output;
    uint64_t start = 0, epoch = 0, main_start = 0, main_stop = 0;
    int result = 0;
    pid_t pid = getpid();
    bool registered = false;
};

Trace *trace = nullptr;

uint64_t clock_ns(clockid_t clock)
{
    timespec ts{};
    clock_gettime(clock, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + ts.tv_nsec;
}

// CUPTI and host instrumentation use the same monotonic clock.
uint64_t CUPTIAPI timestamp()
{
    return clock_ns(CLOCK_MONOTONIC);
}

bool check(CUptiResult result, const char *operation)
{
    if (result == CUPTI_SUCCESS)
        return true;
    const char *description = nullptr;
    cuptiGetResultString(result, &description);
    std::fprintf(stderr, "RATELProf CUDA: %s: %s\n", operation,
                 description ? description : "unknown CUPTI error");
    return false;
}

bool domain_requested(Domain domain)
{
    const std::string key = std::string("RATELPROF_DOMAIN_") + domains[domain];
    const char *value = std::getenv(key.c_str());
    return value && std::strcmp(value, "1") == 0;
}

bool enabled(Domain domain)
{
    return trace->enabled_domains[domain];
}

bool keep(Domain domain, std::string_view name)
{
    const std::string key = std::string("RATELPROF_DOMAIN_") + domains[domain];
    const char *mode = std::getenv((key + "_FILTER_TYPE").c_str());
    const char *list = std::getenv((key + "_FUNCTIONS_FILTERED").c_str());
    if (!mode || !list)
        return true;
    bool listed = false;
    const auto version = (domain == Runtime || domain == Driver || domain == CublasHost)
        ? name.rfind("_v") : name.npos;
    const std::string_view unversioned = name.substr(0, version);
    std::string_view remaining(list);
    while (!remaining.empty()) {
        const auto comma = remaining.find(',');
        const auto filter = remaining.substr(0, comma);
        if (filter == name || filter == unversioned) {
            listed = true;
            break;
        }
        if (comma == std::string_view::npos)
            break;
        remaining.remove_prefix(comma + 1);
    }
    return std::strcmp(mode, "2") == 0 ? listed : !listed;
}

uint64_t api_id(Domain domain, uint32_t correlation)
{
    return correlation_source + uint64_t{correlation} * 2 + (domain == Driver);
}

void CUPTIAPI api_arguments(void *, CUpti_CallbackDomain domain, CUpti_CallbackId callback,
                            const void *data)
{
    const auto &call = *static_cast<const CUpti_CallbackData *>(data);
    if (call.callbackSite != CUPTI_API_ENTER || !call.functionParams || getpid() != trace->pid)
        return;
    const auto trace_domain = domain == CUPTI_CB_DOMAIN_RUNTIME_API ? Runtime : Driver;
    if (!keep(trace_domain, call.functionName))
        return;

    std::vector<Argument> args;
    if (trace_domain == Driver) {
        const auto &p = *static_cast<const cuLaunchKernel_params *>(call.functionParams);
        args = {{"grid_x", "uint32_t", p.gridDimX}, {"grid_y", "uint32_t", p.gridDimY},
                {"grid_z", "uint32_t", p.gridDimZ}, {"block_x", "uint32_t", p.blockDimX},
                {"block_y", "uint32_t", p.blockDimY}, {"block_z", "uint32_t", p.blockDimZ},
                {"shared_bytes", "uint32_t", p.sharedMemBytes},
                {"stream", "CUstream", reinterpret_cast<uintptr_t>(p.hStream)}};
    } else {
        switch (callback) {
        case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000: {
            const auto &p = *static_cast<const cudaLaunchKernel_v7000_params *>(call.functionParams);
            args = {{"grid_x", "uint32_t", p.gridDim.x}, {"grid_y", "uint32_t", p.gridDim.y},
                    {"grid_z", "uint32_t", p.gridDim.z}, {"block_x", "uint32_t", p.blockDim.x},
                    {"block_y", "uint32_t", p.blockDim.y}, {"block_z", "uint32_t", p.blockDim.z},
                    {"shared_bytes", "size_t", p.sharedMem},
                    {"stream", "cudaStream_t", reinterpret_cast<uintptr_t>(p.stream)}};
            break;
        }
        case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020: {
            const auto &p = *static_cast<const cudaMemcpy_v3020_params *>(call.functionParams);
            args = {{"bytes", "size_t", p.count}, {"kind", "cudaMemcpyKind", static_cast<uint64_t>(p.kind)}};
            break;
        }
        case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020: {
            const auto &p = *static_cast<const cudaMemcpyAsync_v3020_params *>(call.functionParams);
            args = {{"bytes", "size_t", p.count}, {"kind", "cudaMemcpyKind", static_cast<uint64_t>(p.kind)},
                    {"stream", "cudaStream_t", reinterpret_cast<uintptr_t>(p.stream)}};
            break;
        }
        case CUPTI_RUNTIME_TRACE_CBID_cudaMemsetAsync_v3020: {
            const auto &p = *static_cast<const cudaMemsetAsync_v3020_params *>(call.functionParams);
            args = {{"bytes", "size_t", p.count}, {"value", "byte", static_cast<uint8_t>(p.value)},
                    {"stream", "cudaStream_t", reinterpret_cast<uintptr_t>(p.stream)}};
            break;
        }
        case CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020: {
            const auto &p = *static_cast<const cudaMalloc_v3020_params *>(call.functionParams);
            args = {{"bytes", "size_t", p.size}};
            break;
        }
        default: return;
        }
    }
    std::lock_guard<std::mutex> guard(trace->mutex);
    trace->arguments.emplace(api_id(trace_domain, call.correlationId), std::move(args));
}

void enable_api_arguments()
{
    if (!enabled(Runtime) && !enabled(Driver))
        return;
    if (!check(cuptiSubscribe(&trace->subscriber, api_arguments, nullptr), "subscribe to CUDA API arguments"))
        return;
    if (enabled(Runtime)) {
        for (auto callback : {CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000,
                              CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020,
                              CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020,
                              CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020,
                              CUPTI_RUNTIME_TRACE_CBID_cudaMemsetAsync_v3020}) {
            check(cuptiEnableCallback(1, trace->subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, callback),
                  "enable CUDA Runtime API arguments");
        }
    }
    if (enabled(Driver)) {
        check(cuptiEnableCallback(1, trace->subscriber, CUPTI_CB_DOMAIN_DRIVER_API,
                                  CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel),
              "enable CUDA Driver API arguments");
    }
}

void add(Event event)
{
    if (event.start && event.stop >= event.start && keep(event.domain, event.name))
        trace->events[event.domain].push_back(std::move(event));
}

bool device_memory(uint32_t kind)
{
    return kind == CUPTI_ACTIVITY_MEMORY_KIND_DEVICE || kind == CUPTI_ACTIVITY_MEMORY_KIND_ARRAY ||
           kind == CUPTI_ACTIVITY_MEMORY_KIND_MANAGED || kind == CUPTI_ACTIVITY_MEMORY_KIND_DEVICE_STATIC ||
           kind == CUPTI_ACTIVITY_MEMORY_KIND_MANAGED_STATIC;
}

template<class Record>
Event gpu_event(Domain domain, const Record &record)
{
    Event event(domain, record.start, record.end);
    event.correlation = record.correlationId;
    event.device = record.deviceId;
    event.context = record.contextId;
    event.stream = record.streamId;
    event.graph_node = record.graphNodeId;
    return event;
}

template<class Record>
Event copy_event(const Record &record, uint32_t src_device, uint32_t dst_device)
{
    Event event = gpu_event(Copy, record);
    event.size = record.bytes;
    event.src_kind = record.srcKind;
    event.dst_kind = record.dstKind;
    event.src = device_memory(event.src_kind) ? src_device : host;
    event.dst = device_memory(event.dst_kind) ? dst_device : host;
    if (record.copyKind == CUPTI_ACTIVITY_MEMCPY_KIND_HTOD ||
        record.copyKind == CUPTI_ACTIVITY_MEMCPY_KIND_HTOA) {
        event.src = host;
        event.dst = dst_device;
    } else if (record.copyKind == CUPTI_ACTIVITY_MEMCPY_KIND_DTOH ||
               record.copyKind == CUPTI_ACTIVITY_MEMCPY_KIND_ATOH) {
        event.src = src_device;
        event.dst = host;
    } else if (record.copyKind == CUPTI_ACTIVITY_MEMCPY_KIND_DTOD ||
               record.copyKind == CUPTI_ACTIVITY_MEMCPY_KIND_PTOP) {
        event.src = src_device;
        event.dst = dst_device;
    }
    event.name = std::string("Copy") + (event.src == host ? "Host" : "Device") +
                 "To" + (event.dst == host ? "Host" : "Device");
    return event;
}

const char *sync_name(CUpti_ActivitySynchronizationType type)
{
    switch (type) {
    case CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_EVENT_SYNCHRONIZE: return "Event synchronize";
    case CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_WAIT_EVENT: return "Stream wait event";
    case CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_SYNCHRONIZE: return "Stream synchronize";
    case CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_CONTEXT_SYNCHRONIZE: return "Context synchronize";
    default: return "CUDA synchronize";
    }
}

std::string uuid_string(const CUuuid &uuid)
{
    const auto *p = reinterpret_cast<const unsigned char *>(uuid.bytes);
    char buffer[37];
    std::snprintf(buffer, sizeof(buffer),
                  "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                  p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7],
                  p[8], p[9], p[10], p[11], p[12], p[13], p[14], p[15]);
    return buffer;
}

void marker(const CUpti_ActivityMarker2 &record)
{
    Event event(Nvtx, record.timestamp, record.timestamp);
    event.name = record.name ? record.name : "";
    event.marker_domain = record.domain ? record.domain : "";
    event.pid = record.objectKind == CUPTI_ACTIVITY_OBJECT_THREAD ||
                record.objectKind == CUPTI_ACTIVITY_OBJECT_PROCESS
                ? record.objectId.pt.processId : trace->pid;
    event.tid = record.objectKind == CUPTI_ACTIVITY_OBJECT_THREAD ? record.objectId.pt.threadId : 0;
    if (record.flags & CUPTI_ACTIVITY_FLAG_MARKER_START) {
        auto end = trace->early_marker_ends.find(record.id);
        if (end != trace->early_marker_ends.end()) {
            event.stop = end->second;
            add(std::move(event));
            trace->early_marker_ends.erase(end);
        } else {
            trace->markers.insert_or_assign(record.id, std::move(event));
        }
    } else if (record.flags & CUPTI_ACTIVITY_FLAG_MARKER_END) {
        auto start = trace->markers.find(record.id);
        if (start != trace->markers.end()) {
            start->second.stop = record.timestamp;
            add(std::move(start->second));
            trace->markers.erase(start);
        } else {
            trace->early_marker_ends[record.id] = record.timestamp;
        }
    } else if (record.flags & CUPTI_ACTIVITY_FLAG_MARKER_INSTANTANEOUS) {
        add(std::move(event));
    }
}

void collect(const CUpti_Activity *activity)
{
    switch (activity->kind) {
    case CUPTI_ACTIVITY_KIND_RUNTIME:
    case CUPTI_ACTIVITY_KIND_DRIVER: {
        const auto &record = *reinterpret_cast<const CUpti_ActivityAPI *>(activity);
        const bool runtime = activity->kind == CUPTI_ACTIVITY_KIND_RUNTIME;
        Event event(runtime ? Runtime : Driver, record.start, record.end);
        const char *name = nullptr;
        cuptiGetCallbackName(runtime ? CUPTI_CB_DOMAIN_RUNTIME_API : CUPTI_CB_DOMAIN_DRIVER_API,
                             record.cbid, &name);
        event.name = name ? name : "CUDA API #" + std::to_string(record.cbid);
        event.correlation = record.correlationId;
        event.pid = record.processId;
        event.tid = record.threadId;
        event.result = record.returnValue;
        add(std::move(event));
        break;
    }
    case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
        const auto &record = *reinterpret_cast<const KernelRecord *>(activity);
        Event event = gpu_event(Kernel, record);
        event.name = record.name ? record.name : "<unnamed CUDA kernel>";
        event.grid = {record.gridX, record.gridY, record.gridZ};
        event.block = {record.blockX, record.blockY, record.blockZ};
        event.local = record.localMemoryPerThread;
        event.shared = record.staticSharedMemory + record.dynamicSharedMemory;
        event.registers = record.registersPerThread;
        event.submitted = record.submitted;
        add(std::move(event));
        break;
    }
    case CUPTI_ACTIVITY_KIND_MEMCPY: {
        const auto &record = *reinterpret_cast<const CUpti_ActivityMemcpy6 *>(activity);
        Event event = copy_event(record, record.deviceId, record.deviceId);
        event.runtime_correlation = record.runtimeCorrelationId;
        add(std::move(event));
        break;
    }
    case CUPTI_ACTIVITY_KIND_MEMCPY2: {
        const auto &record = *reinterpret_cast<const CUpti_ActivityMemcpyPtoP4 *>(activity);
        add(copy_event(record, record.srcDeviceId, record.dstDeviceId));
        break;
    }
    case CUPTI_ACTIVITY_KIND_MEMSET: {
        const auto &record = *reinterpret_cast<const CUpti_ActivityMemset4 *>(activity);
        Event event = gpu_event(Memset, record);
        event.name = "CUDA memset";
        event.size = record.bytes;
        event.value = record.value;
        event.dst_kind = record.memoryKind;
        add(std::move(event));
        break;
    }
    case CUPTI_ACTIVITY_KIND_SYNCHRONIZATION: {
        const auto &record = *reinterpret_cast<const CUpti_ActivitySynchronization2 *>(activity);
        Event event(Sync, record.start, record.end);
        event.name = sync_name(record.type);
        event.correlation = record.correlationId;
        event.context = record.contextId;
        event.stream = record.streamId;
        event.cuda_event = record.cudaEventId;
        event.result = record.returnValue;
        add(std::move(event));
        break;
    }
    case CUPTI_ACTIVITY_KIND_DEVICE: {
        const auto &record = *reinterpret_cast<const CUpti_ActivityDevice5 *>(activity);
        if (!record.isCudaVisible)
            break;
        const auto arch = "sm_" + std::to_string(record.computeCapabilityMajor) +
                          std::to_string(record.computeCapabilityMinor);
        trace->devices[record.id] = {record.name ? record.name : "NVIDIA GPU",
            uuid_string(record.isMigEnabled ? record.migUuid : record.uuid), arch,
            record.numMultiprocessors, record.numThreadsPerWarp, record.globalMemorySize};
        break;
    }
    case CUPTI_ACTIVITY_KIND_CONTEXT: {
        const auto &record = *reinterpret_cast<const ContextRecord *>(activity);
        trace->contexts[record.contextId] = record.deviceId;
        break;
    }
    case CUPTI_ACTIVITY_KIND_MARKER:
        marker(*reinterpret_cast<const CUpti_ActivityMarker2 *>(activity));
        break;
    default:
        break;
    }
}

void CUPTIAPI request_buffer(uint8_t **buffer, size_t *size, size_t *max_records)
{
    *buffer = static_cast<uint8_t *>(std::malloc(activity_buffer_size));
    *size = *buffer ? activity_buffer_size : 0;
    *max_records = 0;
}

void CUPTIAPI complete_buffer(CUcontext context, uint32_t stream, uint8_t *buffer,
                             size_t, size_t valid)
{
    if (getpid() == trace->pid && valid) {
        std::lock_guard<std::mutex> guard(trace->mutex);
        CUpti_Activity *record = nullptr;
        for (;;) {
            CUptiResult result = cuptiActivityGetNextRecord(buffer, valid, &record);
            if (result == CUPTI_ERROR_MAX_LIMIT_REACHED)
                break;
            if (!check(result, "read activity record"))
                break;
            collect(record);
        }
        size_t dropped = 0;
        if (cuptiActivityGetNumDroppedRecords(context, stream, &dropped) == CUPTI_SUCCESS)
            trace->dropped += dropped;
    }
    std::free(buffer);
}

void enable(CUpti_ActivityKind kind, const char *name)
{
    if (check(cuptiActivityEnable(kind), name))
        trace->enabled.push_back(kind);
}

void initialize_cupti()
{
    bool requested = false;
    for (unsigned i = Runtime; i <= Nvtx; ++i)
        requested |= enabled(static_cast<Domain>(i));
    if (!requested)
        return;
    uint32_t version = 0;
    if (!check(cuptiGetVersion(&version), "query CUPTI version"))
        return;
    if (version != CUPTI_API_VERSION) {
        std::fprintf(stderr, "RATELProf CUDA: loaded CUPTI API %u, expected %u\n",
                     version, CUPTI_API_VERSION);
        return;
    }
    if (!check(cuptiActivityRegisterTimestampCallback(timestamp), "set activity clock") ||
        !check(cuptiActivityRegisterCallbacks(request_buffer, complete_buffer), "register activity buffers"))
        return;

    trace->registered = true;
    enable(CUPTI_ACTIVITY_KIND_DEVICE, "device activity");
    enable(CUPTI_ACTIVITY_KIND_CONTEXT, "context activity");
    if (enabled(Runtime))
        enable(CUPTI_ACTIVITY_KIND_RUNTIME, "CUDA Runtime activity");
    if (enabled(Driver))
        enable(CUPTI_ACTIVITY_KIND_DRIVER, "CUDA Driver activity");
    if (enabled(Kernel)) {
        check(cuptiActivityEnableLatencyTimestamps(1), "kernel latency timestamps");
        enable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL, "kernel activity");
    }
    if (enabled(Copy)) {
        enable(CUPTI_ACTIVITY_KIND_MEMCPY, "memory copy activity");
        enable(CUPTI_ACTIVITY_KIND_MEMCPY2, "peer copy activity");
    }
    if (enabled(Memset))
        enable(CUPTI_ACTIVITY_KIND_MEMSET, "memset activity");
    if (enabled(Sync))
        enable(CUPTI_ACTIVITY_KIND_SYNCHRONIZATION, "synchronization activity");
    if (enabled(Nvtx))
        enable(CUPTI_ACTIVITY_KIND_MARKER, "NVTX marker activity");
    enable_api_arguments();
}

void field(msgpack_buffer_t &buf, const char *name, uint64_t value)
{
    msgpack_encode_string_ext(&buf, name);
    msgpack_encode_uint(&buf, value);
}

void field(msgpack_buffer_t &buf, const char *name, const char *value)
{
    msgpack_encode_string_ext(&buf, name);
    msgpack_encode_string_ext(&buf, value);
}

void dimensions(msgpack_buffer_t &buf, const char *name, const std::array<int32_t, 3> &xyz)
{
    msgpack_encode_string_ext(&buf, name);
    msgpack_encode_array(&buf, xyz.size());
    for (auto dim : xyz)
        msgpack_encode_int(&buf, dim);
}

void argument(msgpack_buffer_t &buf, const char *name, uint64_t value, const char *type)
{
    msgpack_encode_string_ext(&buf, name);
    msgpack_encode_map(&buf, 2);
    field(buf, "value", value);
    field(buf, "type", type);
}

void argument(msgpack_buffer_t &buf, const char *name, const char *value)
{
    msgpack_encode_string_ext(&buf, name);
    msgpack_encode_map(&buf, 2);
    field(buf, "value", value);
    field(buf, "type", "string");
}

uint64_t relative(uint64_t timestamp)
{
    return timestamp > trace->start ? timestamp - trace->start : 0;
}

const char *phase(uint64_t timestamp)
{
    if (!trace->main_start || timestamp < trace->main_start)
        return "CONSTRUCTOR_PHASE";
    if (trace->main_stop && timestamp > trace->main_stop)
        return "DESTRUCTOR_PHASE";
    return "MAIN_PHASE";
}

using ApiEvents = std::unordered_map<uint32_t, const Event *>;

const Event *find_api(const ApiEvents &runtime, const ApiEvents &driver, const Event &event)
{
    if (event.domain == Runtime || !event.correlation)
        return nullptr;
    if (event.domain != Driver) {
        auto call = driver.find(event.correlation);
        if (call != driver.end())
            return call->second;
    }
    auto call = runtime.find(event.runtime_correlation ? event.runtime_correlation : event.correlation);
    return call == runtime.end() ? nullptr : call->second;
}

void encode_event(msgpack_buffer_t &buf, const Event &event, const Event *api)
{
    const bool gpu = event.domain == Kernel || event.domain == Copy || event.domain == Memset;
    msgpack_encode_map(&buf, gpu ? 6 : 9 + (event.caller != 0));
    field(buf, "phase", phase(event.start));
    field(buf, "corr_id", api ? api_id(api->domain, api->correlation) : 0);
    field(buf, "start", relative(event.start));
    field(buf, "dur", event.stop - event.start);
    field(buf, "cupti_correlation", event.correlation);

    if (!gpu) {
        field(buf, "name", event.name.c_str());
        field(buf, "pid", event.pid ? event.pid : api ? api->pid : trace->pid);
        field(buf, "tid", event.tid ? event.tid : api ? api->tid : 0);
        if (event.caller)
            field(buf, "return_address", event.caller);
    }

    msgpack_encode_string_ext(&buf, "args");
    switch (event.domain) {
    case Kernel: {
        msgpack_encode_map(&buf, 12);
        field(buf, "kernel_name", event.name.c_str());
        field(buf, "gpu_id", event.device);
        field(buf, "queue_id", event.stream);
        field(buf, "context_id", event.context);
        const auto dispatch = api ? api->start : event.submitted ? event.submitted : event.start;
        field(buf, "dispatch_time", relative(std::min(dispatch, event.start)));
        dimensions(buf, "wrg", event.block);
        dimensions(buf, "grd", event.grid);
        field(buf, "grid_unit", "blocks");
        field(buf, "private_segment_size", event.local);
        field(buf, "group_segment_size", event.shared);
        field(buf, "registers_per_thread", event.registers);
        field(buf, "graph_node", event.graph_node);
        break;
    }
    case Copy:
        msgpack_encode_map(&buf, 10);
        field(buf, "src_agent", event.src);
        field(buf, "dst_agent", event.dst);
        field(buf, "src_type", uint64_t{event.src != host});
        field(buf, "dst_type", uint64_t{event.dst != host});
        field(buf, "size", event.size);
        field(buf, "engine_id", event.stream);
        field(buf, "src_cupti_kind", event.src_kind);
        field(buf, "dst_cupti_kind", event.dst_kind);
        field(buf, "context_id", event.context);
        field(buf, "graph_node", event.graph_node);
        break;
    case Sync: {
        auto device = trace->contexts.find(event.context);
        msgpack_encode_map(&buf, 5);
        argument(buf, "context_id", event.context, "uint32_t");
        argument(buf, "stream_id", event.stream, "uint32_t");
        argument(buf, "event_id", event.cuda_event, "uint32_t");
        argument(buf, "device_id", device == trace->contexts.end() ? unknown : device->second, "uint32_t");
        argument(buf, "retval", event.result, "CUDA result");
        break;
    }
    case Memset:
        msgpack_encode_map(&buf, 7);
        field(buf, "context_id", event.context);
        field(buf, "queue_id", event.stream);
        field(buf, "gpu_id", event.device);
        field(buf, "size", event.size);
        field(buf, "memory_kind", event.dst_kind);
        field(buf, "value", event.value);
        field(buf, "graph_node", event.graph_node);
        break;
    case Nvtx:
        msgpack_encode_map(&buf, 1);
        argument(buf, "domain", event.marker_domain.c_str());
        break;
    case MpiHost:
    case OmpRoutineHost:
    case OmpTargetHost:
    case OmptHost:
    case CublasHost:
    case NcclHost:
        msgpack_encode_map(&buf, event.has_host_result);
        if (event.has_host_result) {
            msgpack_encode_string_ext(&buf, "retval");
            msgpack_encode_map(&buf, 2);
            field(buf, "type", "integer");
            msgpack_encode_string_ext(&buf, "value");
            msgpack_encode_int(&buf, event.host_result);
        }
        break;
    default: {
        auto args = trace->arguments.find(api_id(event.domain, event.correlation));
        msgpack_encode_map(&buf, 1 + (args == trace->arguments.end() ? 0 : args->second.size()));
        if (args != trace->arguments.end()) {
            for (const auto &arg : args->second)
                argument(buf, arg.name, arg.value, arg.type);
        }
        argument(buf, "retval", event.result, "CUDA result");
        break;
    }
    }
}

int mpi_rank()
{
    for (const char *name : {"OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK", "MV2_COMM_WORLD_RANK",
                             "SLURM_PROCID"}) {
        if (std::strcmp(name, "SLURM_PROCID") == 0) {
            const char *tasks = std::getenv("SLURM_NTASKS");
            if (!tasks || std::strtol(tasks, nullptr, 10) < 2)
                continue;
        }
        if (const char *value = std::getenv(name)) {
            char *end = nullptr;
            long rank = std::strtol(value, &end, 10);
            if (*value && !*end && rank >= 0 && rank <= std::numeric_limits<int>::max())
                return static_cast<int>(rank);
        }
    }
    return -1;
}

std::string report_path(int rank)
{
    std::string path = trace->output;
    if (rank >= 0) {
        const auto slash = path.rfind('/');
        path.insert(slash == std::string::npos ? 0 : slash + 1, "rank_" + std::to_string(rank) + "_");
    }
    return path;
}

void write_report(const std::string &path, int rank, uint64_t stop)
{
    ApiEvents runtime, driver;
    for (const auto &event : trace->events[Runtime])
        runtime.emplace(event.correlation, &event);
    for (const auto &event : trace->events[Driver])
        driver.emplace(event.correlation, &event);

    // Some CUPTI records may precede the matching device record or have no such
    // record (for example with restricted visibility). Preserve their ordinal.
    std::map<uintptr_t, Dl_info> locations;
    for (const auto &events : trace->events) {
        for (const auto &event : events) {
            if (event.caller && !locations.count(event.caller)) {
                Dl_info info{};
                if (dladdr(reinterpret_cast<const void *>(event.caller), &info))
                    locations.emplace(event.caller, info);
            }
            if (event.device != unknown)
                trace->devices.try_emplace(event.device, Device{"NVIDIA GPU", "", "", 0, 0, 0});
            for (auto id : {event.src, event.dst}) {
                if (id < unknown)
                    trace->devices.try_emplace(id, Device{"NVIDIA GPU", "", "", 0, 0, 0});
            }
        }
    }

    // The report header stores the interned string table before the trace data.
    msgpack_buffer_t events{};
    if (msgpack_init(&events, 4096, MSGPACK_OVERFLOW_REALLOC, nullptr) != 0)
        return;
    msgpack_ext_string_init();
    size_t domain_count = 0;
    for (const auto &group : trace->events)
        domain_count += !group.empty();
    msgpack_encode_map(&events, domain_count);
    uint64_t next_id = 1;
    for (unsigned i = 0; i < DomainCount; ++i) {
        const auto &group = trace->events[i];
        if (group.empty())
            continue;
        const auto domain = static_cast<Domain>(i);
        const std::string name = std::string("RATELPROF_DOMAIN_") + domains[i];
        msgpack_encode_string_ext(&events, name.c_str());
        msgpack_encode_map(&events, group.size());
        for (const auto &event : group) {
            const bool api = domain == Runtime || domain == Driver;
            msgpack_encode_uint(&events, api ? api_id(domain, event.correlation) : next_id++);
            encode_event(events, event, find_api(runtime, driver, event));
        }
    }

    msgpack_buffer_t report{};
    // Write the temporary file beside the destination so publication is atomic.
    std::string temporary = path + ".tmp.XXXXXX";
    int fd = mkstemp(temporary.data());
    if (fd < 0) {
        std::fprintf(stderr, "RATELProf CUDA: %s: %s\n", path.c_str(), std::strerror(errno));
    } else {
        FILE *file = fdopen(fd, "wb");
        if (!file) {
            std::fprintf(stderr, "RATELProf CUDA: %s: %s\n", path.c_str(), std::strerror(errno));
            close(fd);
        } else if (msgpack_init(&report, 4096, MSGPACK_OVERFLOW_REALLOC, nullptr) != 0) {
            std::fclose(file);
        } else {
            msgpack_encode_ext(&report, MSGPACK_EXT_RATELPROF, nullptr, 0);
            msgpack_encode_array(&report, 3);
            msgpack_encode_uint(&report, RATELPROF_MAJOR);
            msgpack_encode_uint(&report, RATELPROF_MINOR);
            msgpack_encode_uint(&report, RATELPROF_PATCH);
            msgpack_encode_uint(&report, trace->epoch);
            msgpack_encode_int(&report, rank);
            msgpack_encode_array(&report, 3);
            msgpack_encode_uint(&report, relative(trace->main_start));
            msgpack_encode_uint(&report, relative(trace->main_stop ? trace->main_stop : stop));
            msgpack_encode_uint(&report, relative(stop));
            msgpack_encode_array(&report, trace->argv.size() + 1);
            msgpack_encode_int(&report, trace->result);
            for (const auto &arg : trace->argv)
                msgpack_encode_string(&report, arg.c_str());

            msgpack_encode_map(&report, trace->devices.size() + 1);
            msgpack_encode_uint(&report, host);
            msgpack_encode_uint(&report, 0);
            for (const auto &[id, device] : trace->devices) {
                msgpack_encode_uint(&report, id);
                msgpack_encode_uint(&report, uint64_t{id} + 1);
            }
            msgpack_encode_string_table_ext(&report);
            msgpack_encode_map(&report, locations.size());
            for (const auto &[address, info] : locations) {
                msgpack_encode_uint(&report, address);
                msgpack_encode_map(&report, 3);
                msgpack_encode_string(&report, "ofile");
                msgpack_encode_string(&report, info.dli_fname ? info.dli_fname : "<unknown>");
                msgpack_encode_string(&report, "sfun");
                msgpack_encode_string(&report, info.dli_sname ? info.dli_sname : "<unknown>");
                msgpack_encode_string(&report, "offset");
                msgpack_encode_uint(&report, address - reinterpret_cast<uintptr_t>(info.dli_fbase));
            }

            // Flush the header and trace buffer directly to avoid copying the trace.
            bool ok = std::fwrite(report.data, 1, report.size, file) == report.size &&
                      std::fwrite(events.data, 1, events.size, file) == events.size;
            report.size = 0;
            msgpack_encode_map(&report, trace->devices.size());
            for (const auto &[id, device] : trace->devices) {
                msgpack_encode_uint(&report, uint64_t{id} + 1);
                msgpack_encode_map(&report, 10);
                msgpack_encode_string(&report, "backend");
                msgpack_encode_string(&report, "cuda");
                msgpack_encode_string(&report, "device_type_name");
                msgpack_encode_string(&report, "GPU");
                msgpack_encode_string(&report, "node");
                msgpack_encode_uint(&report, uint64_t{id} + 1);
                msgpack_encode_string(&report, "ordinal");
                msgpack_encode_uint(&report, id);
                msgpack_encode_string(&report, "name");
                msgpack_encode_string(&report, device.name.c_str());
                msgpack_encode_string(&report, "uuid");
                msgpack_encode_string(&report, device.uuid.c_str());
                msgpack_encode_string(&report, "arch");
                msgpack_encode_string(&report, device.arch.c_str());
                msgpack_encode_string(&report, "cu_count");
                msgpack_encode_uint(&report, device.sms);
                msgpack_encode_string(&report, "wavefront_size");
                msgpack_encode_uint(&report, device.warp);
                msgpack_encode_string(&report, "global_memory");
                msgpack_encode_uint(&report, device.memory);
            }
            ok = ok && std::fwrite(report.data, 1, report.size, file) == report.size;
            if (std::fclose(file) != 0)
                ok = false;
            if (ok && std::rename(temporary.c_str(), path.c_str()) == 0) {
                std::fprintf(stderr, "RATELProf CUDA: wrote %s\n", path.c_str());
            } else {
                std::fprintf(stderr, "RATELProf CUDA: could not write %s\n", path.c_str());
            }
            msgpack_free(&report);
        }
        unlink(temporary.c_str());
    }
    msgpack_free(&events);
    msgpack_ext_string_free();
}

void finish()
{
    if (!trace || getpid() != trace->pid)
        return;
    trace->accepting_host = false;
    const uint64_t stop = timestamp();
    if (trace->registered) {
        check(cuptiActivityFlushAll(0), "flush completed activities");
        if (trace->subscriber)
            cuptiUnsubscribe(trace->subscriber);
        for (auto kind : trace->enabled)
            cuptiActivityDisable(kind);
        check(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED), "flush remaining activities");
    }

    std::lock_guard<std::mutex> guard(trace->mutex);
    const int rank = mpi_rank();
    if (trace->dropped)
        std::fprintf(stderr, "RATELProf CUDA: CUPTI dropped %llu activity records\n",
                     static_cast<unsigned long long>(trace->dropped.load()));
    const char *plugin = std::getenv("RATELPROF_CUDA_PLUGIN");
    if (plugin && std::strcmp(plugin, "stdout") == 0) {
        for (const auto &group : trace->events) {
            for (const auto &event : group) {
                std::printf("%s\t%llu\t%llu\t%s\n", domains[event.domain],
                            static_cast<unsigned long long>(relative(event.start)),
                            static_cast<unsigned long long>(event.stop - event.start), event.name.c_str());
            }
        }
    } else {
        write_report(report_path(rank), rank, stop);
    }
}

using Main = int (*)(int, char **, char **);
Main application_main;

int traced_main(int argc, char **argv, char **env)
{
    trace->main_start = timestamp();
    trace->result = application_main(argc, argv, env);
    trace->main_stop = timestamp();
    return trace->result;
}

} // namespace

uint64_t host_begin(HostDomain domain)
{
    return trace && trace->accepting_host && getpid() == trace->pid && enabled(host_domains[domain])
        ? timestamp() : 0;
}

void host_end(HostDomain domain, const char *name, uint64_t start, const void *caller,
              int64_t result, bool has_result)
{
    if (!start || !trace || !trace->accepting_host || getpid() != trace->pid)
        return;
    Event event(host_domains[domain], start, timestamp());
    event.name = name;
    event.pid = trace->pid;
    event.tid = syscall(SYS_gettid);
    event.caller = reinterpret_cast<uintptr_t>(caller);
    event.host_result = result;
    event.has_host_result = has_result;
    std::lock_guard<std::mutex> guard(trace->mutex);
    if (trace->accepting_host)
        add(std::move(event));
}

// The interposer runs before application constructors and wraps main so report
// lifecycle and exit status use the same schema as the ROCm collector.
extern "C" int __libc_start_main(Main main, int argc, char **argv, void (*init)(),
                                void (*fini)(), void (*rtld_fini)(), void *stack_end)
{
    using Start = decltype(&__libc_start_main);
    auto start = reinterpret_cast<Start>(dlsym(RTLD_NEXT, "__libc_start_main"));
    if (std::getenv("RATELPROF_CUDA_PLUGIN") && !std::getenv("RATELPROF_CUDA_OWNER")) {
        setenv("RATELPROF_CUDA_OWNER", "1", 1);
        trace = new Trace;
        trace->start = timestamp();
        trace->epoch = clock_ns(CLOCK_REALTIME);
        const char *output = std::getenv("RATELPROF_OUTPUT_FILE");
        trace->output = output && *output ? output : "report.rprof-rep";
        if (trace->output.front() != '/') {
            if (char *cwd = getcwd(nullptr, 0)) {
                trace->output = std::string(cwd) + "/" + trace->output;
                std::free(cwd);
            }
        }
        for (unsigned i = 0; i < DomainCount; ++i)
            trace->enabled_domains[i] = domain_requested(static_cast<Domain>(i));
#ifndef RATELPROF_CUDA_MPI
        if (enabled(MpiHost))
            std::fprintf(stderr, "RATELProf CUDA: rebuild with MPI headers to trace MPI calls\n");
#endif
#ifndef RATELPROF_CUDA_OMPT
        if (enabled(OmpTargetHost) || enabled(OmptHost))
            std::fprintf(stderr, "RATELProf CUDA: rebuild with omp-tools.h to trace OMPT events\n");
#endif
#ifndef RATELPROF_CUDA_CUBLAS
        if (enabled(CublasHost))
            std::fprintf(stderr, "RATELProf CUDA: cuBLAS wrappers were not built\n");
#endif
#ifndef RATELPROF_CUDA_NCCL
        if (enabled(NcclHost))
            std::fprintf(stderr, "RATELProf CUDA: NCCL wrappers were not built\n");
#endif
        for (int i = 0; i < argc; ++i)
            trace->argv.emplace_back(argv[i]);
        // Run after the profiled executable's exit handlers and destructors.
        std::atexit(finish);
        initialize_cupti();
        application_main = main;
        main = traced_main;
    }
    return start(main, argc, argv, init, fini, rtld_fini, stack_end);
}
