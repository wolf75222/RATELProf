#include <cupti.h>
#include <cupti_version.h>
#ifdef RATELPROF_HAVE_CUBIN_CRC
#include <cupti_pcsampling.h>
#endif

#include "../module_capture.h"

#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <map>
#include <mutex>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace {

struct Function {
    uint32_t context, module, index;
    std::string name;
};

struct Location {
    std::string file;
    uint32_t line;
};

struct Session {
    std::mutex mutex;
    std::vector<CUpti_ActivityPCSampling3> samples;
    std::unordered_map<uint32_t, Function> functions;
    std::unordered_map<uint32_t, Location> locations;
    std::map<std::pair<uint32_t, uint32_t>, uint64_t> cubins;
    uint64_t dropped_samples = 0, dropped_records = 0;
    uint32_t contexts = 0, period = 0;
    bool registered = false, dump = false;
    std::atomic<bool> failed{false}, finished{false};
    CUpti_SubscriberHandle subscriber = nullptr;
    std::string path;
};

Session *session = nullptr;
pid_t owner;

bool checked(CUptiResult status, const char *operation)
{
    if (status == CUPTI_SUCCESS)
        return true;
    const char *name = nullptr;
    cuptiGetResultString(status, &name);
    std::fprintf(stderr, "RATELProf legacy PC sampling: %s failed (%s).\n", operation,
                 name ? name : "unknown CUPTI error");
    session->failed = true;
    return false;
}

void json_string(FILE *out, const std::string &value)
{
    std::fputc('"', out);
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') {
            std::fputc('\\', out);
            std::fputc(c, out);
        } else if (c < 32) {
            std::fprintf(out, "\\u%04x", static_cast<unsigned>(c));
        } else {
            std::fputc(c, out);
        }
    }
    std::fputc('"', out);
}

std::string stall_reason(unsigned reason)
{
    static const char *names[] = {
        "invalid", "selected", "instruction_fetch", "execution_dependency",
        "memory_dependency", "texture", "synchronization", "constant_memory_dependency",
        "pipeline_busy", "memory_throttle", "not_selected", "other", "sleeping",
    };
    return reason < sizeof(names) / sizeof(*names) ? names[reason] : std::to_string(reason);
}

void collect(const CUpti_Activity *activity)
{
    auto &s = *session;
    switch (activity->kind) {
    case CUPTI_ACTIVITY_KIND_PC_SAMPLING:
        s.samples.push_back(*reinterpret_cast<const CUpti_ActivityPCSampling3 *>(activity));
        break;
    case CUPTI_ACTIVITY_KIND_PC_SAMPLING_RECORD_INFO:
        s.dropped_samples += reinterpret_cast<const CUpti_ActivityPCSamplingRecordInfo *>(activity)->droppedSamples;
        break;
    case CUPTI_ACTIVITY_KIND_FUNCTION: {
        const auto &fn = *reinterpret_cast<const CUpti_ActivityFunction *>(activity);
        s.functions.insert_or_assign(fn.id, Function{fn.contextId, fn.moduleId, fn.functionIndex, fn.name ? fn.name : ""});
        break;
    }
    case CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR: {
        const auto &loc = *reinterpret_cast<const CUpti_ActivitySourceLocator *>(activity);
        if (loc.fileName)
            s.locations.insert_or_assign(loc.id, Location{loc.fileName, loc.lineNumber});
        break;
    }
    default:
        break;
    }
}

void CUPTIAPI request_buffer(uint8_t **buffer, size_t *size, size_t *records)
{
    constexpr size_t capacity = 1 << 20;
    *buffer = static_cast<uint8_t *>(std::malloc(capacity));
    *size = *buffer ? capacity : 0;
    *records = 0;
}

void CUPTIAPI complete_buffer(CUcontext ctx, uint32_t stream, uint8_t *buffer, size_t, size_t valid)
{
    if (owner == getpid() && valid) {
        std::lock_guard<std::mutex> lock(session->mutex);
        CUpti_Activity *record = nullptr;
        for (;;) {
            const auto status = cuptiActivityGetNextRecord(buffer, valid, &record);
            if (status == CUPTI_ERROR_MAX_LIMIT_REACHED || !checked(status, "read activity records"))
                break;
            collect(record);
        }
        size_t dropped = 0;
        if (checked(cuptiActivityGetNumDroppedRecords(ctx, stream, &dropped), "read dropped activity count"))
            session->dropped_records += dropped;
    }
    std::free(buffer);
}

void CUPTIAPI resource_callback(void *, CUpti_CallbackDomain, CUpti_CallbackId id, const void *data)
{
    if (owner != getpid() || !data)
        return;
    const auto &resource = *static_cast<const CUpti_ResourceData *>(data);
    if (id == CUPTI_CBID_RESOURCE_CONTEXT_CREATED) {
        CUpti_ActivityPCSamplingConfig config{};
        config.size = sizeof(config);
        config.samplingPeriod = CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_MID;
        config.samplingPeriod2 = session->period;
        if (checked(cuptiActivityConfigurePCSampling(resource.context, &config), "configure PC sampling") &&
            checked(cuptiActivityEnableContext(resource.context, CUPTI_ACTIVITY_KIND_PC_SAMPLING), "enable PC sampling")) {
            std::lock_guard<std::mutex> lock(session->mutex);
            ++session->contexts;
        }
    } else if (id == CUPTI_CBID_RESOURCE_MODULE_LOADED) {
        if (session->dump)
            capture_cuda_module(data);
#ifdef RATELPROF_HAVE_CUBIN_CRC
        const auto *module = static_cast<const CUpti_ModuleResourceData *>(resource.resourceDescriptor);
        if (module && module->pCubin && module->cubinSize) {
            uint32_t context_id = 0;
            CUpti_GetCubinCrcParams request{};
            request.size = CUpti_GetCubinCrcParamsSize;
            request.cubin = module->pCubin;
            request.cubinSize = module->cubinSize;
            if (cuptiGetContextId(resource.context, &context_id) == CUPTI_SUCCESS &&
                cuptiGetCubinCrc(&request) == CUPTI_SUCCESS) {
                std::lock_guard<std::mutex> lock(session->mutex);
                session->cubins[{context_id, module->moduleId}] = request.cubinCrc;
            }
        }
#endif
    }
}

void write_report(const Session &s)
{
    FILE *out = std::fopen(s.path.c_str(), "w");
    if (!out) {
        std::perror("RATELProf legacy PC sampling: open report");
        return;
    }
    std::fputs("{\"schema\":\"ratelprof-pcsampling-v1\",\"mode\":\"legacy-serialized\",\"period_exponent\":", out);
    if (s.period)
        std::fprintf(out, "%u", s.period);
    else
        std::fputs("null", out);
    std::fputs(",\"samples\":[", out);
    uint64_t total = 0;
    bool first = true;
    for (const auto &pc : s.samples) {
        const auto found = s.functions.find(pc.functionId);
        const Function *fn = found == s.functions.end() ? nullptr : &found->second;
        const auto crc = fn ? s.cubins.find({fn->context, fn->module}) : s.cubins.end();
        if (!first)
            std::fputc(',', out);
        first = false;
        std::fprintf(out, "\n{\"context\":%u,\"range\":%u,\"cubin_crc\":\"",
                     fn ? fn->context : 0, pc.correlationId);
        if (crc != s.cubins.end())
            std::fprintf(out, "0x%" PRIx64, crc->second);
        std::fprintf(out, "\",\"function_index\":%u,\"pc_offset\":%" PRIu64 ",\"function\":",
                     fn ? fn->index : 0, static_cast<uint64_t>(pc.pcOffset));
        json_string(out, fn ? fn->name : "");
        const auto loc = s.locations.find(pc.sourceLocatorId);
        if (loc != s.locations.end()) {
            std::fputs(",\"source\":{\"file\":", out);
            json_string(out, loc->second.file);
            std::fprintf(out, ",\"line\":%u}", loc->second.line);
        }
        std::fputs(",\"stall_reasons\":{", out);
        json_string(out, stall_reason(pc.stallReason));
        std::fprintf(out, ":%u},\"latency_samples\":%u}", pc.samples, pc.latencySamples);
        total += pc.samples;
    }
    std::fprintf(out, "\n],\"contexts\":%u,\"records\":%zu,\"stall_samples\":%" PRIu64
                      ",\"dropped_samples\":%" PRIu64 ",\"dropped_activity_records\":%" PRIu64
                      ",\"complete\":%s}\n",
                 s.contexts, s.samples.size(), total, s.dropped_samples, s.dropped_records,
                 s.contexts && !s.failed && !s.dropped_records ? "true" : "false");
    if (std::fclose(out) != 0)
        std::perror("RATELProf legacy PC sampling: close report");
}

void finish()
{
    if (!session || owner != getpid())
        return;
    auto &s = *session;
    if (s.finished.exchange(true))
        return;
    if (s.registered)
        checked(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED), "flush activity records");
    if (s.subscriber)
        checked(cuptiUnsubscribe(s.subscriber), "unsubscribe");
    if (s.dump)
        finish_module_capture();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!s.contexts)
        std::fputs("RATELProf legacy PC sampling: no CUDA context could be sampled.\n", stderr);
    if (s.dropped_records)
        std::fprintf(stderr, "RATELProf legacy PC sampling: CUPTI dropped %" PRIu64 " activity records.\n",
                     s.dropped_records);
    write_report(s);
}

using Main = int (*)(int, char **, char **);
Main app_main;

int sampled_main(int argc, char **argv, char **env)
{
    const int status = app_main(argc, argv, env);
    finish();
    return status;
}

} // namespace

extern "C" int InitializeInjection()
{
    if (session)
        return 1;
    const char *pid = std::getenv("RATELPROF_PCSAMPLING_PID");
    const char *path = std::getenv("RATELPROF_PCSAMPLING_OUTPUT");
    if (!pid || std::strtol(pid, nullptr, 10) != getpid() || !path)
        return 1;
    session = new Session;
    owner = getpid();
    auto &s = *session;
    s.path = path;
    std::atexit(finish);
    if (const char *period = std::getenv("RATELPROF_PCSAMPLING_PERIOD"))
        s.period = static_cast<uint32_t>(std::strtoul(period, nullptr, 10));
    uint32_t version = 0;
    if (!checked(cuptiGetVersion(&version), "query CUPTI version"))
        return 1;
    if (version != CUPTI_API_VERSION) {
        std::fprintf(stderr, "RATELProf legacy PC sampling: loaded CUPTI API %u, expected %u.\n",
                     version, CUPTI_API_VERSION);
        s.failed = true;
        return 1;
    }
    s.dump = init_module_capture(std::getenv("RATELPROF_DUMP_CUBINS"));
    s.registered = checked(cuptiActivityRegisterCallbacks(request_buffer, complete_buffer), "register activity buffers");
    if (s.registered && checked(cuptiSubscribe(&s.subscriber, resource_callback, nullptr), "subscribe")) {
        checked(cuptiEnableCallback(1, s.subscriber, CUPTI_CB_DOMAIN_RESOURCE,
                                    CUPTI_CBID_RESOURCE_CONTEXT_CREATED), "subscribe to context creation");
        checked(cuptiEnableCallback(1, s.subscriber, CUPTI_CB_DOMAIN_RESOURCE,
                                    CUPTI_CBID_RESOURCE_MODULE_LOADED), "subscribe to module loads");
    }
    return 1;
}

extern "C" int __libc_start_main(Main main, int argc, char **argv, void (*init)(),
                                void (*fini)(), void (*rtld_fini)(), void *stack_end)
{
    auto start = reinterpret_cast<decltype(&__libc_start_main)>(dlsym(RTLD_NEXT, "__libc_start_main"));
    if (!start)
        return EXIT_FAILURE;
    InitializeInjection();
    if (session) {
        app_main = main;
        main = sampled_main;
    }
    return start(main, argc, argv, init, fini, rtld_fini, stack_end);
}
