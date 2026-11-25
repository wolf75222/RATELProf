#include <cupti.h>
#include <cupti_pcsampling.h>
#include <cupti_version.h>

#include "../module_capture.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <dlfcn.h>
#include <unistd.h>

namespace {

constexpr size_t buffer_records = 4096;

struct SampleBuffer {
    explicit SampleBuffer(size_t reasons)
        : pcs(buffer_records), stalls(buffer_records * reasons)
    {
        data.size = sizeof(data);
        data.collectNumPcs = pcs.size();
        data.pPcData = pcs.data();
        for (size_t i = 0; i < pcs.size(); ++i) {
            pcs[i].size = sizeof(pcs[i]);
            pcs[i].stallReason = stalls.data() + i * reasons;
        }
    }

    CUpti_PCSamplingData data{};
    std::vector<CUpti_PCSamplingPCData> pcs;
    std::vector<CUpti_PCSamplingStallReason> stalls;
};

struct Source {
    std::string file;
    uint32_t line = 0;
};

struct Context {
    explicit Context(uint32_t id) : id(id) {}

    uint32_t id;
    bool enabled = false;
    std::unordered_map<uint32_t, std::string> reasons;
    std::unique_ptr<SampleBuffer> configured;
    std::unique_ptr<SampleBuffer> drain;
    // Accounting is reported by range and may repeat across buffer flushes.
    std::map<uint64_t, uint64_t> dropped;
};

struct Session {
    FILE *output = nullptr;
    CUpti_SubscriberHandle subscriber = nullptr;
    std::mutex mutex;
    std::unordered_map<CUcontext, Context> contexts;
    std::unordered_map<uint64_t, std::vector<char>> cubins;
    std::map<std::tuple<uint64_t, uint32_t, uint64_t>, Source> locations;
    std::unordered_set<char *> function_names;
    uint32_t period = 0;
    uint32_t context_count = 0;
    uint32_t enabled_count = 0;
    uint64_t records = 0;
    uint64_t samples = 0;
    uint64_t dropped = 0;
    bool failed = false;
    bool stopping = false;
};

Session *session = nullptr;
thread_local bool in_callback = false;

struct CallbackGuard {
    CallbackGuard() { in_callback = true; }
    ~CallbackGuard() { in_callback = false; }
};

bool checked(Session &s, const char *operation, CUptiResult result)
{
    if (result == CUPTI_SUCCESS)
        return true;
    const char *name = nullptr;
    cuptiGetResultString(result, &name);
    std::fprintf(stderr, "RATELProf PC sampling: %s failed (%s).\n", operation,
                 name ? name : "unknown CUPTI error");
    if (result == CUPTI_ERROR_INSUFFICIENT_PRIVILEGES)
        std::fputs("RATELProf PC sampling: GPU performance-counter access is restricted by the NVIDIA driver.\n",
                   stderr);
    s.failed = true;
    return false;
}

void json_string(FILE *file, const std::string &value)
{
    std::fputc('"', file);
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') {
            std::fputc('\\', file);
            std::fputc(c, file);
        } else if (c < 32) {
            std::fprintf(file, "\\u%04x", static_cast<unsigned>(c));
        } else {
            std::fputc(c, file);
        }
    }
    std::fputc('"', file);
}

void cache_cubin(Session &s, const CUpti_ResourceData &resource)
{
    auto module = static_cast<const CUpti_ModuleResourceData *>(resource.resourceDescriptor);
    if (!module || !module->pCubin || !module->cubinSize)
        return;
    CUpti_GetCubinCrcParams request{};
    request.size = CUpti_GetCubinCrcParamsSize;
    request.cubinSize = module->cubinSize;
    request.cubin = module->pCubin;
    if (cuptiGetCubinCrc(&request) == CUPTI_SUCCESS && !s.cubins.count(request.cubinCrc))
        s.cubins.emplace(request.cubinCrc,
                         std::vector<char>(module->pCubin, module->pCubin + module->cubinSize));
}

const Source &source_for(Session &s, const CUpti_PCSamplingPCData &pc)
{
    auto [entry, inserted] = s.locations.try_emplace({pc.cubinCrc, pc.functionIndex, pc.pcOffset});
    auto image = s.cubins.find(pc.cubinCrc);
    if (inserted && pc.functionName && image != s.cubins.end()) {
        CUpti_GetSassToSourceCorrelationParams request{};
        request.size = CUpti_GetSassToSourceCorrelationParamsSize;
        request.cubin = image->second.data();
        request.cubinSize = image->second.size();
        request.functionName = pc.functionName;
        request.pcOffset = pc.pcOffset;
        if (cuptiGetSassToSourceCorrelation(&request) == CUPTI_SUCCESS &&
            request.fileName && request.lineNumber) {
            auto &loc = entry->second;
            if (request.fileName[0] != '/' && request.dirName && request.dirName[0])
                loc.file = std::string(request.dirName) + "/";
            loc.file += request.fileName;
            loc.line = request.lineNumber;
        }
        std::free(request.fileName);
        std::free(request.dirName);
    }
    return entry->second;
}

void consume(Session &s, Context &ctx, SampleBuffer &buffer)
{
    const auto &data = buffer.data;
    if (data.droppedSamples) {
        auto &count = ctx.dropped[data.rangeId];
        count = std::max(count, data.droppedSamples);
    }
    auto *out = s.output;
    for (size_t i = 0; i < std::min(data.totalNumPcs, buffer.pcs.size()); ++i) {
        const auto &pc = buffer.pcs[i];
        if (pc.functionName)
            s.function_names.insert(pc.functionName);
        const auto &loc = source_for(s, pc);
        if (s.records++)
            std::fputc(',', out);
        std::fprintf(out, "\n{\"context\":%u,\"range\":%" PRIu64 ",\"cubin_crc\":\"0x%" PRIx64
                          "\",\"function_index\":%u,\"pc_offset\":%" PRIu64 ",\"function\":",
                     ctx.id, data.rangeId, pc.cubinCrc, pc.functionIndex, pc.pcOffset);
        json_string(out, pc.functionName ? pc.functionName : "");
        if (loc.line) {
            std::fputs(",\"source\":{\"file\":", out);
            json_string(out, loc.file);
            std::fprintf(out, ",\"line\":%u}", loc.line);
        }

        std::fputs(",\"stall_reasons\":{", out);
        bool first = true;
        for (size_t reason = 0; reason < std::min(pc.stallReasonCount, ctx.reasons.size()); ++reason) {
            const auto &stall = pc.stallReason[reason];
            if (!stall.samples)
                continue;
            if (!first)
                std::fputc(',', out);
            first = false;
            auto name = ctx.reasons.find(stall.pcSamplingStallReasonIndex);
            json_string(out, name == ctx.reasons.end() ? std::to_string(stall.pcSamplingStallReasonIndex)
                                                        : name->second);
            std::fprintf(out, ":%u", stall.samples);
            s.samples += stall.samples;
        }
        std::fputs("}}", out);
    }
}

void configure(Session &s, CUcontext handle)
{
    auto [node, inserted] = s.contexts.try_emplace(handle, s.context_count + 1);
    if (!inserted)
        return;
    auto &ctx = node->second;
    ++s.context_count;
    CUpti_PCSamplingEnableParams enable{};
    enable.size = CUpti_PCSamplingEnableParamsSize;
    enable.ctx = handle;
    if (!checked(s, "enable", cuptiPCSamplingEnable(&enable)))
        return;
    ctx.enabled = true;

    size_t reason_count = 0;
    CUpti_PCSamplingGetNumStallReasonsParams count{};
    count.size = CUpti_PCSamplingGetNumStallReasonsParamsSize;
    count.ctx = handle;
    count.numStallReasons = &reason_count;
    if (!checked(s, "get stall reason count", cuptiPCSamplingGetNumStallReasons(&count)))
        return;

    std::vector<std::array<char, CUPTI_STALL_REASON_STRING_SIZE>> names(reason_count);
    std::vector<char *> pointers;
    std::vector<uint32_t> ids(reason_count);
    for (auto &name : names)
        pointers.push_back(name.data());
    CUpti_PCSamplingGetStallReasonsParams reasons{};
    reasons.size = CUpti_PCSamplingGetStallReasonsParamsSize;
    reasons.ctx = handle;
    reasons.numStallReasons = reason_count;
    reasons.stallReasonIndex = ids.data();
    reasons.stallReasons = pointers.data();
    if (!checked(s, "get stall reasons", cuptiPCSamplingGetStallReasons(&reasons)))
        return;
    for (size_t i = 0; i < reason_count; ++i)
        ctx.reasons.emplace(ids[i], names[i].data());

    auto configured = std::make_unique<SampleBuffer>(reason_count);
    auto drain = std::make_unique<SampleBuffer>(reason_count);
    std::array<CUpti_PCSamplingConfigurationInfo, 3> attributes{};
    attributes[0].attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_COLLECTION_MODE;
    attributes[0].attributeData.collectionModeData.collectionMode = CUPTI_PC_SAMPLING_COLLECTION_MODE_CONTINUOUS;
    attributes[1].attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_DATA_BUFFER;
    attributes[1].attributeData.samplingDataBufferData.samplingDataBuffer = &configured->data;
    attributes[2].attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_PERIOD;
    attributes[2].attributeData.samplingPeriodData.samplingPeriod = s.period;
    CUpti_PCSamplingConfigurationInfoParams cfg{};
    cfg.size = CUpti_PCSamplingConfigurationInfoParamsSize;
    cfg.ctx = handle;
    cfg.numAttributes = s.period ? attributes.size() : attributes.size() - 1;
    cfg.pPCSamplingConfigurationInfo = attributes.data();
    // Keep the configured buffer alive until disable, including on error.
    ctx.configured = std::move(configured);
    if (checked(s, "configure", cuptiPCSamplingSetConfigurationAttribute(&cfg))) {
        ctx.drain = std::move(drain);
        ++s.enabled_count;
    }
}

void flush(Session &s, CUcontext handle, Context &ctx)
{
    if (!ctx.drain || s.failed ||
        !checked(s, "sampling status", cuptiGetLastError()))
        return;

    auto &data = ctx.drain->data;
    CUpti_PCSamplingGetDataParams request{};
    request.size = CUpti_PCSamplingGetDataParamsSize;
    request.ctx = handle;
    request.pcSamplingData = &data;
    do {
        if (!checked(s, "read samples", cuptiPCSamplingGetData(&request)))
            return;
        consume(s, ctx, *ctx.drain);
        if (data.remainingNumPcs && !data.totalNumPcs) {
            std::fputs("RATELProf PC sampling: CUPTI reports remaining records but did not return data.\n", stderr);
            s.failed = true;
            return;
        }
    } while (data.remainingNumPcs);
}

void close_context(Session &s, CUcontext handle, Context &ctx)
{
    if (!ctx.enabled)
        return;
    flush(s, handle, ctx);
    CUpti_PCSamplingDisableParams disable{};
    disable.size = CUpti_PCSamplingDisableParamsSize;
    disable.ctx = handle;
    const auto result = cuptiPCSamplingDisable(&disable);
    if (!s.failed && checked(s, "disable", result) && ctx.configured) {
        consume(s, ctx, *ctx.configured);
        if (ctx.configured->data.remainingNumPcs) {
            std::fputs("RATELProf PC sampling: unflushed PC records remain after disable.\n", stderr);
            s.failed = true;
        }
    }
    ctx.enabled = false;
    for (const auto &[range, count] : ctx.dropped) {
        (void)range;
        s.dropped += count;
    }
}

void CUPTIAPI on_callback(void *userdata, CUpti_CallbackDomain domain, CUpti_CallbackId callback,
                          const void *data)
{
    if (in_callback)
        return;
    CallbackGuard guard;
    auto &s = *static_cast<Session *>(userdata);
    std::lock_guard<std::mutex> lock(s.mutex);
    if (s.stopping)
        return;
    try {
        if (domain == CUPTI_CB_DOMAIN_DRIVER_API) {
            const auto &api = *static_cast<const CUpti_CallbackData *>(data);
            if (api.callbackSite != CUPTI_API_EXIT)
                return;
            auto ctx = s.contexts.find(api.context);
            if (ctx != s.contexts.end() && ctx->second.drain) {
                const auto &pending = ctx->second.configured->data;
                if (pending.totalNumPcs || pending.remainingNumPcs >= buffer_records)
                    flush(s, api.context, ctx->second);
            }
            return;
        }

        const auto &resource = *static_cast<const CUpti_ResourceData *>(data);
        if (callback == CUPTI_CBID_RESOURCE_MODULE_LOADED)
            capture_cuda_module(data);
        if (callback == CUPTI_CBID_RESOURCE_CONTEXT_CREATED) {
            configure(s, resource.context);
        } else {
            auto ctx = s.contexts.find(resource.context);
            if (ctx == s.contexts.end())
                return;
            if (callback == CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING) {
                close_context(s, resource.context, ctx->second);
                s.contexts.erase(ctx);
            } else if (callback == CUPTI_CBID_RESOURCE_MODULE_LOADED && ctx->second.drain) {
                cache_cubin(s, resource);
                const auto &pending = ctx->second.configured->data;
                if (pending.totalNumPcs || pending.remainingNumPcs)
                    flush(s, resource.context, ctx->second);
            }
        }
    } catch (const std::exception &error) {
        std::fprintf(stderr, "RATELProf PC sampling: collector failed (%s).\n", error.what());
        s.failed = true;
    }
}

void finish()
{
    if (!session)
        return;
    auto &s = *session;
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        if (s.stopping)
            return;
        s.stopping = true;
        CallbackGuard guard;
        // Driver counter permissions can be reported asynchronously after launch.
        checked(s, "sampling status", cuptiGetLastError());
        for (auto &[context, state] : s.contexts)
            close_context(s, context, state);

        if (!s.enabled_count)
            std::fputs("RATELProf PC sampling: no CUDA context could be sampled.\n", stderr);
        if (s.dropped)
            std::fprintf(stderr, "RATELProf PC sampling: CUPTI reported %" PRIu64 " dropped samples.\n",
                         s.dropped);
        std::fprintf(s.output, "\n],\"contexts\":%u,\"records\":%" PRIu64 ",\"stall_samples\":%" PRIu64
                               ",\"dropped_samples\":%" PRIu64 ",\"complete\":%s}\n",
                     s.enabled_count, s.records, s.samples, s.dropped,
                     !s.failed && s.enabled_count ? "true" : "false");
        if (std::fclose(s.output) != 0)
            std::fputs("RATELProf PC sampling: writing results failed.\n", stderr);
        finish_module_capture();
    }
    if (s.subscriber && cuptiUnsubscribe(s.subscriber) == CUPTI_SUCCESS) {
        for (auto name : s.function_names)
            std::free(name);
        delete session;
        session = nullptr;
    }
}

using Main = int (*)(int, char **, char **);
Main application_main = nullptr;

int sampled_main(int argc, char **argv, char **env)
{
    const int status = application_main(argc, argv, env);
    // The CUDA runtime can destroy its primary context during atexit.
    // Ongoing contexts must be drained before that teardown.
    finish();
    return status;
}

} // namespace

extern "C" int __libc_start_main(Main main, int argc, char **argv, void (*init)(),
                                 void (*fini)(), void (*rtld_fini)(), void *stack_end)
{
    using Start = decltype(&__libc_start_main);
    auto start = reinterpret_cast<Start>(dlsym(RTLD_NEXT, "__libc_start_main"));
    application_main = main;
    return start(sampled_main, argc, argv, init, fini, rtld_fini, stack_end);
}

extern "C" int InitializeInjection()
{
    static std::mutex init_mutex;
    std::lock_guard<std::mutex> lock(init_mutex);
    if (session)
        return 1;
    const char *owner = std::getenv("RATELPROF_PCSAMPLING_PID");
    const char *path = std::getenv("RATELPROF_PCSAMPLING_OUTPUT");
    if (!owner || std::strtol(owner, nullptr, 10) != getpid() || !path)
        return 1;

    auto s = std::make_unique<Session>();
    const char *period = std::getenv("RATELPROF_PCSAMPLING_PERIOD");
    if (period && *period) {
        char *end = nullptr;
        const long value = std::strtol(period, &end, 10);
        if (*end || value < 5 || value > 31)
            return 0;
        s->period = static_cast<uint32_t>(value);
    }

    uint32_t version = 0;
    if (!checked(*s, "query CUPTI version", cuptiGetVersion(&version)))
        return 0;
    if (version != CUPTI_API_VERSION) {
        std::fprintf(stderr, "RATELProf PC sampling: loaded CUPTI API %u, expected %u.\n",
                     version, CUPTI_API_VERSION);
        return 0;
    }
    s->output = std::fopen(path, "w");
    if (!s->output) {
        std::perror("RATELProf PC sampling: open report");
        return 0;
    }
    std::fputs("{\"schema\":\"ratelprof-pcsampling-v1\",\"mode\":\"continuous\",\"period_exponent\":",
               s->output);
    if (s->period)
        std::fprintf(s->output, "%u", s->period);
    else
        std::fputs("null", s->output);
    std::fputs(",\"samples\":[", s->output);

    init_module_capture(std::getenv("RATELPROF_DUMP_CUBINS"));
    session = s.release();
    std::atexit(finish);
    if (!checked(*session, "subscribe", cuptiSubscribe(&session->subscriber, on_callback, session)))
        return 1;
    for (auto cb : {CUPTI_CBID_RESOURCE_CONTEXT_CREATED, CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING,
                    CUPTI_CBID_RESOURCE_MODULE_LOADED})
        checked(*session, "subscribe to resource callbacks",
                cuptiEnableCallback(1, session->subscriber, CUPTI_CB_DOMAIN_RESOURCE, cb));
    for (auto cb : {CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz,
                    CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz,
                    CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch, CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz,
                    CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel,
                    CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel_ptsz})
        checked(*session, "subscribe to launch callbacks",
                cuptiEnableCallback(1, session->subscriber, CUPTI_CB_DOMAIN_DRIVER_API, cb));
    return 1;
}
