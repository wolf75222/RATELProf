#include "output_plugin.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <dlfcn.h>
#include <mutex>
#include <thread>
#include <utility>

namespace {

using ProcessActivity = decltype(&ratelprof_cuda_plugin_process_activity_v1);
using ProcessReport = decltype(&ratelprof_cuda_plugin_process_report_v1);
constexpr char activity_symbol[] = "ratelprof_cuda_plugin_process_activity_v1";
constexpr char report_symbol[] = "ratelprof_cuda_plugin_process_report_v1";
constexpr std::size_t activity_queue_capacity = 4096;
constexpr auto flush_interval = std::chrono::milliseconds(250);

thread_local bool in_callback = false;

struct CallbackScope {
    CallbackScope() : previous(in_callback) { in_callback = true; }
    ~CallbackScope() { in_callback = previous; }
    bool previous;
};

void *open_library(const char *path)
{
    CallbackScope scope;
    void *handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!handle)
        std::fprintf(stderr, "RATELProf CUDA: cannot load output plugin %s: %s\n",
                     path, dlerror());
    return handle;
}

template<class Callback, class Record>
bool invoke(Callback process, const char *path, const char *kind, const Record &record,
            bool report_failure)
{
    CallbackScope scope;
    try {
        const int status = process(&record);
        if (status && report_failure)
            std::fprintf(stderr, "RATELProf CUDA: plugin %s %s callback returned %d\n",
                         path, kind, status);
        return status == 0;
    } catch (...) {
        if (report_failure)
            std::fprintf(stderr, "RATELProf CUDA: plugin %s threw from its %s callback\n",
                         path, kind);
        return false;
    }
}

void process_report(ProcessReport process, const char *library, const char *path,
                    std::int64_t rank, std::int64_t pid, std::int64_t exit_code)
{
    if (process) {
        const ratelprof_cuda_report_v1 report{1, sizeof(ratelprof_cuda_report_v1),
                                             path, rank, pid, exit_code};
        invoke(process, library, "report", report, true);
    }
}

ratelprof_cuda_argument_v1 argument_view(const TraceArgument &arg)
{
    ratelprof_cuda_argument_v1 view{};
    view.name = arg.name;
    view.type = arg.type.c_str();
    std::visit([&](const auto &value) {
        using Type = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Type, uint64_t>) {
            view.value_kind = RATELPROF_CUDA_VALUE_U64;
            view.value.u64 = value;
        } else if constexpr (std::is_same_v<Type, int64_t>) {
            view.value_kind = RATELPROF_CUDA_VALUE_I64;
            view.value.i64 = value;
        } else if constexpr (std::is_same_v<Type, double>) {
            view.value_kind = RATELPROF_CUDA_VALUE_F64;
            view.value.f64 = value;
        } else if constexpr (std::is_same_v<Type, bool>) {
            view.value_kind = RATELPROF_CUDA_VALUE_BOOL;
            view.value.boolean = value;
        } else if constexpr (std::is_same_v<Type, std::string>) {
            view.value_kind = RATELPROF_CUDA_VALUE_STRING;
            view.value.string = value.c_str();
        } else {
            view.value_kind = RATELPROF_CUDA_VALUE_DIM3;
            for (std::size_t i = 0; i < value.size(); ++i)
                view.value.dim3[i] = value[i];
        }
    }, arg.value);
    return view;
}

} // namespace

struct CudaActivityPlugin::Instance {
    const std::string path;
    void *handle;
    ProcessActivity const activity;
    ProcessReport const report;
    std::thread worker;
    std::mutex mutex;
    std::mutex flush_mutex;
    bool (*periodic_flush)() = nullptr;
    std::condition_variable ready;
    std::deque<PluginActivity> queue;
    std::uint64_t dropped = 0, failures = 0;
    bool stopping = false;
    std::atomic<bool> accepting{false};

    Instance(const char *path, void *handle)
        : path(path), handle(handle),
          activity(reinterpret_cast<ProcessActivity>(dlsym(handle, activity_symbol))),
          report(reinterpret_cast<ProcessReport>(dlsym(handle, report_symbol)))
    {}

    ~Instance()
    {
        CallbackScope scope;
        if (handle)
            dlclose(handle);
    }

    void run()
    {
        using Clock = std::chrono::steady_clock;
        auto next_flush = Clock::now() + flush_interval;
        for (;;) {
            PluginActivity packet;
            bool have_packet = false;
            {
                std::unique_lock<std::mutex> lock(mutex);
                ready.wait_until(lock, next_flush, [&] { return stopping || !queue.empty(); });
                if (queue.empty() && stopping)
                    return;
                if (!queue.empty()) {
                    packet = std::move(queue.front());
                    queue.pop_front();
                    have_packet = true;
                }
            }
            if (Clock::now() >= next_flush) {
                std::lock_guard<std::mutex> guard(flush_mutex);
                if (periodic_flush && !periodic_flush())
                    periodic_flush = nullptr;
                next_flush = Clock::now() + flush_interval;
            }
            if (!have_packet)
                continue;
            try {
                std::vector<ratelprof_cuda_argument_v1> arguments;
                arguments.reserve(packet.arguments.size());
                for (const auto &argument : packet.arguments)
                    arguments.push_back(argument_view(argument));
                packet.activity.name = packet.name.c_str();
                packet.activity.annotation_domain = packet.annotation_domain.c_str();
                packet.activity.arguments = arguments.empty() ? nullptr : arguments.data();
                packet.activity.argument_count = arguments.size();
                if (!invoke(activity, path.c_str(), "activity", packet.activity, failures == 0))
                    ++failures;
            } catch (const std::bad_alloc &) {
                std::lock_guard<std::mutex> lock(mutex);
                ++dropped;
            }
        }
    }
};

CudaActivityPlugin::CudaActivityPlugin() = default;

CudaActivityPlugin::~CudaActivityPlugin()
{
    close();
}

void CudaActivityPlugin::open()
{
    const char *path = std::getenv("RATELPROF_CUDA_PLUGIN");
    if (!path || !*path)
        return;
    void *handle = open_library(path);
    if (!handle)
        return;
    try {
        instance_ = std::make_unique<Instance>(path, handle);
    } catch (const std::bad_alloc &) {
        std::fprintf(stderr, "RATELProf CUDA: cannot allocate activity queue for %s\n", path);
        CallbackScope scope;
        dlclose(handle);
        return;
    }
    if (!instance_->activity) {
        std::fprintf(stderr, "RATELProf CUDA: plugin %s does not export %s\n", path, activity_symbol);
        return;
    }
    try {
        instance_->worker = std::thread([instance = instance_.get()] { instance->run(); });
        instance_->accepting.store(true, std::memory_order_relaxed);
    } catch (const std::exception &error) {
        std::fprintf(stderr, "RATELProf CUDA: cannot start activity worker for %s: %s\n",
                     path, error.what());
    }
}

bool CudaActivityPlugin::captures_activity() const
{
    return instance_ && instance_->accepting.load(std::memory_order_relaxed);
}

void CudaActivityPlugin::set_periodic_flush(bool (*flush)())
{
    if (instance_) {
        std::lock_guard<std::mutex> guard(instance_->flush_mutex);
        instance_->periodic_flush = flush;
    }
}

void CudaActivityPlugin::enqueue(PluginActivity activity)
{
    auto &instance = *instance_;
    {
        std::lock_guard<std::mutex> guard(instance.mutex);
        if (instance.stopping)
            return;
        if (instance.queue.size() >= activity_queue_capacity) {
            ++instance.dropped;
            return;
        }
        instance.queue.push_back(std::move(activity));
    }
    instance.ready.notify_one();
}

void CudaActivityPlugin::count_dropped_activity()
{
    std::lock_guard<std::mutex> guard(instance_->mutex);
    if (!instance_->stopping)
        ++instance_->dropped;
}

void CudaActivityPlugin::drain()
{
    if (!instance_ || !instance_->worker.joinable())
        return;
    auto &instance = *instance_;
    instance.accepting.store(false, std::memory_order_relaxed);
    set_periodic_flush(nullptr);
    std::uint64_t dropped;
    {
        std::lock_guard<std::mutex> guard(instance.mutex);
        instance.stopping = true;
    }
    instance.ready.notify_one();
    instance.worker.join();
    {
        std::lock_guard<std::mutex> guard(instance.mutex);
        dropped = instance.dropped;
    }
    if (dropped)
        std::fprintf(stderr, "RATELProf CUDA: plugin %s dropped %llu activity records\n",
                     instance.path.c_str(), static_cast<unsigned long long>(dropped));
    if (instance.failures > 1)
        std::fprintf(stderr, "RATELProf CUDA: plugin %s reported %llu activity callback failures\n",
                     instance.path.c_str(), static_cast<unsigned long long>(instance.failures));
}

void CudaActivityPlugin::close()
{
    drain();
    if (instance_ && instance_->handle) {
        CallbackScope scope;
        dlclose(std::exchange(instance_->handle, nullptr));
    }
}

void CudaActivityPlugin::process_report(const char *path, std::int64_t rank,
                                       std::int64_t pid, std::int64_t exit_code)
{
    if (instance_ && instance_->handle)
        ::process_report(instance_->report, instance_->path.c_str(), path, rank, pid, exit_code);
}

bool ratelprof_cuda_in_plugin_callback()
{
    return in_callback;
}

void ratelprof_cuda_run_output_plugin(const char *report_path, std::int64_t rank,
                                    std::int64_t pid, std::int64_t exit_code)
{
    const char *library = std::getenv("RATELPROF_CUDA_PLUGIN");
    if (!library || !*library || std::strcmp(library, "msgpack") == 0 ||
        std::strcmp(library, "stdout") == 0)
        return;
    void *handle = open_library(library);
    if (!handle)
        return;
    auto process = reinterpret_cast<ProcessReport>(dlsym(handle, report_symbol));
    if (!process)
        std::fprintf(stderr, "RATELProf CUDA: plugin %s does not export %s\n", library, report_symbol);
    ::process_report(process, library, report_path, rank, pid, exit_code);
    CallbackScope scope;
    dlclose(handle);
}
