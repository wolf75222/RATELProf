#include "module_capture.h"

#include <cupti.h>
#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>

namespace {

int output_fd = -1;
pid_t owner;
char host[128] = "unknown";
std::string output_dir;
std::atomic<unsigned> sequence{0};
std::atomic<unsigned> emitted{0};

bool write_image(int fd, const char *bytes, size_t size)
{
    while (size > 0) {
        const size_t chunk = size < (1U << 20) ? size : (1U << 20);
        const ssize_t count = write(fd, bytes, chunk);
        if (count < 0 && errno == EINTR)
            continue;
        if (count <= 0) {
            if (!count)
                errno = EIO;
            return false;
        }
        bytes += count;
        size -= static_cast<size_t>(count);
    }
    return true;
}

} // namespace

bool init_module_capture(const char *directory)
{
    if (!directory || !*directory)
        return false;

    std::error_code error;
    const auto path = std::filesystem::absolute(directory, error);
    if (!error)
        std::filesystem::create_directories(path, error);
    if (error) {
        std::fprintf(stderr, "RATELProf: cannot create CUDA module directory '%s': %s\n",
                     directory, error.message().c_str());
        return false;
    }

    output_dir = path.lexically_normal().string();
    output_fd = open(output_dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (output_fd < 0) {
        std::fprintf(stderr, "RATELProf: cannot open CUDA module directory '%s': %s\n",
                     output_dir.c_str(), std::strerror(errno));
        return false;
    }

    owner = getpid();
    if (gethostname(host, sizeof(host) - 1) != 0)
        std::strcpy(host, "unknown");
    for (char *c = host; *c; ++c) {
        if (*c == '/')
            *c = '_';
    }
    return true;
}

void capture_cuda_module(const void *callback_data)
{
    if (output_fd < 0 || getpid() != owner || !callback_data)
        return;
    const auto *resource = static_cast<const CUpti_ResourceData *>(callback_data);
    const auto *module = static_cast<const CUpti_ModuleResourceData *>(resource->resourceDescriptor);
    if (!module || !module->pCubin || !module->cubinSize)
        return;

    char filename[208];
    int fd;
    do {
        const unsigned serial = sequence.fetch_add(1, std::memory_order_relaxed);
        std::snprintf(filename, sizeof(filename), "%s-pid%ld-module%u-%u.cubin",
                      host, static_cast<long>(owner), module->moduleId, serial);
        fd = openat(output_fd, filename, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    } while (fd < 0 && errno == EEXIST);

    if (fd < 0) {
        std::fprintf(stderr, "RATELProf: cannot create CUDA module '%s': %s\n",
                     filename, std::strerror(errno));
        return;
    }

    int error = write_image(fd, module->pCubin, module->cubinSize) ? 0 : errno;
    if (close(fd) != 0 && !error)
        error = errno;
    if (error) {
        unlinkat(output_fd, filename, 0);
        std::fprintf(stderr, "RATELProf: cannot write CUDA module '%s': %s\n",
                     filename, std::strerror(error));
        return;
    }
    emitted.fetch_add(1, std::memory_order_relaxed);
}

void finish_module_capture()
{
    if (output_fd < 0 || getpid() != owner)
        return;
    close(output_fd);
    output_fd = -1;
    std::fprintf(stderr, "RATELProf: wrote %u CUDA module image(s) to '%s'\n",
                 emitted.load(std::memory_order_relaxed), output_dir.c_str());
}
