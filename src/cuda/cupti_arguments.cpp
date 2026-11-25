#include "cupti_arguments.h"

#include <cstddef>

namespace {

ArgumentValue trace_cuda_value(const dim3 &value)
{
    return std::array<uint64_t, 3>{value.x, value.y, value.z};
}

ArgumentValue trace_cuda_value(const cudaExtent &value)
{
    return std::array<uint64_t, 3>{value.width, value.height, value.depth};
}

ArgumentValue trace_cuda_value(const cudaPos &value)
{
    return std::array<uint64_t, 3>{value.x, value.y, value.z};
}

template<class T>
ArgumentValue trace_cuda_value(const T &value)
{
    return trace_argument_value(value);
}

template<class T>
ArgumentValue read_value(const void *address)
{
    return trace_cuda_value(*static_cast<const T *>(address));
}

struct FieldReader {
    const char *name;
    const char *type;
    size_t offset;
    ArgumentValue (*read)(const void *);
};

struct ArgumentReader {
    const FieldReader *fields = nullptr;
    size_t count = 0;
};

ArgumentReader find_reader(CUpti_CallbackDomain domain, CUpti_CallbackId callback)
{
    #include "cupti_arguments.inc"

    return {};
}

} // namespace

TraceArguments cupti_arguments(CUpti_CallbackDomain domain, CUpti_CallbackId callback,
                              const void *parameters)
{
    if (!parameters)
        return {};
    const auto reader = find_reader(domain, callback);
    const auto *data = static_cast<const unsigned char *>(parameters);
    TraceArguments result;
    result.reserve(reader.count);
    for (size_t i = 0; i < reader.count; ++i) {
        const auto &field = reader.fields[i];
        result.push_back({field.name, field.type, field.read(data + field.offset)});
    }
    return result;
}
