#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

using ArgumentValue = std::variant<uint64_t, int64_t, double, bool, std::string,
                                   std::array<uint64_t, 3>>;

struct TraceArgument {
    const char *name;
    std::string type;
    ArgumentValue value;
};

using TraceArguments = std::vector<TraceArgument>;

template<class T>
ArgumentValue trace_argument_value(const T &value)
{
    constexpr uint64_t json_exact = (1ULL << 53) - 1;

    if constexpr (std::is_pointer_v<T>) {
        return trace_argument_value(reinterpret_cast<uintptr_t>(value));
    } else if constexpr (std::is_enum_v<T>) {
        return trace_argument_value(static_cast<std::underlying_type_t<T>>(value));
    } else if constexpr (std::is_same_v<T, bool>) {
        return value;
    } else if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
        const int64_t number = static_cast<int64_t>(value);
        if (number > static_cast<int64_t>(json_exact) || number < -static_cast<int64_t>(json_exact))
            return std::to_string(number);
        return number;
    } else if constexpr (std::is_integral_v<T>) {
        const uint64_t number = static_cast<uint64_t>(value);
        if (number > json_exact)
            return std::to_string(number);
        return number;
    } else if constexpr (std::is_floating_point_v<T>) {
        return static_cast<double>(value);
    } else {
        // A by-value type with no decoder has no safe portable interpretation.
        return "opaque value (" + std::to_string(sizeof(value)) + " bytes)";
    }
}
