local convert   = require ("utils.convert")

local report_helper = {}

function report_helper.get_copy_api(data, copy)
    local E = ratelprof.consts._ENV
    local call, domain = data:find_parent_trace(copy)
    local driver_call, driver_name
    while call do
        if domain == E.DOMAIN_HIP and call.name:match("^hipMemcpy") then
            return call, call.name
        end
        if domain == E.DOMAIN_CUDA_RUNTIME and call.name:match("^cudaMemcpy") then
            return call, (call.name:gsub("_v%d+.*$", ""))
        end
        if domain == E.DOMAIN_CUDA_DRIVER and call.name:match("^cuMemcpy") then
            driver_call = call
            driver_name = call.name:gsub("_v%d+.*$", "")
        end
        call, domain = data:find_parent_trace(call, copy.mpi_rank)
    end
    return driver_call, driver_name
end

function report_helper.get_duration(dur, timeunit)
    if timeunit ~= "ns" then
        return convert.time(dur, "ns", timeunit)
    end
    return dur
end

function report_helper.get_size(size)
    return size
    -- return string.format("%.0f", convert.bytes(size, "bytes", "mb"))
end


return report_helper
