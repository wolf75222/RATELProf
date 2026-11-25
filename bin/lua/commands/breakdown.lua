local BinaryReport = require("utils.Classes.BinaryReport")
local Report = require("utils.Classes.Report")
local E = ratelprof.consts._ENV
local covered = ratelprof.utils.compute_total_covered_duration
local breakdown = {}

local host_domains = {
    [E.DOMAIN_HIP] = "runtime", [E.DOMAIN_HSA] = "runtime",
    [E.DOMAIN_CUDA_RUNTIME] = "runtime", [E.DOMAIN_CUDA_DRIVER] = "runtime",
    [E.DOMAIN_CUDA_SYNC] = "runtime", [E.DOMAIN_CUBLAS] = "runtime", [E.DOMAIN_NCCL] = "runtime",
    [E.DOMAIN_ROCBLAS] = "runtime", [E.DOMAIN_RCCL] = "runtime",
    [E.DOMAIN_OMP_TGT] = "omp", [E.DOMAIN_OMP_TGT_RTL] = "omp", [E.DOMAIN_OMPT] = "omp",
    [E.DOMAIN_MPI] = "mpi",
}
local device_domains = {
    [E.DOMAIN_KERNEL] = "compute", [E.DOMAIN_COPY] = "memory", [E.DOMAIN_CUDA_MEMSET] = "memory",
    [E.DOMAIN_MEMORY_FILL] = "memory",
    [E.DOMAIN_BARRIEROR] = "barrier", [E.DOMAIN_BARRIERAND] = "barrier",
}

local function add(groups, unit, category, interval)
    local group = groups[unit] or {}
    local category_intervals = group[category] or {}
    category_intervals[#category_intervals + 1] = interval
    group[category] = category_intervals
    groups[unit] = group
end

local function collect(data, start, stop, only_main)
    local ranks = {}
    data:for_each_rank(function(rank)
        local first = start or 0
        local last = math.min(stop or math.huge, data:get_app_dur(rank))
        if only_main then
            first = math.max(first, data:get_main_start(rank))
            last = math.min(last, data:get_main_stop(rank))
        end
        ranks[rank] = {first = first, last = last, CPU = {All = {}}, GPU = {All = {}}}
    end)

    data:for_each_traces(function(rank, domain, _, trace)
        local info = ranks[rank]
        local first = math.max(info.first, trace.start)
        local last = math.min(info.last, trace.start + trace.dur)
        if last <= first then return end
        local interval = {start = first, dur = last - first}
        local host = host_domains[domain]
        local device = device_domains[domain]
        if host then
            add(info.CPU, "All", host, interval)
            if trace.pid then add(info.CPU, trace.pid, host, interval) end
        elseif device then
            add(info.GPU, "All", device, interval)
            local args = trace.args
            if domain ~= E.DOMAIN_COPY then
                if args.gpu_id then
                    add(info.GPU, data:get_gpu_id(args.gpu_id, rank), device, interval)
                end
                return
            end
            local source = args.src_type == 1 and data:get_gpu_id(args.src_agent, rank)
            local destination = args.dst_type == 1 and data:get_gpu_id(args.dst_agent, rank)
            if source then add(info.GPU, source, device, interval) end
            if destination and destination ~= source then
                add(info.GPU, destination, device, interval)
            end
        end
    end)
    return ranks
end

local function sort_keys(map)
    local keys = {}
    for key in pairs(map) do keys[#keys + 1] = key end
    table.sort(keys, function(a, b)
        if a == "All" or b == "All" then return a == "All" and b ~= "All" end
        if type(a) ~= type(b) then return type(a) == "number" end
        return a < b
    end)
    return keys
end

local function build_rows(ranks, name, timeunit)
    local rows = {}
    for _, rank in ipairs(sort_keys(ranks)) do
        local info = ranks[rank]
        local elapsed = math.max(0, info.last - info.first)
        local groups = info[name]
        for _, unit in ipairs(sort_keys(groups)) do
            local group = groups[unit]
            local a, b, c, without_compute
            if name == "CPU" then
                a, b, c = group.runtime or {}, group.omp or {}, group.mpi or {}
            else
                a, b, c = group.compute or {}, group.memory or {}, group.barrier or {}
                without_compute = covered(a, b) - covered(a)
            end
            local active = covered(a, b, c)
            local function pct(duration)
                return elapsed == 0 and 0 or 100 * duration / elapsed
            end
            local row = {tostring(rank), tostring(unit), ratelprof.convert.time(elapsed, "ns", timeunit),
                         pct(active), pct(elapsed - active), pct(covered(a)), pct(covered(b)), pct(covered(c))}
            if without_compute then row[#row + 1] = pct(without_compute) end
            rows[#rows + 1] = row
        end
    end
    return rows
end

function breakdown.process(files, opts)
    local reports = {cpu_breakdown = "CPU", gpu_breakdown = "GPU"}
    local requested = opts.report:split(",")
    if #requested == 0 then error("Select at least one breakdown report", 0) end
    for _, report in ipairs(requested) do
        if not reports[report] then error("Unknown breakdown report: " .. report, 0) end
    end
    local timeunit = opts.timeunit == "s" and "sec" or opts.timeunit
    if not ({ns = true, us = true, ms = true, sec = true})[timeunit] then
        error("Invalid breakdown time unit: " .. opts.timeunit, 0)
    end
    local format = opts.format or (opts.output == "-" and "column" or "csv")
    if not ({column = true, table = true, csv = true, tsv = true})[format] then
        error("Invalid breakdown format: " .. format, 0)
    end
    local start, stop = tonumber(opts.start), tonumber(opts.stop)
    local valid_start = not opts.start or start and start >= 0 and start < math.huge
    local valid_stop = not opts.stop or stop and stop >= 0 and stop < math.huge
    if not valid_start or not valid_stop or start and stop and stop < start then
        error("Expected nonnegative timestamps in nanoseconds with start <= stop", 0)
    end

    files = ratelprof.utils.check_report_files(files)
    local ranks = collect(BinaryReport:new(files), start, stop, opts["only-main"])
    for _, id in ipairs(requested) do
        local name = reports[id]
        local header = {"Rank", name == "CPU" and "PID" or "GPU", "Wall Time (" .. opts.timeunit .. ")",
                        "Active (%)", name == "CPU" and "Untraced (%)" or "Idle (%)"}
        local categories = name == "CPU" and {"GPU API (%)", "OpenMP (%)", "MPI (%)"}
                        or {"Compute (%)", "Memory (%)", "Barrier (%)", "Memory without Compute (%)"}
        for _, category in ipairs(categories) do header[#header + 1] = category end
        local report = Report:new({
            format = format, output = opts.output, rprof_rep_files = files,
            report_name = name, report_type = "Breakdown", report_id = id,
            report_path = "breakdown", max_lines = "all", mpi_rank = -1,
            notation = (format == "csv" or format == "tsv") and "raw" or nil,
            header = header, data = build_rows(ranks, name, timeunit)
        })
        report:generate()
    end
end

return breakdown
