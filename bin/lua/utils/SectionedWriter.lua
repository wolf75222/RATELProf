-- The fork's 1.2.0 report format keeps the upstream header and group index.
-- Group domains refer to string IDs, so CUDA domains retain their original names.
local SectionedWriter = {}

local magic = "RATELProf\0"
local sections = {"GLOBAL", "STRING", "LOCATION", "GPU_SPEC", "API_DATA", "KERNEL", "CID", "OFFSETS", "EVENTS"}
local word = 4294967296
local max_exact = 9007199254740991
local unpack_values = table.unpack or unpack
local copy_domain = "RATELPROF_DOMAIN_COPY"
local gpu_domains = {
    [copy_domain] = true,
    RATELPROF_DOMAIN_KERNEL = true,
    RATELPROF_DOMAIN_CUDA_MEMSET = true,
    RATELPROF_DOMAIN_MEMORY_FILL = true,
    RATELPROF_DOMAIN_BARRIERAND = true,
    RATELPROF_DOMAIN_BARRIEROR = true,
}

local function le(value, width)
    local bytes = {}
    for i = 1, width do
        bytes[i] = string.char(value % 256)
        value = math.floor(value / 256)
    end
    return table.concat(bytes)
end

local function uint(value)
    assert(type(value) == "number" and value >= 0 and value < 2^64 and value == math.floor(value),
           "Expected an unsigned report integer")
    if value < 128 then return string.char(value) end
    if value < 256 then return "\xcc" .. le(value, 1) end
    if value < 65536 then return "\xcd" .. le(value, 2) end
    if value < word then return "\xce" .. le(value, 4) end
    return "\xcf" .. le(value, 8)
end

local function int(value)
    if value >= 0 then return uint(value) end
    assert(value == math.floor(value) and value >= -2^63, "Expected a signed report integer")
    if value >= -32 then return string.char(256 + value) end
    if value >= -128 then return "\xd0" .. le(value, 1) end
    if value >= -32768 then return "\xd1" .. le(value, 2) end
    if value >= -2147483648 then return "\xd2" .. le(value, 4) end
    return "\xd3" .. le(value, 8)
end

local function str(value)
    local size = #value
    if size < 32 then return string.char(0xa0 + size) .. value end
    if size < 256 then return "\xd9" .. le(size, 1) .. value end
    if size < 65536 then return "\xda" .. le(size, 2) .. value end
    assert(size < word, "Report string is too long")
    return "\xdb" .. le(size, 4) .. value
end

local function double(value)
    local negative = value < 0 or (value == 0 and 1 / value == -math.huge)
    value = math.abs(value)
    local exponent, fraction
    if value ~= value then exponent, fraction = 2047, 2^51
    elseif value == math.huge then exponent, fraction = 2047, 0
    elseif value < 2^-1022 then exponent, fraction = 0, math.ldexp(value, 1074)
    else
        local significand, power = math.frexp(value)
        exponent, fraction = power + 1022, (significand * 2 - 1) * 2^52
    end
    local hi = (negative and 2^31 or 0) + exponent * 2^20 + math.floor(fraction / word)
    return "\xcb" .. le(fraction % word, 4) .. le(hi, 4)
end

local function keys(map, omit_event_prefix)
    local result = {}
    for key in pairs(map) do
        local kind = type(key)
        assert(kind == "number" or kind == "string", "Report keys must be strings or numbers")
        if not omit_event_prefix or (key ~= "start" and key ~= "dur" and key ~= "corr_id") then
            result[#result + 1] = key
        end
    end
    table.sort(result, function(a, b)
        if type(a) ~= type(b) then return type(a) == "number" end
        return a < b
    end)
    return result
end

local function strings()
    local lookup, list = {}, {}
    local function intern(value)
        local id = lookup[value]
        if id == nil then
            id = #list
            lookup[value] = id
            list[id + 1] = str(value)
        end
        return id
    end
    return intern, list
end

local encode_value

local function encode_map(map, intern, depth, omit_event_prefix)
    local ordered = keys(map, omit_event_prefix)
    local count = #ordered
    local tag = count < 16 and string.char(0x80 + count)
             or count < 65536 and "\xde" .. le(count, 2)
             or "\xdf" .. le(count, 4)
    local fields = {tag}
    for _, key in ipairs(ordered) do
        fields[#fields + 1] = encode_value(key, intern, depth + 1)
        fields[#fields + 1] = encode_value(map[key], intern, depth + 1)
    end
    return table.concat(fields)
end

encode_value = function(value, intern, depth)
    assert(depth <= 64, "Report structure exceeds the nesting limit")
    if type(value) == "string" then return "\xc7" .. uint(intern(value)) end
    if type(value) == "table" then return encode_map(value, intern, depth) end
    if type(value) == "boolean" then return value and "\xc3" or "\xc2" end
    if type(value) == "number" then
        if math.abs(value) <= max_exact and value == math.floor(value) and
           not (value == 0 and 1 / value == -math.huge) then
            return int(value)
        end
        return double(value)
    end
    error("Unsupported report value type: " .. type(value), 0)
end

local function globals(report)
    local version, lifecycle, main, nodes = report.version, report.lifecycle, report.main_data, report.node_id
    local agents = keys(nodes)
    local fields = {int(report.mpi_rank or -1), str(report.hostname or ""),
                    uint(version[1]), uint(version[2]), uint(version[3]), uint(report.experiment_start_epoch),
                    uint(lifecycle[1]), uint(lifecycle[2]), uint(lifecycle[3]), int(main[1]), uint(#main - 1)}
    for i = 2, #main do fields[#fields + 1] = str(main[i]) end
    fields[#fields + 1] = uint(#agents)
    for _, agent in ipairs(agents) do
        fields[#fields + 1] = uint(agent)
        fields[#fields + 1] = uint(nodes[agent])
    end
    return table.concat(fields)
end

local function event_group(domain, event)
    if not gpu_domains[domain] then return 1, event.pid or 0, event.tid or 0 end
    local args = event.args or {}
    local unit = args.gpu_id
    if domain == copy_domain then
        unit = args.dst_type == 1 and args.dst_agent or args.src_agent
    end
    return 2, type(unit) == "number" and unit or 0, args.queue_id or args.engine_id or 0
end

local function encode_event(id, event, intern)
    local correlation = event.corr_id
    local payload = uint(id) .. uint(event.start) .. uint(event.dur) ..
                    (correlation and "\xc6" .. uint(correlation) or "") ..
                    encode_map(event, intern, 0, true)
    local length = uint(#payload)
    return payload .. length:sub(2) .. length:sub(1, 1) .. "\xc1"
end

local function encode_traces(trace_events, intern)
    local trees = {{}, {}}
    local total = 0
    for _, domain in ipairs(keys(trace_events)) do
        local events = trace_events[domain]
        for _, id in ipairs(keys(events)) do
            local event = events[id]
            local tree_id, unit, subunit = event_group(domain, event)
            local tree = trees[tree_id]
            local units = tree[unit] or {}
            local groups = units[subunit] or {}
            local group = groups[domain]
            if not group then
                group = {}
                groups[domain] = group
                total = total + 1
            end
            group[#group + 1] = encode_event(id, event, intern)
            units[subunit], tree[unit] = groups, units
        end
    end

    local index, event_buffers, offset, group_id = {uint(total)}, {}, 0, 0
    for _, tree in ipairs(trees) do
        local units = keys(tree)
        index[#index + 1] = uint(#units)
        for _, unit in ipairs(units) do
            local subunits = keys(tree[unit])
            index[#index + 1] = uint(unit) .. uint(#subunits)
            for _, subunit in ipairs(subunits) do
                local groups = tree[unit][subunit]
                local domains = keys(groups)
                index[#index + 1] = int(subunit) .. uint(#domains)
                for _, domain in ipairs(domains) do
                    local group = groups[domain]
                    local body = table.concat(group)
                    index[#index + 1] = uint(intern(domain)) .. uint(offset) .. uint(#group) .. uint(group_id)
                    event_buffers[#event_buffers + 1] = body
                    group_id, offset = group_id + 1, offset + #body
                end
            end
        end
    end
    index[#index + 1] = uint(0) -- The inherited fixed-width domain mask is reserved.
    return table.concat(index), table.concat(event_buffers)
end

local function report_bytes(report)
    local intern, string_refs = strings()
    local data = {
        GLOBAL = globals(report),
        LOCATION = encode_map(report.locations or {}, intern, 0),
        GPU_SPEC = encode_map(report.gpu_props or {}, intern, 0),
    }
    data.OFFSETS, data.EVENTS = encode_traces(report.trace_events, intern)
    data.STRING = table.concat(string_refs) .. le(#string_refs, 8)

    local header = {magic, string.char(1, 2, 0)}
    local offset = #magic + 3 + 16 * #sections
    for _, section in ipairs(sections) do
        local body = data[section] or ""
        header[#header + 1] = le(#body == 0 and 0 or offset, 8) .. le(#body, 8)
        offset = offset + #body
    end
    local result = {table.concat(header)}
    for _, section in ipairs(sections) do result[#result + 1] = data[section] or "" end
    return result
end

function SectionedWriter.write_normalized(report, path)
    local parts = report_bytes(report)
    assert(not path:find("\n", 1, true), "Report path must not contain a newline")
    local pattern = "'" .. (path .. ".tmp.XXXXXXXX"):gsub("'", "'\\''") .. "'"
    local command = assert(io.popen("mktemp -- " .. pattern, "r"))
    local temporary = command:read("*l")
    command:close()
    assert(temporary, "Could not create temporary file for " .. path)

    local file, message = io.open(temporary, "wb")
    if file then
        local wrote, write_error = file:write(unpack_values(parts))
        local closed, close_error = file:close()
        if wrote and closed then
            local renamed, rename_error = os.rename(temporary, path)
            if renamed then return path end
            message = rename_error
        else
            message = write_error or close_error
        end
    end
    os.remove(temporary)
    error(message or "Could not write " .. path, 0)
end

return SectionedWriter
