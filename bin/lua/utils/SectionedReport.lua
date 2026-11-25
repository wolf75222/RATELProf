-- Reader for the upstream pre-release .rprof-rep binary layout, version 1.1.1.
-- The header and reference counts are little-endian, as is this format's
-- MessagePack dialect. An event ends with its size and a 0xc1 marker.
local SectionedReport = { magic = "RATELProf\0" }

local sections = {"GLOBAL", "STRING", "LOCATION", "GPU_SPEC", "API_DATA", "KERNEL", "CID", "OFFSETS", "EVENTS"}
local domains = {
    [0] = "HSA", "OMP_TGT_RTL", "ROCBLAS", "OMP_ROUTINE", "HIP", "RCCL", "MPI",
    "OMP_REGION", "ROCTX", "PROFILING", "COPY", "KERNEL", "BARRIERAND", "BARRIEROR",
}

local widths = {[0xcc] = 1, [0xcd] = 2, [0xce] = 4, [0xcf] = 8}
local str_widths = {[0xd9] = 1, [0xda] = 2, [0xdb] = 4}
local word = 4294967296
local max_exact = 9007199254740991

local Reader = {}
Reader.__index = Reader

local function reader(data, first, last, name)
    return setmetatable({data = data, pos = first, last = last, name = name}, Reader)
end

function Reader:fail(reason)
    error(("%s at byte %d: %s"):format(self.name, self.pos - 1, reason), 0)
end

function Reader:remaining()
    return self.last - self.pos + 1
end

function Reader:take(size)
    if size > self:remaining() then self:fail("truncated data") end
    local first = self.pos
    self.pos = self.pos + size
    return first
end

function Reader:byte()
    return self.data:byte(self:take(1))
end

function Reader:le(size)
    local first = self:take(size)
    local value = 0
    for i = first + size - 1, first, -1 do
        value = value * 256 + self.data:byte(i)
    end
    return value
end

function Reader:integer(size, signed, wide)
    if size == 8 then
        local lo, hi = self:le(4), self:le(4)
        local neg = signed and hi >= word / 2
        if neg then hi = hi - word end
        local value = hi * word + lo
        if wide and math.abs(value) > max_exact then
            if neg then
                hi, lo = -hi - (lo ~= 0 and 1 or 0), (word - lo) % word
            end
            return ("%s0x%x%08x"):format(neg and "-" or "", hi, lo)
        end
        return value
    end
    local value = self:le(size)
    local modulus = 256 ^ size
    if signed and value >= modulus / 2 then value = value - modulus end
    return value
end

function Reader:uint(wide)
    local tag = self:byte()
    if tag <= 127 then return tag end
    if widths[tag] then return self:integer(widths[tag], false, wide) end
    self:fail("expected unsigned integer")
end

function Reader:int()
    local tag = self:byte()
    if tag <= 127 then return tag end
    if tag >= 0xe0 then return tag - 256 end
    if widths[tag] then return self:le(widths[tag]) end
    if widths[tag - 4] then return self:integer(widths[tag - 4], true) end
    self:fail("expected integer")
end

function Reader:checked_count(count, item_bytes)
    if count > math.floor(self:remaining() / (item_bytes or 1)) then
        self:fail("count exceeds section size")
    end
    return count
end

function Reader:count(item_bytes)
    return self:checked_count(self:uint(), item_bytes)
end

function Reader:str(tag)
    tag = tag or self:byte()
    local size
    if tag >= 0xa0 and tag <= 0xbf then
        size = tag - 0xa0
    elseif str_widths[tag] then
        size = self:le(str_widths[tag])
    else
        self:fail("expected string")
    end
    local first = self:take(size)
    return self.data:sub(first, first + size - 1)
end

function Reader:cstr(maximum)
    local first = self.pos
    local field = self.data:sub(first, math.min(first + maximum - 1, self.last))
    local stop = field:find("\0", 1, true)
    self:take(stop or maximum)
    return stop and field:sub(1, stop - 1) or field
end

function Reader:done()
    if self:remaining() ~= 0 then self:fail("unexpected trailing data") end
end

local function array_size(input, tag)
    if tag >= 0x90 and tag <= 0x9f then return tag - 0x90 end
    if tag == 0xdc or tag == 0xdd then return input:le(tag == 0xdc and 2 or 4) end
end

local function vector(input, size)
    if array_size(input, input:byte()) ~= size then input:fail("unexpected array size") end
    local result = {}
    for i = 1, size do result[i] = input:uint(true) end
    return result
end

local function floating(bits, sign_bit, unit, max_exp, bias, remainder)
    local sign = bits >= sign_bit and -1 or 1
    local exponent = math.floor((bits % sign_bit) / unit)
    local fraction = (bits % unit + remainder) / unit
    if exponent == max_exp then
        if fraction ~= 0 then return "NaN" end
        return sign < 0 and "-Infinity" or "Infinity"
    end
    if exponent == 0 then return sign * fraction * 2^(1 - bias) end
    return sign * (1 + fraction) * 2^(exponent - bias)
end

local function argument(input, depth)
    if depth > 64 then input:fail("argument nesting exceeds limit") end
    local tag = input:byte()
    if tag <= 127 then return tag end
    if tag >= 0xe0 then return tag - 256 end
    if widths[tag] then return input:integer(widths[tag], false, true) end
    if widths[tag - 4] then return input:integer(widths[tag - 4], true, true) end
    if tag == 0xc0 then return nil end
    if tag == 0xc2 or tag == 0xc3 then return tag == 0xc3 end
    if tag == 0xca then return floating(input:le(4), 2^31, 2^23, 255, 127, 0) end
    if tag == 0xcb then
        local lo, hi = input:le(4), input:le(4)
        return floating(hi, 2^31, 2^20, 2047, 1023, lo / word)
    end
    if (tag >= 0xa0 and tag <= 0xbf) or str_widths[tag] then return input:str(tag) end
    local count = array_size(input, tag)
    if count then
        local arr = {}
        for i = 1, input:checked_count(count) do
            arr[i] = {value = argument(input, depth + 1)}
        end
        return arr
    end
    if tag >= 0x80 and tag <= 0x8f then count = tag - 0x80 end
    if tag == 0xde or tag == 0xdf then count = input:le(tag == 0xde and 2 or 4) end
    if count then
        local map = {}
        for _ = 1, input:checked_count(count, 2) do
            local key = argument(input, depth + 1)
            if type(key) ~= "string" and type(key) ~= "number" then
                input:fail("unsupported argument map key")
            end
            map[key] = {value = argument(input, depth + 1)}
        end
        return map
    end
    input:fail(("unsupported argument tag 0x%02x"):format(tag))
end

local function header(data)
    if data:sub(1, #SectionedReport.magic) ~= SectionedReport.magic then
        error("invalid sectioned report magic", 0)
    end
    local input = reader(data, #SectionedReport.magic + 1, #data, "header")
    local major, minor, patch = input:byte(), input:byte(), input:byte()
    if major ~= 1 or minor ~= 1 or patch ~= 1 then
        input:fail(("unsupported sectioned report version %d.%d.%d"):format(major, minor, patch))
    end
    local header_size = input.pos - 1 + #sections * 16
    local result, ranges = {}, {}
    for _, name in ipairs(sections) do
        local offset, size = input:le(8), input:le(8)
        if size > 0 then
            if offset < header_size or offset > #data - size then input:fail("invalid " .. name .. " range") end
            ranges[#ranges + 1] = {offset, offset + size}
        else
            offset = 0
        end
        result[name] = reader(data, offset + 1, offset + size, name)
    end
    table.sort(ranges, function(a, b) return a[1] < b[1] end)
    local last = header_size
    for _, range in ipairs(ranges) do
        if range[1] < last then input:fail("overlapping sections") end
        last = range[2]
    end
    return result
end

local function globals(input)
    local report = {
        mpi_rank = input:int(), hostname = input:str(),
        version = {input:uint(), input:uint(), input:uint()},
        experiment_start_epoch = input:uint(),
        lifecycle = {input:uint(), input:uint(), input:uint()},
        main_data = {input:int()}, node_id = {}, gpu_props = {},
        locations = {}, trace_events = {},
    }
    for i = 1, input:count() do report.main_data[i + 1] = input:str() end
    for _ = 1, input:count(2) do
        local agent = input:uint(true)
        report.node_id[agent] = input:uint()
    end
    input:done()
    return report
end

local function references(input, bytes, decode)
    local result = {}
    if input:remaining() == 0 then return result end
    if input:remaining() < 8 then input:fail("missing reference count") end
    local tail = reader(input.data, input.last - 7, input.last, input.name)
    input.last = input.last - 8
    for i = 0, input:checked_count(tail:le(8), bytes) - 1 do result[i] = decode(input) end
    input:done()
    return result
end

local function reference(input, table, name)
    local value = table[input:uint()]
    if value == nil then input:fail("missing " .. name .. " reference") end
    return value
end

-- Wire order of the GPU_SPEC section, as written by upstream's lsgpu.
local gpu_fields = {
    {"device_type", 4}, {"node", 4}, {"name", "s", 64}, {"product_name", "s", 64},
    {"uuid", "s", 21}, {"nearest_cpu_node_id", 4}, {"driver_node_id", 4}, {"isa", "s", 64},
    {"feature", 4}, {"machine_model", 4}, {"version_major", 2}, {"version_minor", 2},
    {"cache_size", 4, 4}, {"cacheline_size", 4}, {"max_mem_clock", 4}, {"wavefront_size", 4},
    {"fbarrier_max_size", 4}, {"workgroup_max_size", 2}, {"workgroup_max_dim", 2, 3},
    {"grid_max_size", 4}, {"grid_max_dim", 4, 3}, {"queue_type", 4}, {"queues_max", 4},
    {"queue_min_size", 4}, {"queue_max_size", 4}, {"cu_count", 4}, {"max_wave_per_cu", 4},
    {"num_simd_per_cu", 4}, {"num_se", 4}, {"num_sa_per_se", 4}, {"num_sdma_eng", 4},
    {"num_sdma_xgmi_eng", 4}, {"num_xcc", 4}, {"chip_id", 4}, {"max_clock_freq", 4}, {"driver_uid", 4},
}

local function gpu_properties(input, report)
    if input:remaining() == 0 then return end
    local device_types = {[0] = "CPU", "GPU", "DSP", "AIE"}
    for _ = 1, input:checked_count(input:le(4), 36) do
        local gpu = {}
        for _, field in ipairs(gpu_fields) do
            local key, width, count = field[1], field[2], field[3]
            if width == "s" then
                gpu[key] = input:cstr(count)
            elseif count then
                local values = {}
                for i = 1, count do values[i] = input:le(width) end
                gpu[key] = values
            else
                gpu[key] = input:le(width)
            end
        end
        gpu.device_type_name = device_types[gpu.device_type]
        gpu.feature_id, gpu.queue_type_id = gpu.feature, gpu.queue_type
        gpu.feature = ({[0] = "AGENT DISPATCH", "KERNEL DISPATCH"})[gpu.feature_id] or "-"
        gpu.queue_type = ({[0] = "MULTI", "SINGLE", "COOPERATIVE"})[gpu.queue_type_id] or "-"
        if gpu.device_type == 1 then report.gpu_props[gpu.node] = gpu end
    end
    input:done()
end

local function decode_event(input, group, report, refs)
    local domain = group.domain
    local id, start, dur = input:uint(), input:uint(), input:uint()
    local phase
    if start < report.lifecycle[1] then phase = "CONSTRUCTOR_PHASE"
    elseif start < report.lifecycle[2] then phase = "MAIN_PHASE"
    else phase = "DESTRUCTOR_PHASE" end
    local args = {}
    local event = {start = start, dur = dur, args = args, phase = phase, corr_id = 0}
    if input.data:byte(input.pos) == 0xc6 then
        input:byte()
        event.corr_id = input:uint()
    end

    if domain < 10 then
        event.pid, event.tid = group.unit, group.subunit
        local api
        if domain == 8 or domain == 9 then
            event.name = reference(input, refs.strings, "string")
        else
            api = reference(input, refs.apis, "API")
            event.name = api.name
        end
        event.return_address = reference(input, refs.locations, "location")
        if api then
            for _, pair in ipairs(api.args) do
                local kind, name = pair[1], pair[2]
                local value = argument(input, 0)
                if domain == 7 and name == "mapping_flags" and kind == "ompt_target_map_flag_t*" then
                    -- The OMPT mapping writer interns the string names of the flags.
                    for _, flag in ipairs(value) do
                        flag.string_id = flag.value
                        flag.value = refs.strings[flag.string_id]
                        if flag.value == nil then input:fail("missing OMPT mapping flag string reference") end
                    end
                end
                args[name] = {type = kind, value = value}
            end
        end
    elseif domain == 10 then
        local name = reference(input, refs.strings, "string")
        local op = input:uint()
        if op < 1 or op > 7 then input:fail("unknown memory operation") end
        event.sig = input:uint(true)
        args.size, args.operation, args.operation_name = input:uint(true), op, name
        args.gpu_id = group.unit
        if op <= 2 then
            args.queue_id, domain = group.subunit, "MEMORY_FILL"
        else
            local other = input:uint(true)
            if name == "CopyHostToDevice" then
                args.src_agent, args.dst_agent, args.src_type, args.dst_type = other, group.unit, 0, 1
            elseif name == "CopyDeviceToHost" then
                args.src_agent, args.dst_agent, args.src_type, args.dst_type = group.unit, other, 1, 0
            elseif name == "CopyDeviceToDevice" then
                args.src_agent, args.dst_agent, args.src_type, args.dst_type = group.unit, other, 1, 1
            else
                args.src_agent, args.dst_agent, args.src_type, args.dst_type = other, group.unit, 0, 0
            end
            if op == 7 then args.engine_id = group.subunit < -1 and -group.subunit - 2 or -1
            else args.queue_id, args.engine_id = group.subunit, -1 end
        end
        if op ~= 7 then args.dispatch_time, args.workgroup_size_x = input:uint(), input:uint() end
    else
        args.gpu_id, args.queue_id = group.unit, group.subunit
        if domain == 11 then
            local kernel = reference(input, refs.kernels, "kernel")
            args.kernel_name, args.kernel_object = kernel.name, kernel.object
            args.group_segment_size, args.private_segment_size = kernel.group, kernel.private
        end
        event.sig, args.dispatch_time = input:uint(true), input:uint()
        if domain == 11 then
            args.wrg, args.grd = vector(input, 3), vector(input, 3)
            args.kernarg_address = input:uint(true)
        else
            args.dep_signal = vector(input, 5)
        end
    end
    input:done()

    local key = "RATELPROF_DOMAIN_" .. (domains[domain] or domain)
    local events = report.trace_events[key]
    if not events then events = {}; report.trace_events[key] = events end
    if events[id] then input:fail("duplicate event ID") end
    events[id] = event
end

local function groups(input)
    local count = input:count(4)
    local result, ids = {}, {}
    for tree = 1, 2 do
        for _ = 1, input:count(2) do
            local unit = input:uint(true)
            for _ = 1, input:count(2) do
                local subunit = input:int()
                for _ = 1, input:count(4) do
                    local domain, offset, events, id = input:uint(), input:uint(), input:uint(), input:uint()
                    if not domains[domain] or (domain < 10) ~= (tree == 1) then input:fail("invalid group domain") end
                    if id >= count or ids[id] then input:fail("invalid group ID") end
                    ids[id] = true
                    result[#result + 1] = {unit = unit, subunit = subunit, domain = domain, offset = offset, count = events}
                end
            end
        end
    end
    input:uint() -- Domain mask is redundant with the groups.
    input:done()
    if #result ~= count then input:fail("group count does not match the trees") end
    return result
end

local function events(input, group_list, report, refs)
    if #group_list == 0 and input:remaining() > 0 then input:fail("events are missing group offsets") end
    for i, group in ipairs(group_list) do
        local group_stop = group_list[i + 1] and group_list[i + 1].offset or input:remaining()
        if (i == 1 and group.offset ~= 0) or group.offset >= group_stop or group_stop > input:remaining() then
            input:fail("invalid event group range")
        end
        local first, last = input.pos + group.offset, input.pos + group_stop - 1
        local cursor = last
        if group.count > math.floor((last - first + 1) / 2) then input:fail("event count exceeds group size") end
        for _ = 1, group.count do
            if cursor <= first or input.data:byte(cursor) ~= 0xc1 then input:fail("missing event marker") end
            local tag = input.data:byte(cursor - 1)
            local width = widths[tag] or 0
            local size = tag
            if width > 0 then
                local prefix = cursor - 1 - width
                if prefix < first then input:fail("truncated event size") end
                size = reader(input.data, prefix, cursor - 2, "event size"):le(width)
            elseif tag > 127 then
                input:fail("invalid event size tag")
            end
            local payload_last = cursor - width - 2
            local payload_first = payload_last - size + 1
            if size == 0 or payload_first < first then input:fail("invalid event payload range") end
            decode_event(reader(input.data, payload_first, payload_last, "event"), group, report, refs)
            cursor = payload_first - 1
        end
        if cursor ~= first - 1 then input:fail("event count does not match the group") end
    end
end

function SectionedReport.decode(data)
    local s = header(data)
    local report = globals(s.GLOBAL)
    local refs = {}
    refs.strings = references(s.STRING, 1, function(r) return r:str() end)
    local function str(r) return reference(r, refs.strings, "string") end
    refs.locations = references(s.LOCATION, 6, function(r)
        local address, base = r:uint(true), r:uint(true)
        local loc = {ofile = str(r), sfun = str(r), source = str(r), sline = r:uint(), base_address = base}
        if loc.source ~= "??" and loc.sline > 0 then loc.sfile = loc.source end
        if type(address) == "number" and type(base) == "number" and base > 0 and address >= base then
            loc.offset = address - base
        end
        report.locations[address] = loc
        return address
    end)
    refs.apis = references(s.API_DATA, 2, function(r)
        local api = {name = str(r), args = {}}
        for i = 1, r:count(2) do api.args[i] = {str(r), str(r)} end
        return api
    end)
    refs.kernels = references(s.KERNEL, 4, function(r)
        return {name = str(r), object = r:uint(true), group = r:uint(), private = r:uint()}
    end)
    gpu_properties(s.GPU_SPEC, report)
    local group_list = groups(s.OFFSETS)
    events(s.EVENTS, group_list, report, refs)
    return report
end

return SectionedReport
