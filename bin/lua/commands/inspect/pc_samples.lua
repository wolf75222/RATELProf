local pc_samples = {}

local function is_count(value)
    return type(value) == "number" and value >= 0 and value < math.huge and value % 1 == 0
end

local function merge(cubins, crc, symbol, offset, sample)
    cubins[crc] = cubins[crc] or {}
    local functions = cubins[crc]
    functions[symbol] = functions[symbol] or {}
    local offsets = functions[symbol]
    offsets[offset] = offsets[offset] or {stall_reasons = {}}
    local entry = offsets[offset]
    for reason, count in pairs(sample.stall_reasons) do
        if type(reason) ~= "string" or not is_count(count) then
            return nil, "Invalid stall reason count in a PC sample."
        end
        entry.stall_reasons[reason] = (entry.stall_reasons[reason] or 0) + count
    end
    if not entry.source and type(sample.source) == "table"
        and type(sample.source.file) == "string" and is_count(sample.source.line) then
        entry.source = sample.source
    end
    return true
end

function pc_samples.load(path)
    local ok, report = pcall(ratelprof.utils.load_json, path)
    if not ok then return nil, tostring(report) end
    if type(report) ~= "table" or report.schema ~= "ratelprof-pcsampling-v1" or type(report.samples) ~= "table" then
        return nil, "Expected a ratelprof-pcsampling-v1 JSON report."
    end

    local cubins = {}
    for _, sample in ipairs(report.samples) do
        if type(sample) ~= "table" then return nil, "Invalid PC sample." end
        local crc = type(sample.cubin_crc) == "string" and sample.cubin_crc:lower():match("^0x(%x+)$")
        local symbol, offset = sample["function"], sample.pc_offset
        if not crc or #crc > 16 or type(symbol) ~= "string"
            or not is_count(offset) or type(sample.stall_reasons) ~= "table" then
            return nil, "Invalid cubin, symbol, offset or stall reasons in a PC sample."
        end
        -- CUPTI can omit the function name. There is no symbol to join in that case.
        if symbol ~= "" then
            crc = "0x"..crc:gsub("^0+(%x)", "%1")
            local merged, message = merge(cubins, crc, symbol, offset, sample)
            if not merged then return nil, message end
        end
    end
    return cubins
end

return pc_samples
