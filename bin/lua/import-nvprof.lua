require("globals")

local Script = require("utils.Classes.Script")

local function fail(message)
    Message:error(message)
    os.exit(1)
end

local function numeric_keys(data)
    local result = {}
    for key, value in pairs(data) do result[tonumber(key) or key] = value end
    return result
end

local function process(_, options)
    local utils = ratelprof.utils
    local quote = utils.shell_quote
    local output = options.output
    if not ratelprof.fs.has_extension(output, ".rprof-rep") then
        fail("The output filename must end in .rprof-rep.")
    end
    if options.profile and (options.gpu or options.api) or not (options.profile or options.gpu or options.api) then
        fail("Use --profile or one or both of --gpu and --api.")
    end
    local argv = {"python3", quote(ratelprof.consts._MODULES_DIR.."metrics/nvprof_timeline.py")}
    for _, key in ipairs({"profile", "gpu", "api", "arch", "nvprof"}) do
        if options[key] then argv[#argv+1] = quote("--"..key.."="..options[key]) end
    end
    local temp = os.tmpname()
    local status = utils.exit_status(os.execute(table.concat(argv, " ").." >"..quote(temp)))
    if status ~= 0 then
        os.remove(temp)
        fail("Could not import the nvprof traces.")
    end
    local file, message = io.open(temp, "r")
    if not file then
        os.remove(temp)
        fail(message)
    end
    local json = file:read("*a")
    file:close()
    os.remove(temp)
    local ok, result = pcall(JSON.decode, JSON, json)
    if not ok then fail(tostring(result)) end
    for _, field in ipairs({"node_id", "gpu_props", "locations"}) do result[field] = numeric_keys(result[field]) end
    for domain, events in pairs(result.trace_events) do result.trace_events[domain] = numeric_keys(events) end
    local writer = require("utils.SectionedWriter")
    ok, result = pcall(writer.write_normalized, result, output)
    if not ok then fail(tostring(result)) end
    Message:print("RPROF: nvprof trace written to "..output)
end

local script = Script:new({name = "ratelprof import-nvprof", version = ratelprof.consts._VERSION})
script:set_desc("Import an nvprof recording or GPU/API trace CSV as a sectioned RATELProf report.")
script:set_execute_function(process)
script:add_option("profile", nil, "FILE", "Original recording from nvprof --export-profile.", true)
script:add_option("gpu", nil, "CSV", "Output of nvprof --print-gpu-trace --csv.", true)
script:add_option("api", nil, "CSV", "Output of nvprof --print-api-trace --csv from the same recording.", true)
script:add_option("arch", nil, "SM", "Tag the imported devices with a CUDA architecture.", true)
script:add_option("nvprof", nil, "PATH", "Path to the nvprof executable for binary import.", true)
script:add_option("output", "o", "FILE", "Write a sectioned RATELProf report.", true, "nvprof.rprof-rep")
script:execute(arg)
