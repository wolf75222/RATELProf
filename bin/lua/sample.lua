require("globals")
local Script = require("utils.Classes.Script")

local function fail(message)
    Message:error(message)
    os.exit(1)
end

local function process(arguments, options)
    local modern = ratelprof.consts._LIBS_DIR.."libratelprof_cuda_pcsampling.so"
    local legacy = ratelprof.consts._LIBS_DIR.."libratelprof_cuda_pcsampling_legacy.so"
    if options.backend ~= "auto" and options.backend ~= "modern" and options.backend ~= "legacy" then
        fail("--backend expects auto, modern or legacy.")
    end
    local arch
    if options.arch then
        arch = tonumber(options.arch:lower():match("^sm_(%d+)[af]?$"))
        if not arch then fail("--arch expects a CUDA architecture such as sm_61.") end
        if arch < 52 then fail("CUPTI PC sampling requires sm_52 or newer.") end
        if arch < 70 and options.backend == "modern" then
            fail("Use the legacy PC sampling backend for Maxwell and Pascal.")
        end
    end
    local lib = modern
    if options.backend == "legacy" or (options.backend == "auto" and
        ((arch and arch < 70) or not ratelprof.fs.exists(modern))) then
        lib = legacy
    end
    if not ratelprof.fs.exists(lib) then
        fail("The requested CUPTI PC sampling backend is unavailable in this build. Use CUDA 10.2–12.x for the legacy backend.")
    end

    local period = tonumber(options.period)
    if options.period and (not period or period % 1 ~= 0 or period < 5 or period > 31) then
        fail("The PC sampling period exponent must be an integer from 5 to 31.")
    end

    local utils = ratelprof.utils
    local quote = utils.shell_quote
    local out = options.output
    local cubins = options["dump-cubins"] or ""
    if out:sub(1, 1) ~= "/" then
        out = utils.execute_command("pwd -P").."/"..out
    end
    if cubins ~= "" and cubins:sub(1, 1) ~= "/" then
        cubins = utils.execute_command("pwd -P").."/"..cubins
    end
    local temporary = utils.execute_command("mktemp -- "..quote(out..".tmp.XXXXXXXX"))
    if temporary == "" then fail("Could not create a temporary PC sampling report next to "..out) end

    local inherited = os.getenv("LD_PRELOAD")
    local preload = lib..(inherited and inherited ~= "" and ":"..inherited or "")
    local command = {
        "RATELPROF_PCSAMPLING_PID=$$ exec env",
        quote("CUDA_INJECTION64_PATH="..lib),
        quote("LD_PRELOAD="..preload),
        quote("RATELPROF_PCSAMPLING_OUTPUT="..temporary),
        quote("RATELPROF_PCSAMPLING_PERIOD="..(period and tostring(period) or "")),
        quote("RATELPROF_DUMP_CUBINS="..cubins),
    }
    for _, value in ipairs(arguments) do command[#command + 1] = quote(value) end

    local code = utils.exit_status(os.execute(table.concat(command, " ")))
    if code ~= 0 then
        os.remove(temporary)
        Message:error("The application exited with status "..code..".")
        os.exit(code)
    end

    local file = io.open(temporary, "rb")
    local complete = false
    if file then
        local size = file:seek("end") or 0
        file:seek("set", math.max(0, size - 128))
        complete = file:read("*a"):match('"complete":true}%s*$')
        file:close()
    end
    if not complete then
        os.remove(temporary)
        fail("CUPTI did not produce a complete PC sampling report. Check the collector messages above.")
    end

    local ok, err = os.rename(temporary, out)
    if not ok then
        os.remove(temporary)
        fail("Could not write "..out..". "..err)
    end
    Message:print("RPROF: PC sampling report written to "..out)
end

local script = Script:new({name = "ratelprof sample", version = ratelprof.consts._VERSION})
script:set_desc("Collect PC and warp stall samples using CUPTI in a dedicated run.")
script:set_execute_function(process)
script:add_argument("application", false, false, "CUDA application to sample.")
script:add_argument("application-args", true, true, "Arguments for the application.")
script:add_option("output", "o", "FILE", "Write the samples as JSON.", true, "pcsamples.json")
script:add_option("backend", nil, "NAME", "Choose auto, modern, or legacy (Maxwell sm_52+ and Pascal).", true, "auto")
script:add_option("arch", nil, "SM", "Select the sampler for an architecture such as sm_61.", true)
script:add_option("period", nil, "EXP", "Sample every 2^EXP cycles. CUPTI chooses a device default when omitted.", true)
script:add_option("dump-cubins", nil, "DIR", "Save the loaded cubins from this sampling run.", true)
script:execute(arg)
