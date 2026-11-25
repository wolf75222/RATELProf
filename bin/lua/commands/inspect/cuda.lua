local inspect = {}
local quote = ratelprof.utils.shell_quote

local function fail(message)
    Message:error(message)
    os.exit(1)
end

local function tool(name)
    local root = os.getenv("CUDAToolkit_ROOT") or os.getenv("CUDA_HOME")
    return root and quote(root.."/bin/"..name) or name
end

local function capture(command)
    local path = os.tmpname()
    local status = ratelprof.utils.exit_status(os.execute(command.." >"..quote(path).." 2>&1"))
    local file = ratelprof.fs.open_file(path, "r")
    local output = file:read("*a")
    file:close()
    os.remove(path)
    return output, status
end

local function parse_resources(raw)
    local kernels = {}
    local arch, entry
    for line in raw:gmatch("[^\r\n]+") do
        local architecture = line:match("^%s*arch%s*=%s*(%S+)")
        local name = line:match("^%s*Function%s+(.+):%s*$")
        if architecture then
            arch, entry = architecture, nil
        elseif name then
            entry = {architecture = arch or "unknown", symbol = name, constant_bytes = 0}
            kernels[#kernels + 1] = entry
        elseif entry then
            for key, number in line:gmatch("([A-Z]+[%d%[%]]*):%s*(%d+)") do
                local value = tonumber(number)
                if key:match("^CONSTANT") then
                    entry.constant_bytes = entry.constant_bytes + value
                else
                    entry[key] = value
                end
            end
        end
    end
    return kernels
end

local function emit_sass(application, consume)
    local disassembler = os.getenv("RATELPROF_NVDISASM")
    disassembler = disassembler and quote(disassembler) or tool("nvdisasm")
    local _, available = capture("command -v "..disassembler)
    if available ~= 0 then
        local output, status = capture(tool("cuobjdump").." --dump-sass "..quote(application))
        if status ~= 0 then fail("cuobjdump could not disassemble the CUDA binary. "..output) end
        consume(ratelprof.fs.basename(application), output)
        return
    end
    local absolute = ratelprof.utils.execute_command("realpath -- "..quote(application))
    local directory = ratelprof.utils.execute_command("mktemp -d")
    if absolute == "" or directory == "" then
        fail("Could not prepare CUDA binaries for disassembly.")
    end

    local ok, err = pcall(function()
        local extract = "cd "..quote(directory).." && "..tool("cuobjdump").." --extract-elf all "..quote(absolute)
        local _, status = capture(extract)
        local list = io.popen("find "..quote(directory).." -maxdepth 1 -type f -name '*.cubin' -print")
        local images = {}
        for path in list:lines() do images[#images + 1] = path end
        list:close()
        table.sort(images)
        if status ~= 0 or #images == 0 then images = {absolute} end

        for _, path in ipairs(images) do
            local sass, result = capture(disassembler.." --print-code --no-dataflow "..quote(path))
            if result ~= 0 then error(sass, 0) end
            consume(ratelprof.fs.basename(path), sass)
        end
    end)
    os.execute("rm -rf -- "..quote(directory))
    if not ok then fail("nvdisasm failed. "..err) end
end

local function instruction_counts(image, sass)
    local kernels, kernel = {}, nil
    local architecture = image:match("sm_(%d+[af]?)") or "unknown"
    for line in sass:gmatch("[^\r\n]+") do
        local arch = line:match("^%s*arch%s*=%s*sm_(%d+[af]?)")
        local name = line:match("^%s*Function%s*:%s*(%S+)")
                  or line:match("^%s*//%-+%s*%.text%.(%S+)")
        if arch then architecture = arch end
        if name then
            kernel = {symbol = name, architecture = "sm_"..architecture, image = image,
                      instruction_count = 0, opcodes = {}}
            kernels[#kernels + 1] = kernel
        elseif kernel then
            local instruction = line:match("/%*[%da-fA-F]+%*/%s*(.-);")
            if instruction then
                local token = instruction:gsub("^@!?[%w_]+%s+", ""):match("^(%u[%u%d]*)")
                if token then
                    kernel.opcodes[token] = (kernel.opcodes[token] or 0) + 1
                    kernel.instruction_count = kernel.instruction_count + 1
                end
            end
        end
    end
    return kernels
end

local function csv_value(value)
    local str = tostring(value or "")
    return '"'..str:gsub('"', '""')..'"'
end

function inspect.kernels(application, options)
    if not application or not ratelprof.fs.exists(application) then
        fail("Specify a CUDA application or cubin to inspect.")
    end
    local raw, status = capture(tool("cuobjdump").." --dump-resource-usage "..quote(application))
    if status ~= 0 then fail("cuobjdump failed. "..raw) end
    local kernels = parse_resources(raw)
    if #kernels == 0 then fail("cuobjdump found no compiled CUDA kernels.") end

    local output = options.output or "inspect_"..ratelprof.fs.basename(application)..".csv"
    if not ratelprof.fs.has_extension(output, ".csv") then fail("Output filename must end in .csv.") end

    local file = ratelprof.fs.open_file(output, "w")
    file:write("Name,Architecture,Registers,Stack bytes,Local bytes,Shared bytes,Constant bytes,Texture count,Surface count\n")
    for _, kernel in ipairs(kernels) do
        local name = options.mangled and kernel.symbol or ratelprof.utils.demangle(kernel.symbol, options.trunc)
        kernel.name = name
        local row = {name, kernel.architecture, kernel.REG or "", kernel.STACK or "",
                     kernel.LOCAL or "", kernel.SHARED or "", kernel.constant_bytes,
                     kernel.TEXTURE or "", kernel.SURFACE or ""}
        for i, value in ipairs(row) do row[i] = csv_value(value) end
        file:write(table.concat(row, ","), "\n")
    end
    file:close()
    Message:print("RPROF: CSV output written to "..output)

    local base = ratelprof.fs.remove_extension(output, ".csv")
    if options["save-json"] then
        ratelprof.utils.store_json({backend = "cuda", kernels = kernels}, base..".json")
        Message:print("RPROF: JSON output written to "..base..".json")
    end
    if options.sass or options["sass-summary"] then
        local stats = {}
        local listing = options.sass and ratelprof.fs.open_file(base..".sass", "w")
        emit_sass(application, function(image, sass)
            if listing then listing:write("// ", image, "\n", sass, "\n") end
            if options["sass-summary"] then
                for _, kernel in ipairs(instruction_counts(image, sass)) do
                    if kernel.instruction_count > 0 then stats[#stats + 1] = kernel end
                end
            end
        end)
        if listing then
            listing:close()
            Message:print("RPROF: SASS output written to "..base..".sass")
        end
        if options["sass-summary"] then
            if #stats == 0 then fail("No SASS instructions could be parsed from the disassembly.") end
            ratelprof.utils.store_json({kernels = stats}, base..".sass.json")
            Message:print("RPROF: SASS instruction counts written to "..base..".sass.json")
        end
    end
end

function inspect.gpus()
    local devices, message = require("ratelprof_cuda_agent").get_devices()
    if not devices then fail(message) end
    for _, gpu in ipairs(devices) do
        Message:print(string.format("GPU %d  %s  %s  %s  %d SMs  warp %d  %.2f GiB",
            gpu.ordinal, gpu.name, gpu.arch, gpu.uuid, gpu.sm_count, gpu.warp_size,
            gpu.memory / 1024^3))
    end
end

return inspect
