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

local function parse_architectures(spec)
    if not spec then return {false} end
    local archs, seen = {}, {}
    for token in (spec..","):gmatch("(.-),") do
        local arch = token:lower():match("^%s*sm_(%d+[af]?)%s*$") or token:lower():match("^%s*(%d+[af]?)%s*$")
        if not arch then fail("Invalid CUDA architecture '"..token.."'. Use names such as sm_90 or sm_90a.") end
        arch = "sm_"..arch
        if not seen[arch] then
            archs[#archs + 1] = arch
            seen[arch] = true
        end
    end
    return archs
end

local function architecture_option(arch)
    return arch and " --gpu-architecture "..quote(arch) or ""
end

local function emit_sass(application, arch, consume, identify_cubins)
    local disassembler = os.getenv("RATELPROF_NVDISASM")
    disassembler = disassembler and quote(disassembler) or tool("nvdisasm")
    local _, available = capture("command -v "..disassembler)
    if available ~= 0 then
        if identify_cubins then fail("--pc-samples requires nvdisasm.") end
        local output, status = capture(tool("cuobjdump").." --dump-sass"..architecture_option(arch).." "..quote(application))
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
        local extract = "cd "..quote(directory).." && "..tool("cuobjdump").." --extract-elf all"..architecture_option(arch).." "..quote(absolute)
        local _, status = capture(extract)
        local list = io.popen("find "..quote(directory).." -maxdepth 1 -type f -name '*.cubin' -print")
        local images = {}
        for path in list:lines() do images[#images + 1] = path end
        list:close()
        table.sort(images)
        if status ~= 0 or #images == 0 then images = {absolute} end

        for _, path in ipairs(images) do
            local image_arch = path:match("%.(sm_%d+[af]?)%.cubin$")
            if not arch or not image_arch or image_arch == arch then
                local sass, result = capture(disassembler.." --print-code --no-dataflow "..quote(path))
                if result ~= 0 then error(sass, 0) end
                consume(ratelprof.fs.basename(path), sass, path)
            end
        end
    end)
    os.execute("rm -rf -- "..quote(directory))
    if not ok then fail("Could not inspect CUDA machine code. "..err) end
end

local function instruction_counts(image, sass, fallback_arch, crc, functions)
    local kernels, kernel = {}, nil
    local architecture = image:match("(sm_%d+[af]?)") or fallback_arch or "unknown"
    for line in sass:gmatch("[^\r\n]+") do
        local arch = line:match("^%s*arch%s*=%s*(sm_%d+[af]?)")
                 or line:match("^%s*%.target%s+(sm_%d+[af]?)")
        local elf_arch = line:match("^%s*%.headerflags%s+[^%d]*EF_CUDA_SM(%d+[af]?)")
        if not arch and elf_arch and architecture == "unknown" then arch = "sm_"..elf_arch end
        local name = line:match("^%s*Function%s*:%s*(%S+)")
                  or line:match("^%s*//%-+%s*%.text%.(%S+)")
        if arch then architecture = arch end
        if name then
            kernel = {symbol = name, architecture = architecture, image = image,
                      instruction_count = 0, opcodes = {}, cubin_crc = crc}
            kernels[#kernels + 1] = kernel
        elseif kernel then
            local offset, instruction = line:match("/%*(%x+)%*/%s*(.-);")
            if instruction then
                local token = instruction:gsub("^@!?[%w_]+%s+", ""):match("^(%u[%u%d]*)")
                if token then
                    kernel.opcodes[token] = (kernel.opcodes[token] or 0) + 1
                    kernel.instruction_count = kernel.instruction_count + 1
                    local pc = tonumber(offset, 16)
                    local samples = functions and functions[kernel.symbol]
                    local sample = samples and samples[pc]
                    if sample then
                        local annotations = kernel.sampled_instructions or {}
                        kernel.sampled_instructions = annotations
                        annotations[#annotations + 1] = {
                            pc_offset = pc, opcode = token, sass = instruction,
                            stall_reasons = sample.stall_reasons, source = sample.source,
                        }
                    end
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

local function inspect_inputs(application, linked)
    if not linked then return {application}, {} end
    local inputs, unresolved = require("commands.inspect.elf_dependencies").discover(application, capture)
    if not inputs then fail(unresolved) end
    for _, dependency in ipairs(unresolved) do
        Message:print("RPROF: unresolved ELF dependency "..dependency.name.." required by "..dependency.required_by)
    end
    return inputs, unresolved
end

local function resource_usage(input, arch)
    local raw, status = capture(tool("cuobjdump").." --dump-resource-usage"..architecture_option(arch).." "..quote(input))
    if status ~= 0 then
        local message = raw:lower()
        if message:find("does not contain device code", 1, true)
            or message:find("could not find any device code", 1, true)
            or message:find("no device code", 1, true) then
            return {}
        end
        fail("cuobjdump failed for "..input..". "..raw)
    end
    local kernels = parse_resources(raw)
    if kernels[1] and kernels[1].architecture == "unknown" then
        -- Standalone cubins have no fatbin header in --dump-resource-usage.
        local header = capture(tool("cuobjdump").." --dump-elf "..quote(input).." | head -n 4")
        local sm = header:match("ELF:[^\r\n]-[,%s]sm=(%d+[af]?)")
        if sm then
            for _, kernel in ipairs(kernels) do
                if kernel.architecture == "unknown" then kernel.architecture = "sm_"..sm end
            end
        end
    end
    return kernels
end

function inspect.kernels(application, options)
    if not application or not ratelprof.fs.exists(application) then
        fail("Specify a CUDA application or cubin to inspect.")
    end
    local archs = parse_architectures(options.archs)
    local sample_index, get_crc
    if options["pc-samples"] then
        local message
        sample_index, message = require("commands.inspect.pc_samples").load(options["pc-samples"])
        if not sample_index then fail(message) end
        get_crc = require("ratelprof_cubin").get_crc
    end
    local inputs, unresolved = inspect_inputs(application, options.linked)
    local kernels, disassemblies = {}, {}
    for _, input in ipairs(inputs) do
        for _, arch in ipairs(archs) do
            local found, image_arch
            for _, kernel in ipairs(resource_usage(input, arch)) do
                if not arch or arch == kernel.architecture then
                    kernel.source = input
                    kernels[#kernels + 1] = kernel
                    if not found then
                        image_arch = kernel.architecture
                    elseif image_arch ~= kernel.architecture then
                        image_arch = nil
                    end
                    found = true
                end
            end
            if found then
                disassemblies[#disassemblies + 1] = {path = input, arch = arch, image_arch = image_arch}
            end
        end
    end
    if #kernels == 0 then fail("cuobjdump found no compiled CUDA kernels for the selected files and architectures.") end

    local output = options.output or "inspect_"..ratelprof.fs.basename(application)..".csv"
    if not ratelprof.fs.has_extension(output, ".csv") then fail("Output filename must end in .csv.") end

    local file = ratelprof.fs.open_file(output, "w")
    file:write("Name,Architecture,Registers,Stack bytes,Local bytes,Shared bytes,Constant bytes,Texture count,Surface count")
    if options.linked then file:write(",Source file") end
    file:write("\n")
    for _, kernel in ipairs(kernels) do
        local name = options.mangled and kernel.symbol or ratelprof.utils.demangle(kernel.symbol, options.trunc)
        kernel.name = name
        local row = {name, kernel.architecture, kernel.REG or "", kernel.STACK or "",
                     kernel.LOCAL or "", kernel.SHARED or "", kernel.constant_bytes,
                     kernel.TEXTURE or "", kernel.SURFACE or ""}
        if options.linked then row[#row + 1] = kernel.source end
        for i, value in ipairs(row) do row[i] = csv_value(value) end
        file:write(table.concat(row, ","), "\n")
    end
    file:close()
    Message:print("RPROF: CSV output written to "..output)

    local base = ratelprof.fs.remove_extension(output, ".csv")
    if options["save-json"] then
        ratelprof.utils.store_json({backend = "cuda", kernels = kernels, inputs = inputs,
                                    unresolved_dependencies = unresolved}, base..".json")
        Message:print("RPROF: JSON output written to "..base..".json")
    end
    local summarize = options["sass-summary"] or sample_index
    if options.sass or summarize then
        local stats = {}
        local matched_instructions = 0
        local listing = options.sass and ratelprof.fs.open_file(base..".sass", "w")
        for _, input in ipairs(disassemblies) do
            emit_sass(input.path, input.arch, function(image, sass, image_path)
                if listing then listing:write("// ", input.path, " ("..image..")\n", sass, "\n") end
                if summarize then
                    local crc
                    if get_crc then
                        local message
                        crc, message = get_crc(image_path)
                        if not crc then error("Could not identify "..image..". "..message, 0) end
                    end
                    for _, kernel in ipairs(instruction_counts(image, sass, input.image_arch, crc, sample_index and sample_index[crc])) do
                        if kernel.instruction_count > 0 and (not input.arch or kernel.architecture == input.arch) then
                            kernel.source = input.path
                            stats[#stats + 1] = kernel
                            if kernel.sampled_instructions then
                                matched_instructions = matched_instructions + #kernel.sampled_instructions
                            end
                        end
                    end
                end
            end, sample_index ~= nil)
        end
        if listing then
            listing:close()
            Message:print("RPROF: SASS output written to "..base..".sass")
        end
        if summarize then
            if #stats == 0 then fail("No SASS instructions could be parsed from the disassembly.") end
            if sample_index and matched_instructions == 0 then
                fail("No PC samples matched an inspected cubin CRC, function symbol and instruction offset.")
            end
            ratelprof.utils.store_json({kernels = stats}, base..".sass.json")
            Message:print("RPROF: SASS instruction counts written to "..base..".sass.json")
            if sample_index then
                Message:print("RPROF: annotated "..matched_instructions.." SASS instruction(s) with PC samples")
            end
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
