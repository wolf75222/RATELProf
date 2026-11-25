local env           = require ("commands.profile.env")
local agent_helper  = require ("commands.inspect.agent_helper")

local profile = {}
local cuda = ratelprof.consts._BACKEND == "cuda"
local quote = ratelprof.utils.shell_quote

-- Function to handle the options for the profile command
local function handle_profile_option(options)
    local output = ratelprof.get_opt_val(options, "output")
    if output then env.set_env(ratelprof.consts._ENV.OUTPUT_FILE, output) end

    local whitelist = ratelprof.get_opt_val(options, "whitelist")
    if whitelist then env.set_filter_env_var(whitelist, 2) end

    local blacklist = ratelprof.get_opt_val(options, "blacklist")
    if blacklist then env.set_filter_env_var(blacklist, 1) end

    local log_level = ratelprof.get_opt_val(options, "log-level")
    if log_level then env.set_env(ratelprof.consts._ENV.LOG_LEVEL, log_level) end

    local enabled_domain, domain_flags = {}, {}
    local function enable_domain(var)
        if var then
            domain_flags[var] = true
            env.set_env(var, "1")
        end
    end
    local traces = ratelprof.get_opt_val(options, "trace")
    local traces_list = (cuda and traces:match("^%s*none%s*$")) and {"none"} or traces:split(",")
    for _, trace_name in ipairs(traces_list) do
        trace_name = trace_name:trim()
        local trace = ratelprof.consts._TRACES[trace_name]
        if trace then
            if type(trace.var) == "table" then
                for _, var in ipairs(trace.var) do
                    enable_domain(var)
                end
            else
                enable_domain(trace.var)
            end
            table.insert(enabled_domain, trace.name)
        else
            Message:warn("Trace '"..trace_name.."' is not supported. Skipping it.")
        end
    end

    local plugin_path = ratelprof.get_opt_val(options, "plugin")
    if cuda then
        if plugin_path ~= "msgpack" and plugin_path ~= "stdout" then
            if not ratelprof.fs.exists(plugin_path) or ratelprof.fs.is_dir(plugin_path) then
                Message:error("CUDA output plugin library '"..plugin_path.."' not found.")
                os.exit(1)
            end
            plugin_path = ratelprof.utils.execute_command("realpath -- " .. quote(plugin_path))
            if plugin_path == "" then
                Message:error("Cannot resolve the CUDA output plugin path.")
                os.exit(1)
            end
        end
        env.set_env("RATELPROF_CUDA_PLUGIN", plugin_path)
    elseif plugin_path then
        if plugin_path == "stdout"        then plugin_path = ratelprof.consts._LIBS.PLUGIN_STDOUT
        elseif plugin_path == "msgpack"   then plugin_path = ratelprof.consts._LIBS.PLUGIN_MSGPACK
        elseif plugin_path == "json"      then plugin_path = ratelprof.consts._LIBS.PLUGIN_JSON
        elseif plugin_path == "json-dyn"  then plugin_path = ratelprof.consts._LIBS.PLUGIN_JSON_DYN end
        if not ratelprof.fs.exists(plugin_path) then
            Message:error("Plugin library '"..plugin_path.."' not found.")
            os.exit(1)
        end
        env.set_env(ratelprof.consts._ENV.PLUGIN_PATH, plugin_path) 
    end

    local launch_script     = ratelprof.get_opt_val(options, "launch-script")
    local launch_command    = ratelprof.get_opt_val(options, "launch-command")
    local prefix            = ratelprof.get_opt_val(options, "prefix")
    
    return {
        launch_script = launch_script,
        launch_command = launch_command,
        prefix = prefix,
        plugin = plugin_path,
        enabled_domain = enabled_domain,
        domain_flags = domain_flags,
    }
end


local function normal_execution(cmd)
    local ret_code = ratelprof.utils.exit_status(os.execute(cmd))
    if ret_code ~= 0 then
        Message:error("Application execution failed - code("..ret_code..").")
        Message:error("Run command : '"..cmd.."'")
        os.exit(ret_code)
    end
end

local function script_execution(cmd, launch_command, launch_script)
    local file = ratelprof.fs.open_file(launch_script, "r")
    local script_content = file:read("*all")
    file:close()
    
    file = ratelprof.fs.open_file(launch_script..".tmp", "w")
    file:write((script_content:gsub("<run_command>", function() return cmd end)))
    file:close()

    normal_execution(launch_command .. " " .. quote(launch_script..".tmp"))
end

local function run_application(run_command, opt)
    if opt.launch_script and opt.launch_command then
        script_execution(run_command, opt.launch_command, opt.launch_script)
    else
        normal_execution(run_command)
    end
end

function profile.process_profiling(positional_args, options_values)
    local application_command = positional_args
    local application = application_command[1]
    
    local opt = handle_profile_option(options_values)
    local launch_script     = opt.launch_script
    local launch_command    = opt.launch_command
    local prefix            = opt.prefix .. " "
    local plugin_path       = opt.plugin

    if not application or (not cuda and not ratelprof.fs.exists(application)) then
        Message:print("No application to profile or '"..application.."' doesn't exists.")
        os.exit(1)
    end

    local command = {}
    for _, arg in ipairs(application_command) do
        command[#command + 1] = quote(arg)
    end
    local app_cmd = table.concat(command, " ")

    if cuda then
        local libdir = ratelprof.consts._LIBS_DIR
        local lib = libdir.."libratelprof_cuda.so"
        local libraries = {lib}
        local flags, E = opt.domain_flags, ratelprof.consts._ENV
        if flags[E.DOMAIN_OMP_TGT] or flags[E.DOMAIN_OMPT] or flags[E.DOMAIN_OMP_TGT_RTL] then
            libraries[#libraries + 1] = libdir.."libratelprof_cuda_omp.so"
        end
        local mpi = libdir.."libratelprof_cuda_mpi.so"
        if flags[E.DOMAIN_MPI] and ratelprof.fs.exists(mpi) then
            libraries[#libraries + 1] = mpi
        end
        for domain, library in pairs(ratelprof.consts._CUDA_LIBRARY_WRAPPERS) do
            if flags[domain] then
                libraries[#libraries + 1] = library
            end
        end
        local inherited = os.getenv("LD_PRELOAD")
        if inherited and inherited ~= "" then libraries[#libraries + 1] = inherited end
        env.set_env("LD_PRELOAD", table.concat(libraries, ":"))
        env.set_env("NVTX_INJECTION64_PATH", lib)
        local api_args = options_values["api-args"]
        if api_args ~= "basic" and api_args ~= "all" and api_args ~= "none" then
            Message:error("--api-args expects basic, all or none.")
            os.exit(1)
        end
        env.set_env("RATELPROF_CUDA_API_ARGS", api_args)
        if options_values["dump-cubins"] then
            local cubins = options_values["dump-cubins"]
            if cubins:sub(1, 1) ~= "/" then
                cubins = ratelprof.utils.execute_command("pwd -P").."/"..cubins
            end
            env.set_env("RATELPROF_DUMP_CUBINS", cubins)
        end
        Message:print("RPROF: CUDA profiling enabled for "..table.concat(opt.enabled_domain, ", "))
        run_application(prefix.."env -u RATELPROF_CUDA_OWNER "..env.get_env().." "..app_cmd, opt)
        return
    end
    
    local ld_library_path = os.getenv ("LD_LIBRARY_PATH")
    local preload_libs = ratelprof.consts._PRELOADED_LIBS
    local ld_preload = {}

    for _, lib in ipairs(preload_libs) do
        if not ratelprof.fs.exists(lib.path) then
            lib.path = nil
            if not ld_library_path then
                Message:error("LD_LIBRARY_PATH is not set.")
                os.exit (1)
            end
            for path in ld_library_path:gmatch ("([^:]+)") do
                local lib_path = path .. "/" .. lib.name
                if ratelprof.fs.exists (lib_path) then
                    lib.path = lib_path
                    break
                end
            end
            if not lib.path then
                Message:error("Required library '"..lib.name.."' can't be find.")
                os.exit (1)
            end
        end
        table.insert(ld_preload, lib.path)
    end


    env.set_number_of_kernel_env_var(application)
    env.set_number_of_queue_env_var()
    env.set_env("LD_PRELOAD", table.concat(ld_preload, ":"))

    local env_var = env.get_env()
    local run_command = prefix.."env "..env_var.." "..app_cmd

    Message:print ([[
     ____      _  _____ _____ _     ____             __ 
    |  _ \    / \|_   _| ____| |   |  _ \ _ __ ___  / _|
    | |_) |  / _ \ | | |  _| | |   | |_) | '__/ _ \| |_ 
    |  _ <  / ___ \| | | |___| |___|  __/| | | (_) |  _|
    |_| \_\/_/   \_\_| |_____|_____|_|   |_|  \___/|_|  
]])
    Message:print ("RPROF: Application profiled :    '"..application.."'")
    Message:print ("RPROF: Application Command :     '"..prefix..app_cmd.."'")
    Message:print ("RPROF: Preloaded tool :          '"..ld_preload[1].."'")
    Message:print ("RPROF: Preloaded wrapper :       '"..ld_preload[2].."'")
    Message:print ("RPROF: Plugin used :             '"..plugin_path.."'")
    Message:print ("RPROF: Profiling enabled for :   '"..table.concat(opt.enabled_domain, ", ").."'")

    if launch_command and launch_script and ratelprof.fs.exists(launch_script) then
        Message:print ("RPROF: Running :                 '"..launch_command.." "..launch_script.."'")
        run_application(run_command, opt)
    else
        normal_execution (run_command)
    end

    local f = ratelprof.fs.open_file("/tmp/rprof_output_filename.txt", "r")
    for output_file in f:lines() do
        agent_helper.set_gpu_props_to_msgpack(output_file)
        local bytes_written = ratelprof.fs.get_size(output_file)
        Message:print ("RPROF: Bytes written in '"..output_file.."' : "..bytes_written)
    end
    f:close()
    os.remove("/tmp/rprof_output_filename.txt")

    Message:print ("RPROF: Exiting tool . . .")
end

return profile
