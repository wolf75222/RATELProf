-- visualize.lua
local preprocess    = require ("commands.visualize.preprocess")
local reports       = require ("commands.visualize.reports")
local main_panel    = require ("commands.visualize.main_panel")

local summarize         = require ("commands.summarize")
local summarize_report  = require ("commands.summarize.summarize_report")

local BinaryReport  = require ("utils.Classes.BinaryReport")

local visualize = {}

local function webgl(data, output, options, measurements)
    local quote = ratelprof.utils.shell_quote
    local directory = ratelprof.utils.execute_command("mktemp -d")
    if directory == "" then error("Could not create a temporary WebGL export directory") end

    local ok, result = pcall(function()
        options.report_prefix = directory.."/report"
        local summary = summarize.get_summary(data)
        local analyzed, statistics, rules = summarize_report.get_analyzed_data(data, options)
        local payload = {
            report = data:to_json(),
            summary = JSON:decode(main_panel.get_summary_data_as_js_string(summary, analyzed)),
            statistics = statistics[-1] or {},
            analyze = rules[-1] or {},
            measurements = measurements,
            domains = ratelprof.consts._DOMAIN_NAME,
            start = tonumber(options.start),
            stop = tonumber(options.stop),
        }
        local input = directory.."/report.json"
        local file = ratelprof.fs.open_file(input, "w")
        file:write(ratelprof.utils.generate_json(payload))
        file:close()
        local exporter = ratelprof.consts._MODULES_DIR.."webgl/export.py"
        local command = "python3 "..quote(exporter).." "..quote(input).." "..
                        quote(options.viewer).." "..quote(output)
        return ratelprof.utils.exit_status(os.execute(command))
    end)
    os.execute("rm -rf -- "..quote(directory))
    if not ok then error(result, 0) end
    if result ~= 0 then os.exit(result) end
    Message:print("RPROF: RATELProf-Visualize report written to "..output.."/index.html")
end

function visualize.process_visualizing(positional_args, options_values)
    local measurements = {}
    for _, name in ipairs({"metrics-json", "pc-samples", "source-csv"}) do
        if options_values[name] then
            if not options_values.viewer then
                Message:error("--"..name.." requires --viewer")
                os.exit(1)
            end
            measurements[name] = options_values[name]
        end
    end

    local data = BinaryReport:new(positional_args, {
        start = tonumber(options_values.start),
        stop = tonumber(options_values.stop)
    })

    local output = ratelprof.get_opt_val(options_values, "output")
    if not output then
        output = data:get_report_basename()
        if ratelprof.fs.exists(output) and not ratelprof.fs.is_dir(output) then
            output = output.."_visualize"
        end
    end
    if options_values.viewer then
        webgl(data, output, options_values, measurements)
        return
    end
    local tmp_dir        = ratelprof.get_opt_val(options_values, "tmp")
    ratelprof.fs.mkdir(tmp_dir)
    if not ratelprof.fs.mkdir(output) then
        Message:error("Could not create visualization directory "..output)
        os.exit(1)
    end

    local timeline_data = preprocess.start(data, output, options_values)

    options_values["with-analysis"] = true

    local gpu_labels, gpu_props = main_panel.get_gpu_data_as_js_string(data)
    local global_data = summarize.get_summary(data)
    local analyzed_data, stats_ret_vals, analyze_ret_vals = summarize_report.get_analyzed_data(data, options_values)
    local summary = main_panel.get_summary_data_as_js_string(global_data, analyzed_data)

    local string_ext_array = ratelprof.msgpack.encode.get_ext_string_table()

    local csv_content = reports.get_csv_data(stats_ret_vals, analyze_ret_vals)

    local timeline_lifecycle = {}
    data:for_each_rank(function (mpi_rank, _)
        timeline_lifecycle[tostring(mpi_rank)] = {
            data:get_main_start(mpi_rank),
            data:get_main_stop(mpi_rank)
        }
    end)

    local handlers = {
        {HANDLE = "HTML_TITLE",             data = JSON:encode(data:get_report_basename())},
        {HANDLE = "TIMELINE_LIFECYCLE",     data = JSON:encode(timeline_lifecycle)},
        {HANDLE = "TIMELINE_MAX",           data = data:get_max_stop_time()},
        {HANDLE = "TIMELINE_DATA",          data = timeline_data},
        {HANDLE = "CHART_DATA",             data = JSON:encode(global_data.plot_data)},
        {HANDLE = "EXT_STRINGS",            data = JSON:encode(string_ext_array)},
        {HANDLE = "GPU_LABELS",             data = gpu_labels},
        {HANDLE = "GPU_DATA",               data = gpu_props},
        {HANDLE = "REPORT_DATA",            data = summary},
        {HANDLE = "CSV_DATA_PER_TAB",       data = csv_content},
    }

    local data_content = "let "
    for i, h in ipairs(handlers) do
        data_content = data_content .. h.HANDLE.. '=' .. h.data..(i ~= #handlers and ',' or '')
    end

    local data_output = ratelprof.fs.concat_path(output, '.data/.data.js')
    local html_output = ratelprof.fs.concat_path(output, ratelprof.fs.basename(output)..'.html')

    if not ratelprof.fs.copy_file(ratelprof.consts._HTML_REPORT_PATH, html_output) then
        Message:error("Could not write visualization "..html_output)
        os.exit(1)
    end

    local file = ratelprof.fs.open_file(data_output, "w")
    file:write(data_content)
    file:close()

    Message:print("RPROF: HTML report written to '" .. html_output.."'")
end

return visualize
