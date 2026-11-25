local stats = require("utils.stats_helper")

return function(domain, summary)
    return function(data, _, options)
        local name = ratelprof.consts._DOMAIN_NAME[domain]
        local process = summary and stats.process_api_raw_data_for_sum_report
                                 or stats.process_api_raw_data_for_trace_report
        local rows, header = process(data, domain, options)
        return {NAME = name, TYPE = summary and "Summary" or "Traces", DATA = rows, HEADER = header}
    end
end
