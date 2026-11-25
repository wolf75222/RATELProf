local elf = {}
local quote = ratelprof.utils.shell_quote

local function normalize(path, capture)
    local output, status = capture("realpath -- "..quote(path))
    if status == 0 then return output:gsub("\n$", "") end
end

local function elf_magic(path)
    local file = io.open(path, "rb")
    if not file then return false end
    local magic = file:read(4)
    file:close()
    return magic == "\127ELF"
end

local function metadata(path, capture)
    local output, status = capture("LC_ALL=C readelf -hW -dW -- "..quote(path))
    if status ~= 0 then return nil, output end

    local info = {needed = {}}
    for line in output:gmatch("[^\r\n]+") do
        local tag, value = line:match("%(([A-Z]+)%)%s+[^%[]*%[([^%]]*)%]")
        if tag == "NEEDED" then
            info.needed[#info.needed + 1] = value
        elseif tag == "RPATH" then
            info.rpath = value
        elseif tag == "RUNPATH" then
            info.runpath = value
        elseif tag == "SONAME" then
            info.soname = value
        end
        info.class = info.class or line:match("^%s*Class:%s+(ELF%d+)")
        info.machine = info.machine or line:match("^%s*Machine:%s+(.+)")
        info.endian = info.endian or line:match("^%s*Data:%s+(.+)")
    end

    if not info.class or not info.machine or not info.endian then
        return nil, "Unable to read the ELF header from "..path
    end
    return info
end

local platforms = {
    ["Advanced Micro Devices X86-64"] = "x86_64",
    ["Intel 80386"] = "i386",
    ["AArch64"] = "aarch64",
    ["ARM"] = "arm",
}

local function profile_paths(info)
    local platform = platforms[info.machine]
    local multiarch = platform and platform.."-linux-gnu"
    local lib = info.class == "ELF64" and "lib64" or "lib"
    local default_dirs = {"/"..lib, "/usr/"..lib}
    if multiarch then
        default_dirs[#default_dirs + 1] = "/lib/"..multiarch
        default_dirs[#default_dirs + 1] = "/usr/lib/"..multiarch
    end
    if lib ~= "lib" then
        default_dirs[#default_dirs + 1] = "/lib"
        default_dirs[#default_dirs + 1] = "/usr/lib"
    end
    return platform or "", lib, default_dirs
end

local function loader_cache(capture)
    local output, status = capture("LC_ALL=C ldconfig -p")
    if status ~= 0 then
        output, status = capture("LC_ALL=C /sbin/ldconfig -p")
    end
    local cache = {}
    if status ~= 0 then return cache end
    for line in output:gmatch("[^\r\n]+") do
        local name, path = line:match("^%s*(%S+)%s+%b()%s+=>%s+(/.+)$")
        if name then
            cache[name] = cache[name] or {}
            table.insert(cache[name], path)
        end
    end
    return cache
end

local function token_expander(origin, platform, lib)
    local substitutions = {ORIGIN = origin, PLATFORM = platform, LIB = lib}
    local function expand(key)
        return substitutions[key]
    end

    return function(s)
        return (s:gsub("%${([%w_]+)}", expand):gsub("%$([%w_]+)", expand))
    end
end

local function append_paths(output, list, expand)
    if list == nil then return end
    for path in (list..":"):gmatch("([^:]*):") do
        output[#output + 1] = path == "" and "." or expand(path)
    end
end

local function append(output, items)
    for _, item in ipairs(items) do output[#output + 1] = item end
end

function elf.discover(application, capture)
    local _, status = capture("command -v readelf")
    if status ~= 0 then return nil, "--linked requires GNU readelf on PATH." end

    local root_path = normalize(application, capture)
    if not root_path or not elf_magic(root_path) then
        return nil, "--linked requires an ELF executable or shared library."
    end
    local root, err = metadata(root_path, capture)
    if not root then return nil, err end

    local platform, lib, default_dirs = profile_paths(root)
    local cache = loader_cache(capture)
    local visited, names, unresolved = {}, {}, {}
    local function load_candidate(path)
        if not elf_magic(path) then return nil end
        local full_path = normalize(path, capture)
        if not full_path then return nil end
        local info = visited[full_path] or metadata(full_path, capture)
        if not info or info.class ~= root.class or info.machine ~= root.machine or info.endian ~= root.endian then
            return nil
        end
        return full_path, info
    end

    local env_paths = {}
    local env = os.getenv("LD_LIBRARY_PATH")
    append_paths(env_paths, env and env:gsub(";", ":"), token_expander(ratelprof.fs.dirname(root_path), platform, lib))
    local queue = {{path = root_path, info = root, inherited_rpath = {}}}
    visited[root_path] = root
    local position = 1
    while position <= #queue do
        local node = queue[position]
        position = position + 1
        local expand = token_expander(ratelprof.fs.dirname(node.path), platform, lib)
        local legacy_dirs = {}
        if not node.info.runpath then append_paths(legacy_dirs, node.info.rpath, expand) end
        append(legacy_dirs, node.inherited_rpath)

        local search = {}
        if not node.info.runpath then append(search, legacy_dirs) end
        append(search, env_paths)
        append_paths(search, node.info.runpath, expand)
        local function resolve(needed)
            if needed:find("/", 1, true) then
                return load_candidate(needed)
            end
            for _, dir in ipairs(search) do
                local path, info = load_candidate(dir.."/"..needed)
                if path then return path, info end
            end
            for _, candidate in ipairs(cache[needed] or {}) do
                local path, info = load_candidate(candidate)
                if path then return path, info end
            end
            for _, dir in ipairs(default_dirs) do
                local path, info = load_candidate(dir.."/"..needed)
                if path then return path, info end
            end
        end

        for _, dependency in ipairs(node.info.needed) do
            local needed = expand(dependency)
            if not names[needed] then
                local path, info = resolve(needed)
                if path then
                    names[needed] = path
                    if info.soname then names[info.soname] = path end
                    if not visited[path] then
                        visited[path] = info
                        queue[#queue + 1] = {path = path, info = info, inherited_rpath = legacy_dirs}
                    end
                else
                    unresolved[#unresolved + 1] = {name = needed, required_by = node.path}
                end
            end
        end
    end

    local inputs = {}
    for _, node in ipairs(queue) do inputs[#inputs + 1] = node.path end
    return inputs, unresolved
end

return elf
