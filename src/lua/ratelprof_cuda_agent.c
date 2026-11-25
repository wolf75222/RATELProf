#define _GNU_SOURCE
#include <cuda.h>
#include <lauxlib.h>
#include <lua.h>

#include <dlfcn.h>
#include <stdio.h>

static void string_field(lua_State *lua, const char *key, const char *value)
{
    lua_pushstring(lua, value);
    lua_setfield(lua, -2, key);
}

static void number_field(lua_State *lua, const char *key, lua_Number value)
{
    lua_pushnumber(lua, value);
    lua_setfield(lua, -2, key);
}

static int failure(lua_State *lua, CUresult status)
{
    const char *error = "CUDA driver error";
    cuGetErrorString(status, &error);
    lua_pushnil(lua);
    lua_pushstring(lua, error);
    return 2;
}

static int get_devices(lua_State *lua)
{
    int count;
    CUresult status = cuInit(0);
    if (status != CUDA_SUCCESS)
        return failure(lua, status);
    status = cuDeviceGetCount(&count);
    if (status != CUDA_SUCCESS)
        return failure(lua, status);

    typedef CUresult (CUDAAPI *uuid_query)(CUuuid *, CUdevice);
    uuid_query get_uuid = (uuid_query)dlsym(RTLD_DEFAULT, "cuDeviceGetUuid_v2");
    if (!get_uuid)
        get_uuid = (uuid_query)dlsym(RTLD_DEFAULT, "cuDeviceGetUuid");
    lua_createtable(lua, count, 0);
    for (int i = 0; i < count; ++i) {
        CUdevice device;
        CUuuid uuid;
        char name[256], arch[32], id[37] = "";
        int major, minor, sms, warp;
        size_t memory;

        if ((status = cuDeviceGet(&device, i)) != CUDA_SUCCESS ||
            (status = cuDeviceGetName(name, sizeof(name), device)) != CUDA_SUCCESS ||
            (status = cuDeviceTotalMem(&memory, device)) != CUDA_SUCCESS ||
            (status = cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device)) != CUDA_SUCCESS ||
            (status = cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device)) != CUDA_SUCCESS ||
            (status = cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device)) != CUDA_SUCCESS ||
            (status = cuDeviceGetAttribute(&warp, CU_DEVICE_ATTRIBUTE_WARP_SIZE, device)) != CUDA_SUCCESS)
            return failure(lua, status);

        if (get_uuid && get_uuid(&uuid, device) == CUDA_SUCCESS) {
            const unsigned char *b = (const unsigned char *)uuid.bytes;
            snprintf(id, sizeof(id), "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                     b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]);
        }
        snprintf(arch, sizeof(arch), "sm_%d%d", major, minor);

        lua_createtable(lua, 0, 7);
        string_field(lua, "name", name);
        string_field(lua, "uuid", id);
        string_field(lua, "arch", arch);
        number_field(lua, "ordinal", i);
        number_field(lua, "sm_count", sms);
        number_field(lua, "warp_size", warp);
        number_field(lua, "memory", memory);
        lua_rawseti(lua, -2, i + 1);
    }
    return 1;
}

int luaopen_ratelprof_cuda_agent(lua_State *lua)
{
    lua_newtable(lua);
    lua_pushcfunction(lua, get_devices);
    lua_setfield(lua, -2, "get_devices");
    return 1;
}
