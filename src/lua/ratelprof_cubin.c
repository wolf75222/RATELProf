#include <cupti_pcsampling.h>
#include <lauxlib.h>
#include <lua.h>

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

static int failure(lua_State *lua, const char *message)
{
    lua_pushnil(lua);
    lua_pushstring(lua, message);
    return 2;
}

static int get_crc(lua_State *lua)
{
    const char *path = luaL_checkstring(lua, 1);
    FILE *file = fopen(path, "rb");
    if (!file)
        return failure(lua, "cannot open the cubin");

    long size = fseek(file, 0, SEEK_END) == 0 ? ftell(file) : -1;
    if (size <= 0 || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return failure(lua, "cannot determine the cubin size");
    }
    void *image = malloc((size_t) size);
    if (!image) {
        fclose(file);
        return failure(lua, "cannot allocate memory for the cubin");
    }
    size_t bytes_read = fread(image, 1, (size_t) size, file);
    fclose(file);
    if (bytes_read != (size_t) size) {
        free(image);
        return failure(lua, "cannot read the whole cubin");
    }

    CUpti_GetCubinCrcParams request = {0};
    request.size = CUpti_GetCubinCrcParamsSize;
    request.cubinSize = bytes_read;
    request.cubin = image;
    CUptiResult status = cuptiGetCubinCrc(&request);
    free(image);
    if (status != CUPTI_SUCCESS) {
        const char *error = "cuptiGetCubinCrc failed";
        cuptiGetResultString(status, &error);
        return failure(lua, error);
    }

    char crc[sizeof(uint64_t) * 2 + 3];
    snprintf(crc, sizeof(crc), "0x%" PRIx64, request.cubinCrc);
    lua_pushstring(lua, crc);
    return 1;
}

int luaopen_ratelprof_cubin(lua_State *lua)
{
    lua_newtable(lua);
    lua_pushcfunction(lua, get_crc);
    lua_setfield(lua, -2, "get_crc");
    return 1;
}
