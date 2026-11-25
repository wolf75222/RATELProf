#!/bin/bash

# Function to display help message
function display_help() {
    cat << EOF
Usage: ratelprof [--help] <command> [options] [command args] 

Commands:
  profile       Run an application and capture its profile into a .rprof-rep file.
  metrics       Collect or import hardware counters with Nsight Compute or nvprof.
  sample        Collect CUDA program counter and warp stall samples with CUPTI.
  import-nvprof Read a legacy nvprof recording into a RATELProf trace report.
  stats         Generate statistics from an existing .rprof-rep file.
  analyze       Run optimization rules on a profile.
  breakdown     Break down CPU and GPU activity by rank, process and device.
  visualize     Generate a HTML Report Viewer to see the traces into a timeline.
  inspect       Inspect an application and output data about kernels from the application.
  summarize     Summarize global information/metrics about a profiling report.
  export        Export a profiling report as JSON or function call information.

Use '$(basename $0) <command> --help' for more information about a specific command.
EOF
    exit 0
}

BIN_DIR=$(dirname "$0")
LUA_DIR=$BIN_DIR/lua
export LUA_PATH="$LUA_DIR/?.lua;;"
export INSTALL_DIR="$(realpath "$BIN_DIR/..")"

if [ "$1" = metrics ]; then
    shift
    exec python3 "$INSTALL_DIR/share/modules/metrics/metrics.py" "$@"
fi

# Function to check if a command exists
command_exists() {
    which "$1" >/dev/null 2>&1
}

# Determine which Lua interpreter to use
if command_exists luajit; then
    LUA_EXEC="luajit"
elif command_exists lua; then
    LUA_EXEC="lua"
else
    echo "Error: Neither LuaJIT nor Lua is installed."
    exit 1
fi

# Check if no argument is provided
if [ -z "$1" ]; then
    echo "Error: No argument provided."
    display_help
fi

CMD="$1"
shift

case "$CMD" in
    --help|-h)
        display_help
        ;;

    profile|stats|analyze|breakdown|visualize|inspect|export|summarize|sample|import-nvprof)
        exec "$LUA_EXEC" "$LUA_DIR/$CMD.lua" "$@"
        ;;

    *)
        echo "Error: Invalid command '$CMD'."
        display_help
        ;;
esac



