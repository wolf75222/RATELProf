#pragma once

#include <stdint.h>

#include "argument_values.h"

enum HostDomain { Mpi, OmpRoutine, OmpTarget, Ompt, Cublas, Nccl };

uint64_t host_begin(HostDomain domain);
bool host_arguments_enabled(HostDomain domain);
void host_end(HostDomain domain, const char *name, uint64_t start, const void *caller,
              ArgumentValue result = int64_t{0}, bool has_result = false,
              TraceArguments arguments = {}, bool instantaneous = false);
