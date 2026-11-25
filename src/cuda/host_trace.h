#pragma once

#include <stdint.h>

enum HostDomain { Mpi, OmpRoutine, OmpTarget, Ompt, Cublas, Nccl };

uint64_t host_begin(HostDomain domain);
void host_end(HostDomain domain, const char *name, uint64_t start, const void *caller,
              int64_t result = 0, bool has_result = false);
