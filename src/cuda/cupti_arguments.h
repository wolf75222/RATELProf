#pragma once

#include <cupti.h>

#include "argument_values.h"

TraceArguments cupti_arguments(CUpti_CallbackDomain domain, CUpti_CallbackId callback,
                              const void *parameters);
