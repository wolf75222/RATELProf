#pragma once

bool init_module_capture(const char *directory);
void capture_cuda_module(const void *callback_data);
void finish_module_capture();
