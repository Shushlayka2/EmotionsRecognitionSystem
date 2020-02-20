#pragma once

extern unsigned int seed;

float* set_normal_random(const int arr_size, const int depth, size_t& pitch, const float sigma, bool is2dim = false);

float* set_repeatable_values(const int arr_size, const float custom_val);