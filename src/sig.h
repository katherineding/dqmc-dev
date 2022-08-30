#pragma once

#include <stdio.h>
#include "time_.h"

void sig_init(FILE *_log, const int64_t _wall_start, const int64_t _max_time);

int sig_check_state(const int sweep, const int n_sweep_warm, const int n_sweep);
