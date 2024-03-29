#pragma once

#include <stdint.h>
#include <time.h>

#define TICK_PER_SEC INT64_C(1000000000) //int64
#define SEC_PER_TICK 1e-9
#define US_PER_TICK 1e-3

//return monotonic wall time in units of nanoseconds
// clock_gettime is part of time.h on POSIX compliant systems
static inline int64_t time_wall(void)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec * TICK_PER_SEC + t.tv_nsec;
}