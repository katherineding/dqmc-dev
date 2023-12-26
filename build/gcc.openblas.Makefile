CC = gcc

#remember to adjust march flag before compiling
CFLAGS = -std=gnu17 -O0 -g -march=native
CFLAGS += -Wall -Wextra -Wno-unused-variable -Wno-unused-parameter
CFLAGS += -DMKL_DIRECT_CALL_SEQ 
CFLAGS += -DGIT_ID=\"$(shell git rev-parse --short HEAD)\"
CFLAGS += -DGIT_REPO=\"$(shell git config --get remote.origin.url)\"
CFLAGS += -DOMP_MEAS_NUM_THREADS=2
CFLAGS += -DPROFILE_ENABLE
CFLAGS += -DUSE_CPLX  # uncomment to use complex numbers
CFLAGS += -fopenmp  

# Linear algebra library: FIXME: change lapacke.h -> lapack.h
CFLAGS += -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.5/compilers/include/lp64/
LDFLAGS = -lm -lopenblas

# HDF5
CFLAGS += -I/usr/include/hdf5/serial
LDFLAGS += -L/usr/lib/x86_64-linux-gnu/hdf5/serial
LDFLAGS += -lhdf5 -lhdf5_hl 

#LDFLAGS += -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
SRCFILES = data.o dqmc.o greens.o meas.o prof.o sig.o updates.o

all: one stack

one: ${SRCFILES} main_1.o
	@echo linking dqmc_1
	@${CC} ${CFLAGS} -o dqmc_1 $? ${LDFLAGS}

stack: ${SRCFILES} main_stack.o
	@echo linking dqmc_stack
	@${CC} ${CFLAGS} -o dqmc_stack $? ${LDFLAGS}

%.o: ../src/%.c
	@echo compiling $<
	@${CC} -c ${CFLAGS} $<

clean:
	rm -f *.o *.optrpt *.seq *.par

