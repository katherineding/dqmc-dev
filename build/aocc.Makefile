CC = clang

CFLAGS = -std=gnu17 -Ofast -g -march=znver3 #-xHost
CFLAGS += -Wall -Wextra -Wno-unused-variable #-Wno-unused-parameter
CFLAGS += -DGIT_ID=\"$(shell git rev-parse --short HEAD)\"
CFLAGS += -DGIT_REPO=\"$(shell git config --get remote.origin.url)\"
CFLAGS += -DOMP_MEAS_NUM_THREADS=2
CFLAGS += -DPROFILE_ENABLE
CFLAGS += -DGENERIC_LINALG
CFLAGS += -DUSE_CPLX  # uncomment to use complex numbers
CFLAGS += -fopenmp  # to disable openmp, use -qopenmp-stubs

# Careful: CFLAGS and LDFLAGS are suspect to contamination by Ana/Miniconda
CFLAGS += -I/usr/include/hdf5/serial/
LDFLAGS = -L/usr/lib/x86_64-linux-gnu/hdf5/serial/
LDFLAGS += -lhdf5 -lhdf5_hl 

LDFLAGS += -lamdlibm -fsclrlib=AMDLIBM -lamdlibmfast -lm -lopenblas
#LDFLAGS += -Wl,--verbose

#multithreaded version of MKL
# CFLAGS += -DMKL_DIRECT_CALL #-Wno-ignored-pragmas
# LDFLAGS += -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl

OBJS = data.o dqmc.o greens.o meas.o prof.o sig.o updates.o

TARGET = dqmc

all: one stack

one: ${OBJS} main_1.o
	@echo linking ${TARGET}_1
	@${CC} ${CFLAGS} -o ${TARGET}_1 $? ${LDFLAGS}

stack: ${OBJS} main_stack.o
	@echo linking ${TARGET}_stack
	@${CC} ${CFLAGS} -o ${TARGET}_stack $? ${LDFLAGS}

%.o: ../src/%.c
	@echo compiling $<
	@${CC} -c ${CFLAGS} $<

clean:
	rm -f *.o *.optrpt *.seq *.par
