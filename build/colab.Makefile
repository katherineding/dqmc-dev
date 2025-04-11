#Req: PrgEnv-gnu. cray-hdf5, cray-libsci
CC = gcc

CFLAGS = -std=gnu17 -Ofast -march=native
CFLAGS += -Wall -Wextra -Wno-discarded-qualifiers
#CFLAGS += -DMKL_DIRECT_CALL_SEQ -Wno-ignored-pragmas
CFLAGS += -DGIT_ID=\"$(shell git rev-parse --short HEAD)\"
CFLAGS += -DGIT_REPO=\"$(shell git config --get remote.origin.url)\"
CFLAGS += -DOMP_MEAS_NUM_THREADS=6
CFLAGS += -DPROFILE_ENABLE
CFLAGS += -DGDRIVE # only use softlinks if stack file is on google Drive
CFLAGS += -DUSE_CPLX  # uncomment to use complex numbers
CFLAGS += -fopenmp  # to disable openmp, use -qopenmp-stubs

#Extra -I flags to specify header locations
CFLAGS += -I/usr/include/hdf5/serial/
CFLAGS += -I/usr/include/mkl/
#Extra -L flags to specify library locations
LDFLAGS += -L/usr/lib/x86_64-linux-gnu/hdf5/serial/
LDFLAGS += -L/usr/lib/x86_64-linux-gnu/mkl/

# these lines are not necessary after running ICX setvars.sh, which sets
# $CPATH and $LD_LIBRARY_PATH
# CFLAGS += -I${MKLROOT}/include
# LDFLAGS += -L${MKLROOT}/lib/intel64

LDFLAGS += -lhdf5 -lhdf5_hl 
LDFLAGS += -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

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
