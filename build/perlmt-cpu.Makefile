#Req: PrgEnv-gnu. cray-hdf5, cray-libsci
CC = cc

CFLAGS = -std=gnu17 -Ofast -march=native
CFLAGS += -Wall -Wextra -Wno-discarded-qualifiers
#CFLAGS += -DMKL_DIRECT_CALL_SEQ -Wno-ignored-pragmas
CFLAGS += -DGIT_ID=\"$(shell git rev-parse --short HEAD)\"
CFLAGS += -DGIT_REPO=\"$(shell git config --get remote.origin.url)\"
CFLAGS += -DOMP_MEAS_NUM_THREADS=6
CFLAGS += -DPROFILE_ENABLE
CFLAGS += -DUSE_CPLX  # uncomment to use complex numbers
CFLAGS += -fopenmp  # to disable openmp, use -qopenmp-stubs

LDFLAGS += -lhdf5 -lhdf5_hl

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
