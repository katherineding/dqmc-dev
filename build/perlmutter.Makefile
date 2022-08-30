CC = nvc++ #use nvc to compile all files

CFLAGS = -fast -mp=gpu -gpu=cc80,cuda11.7 -Minfo 
CFLAGS += -cuda #tell nvc that .c files might contain GPU device code
CFLAGS += -cudalib=cublas,cusolver,curand #link to libraries
CFLAGS += -DGIT_ID=\"$(shell git describe --always)\"
#don't need Edwin's profiling if using Nsight instead
#CFLAGS += -DPROFILE_ENABLE 
#CFLAGS += -DUSE_CPLX  # uncomment to use complex numbers
CFLAGS += -I/opt/cray/pe/hdf5/1.12.0.7/nvidia/20.7/include/

#LDFLAGS += -L/usr/lib/x86_64-linux-gnu/hdf5/serial
LDFLAGS += -lhdf5 -lhdf5_hl
LDFLAGS += -L/opt/cray/pe/hdf5/default/nvidia/20.7/lib #might cause problem b/c this is old
LDFLAGS += -L/opt/nvidia/hpc_sdk/Linux_x86_64/21.9/math_libs/11.4/lib64 #cuda library locations
SRCFILES = data.o dqmc.o greens.o meas.o updates.o

all: one stack

test: ../src/testlinalg.c
	@echo compiling $<, testing cuda api
	@${CC} ${CFLAGS} -o testlinalg $? ${LDFLAGS}

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
