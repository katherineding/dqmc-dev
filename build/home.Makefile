CC = nvc++ #use nvc to compile all files

CFLAGS = -fast -mp=gpu -gpu=cc75,cuda11.6 -Minfo 
CFLAGS += -cuda #tell nvc that .c files might contain GPU device code
CFLAGS += -cudalib=cublas,cusolver,curand #link to libraries
CFLAGS += -DGIT_ID=\"$(shell git describe --always)\"
#don't need Edwin's profiling if using Nsight instead
#CFLAGS += -DPROFILE_ENABLE 
#CFLAGS += -DUSE_CPLX  # uncomment to use complex numbers
CFLAGS += -I/usr/include/hdf5/serial

LDFLAGS += -L/usr/lib/x86_64-linux-gnu/hdf5/serial
LDFLAGS += -lhdf5 -lhdf5_hl 

SRCFILES = data.o dqmc.o greens.o meas.o prof.o sig.o updates.o

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
