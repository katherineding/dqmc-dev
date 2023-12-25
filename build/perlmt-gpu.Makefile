#Prereq: 
#module load PrgEnv-nvidia nvidia/22.7 cudatoolkit/11.7 craype-accel-nvidia80
#module load cray-hdf5 (1.12) 
#module load cray-libsci
CC = cc #use nvc to compile all files

CFLAGS = -fast -mp=gpu -gpu=cc80,cuda11.7 -Minfo=mp 
CFLAGS += -DGIT_ID=\"$(shell git rev-parse --short HEAD)\"
CFLAGS += -DPROFILE_ENABLE 
CFLAGS += -DGIT_REPO=\"$(shell git config --get remote.origin.url)\"
CFLAGS += -DOMP_MEAS_NUM_THREADS=2
CFLAGS += -DUSE_CPLX  # uncomment to use complex numbers

#LDFLAGS += -L/usr/lib/x86_64-linux-gnu/hdf5/serial
LDFLAGS += -lhdf5 -lhdf5_hl

SRCFILES = meas.o updates.o greens.o dqmc.o data.o prof.o sig.o

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
