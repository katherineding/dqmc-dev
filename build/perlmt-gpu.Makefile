#Prereq (add to .bashrc): 
#module load PrgEnv-nvidia nvidia cudatoolkit craype-accel-nvidia80
#module load cray-hdf5
#module load cray-libsci

# Compilation works with this module configuration:
#  1) craype-x86-milan     4) xpmem/2.6.2-2.5_2.38__gd067c3f.shasta   7) gpu/1.0               10) PrgEnv-nvidia/8.5.0 (cpe)  13) cudatoolkit/12.2      (g)     16) cray-hdf5/1.12.2.3 (io)
#  2) libfabric/1.15.2.0   5) perftools-base/23.12.0                  8) craype/2.7.30    (c)  11) nvidia/23.9         (g,c)  14) craype-accel-nvidia80
#  3) craype-network-ofi   6) cpe/23.12                               9) cray-dsmml/0.2.2      12) cray-mpich/8.1.28   (mpi)  15) cray-libsci/23.12.5   (math)
CC = cc #use nvc to compile all files

CFLAGS = -fast -mp=gpu -gpu=cc80,cuda12.2 -Minfo=mp 
CFLAGS += -DGIT_ID=\"$(shell git rev-parse --short HEAD)\"
CFLAGS += -DPROFILE_ENABLE 
CFLAGS += -DGIT_REPO=\"$(shell git config --get remote.origin.url)\"
CFLAGS += -DOMP_MEAS_NUM_THREADS=2
CFLAGS += -DGENERIC_LINALG
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
