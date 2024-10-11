# Only need to compile for milan: AMD EPYC 7543
CFLAGS = -std=gnu17 -Ofast -march=znver3
CFLAGS += -Wall -Wextra -Wno-unused-variable -Wno-unused-parameter
CFLAGS += -DMKL_DIRECT_CALL_SEQ 
CFLAGS += -DGIT_ID=\"$(shell git rev-parse --short HEAD)\"
CFLAGS += -DGIT_REPO=\"$(shell git config --get remote.origin.url)\"
CFLAGS += -DOMP_MEAS_NUM_THREADS=2
CFLAGS += -DPROFILE_ENABLE
CFLAGS += -DUSE_CPLX  # uncomment to use complex numbers
CFLAGS += -fopenmp  

#might need extra -I flags to specify header locations
#might need extra -L flags to specify library locations
CFLAGS += -I/data1/conda/mobius/include/
LDFLAGS += -L/data1/conda/mobius/lib/

# these lines are not necessary after running ICX setvars.sh, which sets
# $CPATH and $LD_LIBRARY_PATH
# CFLAGS += -I${MKLROOT}/include
# LDFLAGS += -L${MKLROOT}/lib/intel64

LDFLAGS += -lhdf5 -lhdf5_hl 
LDFLAGS += -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

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

