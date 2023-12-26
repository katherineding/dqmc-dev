#!/bin/bash

cp gcc.openblas.Makefile gcc.openblas.Makefile2;
make clean && make -f gcc.openblas.Makefile && mv dqmc_1 dqmc_1_c && mv dqmc_stack dqmc_stack_c;
sed -i -e 's/CFLAGS += -DUSE_CPLX/#CFLAGS += -DUSE_CPLX/g' gcc.openblas.Makefile2;
cat gcc.openblas.Makefile2;
make clean && make -f gcc.openblas.Makefile2 && mv dqmc_1 dqmc_1_r && mv dqmc_stack dqmc_stack_r;
make clean;
rm gcc.openblas.Makefile2
