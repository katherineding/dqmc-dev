#!/bin/bash

cp perlmt-gpu.Makefile perlmt-gpu.Makefile2;
make clean -f perlmt-gpu.Makefile && make -f perlmt-gpu.Makefile && mv dqmc_1 dqmc_1_c && mv dqmc_stack dqmc_stack_c;
sed -i -e 's/CFLAGS += -DUSE_CPLX/#CFLAGS += -DUSE_CPLX/g' perlmt-gpu.Makefile2;
cat perlmt-gpu.Makefile2;
make clean -f perlmt-gpu.Makefile && make -f perlmt-gpu.Makefile2 && mv dqmc_1 dqmc_1_r && mv dqmc_stack dqmc_stack_r;
make clean -f perlmt-gpu.Makefile;
rm perlmt-gpu.Makefile2

