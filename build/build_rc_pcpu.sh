#!/bin/bash

cp perlmt-cpu.Makefile perlmt-cpu.Makefile2;
make clean -f perlmt-cpu.Makefile && make -f perlmt-cpu.Makefile && mv dqmc_1 dqmc_1_c && mv dqmc_stack dqmc_stack_c;
sed -i -e 's/CFLAGS += -DUSE_CPLX/#CFLAGS += -DUSE_CPLX/g' perlmt-cpu.Makefile2;
cat perlmt-cpu.Makefile2;
make clean -f perlmt-cpu.Makefile && make -f perlmt-cpu.Makefile2 && mv dqmc_1 dqmc_1_r && mv dqmc_stack dqmc_stack_r;
make clean -f perlmt-cpu.Makefile;
rm perlmt-cpu.Makefile2

