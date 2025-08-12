#!/bin/bash

cp Makefile Makefile2;
make clean -f Makefile2;
sed -i -e 's/-xHost/-axIVYBRIDGE,HASWELL,BROADWELL,SKYLAKE,SKYLAKE-AVX512/g' Makefile2 && cat Makefile2;
make -f Makefile2 && mv dqmc_1 dqmc_1_r && mv dqmc_stack dqmc_stack_r;
sed -i -e 's/#CFLAGS += -DUSE_CPLX/CFLAGS += -DUSE_CPLX/g' Makefile2 && cat Makefile2;
make clean -f Makefile2 && make -f Makefile2 && mv dqmc_1 dqmc_1_c && mv dqmc_stack dqmc_stack_c;
make clean -f Makefile2;
rm Makefile2

