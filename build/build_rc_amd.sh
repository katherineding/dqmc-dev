#!/bin/bash

cp aocc.Makefile aocc.Makefile2;
make clean -f aocc.Makefile && make -f aocc.Makefile && mv dqmc_1 dqmc_1_c && mv dqmc_stack dqmc_stack_c;
sed -i -e 's/CFLAGS += -DUSE_CPLX/#CFLAGS += -DUSE_CPLX/g' aocc.Makefile2;
cat aocc.Makefile2;
make clean -f aocc.Makefile && make -f aocc.Makefile2 && mv dqmc_1 dqmc_1_r && mv dqmc_stack dqmc_stack_r;
make clean -f aocc.Makefile;
rm aocc.Makefile2
