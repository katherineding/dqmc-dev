#!/bin/bash

cp mobius.Makefile mobius.Makefile2;
make clean -f mobius.Makefile && make -f mobius.Makefile && mv dqmc_1 dqmc_1_c && mv dqmc_stack dqmc_stack_c;
sed -i -e 's/CFLAGS += -DUSE_CPLX/#CFLAGS += -DUSE_CPLX/g' mobius.Makefile2;
cat mobius.Makefile2;
make clean -f mobius.Makefile && make -f mobius.Makefile2 && mv dqmc_1 dqmc_1_r && mv dqmc_stack dqmc_stack_r;
make clean -f mobius.Makefile;
rm mobius.Makefile2
