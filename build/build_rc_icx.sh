#!/bin/bash

cp Makefile.icx Makefile2.icx;
make clean -f Makefile.icx && make -f Makefile.icx && mv dqmc_1 dqmc_1_c && mv dqmc_stack dqmc_stack_c;
sed -i -e 's/CFLAGS += -DUSE_CPLX/#CFLAGS += -DUSE_CPLX/g' Makefile2.icx;
cat Makefile2.icx;
make clean -f Makefile.icx && make -f Makefile2.icx && mv dqmc_1 dqmc_1_r && mv dqmc_stack dqmc_stack_r;
make clean -f Makefile.icx;
rm Makefile2.icx
