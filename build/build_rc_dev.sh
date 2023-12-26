#!/bin/bash

cp dev-gpu.Makefile dev-gpu.Makefile2;
make clean -f dev-gpu.Makefile && make -f dev-gpu.Makefile && mv dqmc_1 dqmc_1_c && mv dqmc_stack dqmc_stack_c;
sed -i -e 's/CFLAGS += -DUSE_CPLX/#CFLAGS += -DUSE_CPLX/g' dev-gpu.Makefile2;
cat dev-gpu.Makefile2;
make clean -f dev-gpu.Makefile && make -f dev-gpu.Makefile2 && mv dqmc_1 dqmc_1_r && mv dqmc_stack dqmc_stack_r;
make clean -f dev-gpu.Makefile ;
rm dev-gpu.Makefile2
