#!/bin/bash

cp colab.Makefile colab.Makefile2;
make clean -f colab.Makefile && make -f colab.Makefile && mv dqmc_1 dqmc_1_c && mv dqmc_stack dqmc_stack_c;
sed -i -e 's/CFLAGS += -DUSE_CPLX/#CFLAGS += -DUSE_CPLX/g' colab.Makefile2;
cat colab.Makefile2;
make clean -f colab.Makefile && make -f colab.Makefile2 && mv dqmc_1 dqmc_1_r && mv dqmc_stack dqmc_stack_r;
make clean -f colab.Makefile;
rm colab.Makefile2

