#!/bin/bash
rm *.gcda *.gcno *.gcov
make clean && make \
    && sh genfiles.sh \
    && ./dqmc_stack stack \
    && ./dqmc_1 single_0.h5 \
    && gcov *.gcda -b \
    && gcovr --use-gcov-files --keep --root ../src/ . --html-details --output cover.html
