# Instructions for testing

## Unit tests

// TODO

## Integrated correctness tests against reference commits

Prerequisite: Generate reference files and run with known good dqmc implementation and put completed runs in test/ref. Script for generating reference files: util/makeref.py

With reference files present in test/ref,
1. Go to build/
2. Based on branch, compiler, linalg libary, and gpu offloading yes/no, pick a `xxx`, do `bash build_rc_xxx.sh` to generate dqmc_1_c, dqmc_1_r, dqmc_stack_c, dqmc_stack_r executables
3. Go to test/
4. Run `pytest -v -s`

