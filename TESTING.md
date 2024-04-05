# Instructions for testing

## Unit tests

// TODO

## Integrated correctness tests against reference commits

Prerequisite: Generate reference files and run with known good dqmc implementation and put completed runs in test/ref. Script for generating reference files: `util/makeref.py`

With reference files present in test/ref,
1. Go to build/
2. Based on branch, compiler, linalg libary, gpu offloading yes/no, host, pick a `xxx`, do `bash build_rc_xxx.sh` to generate dqmc_1_c, dqmc_1_r, dqmc_stack_c, dqmc_stack_r executables
3. Go to test/
4. Run `pytest -v -s test_gen_1band_hub_unified.py`

To check Python code coverage of integrated tests, do `coverage run -m pytest -s test_gen_1band_hub_unified.py && coverage report -m`. If `pytest-cov` is installed, then the same result can be achieved by `pytest -s test_gen_1band_hub_unified.py --cov --cov-report term-missing`

## Rough profiling:

It's a bit weird to use pytest to run this since I'm not asserting anything, but w/e

1. Set environment variables for compilation
2. Pick a git branch
3. Go to build/
4. Based on branch, compiler, linalg libary, and gpu offloading yes/no, host, pick a `xxx`, do `bash build_rc_xxx.sh` to generate dqmc_1_c, dqmc_1_r, dqmc_stack_c, dqmc_stack_r executables
5. Go to test/
6. Run `pytest -v -s test_profiling.py`
7. Move resulting log files with rough timing, memory and storage info into corresponding `test/prof/<branch>/<compiler>-<lib>-<gpuoffload>-<host>/` folder

- Options for `branch`: {`master`, `perlmt-gpu`}
- Options for `compiler`: {`icx`, `icc`, `gcc`, `nvc`}
- Options for `lib`: {`imkl`, `openblas`, `libsci`(stands for `cray-libsci`)}
- Options for `gpuoffload`: {`0`, `1`}
- Options for `host` (where the DQMC was run): {`home`, `laptop`, `sherlock`, `perlmt`}
