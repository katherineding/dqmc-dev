# Instructions for testing

## Unit tests

// TODO

## Integrated correctness tests against reference commits

Prerequisite: Generate reference files and run with known good dqmc implementation. Steps:
1. Go to test/ 
2. Run script for generating reference files: `test/makeref.py` or `test/makeref.py` 
3. Put them on a stack file: `python3 ../util/push.py stack *.h5`
3. Meanwhile, compile dqmc excutables by going to build/ and do `bash build_rc_xxx.sh`, chosing `xxx` based on your architecture and compiler.
4. Back in test/, do `../build/dqmc_stack_x stack`, choosing `x` based on if you need complex numbers.
5. In test/ref/ make a new folder named with the current short commit hash. Move completed .h5 files into this folder.

With reference files present in test/ref,
1. Compile dqmc excutables by going to build/ and do `bash build_rc_xxx.sh`, chosing `xxx` based on your architecture and compiler.
2. Go to test/
3. Run `pytest -v -s test_gen_1band_hub_unified.py`
	- `-v` means verbose output
	- `-s` means show stdout from the code

	Other useful flags:
		- `-x` means stop after first fail
		- `-k keyword` only runs tests containing keyword

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
