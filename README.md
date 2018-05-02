
# Parallel Nonlinear Programming

PNLP is a software tool that automatically generates CUDA code for a given NLP problem.

Input to PNLP includes definition of the functions to be evaluated and a JSON file that describes the NLP problem.
PNLP generates a standalone CUDA code that can be compiled to run on CPU plus GPU or on CPU only.

## Generate CUDA code

After you've cloned the code, run `make` and then `./pnlp nlp051`, this will create a folder at generated/nlp051 which contains the generated code.
Current implementation uses RapidJSON to parse the JSON file, so you'll need to specify its include path in the Makefile.

## Compile CUDA code

Go to generated/nlp051, run `make`.
By default the code is compiled to run on CPU and GPU, to compile for CPU only, change the RUN_ON variable in the Makefile to CPU and recompile.
The default Makefile assumes the following environment variables are defined and they are used by the compiler to find shared libraries, you'll need to change or comment out some of them depending on your environment setup.

- CUDA_HOME - CUDA tool kit directory
- MKL_HOME - Intel MKL directory
- IPOPT_HOME - IPOPT directory
- COINHSL_HOME - HSL directory

