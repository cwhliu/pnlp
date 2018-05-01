
#include <vector>

#include <cstdio>

#include "PH_pnlpGpuEval.h"

#include "evalFuncs.h"

using std::vector;

#ifdef PNLP_ON_CPU
__host__ void PH_PnlpGpuEvalKernel(int idx, double *input, double *output);
#else
__global__ void PH_PnlpGpuEvalKernel(double *input, double *output);
#endif

double *PH_PnlpGpuEval_pInMemCpu;
double *PH_PnlpGpuEval_pOutMemCpu;
double *PH_PnlpGpuEval_pInMemGpu;
double *PH_PnlpGpuEval_pOutMemGpu;

vector<int> PH_PnlpGpuEval_InRemapMem;
vector<int> PH_PnlpGpuEval_InRemapIdx;
vector<int> PH_PnlpGpuEval_InRemapAux;

vector<int> PH_PnlpGpuEval_OutRemapIdx;
vector<int> PH_PnlpGpuEval_OutRemapMem;

// -----------------------------------------------------------------------------
__host__ bool PH_PnlpGpuEval_initialize()
{
  FILE *fp;

  // Load the mapping for input to memory from file
  fp = fopen("constant/PH_pnlpGpuEvalIn.txt", "r");
  while (1) {
    int mem, idx;
    double aux;

    if (fscanf(fp, "%d %d %lf", &mem, &idx, &aux) != 3) break;

    PH_PnlpGpuEval_InRemapMem.push_back(mem);
    PH_PnlpGpuEval_InRemapIdx.push_back(idx);
    PH_PnlpGpuEval_InRemapAux.push_back(aux);
  }
  fclose(fp);

  // Load the mapping for memory to output from file
  fp = fopen("constant/PH_pnlpGpuEvalOut.txt", "r");
  while (1) {
    int idx, mem;

    if (fscanf(fp, "%d %d", &idx, &mem) != 2) break;

    PH_PnlpGpuEval_OutRemapIdx.push_back(idx);
    PH_PnlpGpuEval_OutRemapMem.push_back(mem);
  }
  fclose(fp);

//PH_INIT

  return true;
}

// -----------------------------------------------------------------------------
__host__ bool PH_PnlpGpuEval_cleanup()
{
  delete PH_PnlpGpuEval_pInMemCpu;
  delete PH_PnlpGpuEval_pOutMemCpu;

  #ifdef PNLP_ON_GPU
  cudaFree(PH_PnlpGpuEval_pInMemGpu);
  cudaFree(PH_PnlpGpuEval_pOutMemGpu);
  #endif // PNLP_ON_GPU

  return true;
}

// -----------------------------------------------------------------------------
__host__ bool PH_PnlpGpuEval_evaluate(const double *input, double *output)
{
//PH_EVAL

  return true;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
#ifdef PNLP_ON_CPU
__host__ void PH_PnlpGpuEvalKernel(int idx, double *input, double *output)
{
#else
__global__ void PH_PnlpGpuEvalKernel(double *input, double *output)
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
#endif

//PH_KERNEL
}

