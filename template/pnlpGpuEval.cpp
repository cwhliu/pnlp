
#include "PH_pnlpGpuEval.h"

#include "evalFuncs.h"

#ifdef PNLP_ON_CPU
__host__ void PH_PnlpGpuEvalKernel(int idx, double *input, double *output);
#else
__global__ void PH_PnlpGpuEvalKernel(double *input, double *output);
#endif

double *PH_PnlpGpuEval_pInMemCpu;
double *PH_PnlpGpuEval_pOutMemCpu;
double *PH_PnlpGpuEval_pInMemGpu;
double *PH_PnlpGpuEval_pOutMemGpu;

// -----------------------------------------------------------------------------
__host__ bool PH_PnlpGpuEval_initialize()
{
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

