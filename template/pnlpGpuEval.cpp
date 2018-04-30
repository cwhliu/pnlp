
#include "PH_pnlpGpuEval.h"

#include "evalFuncs.h"

__global__ void PH_PnlpGpuEvalKernel(double *input, double *output);

double *PH_PnlpGpuEval_pInMemCpu;
double *PH_PnlpGpuEval_pOutMemCpu;
double *PH_PnlpGpuEval_pInMemGpu;
double *PH_PnlpGpuEval_pOutMemGpu;

// -----------------------------------------------------------------------------
bool PH_PnlpGpuEval_initialize()
{
//PH_INIT

  return true;
}

// -----------------------------------------------------------------------------
bool PH_PnlpGpuEval_cleanup()
{
  delete PH_PnlpGpuEval_pInMemCpu;
  delete PH_PnlpGpuEval_pOutMemCpu;

  cudaFree(PH_PnlpGpuEval_pInMemGpu);
  cudaFree(PH_PnlpGpuEval_pOutMemGpu);

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
__global__ void PH_PnlpGpuEvalKernel(double *input, double *output)
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x;

//PH_KERNEL
}

//// -----------------------------------------------------------------------------
//bool PH_PnlpGpuEval::initialize()
//{
////PH_INIT
//
//  return true;
//}
//
//// -----------------------------------------------------------------------------
//bool PH_PnlpGpuEval::cleanup()
//{
//  delete _pInMemCpu;
//  delete _pOutMemCpu;
//
//  cudaFree(_pInMemGpu);
//  cudaFree(_pOutMemGpu);
//
//  return true;
//}
//
//// -----------------------------------------------------------------------------
//__host__ bool PH_PnlpGpuEval::evaluate(const double *input, double *output)
//{
////PH_EVAL
//
//  return true;
//}
//
//// -----------------------------------------------------------------------------
//// -----------------------------------------------------------------------------
//__global__ void PH_PnlpGpuEvalKernel(double *input, double *output)
//{
//  int idx = blockIdx.x*blockDim.x + threadIdx.x;
//
////PH_KERNEL
//}

