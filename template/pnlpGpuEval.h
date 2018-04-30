
#ifndef PH_PNLP_GPU_EVAL_H
#define PH_PNLP_GPU_EVAL_H

bool PH_PnlpGpuEval_initialize();
bool PH_PnlpGpuEval_cleanup();

bool PH_PnlpGpuEval_evaluate(const double *input, double *output);

//class PH_PnlpGpuEval
//{
//public:
//  PH_PnlpGpuEval() {};
//
//  bool initialize();
//  bool cleanup();
//
//  bool evaluate(const double *input, double *output);
//
//private:
//  double *_pInMemCpu;
//  double *_pOutMemCpu;
//  double *_pInMemGpu;
//  double *_pOutMemGpu;
//};

#endif // PH_PNLP_GPU_EVAL_H

