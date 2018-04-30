
#ifndef PH_PNLP_GPU_EVAL_H
#define PH_PNLP_GPU_EVAL_H

bool PH_PnlpGpuEval_initialize();

bool PH_PnlpGpuEval_cleanup();

bool PH_PnlpGpuEval_evaluate(const double *input, double *output);

#endif // PH_PNLP_GPU_EVAL_H

