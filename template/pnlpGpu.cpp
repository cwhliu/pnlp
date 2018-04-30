
#include <cstdio>
#include <cassert>

#include "pnlpGpu.h"

#include "pnlpGpuEvalObj.h"
#include "pnlpGpuEvalObjGra.h"
#include "pnlpGpuEvalCon.h"
#include "pnlpGpuEvalConJac.h"

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
PnlpGpu::PnlpGpu()
{
  PnlpGpuEvalObj_initialize();
  PnlpGpuEvalObjGra_initialize();
  PnlpGpuEvalCon_initialize();
  PnlpGpuEvalConJac_initialize();
}

PnlpGpu::~PnlpGpu()
{
  PnlpGpuEvalObj_cleanup();
  PnlpGpuEvalObjGra_cleanup();
  PnlpGpuEvalCon_cleanup();
  PnlpGpuEvalConJac_cleanup();
}

// Gives IPOPT the information about the size of the problem
// -----------------------------------------------------------------------------
bool PnlpGpu::get_nlp_info(Index &n, Index &m, Index &nnz_jac_g,
                           Index &nnz_h_lag, IndexStyleEnum &index_style)
{
//PH_NLP_INFO

  index_style = TNLP::C_STYLE;

  return true;
}

// Gives IPOPT the value of the bounds on the variables and constraints
// -----------------------------------------------------------------------------
bool PnlpGpu::get_bounds_info(Index n, Number *x_l, Number *x_u,
                              Index m, Number *g_l, Number *g_u)
{
//PH_BOUNDS_INFO

  return true;
}

// Gives IPOPT the starting point before it begins iteratin
// -----------------------------------------------------------------------------
bool PnlpGpu::get_starting_point(Index n, bool init_x, Number *x,
                                 bool init_z, Number *z_L, Number *z_U,
                                 Index m, bool init_lambda, Number *lambda)
{
  assert(init_x == true);
  assert(init_z == false);
  assert(init_lambda == false);

//PH_STARTING_POINT

  return true;
}

// Called by IPOPT after the algorithm has finished
// -----------------------------------------------------------------------------
void PnlpGpu::finalize_solution(SolverReturn status,
                                Index n, const Number *x,
                                const Number *z_L, const Number *z_U,
                                Index m, const Number *g, const Number *lambda,
                                Number obj_value,
                                const IpoptData *ip_data,
                                IpoptCalculatedQuantities *ip_cq)
{
  printf("Solution\n");
  for (int i = 0; i < n; i++)
    printf("  x[%d] = %e\n", i, x[i]);

  printf("Objective value\n");
  printf("  f(x*) = %e\n", obj_value);

  printf("Constraint value\n");
  for (int i = 0; i < m; i++)
    printf("  g[%d] = %e\n", i, g[i]);
}

// Return the value of the objective function at the point x
// -----------------------------------------------------------------------------
bool PnlpGpu::eval_f(Index n, const Number *x, bool new_x, Number &obj_value)
{
  double output;

  PnlpGpuEvalObj_evaluate(x, &output);

  obj_value = output;

  return true;
}

// Return the gradient of the objective function at the point x
// -----------------------------------------------------------------------------
bool PnlpGpu::eval_grad_f(Index n, const Number *x, bool new_x, Number *grad_f)
{
  PnlpGpuEvalObjGra_evaluate(x, grad_f);

  return true;
}

// Return the value of the constraint function at the point x
// -----------------------------------------------------------------------------
bool PnlpGpu::eval_g(Index n, const Number *x, bool new_x, Index m, Number *g)
{
  PnlpGpuEvalCon_evaluate(x, g);

  return true;
}

// Return either the sparsity structure of the Jacobian of the constraints, or
// the values for the Jacobian of the constraints at the point x
// -----------------------------------------------------------------------------
bool PnlpGpu::eval_jac_g(Index n, const Number *x, bool new_x,
                         Index m, Index nele_jac,
                         Index *iRow, Index *jCol, Number *values)
{
  if (values == NULL) {
//PH_JAC_STRUCTURE
  }
  else
    PnlpGpuEvalConJac_evaluate(x, values);

  return true;
}

// Return either the sparsity structure of the Hessian of the Lagrangian, or the
// values of the Hessian of the Lagrangian
// -----------------------------------------------------------------------------
bool PnlpGpu::eval_h(Index n, const Number *x, bool new_x,
                     Number obj_factor, Index m, const Number *lambda,
                     bool new_lambda, Index nele_hess,
                     Index *iRow, Index *jCol, Number *values)
{
  return true;
}

