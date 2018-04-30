
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
  //_pObjFuncs    = new PnlpGpuEvalObj;
  //_pObjGraFuncs = new PnlpGpuEvalObjGra;
  //_pConFuncs    = new PnlpGpuEvalCon;
  //_pConJacFuncs = new PnlpGpuEvalConJac;

  //_pObjFuncs->initialize();
  //_pObjGraFuncs->initialize();
  //_pConFuncs->initialize();
  //_pConJacFuncs->initialize();

  PnlpGpuEvalObj_initialize();
  PnlpGpuEvalObjGra_initialize();
  PnlpGpuEvalCon_initialize();
  PnlpGpuEvalConJac_initialize();
}

PnlpGpu::~PnlpGpu()
{
  //_pObjFuncs->cleanup();
  //_pObjGraFuncs->cleanup();
  //_pConFuncs->cleanup();
  //_pConJacFuncs->cleanup();

  //delete _pObjFuncs;
  //delete _pObjGraFuncs;
  //delete _pConFuncs;
  //delete _pConJacFuncs;

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

  //n = 4;
  //m = 2;
  //nnz_jac_g = 8;
  //nnz_h_lag = 10;

  index_style = TNLP::C_STYLE;

  return true;
}

// Gives IPOPT the value of the bounds on the variables and constraints
// -----------------------------------------------------------------------------
bool PnlpGpu::get_bounds_info(Index n, Number *x_l, Number *x_u,
                              Index m, Number *g_l, Number *g_u)
{
//PH_BOUNDS_INFO

  //for (int i = 0; i < 4; i++) x_l[i] = 1.0;
  //for (int i = 0; i < 4; i++) x_u[i] = 5.0;

  //g_l[0] = 25;
  //g_u[0] = 2e19;

  //g_l[1] = g_u[1] = 40.0;

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

  //x[0] = 1.0;
  //x[1] = 5.0;
  //x[2] = 5.0;
  //x[3] = 1.0;

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
  //obj_value = x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2];

  double output;

  PnlpGpuEvalObj_evaluate(x, &output);

  obj_value = output;

  return true;
}

// Return the gradient of the objective function at the point x
// -----------------------------------------------------------------------------
bool PnlpGpu::eval_grad_f(Index n, const Number *x, bool new_x, Number *grad_f)
{
  //grad_f[0] = x[0] * x[3] + x[3] * (x[0] + x[1] + x[2]);
  //grad_f[1] = x[0] * x[3];
  //grad_f[2] = x[0] * x[3] + 1;
  //grad_f[3] = x[0] * (x[0] + x[1] + x[2]);

  PnlpGpuEvalObjGra_evaluate(x, grad_f);

  return true;
}

// Return the value of the constraint function at the point x
// -----------------------------------------------------------------------------
bool PnlpGpu::eval_g(Index n, const Number *x, bool new_x, Index m, Number *g)
{
  //g[0] = x[0] * x[1] * x[2] * x[3];
  //g[1] = x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3];

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

  //if (values == NULL) {
  //  iRow[0] = 0; jCol[0] = 0;
  //  iRow[1] = 0; jCol[1] = 1;
  //  iRow[2] = 0; jCol[2] = 2;
  //  iRow[3] = 0; jCol[3] = 3;
  //  iRow[4] = 1; jCol[4] = 0;
  //  iRow[5] = 1; jCol[5] = 1;
  //  iRow[6] = 1; jCol[6] = 2;
  //  iRow[7] = 1; jCol[7] = 3;
  //}
  //else {
  //  values[0] = x[1]*x[2]*x[3];
  //  values[1] = x[0]*x[2]*x[3];
  //  values[2] = x[0]*x[1]*x[3];
  //  values[3] = x[0]*x[1]*x[2];

  //  values[4] = 2*x[0];
  //  values[5] = 2*x[1];
  //  values[6] = 2*x[2];
  //  values[7] = 2*x[3];
  //}

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
  //if (values == NULL) {
  //  int idx = 0;

  //  for (int row = 0; row < 4; row++)
  //    for (int col = 0; col <= row; col++) {
  //      iRow[idx] = row;
  //      jCol[idx] = col;
  //      idx++;
  //    }
  //}
  //else {
  //  values[0] = obj_factor * (2*x[3]);

  //  values[1] = obj_factor * x[3];
  //  values[2] = 0;

  //  values[3] = obj_factor * x[3];
  //  values[4] = 0;
  //  values[5] = 0;

  //  values[6] = obj_factor * (2*x[0] + x[1] + x[2]);
  //  values[7] = obj_factor * x[0];
  //  values[7] = obj_factor * x[0];
  //  values[7] = 0;

  //  values[1] += lambda[0] * (x[2] * x[3]);

  //  values[3] += lambda[0] * (x[1] * x[3]);
  //  values[4] += lambda[0] * (x[0] * x[3]);

  //  values[6] += lambda[0] * (x[1] * x[2]);
  //  values[7] += lambda[0] * (x[0] * x[2]);
  //  values[8] += lambda[0] * (x[0] * x[1]);

  //  values[0] += lambda[1] * 2;
  //  values[2] += lambda[1] * 2;
  //  values[5] += lambda[1] * 2;
  //  values[9] += lambda[1] * 2;
  //}

  return true;
}

