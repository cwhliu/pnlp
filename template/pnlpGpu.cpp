
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

  FILE *fp;

  // Load bounds from file
  fp = fopen("constant/bounds.txt", "r");
  while (1) {
    double lb, ub;

    // Pitfall: need to use %lf here, %f will read all zero
    if (fscanf(fp, "%lf %lf", &lb, &ub) != 2) break;

    _lowerBounds.push_back(lb);
    _upperBounds.push_back(ub);
  }
  fclose(fp);

  // Load starting point values from file
  fp = fopen("constant/initValues.txt", "r");
  while (1) {
    double value;

    if (fscanf(fp, "%lf", &value) != 1) break;

    _initValues.push_back(value);
  }
  fclose(fp);

  // Load Jacobian sparsity structure from file
  fp = fopen("constant/jacRowCol.txt", "r");
  while (1) {
    int row, col;

    if (fscanf(fp, "%d %d", &row, &col) != 2) break;

    _jacRows.push_back(row);
    _jacCols.push_back(col);
  }
  fclose(fp);
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
  assert((n+m) == _lowerBounds.size());

  // First n bounds are for x
  for (int i = 0; i < n; i++) {
    x_l[i] = _lowerBounds[i];
    x_u[i] = _upperBounds[i];
  }

  // Last m bounds are for g
  for (int i = 0; i < m; i++) {
    g_l[i] = _lowerBounds[n+i];
    g_u[i] = _upperBounds[n+i];
  }

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

  assert(n == _initValues.size());

  for (int i = 0; i < _initValues.size(); i++)
    x[i] = _initValues[i];

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
  return;

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
    assert(nele_jac == _jacRows.size());

    for (int i = 0; i < _jacRows.size(); i++) {
      iRow[i] = _jacRows[i];
      jCol[i] = _jacCols[i];
    }
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

