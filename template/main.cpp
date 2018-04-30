
#include "pnlpGpu.h"

#include "IpIpoptApplication.hpp"

using namespace Ipopt;

int main()
{
  SmartPtr<TNLP> nlp = new PnlpGpu();

  SmartPtr<IpoptApplication> app = IpoptApplicationFactory();

  //app->Options()->SetNumericValue("tol", 1e-9);
  //app->Options()->SetStringValue("mu_strategy", "adaptive");
  app->Options()->SetStringValue("option_file_name", "ipopt.opt");

  ApplicationReturnStatus status;

  status = app->Initialize();

  if (status != Solve_Succeeded) {
    printf("\nError: IPOPT solver initialization failed\n\n");
    return (int)status;
  }

  status = app->OptimizeTNLP(nlp);

  return (int)status;
}

