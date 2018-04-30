
#include <cstdio>

#include "pnlpProblem.h"

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
int main()
{
  printf("\nPNLP - Parallel Nonlinear Programming\n\n");

  PnlpProblem *pProb = new PnlpProblem("nlp051");

  if (!pProb->load()) return 1;

  pProb->build();

  pProb->generate();

  printf("\nbye ...\n\n");

  return 0;
}

