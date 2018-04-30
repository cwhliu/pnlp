
#include <cstdio>

#include "pnlpProblem.h"

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
int main()
{
  printf("\nPNLP - Parallel Nonlinear Programming\n\n");

  PnlpProblem *pProb = new PnlpProblem;

  pProb->load("test/nlp051/problem.json");
  pProb->build();
  pProb->generate();

  printf("\nbye ...\n\n");

  return 0;
}

