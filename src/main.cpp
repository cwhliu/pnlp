
#include <cstdio>

#include "pnlpProblem.h"

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
int main()
{
  printf("\nPNLP - Parallel Nonlinear Programming\n\n");

  PnlpProblem *pProb;

  pProb = new PnlpProblem("nlp051");
  //pProb = new PnlpProblem("atlas");

  if (!pProb->load())
    return 1;

  pProb->build();

  pProb->generate();

  printf("\nbye ...\n\n");

  return 0;
}

