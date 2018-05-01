
#include <cstdio>

#include "pnlpProblem.h"

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
int main(int argc, char **argv)
{
  printf("\nPNLP - Parallel Nonlinear Programming\n\n");

  if (argc != 2) {
    printf("Usage: %s problem_name\n\n", argv[0]);
    return 1;
  }

  PnlpProblem *pProb;

  pProb = new PnlpProblem(argv[1]);

  if (!pProb->load())
    return 1;

  pProb->build();

  pProb->generate();

  printf("\nbye ...\n\n");

  return 0;
}

