// Generated on Mon 30 Apr 2018 14:58:20 Eastern Daylight Time

#include <stdio.h>

#include "f_cost3.hh"

// Inner functions
// -----------------------------------------------------------------------------
#ifdef PNLP_ON_CPU
__host__
#else
__device__
#endif
void f_cost3_output1(double *p_output1, const double *var1)
{
  double _NotUsed;
  NULL;
  p_output1[0]=Power(-1 + var1[0],2) + Power(-1 + var1[1],2);
}


// Wrapper function
// -----------------------------------------------------------------------------
#ifdef PNLP_ON_CPU
__host__
#else
__device__
#endif
void f_cost3 (
        double *p_output1,
  const double *var1
)
{
  // Call inner functions
  f_cost3_output1(p_output1, var1);
}