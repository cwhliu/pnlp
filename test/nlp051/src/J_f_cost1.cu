// Generated on Mon 30 Apr 2018 14:58:20 Eastern Daylight Time

#include <stdio.h>

#include "J_f_cost1.hh"

// Inner functions
// -----------------------------------------------------------------------------
#ifdef PNLP_ON_CPU
__host__
#else
__device__
#endif
void J_f_cost1_output1(double *p_output1, const double *var1)
{
  double t31;
  double t32;
  t31 = -1.*var1[1];
  t32 = var1[0] + t31;
  p_output1[0]=2.*t32;
  p_output1[1]=-2.*t32;
}


// Wrapper function
// -----------------------------------------------------------------------------
#ifdef PNLP_ON_CPU
__host__
#else
__device__
#endif
void J_f_cost1 (
        double *p_output1,
  const double *var1
)
{
  // Call inner functions
  J_f_cost1_output1(p_output1, var1);
}