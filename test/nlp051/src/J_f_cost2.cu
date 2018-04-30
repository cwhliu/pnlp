// Generated on Mon 30 Apr 2018 14:58:21 Eastern Daylight Time

#include <stdio.h>

#include "J_f_cost2.hh"

// Inner functions
// -----------------------------------------------------------------------------
#ifdef PNLP_ON_CPU
__host__
#else
__device__
#endif
void J_f_cost2_output1(double *p_output1, const double *var1)
{
  double t33;
  double t34;
  t33 = -2. + var1[1] + var1[2];
  t34 = 2.*t33;
  p_output1[0]=t34;
  p_output1[1]=t34;
}


// Wrapper function
// -----------------------------------------------------------------------------
#ifdef PNLP_ON_CPU
__host__
#else
__device__
#endif
void J_f_cost2 (
        double *p_output1,
  const double *var1
)
{
  // Call inner functions
  J_f_cost2_output1(p_output1, var1);
}