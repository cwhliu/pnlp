// Generated on Wed 25 Apr 2018 21:15:33 Eastern Daylight Time

#include <stdio.h>

#include "J_f_cost1.hh"

// Inner functions
// -----------------------------------------------------------------------------
__device__
void J_f_cost1_output1(double *p_output1, const double *var1)
{
  double t83;
  double t84;
  t83 = -1.*var1[1];
t84 = var1[0] + t83;
  p_output1[0]=2.*t84;
p_output1[1]=-2.*t84;
}


// Wrapper function
// -----------------------------------------------------------------------------
__device__
void J_f_cost1 (
        double *p_output1,
  const double *var1
)
{
  // Call inner functions
  J_f_cost1_output1(p_output1, var1);
}