// Generated on Wed 25 Apr 2018 21:15:33 Eastern Daylight Time

#include <stdio.h>

#include "J_f_cost2.hh"

// Inner functions
// -----------------------------------------------------------------------------
__device__
void J_f_cost2_output1(double *p_output1, const double *var1)
{
  double t85;
  double t86;
  t85 = -2. + var1[1] + var1[2];
t86 = 2.*t85;
  p_output1[0]=t86;
p_output1[1]=t86;
}


// Wrapper function
// -----------------------------------------------------------------------------
__device__
void J_f_cost2 (
        double *p_output1,
  const double *var1
)
{
  // Call inner functions
  J_f_cost2_output1(p_output1, var1);
}