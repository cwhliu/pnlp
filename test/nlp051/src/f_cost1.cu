// Generated on Wed 25 Apr 2018 21:15:33 Eastern Daylight Time

#include <stdio.h>

#include "f_cost1.hh"

// Inner functions
// -----------------------------------------------------------------------------
__device__
void f_cost1_output1(double *p_output1, const double *var1)
{
  double _NotUsed;
  NULL;
  p_output1[0]=Power(var1[0] - var1[1],2);
}


// Wrapper function
// -----------------------------------------------------------------------------
__device__
void f_cost1 (
        double *p_output1,
  const double *var1
)
{
  // Call inner functions
  f_cost1_output1(p_output1, var1);
}