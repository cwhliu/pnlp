// Generated on Wed 25 Apr 2018 21:15:32 Eastern Daylight Time

#include <stdio.h>

#include "f_constr1.hh"

// Inner functions
// -----------------------------------------------------------------------------
__device__
void f_constr1_output1(double *p_output1, const double *var1)
{
  double _NotUsed;
  NULL;
  p_output1[0]=var1[0] + 3*var1[1];
}


// Wrapper function
// -----------------------------------------------------------------------------
__device__
void f_constr1 (
        double *p_output1,
  const double *var1
)
{
  // Call inner functions
  f_constr1_output1(p_output1, var1);
}