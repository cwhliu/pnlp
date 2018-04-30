// Generated on Wed 25 Apr 2018 21:15:33 Eastern Daylight Time

#include <stdio.h>

#include "J_f_constr1.hh"

// Inner functions
// -----------------------------------------------------------------------------
__device__
void J_f_constr1_output1(double *p_output1, const double *var1)
{
  double _NotUsed;
  NULL;
  p_output1[0]=1;
p_output1[1]=3;
}


// Wrapper function
// -----------------------------------------------------------------------------
__device__
void J_f_constr1 (
        double *p_output1,
  const double *var1
)
{
  // Call inner functions
  J_f_constr1_output1(p_output1, var1);
}