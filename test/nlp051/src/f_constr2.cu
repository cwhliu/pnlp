// Generated on Mon 30 Apr 2018 14:58:20 Eastern Daylight Time

#include <stdio.h>

#include "f_constr2.hh"

// Inner functions
// -----------------------------------------------------------------------------
#ifdef PNLP_ON_CPU
__host__
#else
__device__
#endif
void f_constr2_output1(double *p_output1, const double *var1)
{
  double _NotUsed;
  NULL;
  p_output1[0]=var1[2] + var1[3] - 2*var1[4];
}


// Wrapper function
// -----------------------------------------------------------------------------
#ifdef PNLP_ON_CPU
__host__
#else
__device__
#endif
void f_constr2 (
        double *p_output1,
  const double *var1
)
{
  // Call inner functions
  f_constr2_output1(p_output1, var1);
}