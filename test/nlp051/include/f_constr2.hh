// Generated on Mon 30 Apr 2018 15:00:03 Eastern Daylight Time

#ifndef F_CONSTR2_H
#define F_CONSTR2_H

#include "inline_math.h"

#ifdef PNLP_ON_CPU
__host__
#else
__device__
#endif
void f_constr2 (
        double *p_output1,
  const double *var1
);

#endif // F_CONSTR2_H
