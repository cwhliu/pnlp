// Generated on Mon 30 Apr 2018 15:00:04 Eastern Daylight Time

#ifndef F_COST1_H
#define F_COST1_H

#include "inline_math.h"

#ifdef PNLP_ON_CPU
__host__
#else
__device__
#endif
void f_cost1 (
        double *p_output1,
  const double *var1
);

#endif // F_COST1_H
