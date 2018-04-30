// Generated on Mon 30 Apr 2018 15:00:03 Eastern Daylight Time

#ifndef F_CONSTR3_H
#define F_CONSTR3_H

#include "inline_math.h"

#ifdef PNLP_ON_CPU
__host__
#else
__device__
#endif
void f_constr3 (
        double *p_output1,
  const double *var1
);

#endif // F_CONSTR3_H
