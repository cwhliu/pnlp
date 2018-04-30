// Generated on Mon 30 Apr 2018 15:00:03 Eastern Daylight Time

#ifndef J_F_CONSTR3_H
#define J_F_CONSTR3_H

#include "inline_math.h"

#ifdef PNLP_ON_CPU
__host__
#else
__device__
#endif
void J_f_constr3 (
        double *p_output1,
  const double *var1
);

#endif // J_F_CONSTR3_H
