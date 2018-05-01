
#ifndef PNLP_H
#define PNLP_H

typedef enum {
  PnlpObjFunc    = 0,
  PnlpConFunc    = 1,
  PnlpObjGraFunc = 2,
  PnlpConJacFunc = 3,
  // This is not an actual evaluation function type, it is defined to simplify
  // the code generation part
  PnlpGpu        = 4
} PnlpFuncType;

#endif // PNLP_H

