
#ifndef PNLP_PROBLEM_H
#define PNLP_PROBLEM_H

#include <vector>

#include "pnlp.h"

#include "rapidjson/document.h"

using std::vector;

class PnlpFuncs;

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
class PnlpProblem
{
public:
  PnlpProblem() {};

  bool load(const char *fileName);

  bool build();

  void generate();

private:
  bool _buildBounds();

  bool _buildStartingPoint();

  bool _buildSparsityStructure();

  bool _buildObjFuncs();
  bool _buildObjGraFuncs();
  bool _buildConFuncs();
  bool _buildConJacFuncs();

  void _generateGpuCpp();
  void _generateGpuEvalCpp(PnlpFuncType funcType);

  void _copyFile(const char *srcFile, const char *dstFile);
  void _copyFile(const char *srcFile, const char *dstFile,
                 const char *oldName, const char *newName);

private:
  rapidjson::Document _doc;

  int _numVars;
  int _numCons;

  // Normally we don't need to build sparsity structure for objective's gradient
  // functions. But doing so provides better flexibity because now gradient
  // functions can be decomposed into smaller common subfunctions.
  int         _numNzGra;
  vector<int> _nzGraRows; // this should contain all 1
  vector<int> _nzGraCols;

  int         _numNzJac;
  vector<int> _nzJacRows;
  vector<int> _nzJacCols;

  vector<double> _varLowerBounds;
  vector<double> _varUpperBounds;
  vector<double> _conLowerBounds;
  vector<double> _conUpperBounds;

  vector<double> _varInitValues;

  PnlpFuncs *_pObjFuncs;
  PnlpFuncs *_pObjGraFuncs;
  PnlpFuncs *_pConFuncs;
  PnlpFuncs *_pConJacFuncs;
};

#endif // PNLP_PROBLEM_H

