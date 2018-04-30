
#ifndef PNLP_PROBLEM_H
#define PNLP_PROBLEM_H

#include <string>
#include <vector>

#include "pnlp.h"

#include "rapidjson/document.h"

using std::string;
using std::vector;

class PnlpFuncs;

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
class PnlpProblem
{
public:
  PnlpProblem(const char *name);

  bool load();

  void build();

  void generate();

private:
  bool _buildBounds();

  bool _buildStartingPoint();

  bool _buildSparsityStructure();

  bool _buildEvalFuncs(PnlpFuncType funcType);

  void _generateEvalFuncsH();
  void _generateGpuCpp();
  void _generateGpuEvalCppH(PnlpFuncType funcType);

  void _replaceString(char *str, const char *oldToken, const char *newToken);
  void _copyFile(const char *srcFile, const char *dstFile, bool force=true);
  void _copyFile(const char *srcFile, const char *dstFile,
                 const char *oldName, const char *newName, bool force=true);

private:
  string _name;

  int   _bufferSize;
  char *_buffer;

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

  PnlpFuncs *_pFuncs[4];
};

#endif // PNLP_PROBLEM_H

