
#ifndef PNLP_FUNCS_H
#define PNLP_FUNCS_H

#include <string>
#include <vector>

#include "pnlp.h"

#include "rapidjson/document.h"

using std::string;
using std::vector;

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
class PnlpFuncs
{
public:
  PnlpFuncs(PnlpFuncType funcType, const rapidjson::Document &doc);

  bool load(const rapidjson::Document &doc);

  bool build();

  inline int getNumFuncs() const { return _numFuncs; };

  inline string getFuncName(int idx) const { return _funcNames[idx]; };

  inline int getOutSize() const { return _outSize; };

  inline int getInMemSize()  const { return _inMemSize;  };
  inline int getOutMemSize() const { return _outMemSize; };

  inline int getFuncInMemOffset(int f)  const { return _funcInMemOffsets[f];  };
  inline int getFuncOutMemOffset(int f) const { return _funcOutMemOffsets[f]; };

  inline int    getFuncInDepSize(int f)     const { return _funcInDeps[f].size(); };
  inline int    getFuncInDep(int f, int i)  const { return _funcInDeps[f][i]; };
  inline int    getFuncInAuxSize(int f)     const { return _funcInAuxs[f].size(); };
  inline double getFuncInAux(int f, int i)  const { return _funcInAuxs[f][i]; };
  inline int    getFuncOutDepSize(int f)    const { return _funcOutDeps[f].size(); };
  inline int    getFuncOutDep(int f, int i) const { return _funcOutDeps[f][i]; };

private:
  PnlpFuncType _funcType;

  // Names in the JSON file, depend on function type
  char *_docH1Name;
  char *_docH2Name;
  char *_docH2InDepName;
  char *_docH2OutDepName;
  bool  _idxToCol; // TODO see if there's a better way

  int _numFuncs;
  int _outSize;

  vector< string         > _funcNames;
  vector< vector<int>    > _funcInDeps;
  vector< vector<double> > _funcInAuxs;
  vector< vector<int>    > _funcOutDeps;

  vector<int> _funcInMemOffsets;
  vector<int> _funcOutMemOffsets;

  int _inMemSize;
  int _outMemSize;
};

#endif // PNLP_FUNCS_H

