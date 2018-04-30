
#include <cstdio>

#include "pnlpFuncs.h"

using namespace rapidjson;

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
PnlpFuncs::PnlpFuncs(PnlpFuncType funcType, const rapidjson::Document &doc)
{
  _funcType = funcType;

  // Make sure these are big enough to store their value
  _docH1Name       = new char[16];
  _docH2Name       = new char[16];
  _docH2InDepName  = new char[16];
  _docH2OutDepName = new char[16];

  switch (_funcType) {
    case PnlpObjFunc: {
      strcpy(_docH1Name,       "Objective");
      strcpy(_docH2Name,       "Funcs");
      strcpy(_docH2InDepName,  "DepIndices");
      strcpy(_docH2OutDepName, "FuncIndices");
      _idxToCol = false;
      _outSize = 1;
      break;
    }

    case PnlpObjGraFunc: {
      strcpy(_docH1Name,       "Objective");
      strcpy(_docH2Name,       "JacFuncs");
      strcpy(_docH2InDepName,  "DepIndices");
      strcpy(_docH2OutDepName, "nzJacIndices");
      _idxToCol = true;
      _outSize = doc["Variable"]["dimVars"].GetInt();
      break;
    }

    case PnlpConFunc: {
      strcpy(_docH1Name,       "Constraint");
      strcpy(_docH2Name,       "Funcs");
      strcpy(_docH2InDepName,  "DepIndices");
      strcpy(_docH2OutDepName, "FuncIndices");
      _idxToCol = false;
      _outSize = doc["Constraint"]["Dimension"].GetInt();
      break;
    }

    case PnlpConJacFunc: {
      strcpy(_docH1Name,       "Constraint");
      strcpy(_docH2Name,       "JacFuncs");
      strcpy(_docH2InDepName,  "DepIndices");
      strcpy(_docH2OutDepName, "nzJacIndices");
      _idxToCol = false;
      _outSize = doc["Constraint"]["nnzJac"].GetInt();
      break;
    }

    default: assert(false);
  };
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
bool PnlpFuncs::load(const rapidjson::Document &doc)
{
  const Value &docH1 = (const_cast<rapidjson::Document&>(doc))[_docH1Name];

  _numFuncs = docH1["numFuncs"].GetInt();

  //printf("Parsing functions ...\n");

  vector< string                   > funcNames;
  vector< vector< vector<int>    > > funcInDeps;
  vector< vector< vector<double> > > funcInAuxs;
  vector< vector< vector<int>    > > funcOutDeps;

  // Temporary vector used to buffer input/output dependencies and auxiliary data
  vector<int>    depVec;
  vector<double> auxVec;

  // Loop over all functions
  for (int f = 0; f < _numFuncs; f++) {
    int funcId = docH1[_docH2Name][f].GetInt()-1;

    // Resize vectors if the function ID exceeds the capacity of the vectors
    // At this point we're using function ID in the JSON file as vector index
    // so there will be unused entries, we'll delete them later on
    if (funcId >= funcNames.size()) {
      funcNames.resize   (funcId+1);
      funcInDeps.resize  (funcId+1);
      funcInAuxs.resize  (funcId+1);
      funcOutDeps.resize (funcId+1);

      // Get function names
      funcNames[funcId] = docH1["Names"][f].GetString();
    }

    // Get function input dependencies, ie, which input variables are needed for
    // this function
    depVec.clear();
    for (int i = 0; i < docH1[_docH2InDepName][f].Size(); i++)
      depVec.push_back(docH1[_docH2InDepName][f][i].GetInt()-1);
    funcInDeps[funcId].push_back(depVec);

    // Get function input auxiliary data
    auxVec.clear();
    if (docH1["AuxData"][f].IsArray())
      for (int i = 0; i < docH1["AuxData"][f].Size(); i++)
        auxVec.push_back(docH1["AuxData"][f][i].GetDouble());
    funcInAuxs[funcId].push_back(auxVec);

    // Get function output dependencies, ie, which output variables depend on
    // the result of this function
    depVec.clear();
    for (int i = 0; i < docH1[_docH2OutDepName][f].Size(); i++)
      if (_idxToCol)
        depVec.push_back(
          docH1["nzJacCols"][ docH1[_docH2OutDepName][f][i].GetInt()-1 ].GetInt()-1
        );
      else
        depVec.push_back(docH1[_docH2OutDepName][f][i].GetInt()-1);
    funcOutDeps[funcId].push_back(depVec);

    // Display function information
    //printf("  Function %d: %s\n", funcId, funcNames[funcId].c_str());
    //printf("    Input dependencies: ");
    //for (int i = 0; i < funcInDeps[funcId].size(); i++) {
    //  printf("[ ");
    //  for (int j = 0; j < funcInDeps[funcId][i].size(); j++)
    //    printf("%d ", funcInDeps[funcId][i][j]);
    //  printf("] ");
    //}
    //printf("\n");
    //printf("    Input auxiliary data: ");
    //for (int i = 0; i < funcInAuxs[funcId].size(); i++) {
    //  printf("[ ");
    //  for (int j = 0; j < funcInAuxs[funcId][i].size(); j++)
    //    printf("%f ", funcInAuxs[funcId][i][j]);
    //  printf("] ");
    //}
    //printf("\n");
    //printf("    Output dependencies: ");
    //for (int i = 0; i < funcOutDeps[funcId].size(); i++) {
    //  printf("[ ");
    //  for (int j = 0; j < funcOutDeps[funcId][i].size(); j++)
    //    printf("%d ", funcOutDeps[funcId][i][j]);
    //  printf("] ");
    //}
    //printf("\n");
  }

  _inMemSize = 0;
  _outMemSize = 0;

  for (int f = 0; f < funcNames.size(); f++) {
    for (int i = 0; i < funcInDeps[f].size(); i++) {
      _funcNames.push_back(funcNames[f]);

      _funcInMemOffsets.push_back(_inMemSize);
      _funcOutMemOffsets.push_back(_outMemSize);

      _funcInDeps.push_back(funcInDeps[f][i]);
      _funcInAuxs.push_back(funcInAuxs[f][i]);

      _inMemSize += funcInDeps[f][i].size();
      _inMemSize += funcInAuxs[f][i].size();

      _funcOutDeps.push_back(funcOutDeps[f][i]);

      _outMemSize += funcOutDeps[f][i].size();
    }
  }

  // Now delete unused vector entries
  //int i = 0;
  //while (i < _funcNames.size())
  //  if (_funcNames[i] == "") {
  //    _funcNames.erase(_funcNames.begin()+i);
  //    _funcInDeps.erase(_funcInDeps.begin()+i);
  //    _funcInAuxs.erase(_funcInAuxs.begin()+i);
  //    _funcOutDeps.erase(_funcOutDeps.begin()+i);
  //  }
  //  else
  //    i++;

  //printf("\n");

  return true;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
bool PnlpFuncs::build()
{
//  // Calculate the size of the input memory
//  _inMemSize = 0;
//
//  for (int f = 0; f < _funcInDeps.size(); f++)
//    for (int i = 0; i < _funcInDeps[f].size(); i++)
//      _inMemSize += _funcInDeps[f][i].size();
//
//  for (int f = 0; f < _funcInAuxs.size(); f++)
//    for (int i = 0; i < _funcInAuxs[f].size(); i++)
//      _inMemSize += _funcInAuxs[f][i].size();
//
//  // Calculate the size of the output buffer
//  _outBufSize = 0;
//
//  for (int f = 0; f < _funcOutDeps.size(); f++)
//    for (int i = 0; i < _funcOutDeps[f].size(); i++)
//      _outBufSize += _funcOutDeps[f][i].size();

  return true;
}

