
#include <string>

#include <cstdio>

#include "pnlpProblem.h"
#include "pnlpFuncs.h"

#include "rapidjson/filereadstream.h"

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
PnlpProblem::PnlpProblem(const char *name)
{
  _name = name;

  _bufferSize = 1024;
  _buffer     = (char *)malloc(_bufferSize*sizeof(char));
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
bool PnlpProblem::load()
{
  // Open problem definition JSON file
  string fileName = "test/" + _name + "/problem.json";

  printf("Loading \"%s\" ... ", fileName.c_str());

  FILE *fp = fopen(fileName.c_str(), "r");

  if (!fp) {
    printf("failed\n\n");
    printf("Error: can not open file \"%s\"\n\n", fileName.c_str());
    return false;
  }

  // Parse the JSON file
  rapidjson::FileReadStream fileStream(fp, _buffer, _bufferSize);

  _doc.ParseStream(fileStream);

  if (!_doc.IsObject()) {
    printf("failed\n\n");
    printf("Error: parse file \"%s\" failed\n\n", fileName.c_str());
    return false;
  }

  printf("okay\n\n");

  fclose(fp);

  return true;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void PnlpProblem::build()
{
  // Problem dimensions
  _numVars = _doc["Variable"]["dimVars"].GetInt();
  _numCons = _doc["Constraint"]["Dimension"].GetInt();

  // Problem bounds
  _buildBounds();

  // Problem starting point
  _buildStartingPoint();

  // Problem structure
  _buildSparsityStructure();

  // Problem evaluation functions
  _buildEvalFuncs(PnlpObjFunc);
  _buildEvalFuncs(PnlpObjGraFunc);
  _buildEvalFuncs(PnlpConFunc);
  _buildEvalFuncs(PnlpConJacFunc);

  printf(" Problem name:               %s\n", _name.c_str());
  printf(" Number of variables:        %d\n", _numVars);
  printf(" Number of constraints:      %d\n", _numCons);
  printf(" Number of nonzero Jacobian: %d\n", _numNzJac);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void PnlpProblem::generate()
{
  // Create directories
  vector<string> mkdir;

  mkdir.push_back("include");
  mkdir.push_back("src");
  mkdir.push_back("obj");
  mkdir.push_back("dep");
  mkdir.push_back("eval/include");
  mkdir.push_back("eval/src");

  for (int i = 0; i < mkdir.size(); i++) {
    string cmd = "mkdir -p generated/" + _name + "/" + mkdir[i];

    system(cmd.c_str());
  }

  // Copy files
  vector<string> cpSrcFile, cpDstDir;
  vector<bool>   cpForce;

  cpSrcFile.push_back("inline_math.h"); cpDstDir.push_back("include"); cpForce.push_back(false);
  cpSrcFile.push_back("pnlpGpu.h");     cpDstDir.push_back("include"); cpForce.push_back(true);
  cpSrcFile.push_back("main.cpp");      cpDstDir.push_back("src");     cpForce.push_back(true);
  cpSrcFile.push_back("ipopt.opt");     cpDstDir.push_back("");        cpForce.push_back(false);

  for (int i = 0; i < cpSrcFile.size(); i++) {
    string srcFile = "template/" + cpSrcFile[i];
    string dstFile = "generated/" + _name + "/" + cpDstDir[i] + "/" + cpSrcFile[i];

    _copyFile(srcFile.c_str(), dstFile.c_str(), cpForce[i]);
  }

  string makefileName = "generated/" + _name + "/Makefile";

  _copyFile("template/Makefile", makefileName.c_str(), "PH_TARGET", _name.c_str(), false);

  // Generate files
  _generateEvalFuncsH();

  _generateGpuCpp();

  _generateGpuEvalCppH(PnlpObjFunc);
  _generateGpuEvalCppH(PnlpObjGraFunc);
  _generateGpuEvalCppH(PnlpConFunc);
  _generateGpuEvalCppH(PnlpConJacFunc);

  // Copy evaluation functions
  for (int i = PnlpObjFunc; i <= PnlpConJacFunc; i++)
    for (int f = 0; f < _pFuncs[i]->getNumFuncs(); f++) {
      string fileName = _pFuncs[i]->getFuncName(f) + ".hh";
      string srcFile  = "test/" + _name + "/include/" + fileName;
      string dstFile  = "generated/" + _name + "/eval/include/" + fileName;

      _copyFile(srcFile.c_str(), dstFile.c_str(), false);

      fileName = _pFuncs[i]->getFuncName(f) + ".cu";
      srcFile  = "test/" + _name + "/src/" + fileName;
      dstFile  = "generated/" + _name + "/eval/src/" + fileName;

      _copyFile(srcFile.c_str(), dstFile.c_str(), false);
    }
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
bool PnlpProblem::_buildBounds()
{
  // Get variable lower/upper bounds
  for (int i = 0; i < _numVars; i++) {
    _varLowerBounds.push_back(_doc["Variable"]["lb"][i].GetDouble());
    _varUpperBounds.push_back(_doc["Variable"]["ub"][i].GetDouble());
  }

  // Get constraint lower/upper bounds
  for (int i = 0; i < _numCons; i++) {
    _conLowerBounds.push_back(_doc["Constraint"]["LowerBound"][i].GetDouble());
    _conUpperBounds.push_back(_doc["Constraint"]["UpperBound"][i].GetDouble());
  }

  return true;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
bool PnlpProblem::_buildStartingPoint()
{
  // Get variable initial value
  for (int i = 0; i < _numVars; i++)
    _varInitValues.push_back(_doc["Variable"]["initial"][i].GetDouble());

  return true;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
bool PnlpProblem::_buildSparsityStructure()
{
  // Get gradient function's nonzero rows/columns
  _numNzGra = _doc["Objective"]["nnzJac"].GetInt();

  for (int i = 0; i < _numNzGra; i++) {
    _nzGraRows.push_back(_doc["Objective"]["nzJacRows"][i].GetInt()-1);
    _nzGraCols.push_back(_doc["Objective"]["nzJacCols"][i].GetInt()-1);
  }

  // Get Jacobian function's nonzero rows/columns
  _numNzJac = _doc["Constraint"]["nnzJac"].GetInt();

  for (int i = 0; i < _numNzJac; i++) {
    _nzJacRows.push_back(_doc["Constraint"]["nzJacRows"][i].GetInt()-1);
    _nzJacCols.push_back(_doc["Constraint"]["nzJacCols"][i].GetInt()-1);
  }

  return true;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
bool PnlpProblem::_buildEvalFuncs(PnlpFuncType funcType)
{
  PnlpFuncs *pFuncs;

  pFuncs = new PnlpFuncs(funcType, _doc);

  pFuncs->load(_doc);
  pFuncs->build();

  _pFuncs[funcType] = pFuncs;

  return true;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void PnlpProblem::_generateEvalFuncsH()
{
  string fileName = "generated/" + _name + "/include/evalFuncs.h";

  FILE *fp = fopen(fileName.c_str(), "w");

  fprintf(fp, "\n#ifndef EVAL_FUNCS_H\n#define EVAL_FUNCS_H\n\n");

  for (int i = PnlpObjFunc; i <= PnlpConJacFunc; i++)
    for (int f = 0; f < _pFuncs[i]->getNumFuncs(); f++)
      fprintf(fp, "#include \"%s.hh\"\n", _pFuncs[i]->getFuncName(f).c_str());

  fprintf(fp, "\n#endif\n\n");

  fclose(fp);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void PnlpProblem::_generateGpuCpp()
{
  string oFileName = "generated/" + _name + "/src/pnlpGpu.cpp";

  FILE *iFile = fopen("template/pnlpGpu.cpp", "r");
  FILE *oFile = fopen(oFileName.c_str(), "w");

  while (fgets(_buffer, _bufferSize, iFile)) {
    // Replace the placeholder in get_nlp_info
    if (strcmp(_buffer, "//PH_NLP_INFO\n") == 0) {
      fprintf(oFile, "  n = %d;\n", _numVars);
      fprintf(oFile, "  m = %d;\n", _numCons);
      fprintf(oFile, "  nnz_jac_g = %d;\n", _numNzJac);
    }
    else
    // Replace the placeholder in get_bounds_info
    if (strcmp(_buffer, "//PH_BOUNDS_INFO\n") == 0) {
      fprintf(oFile, "  assert(n == %d);\n", _numVars);
      fprintf(oFile, "  assert(m == %d);\n\n", _numCons);

      for (int i = 0; i < _numVars; i++)
        fprintf(oFile, "  x_l[%d] = %f;\n", i, _varLowerBounds[i]);
      fprintf(oFile, "\n");
      for (int i = 0; i < _numVars; i++)
        fprintf(oFile, "  x_u[%d] = %f;\n", i, _varUpperBounds[i]);
      fprintf(oFile, "\n");

      for (int i = 0; i < _numCons; i++)
        fprintf(oFile, "  g_l[%d] = %f;\n", i, _conLowerBounds[i]);
      fprintf(oFile, "\n");
      for (int i = 0; i < _numCons; i++)
        fprintf(oFile, "  g_u[%d] = %f;\n", i, _conUpperBounds[i]);
    }
    else
    // Replace the placeholder in get_starting_point
    if (strcmp(_buffer, "//PH_STARTING_POINT\n") == 0) {
      for (int i = 0; i < _numVars; i++)
        fprintf(oFile, "  x[%d] = %f;\n", i, _varInitValues[i]);
    }
    else
    // Replace the placeholder for Jacobian sparsity structure
    if (strcmp(_buffer, "//PH_JAC_STRUCTURE\n") == 0) {
      for (int i = 0; i < _numNzJac; i++) {
        fprintf(oFile, "    iRow[%d] = %d;", i, _nzJacRows[i]);
        fprintf(oFile, "  jCol[%d] = %d;\n", i, _nzJacCols[i]);
      }
    }
    else
      fprintf(oFile, "%s", _buffer);
  }

  fclose(iFile);
  fclose(oFile);
}

// -----------------------------------------------------------------------------
void PnlpProblem::_generateGpuEvalCppH(PnlpFuncType funcType)
{
  string fileName;
  string className;
  string macroName;
  PnlpFuncs *pFuncs;

  switch (funcType) {
    case PnlpObjFunc: {
      fileName   = "pnlpGpuEvalObj";
      className  = "PnlpGpuEvalObj";
      macroName  = "PNLP_GPU_EVAL_OBJ";
      break;
    }
    case PnlpObjGraFunc: {
      fileName   = "pnlpGpuEvalObjGra";
      className  = "PnlpGpuEvalObjGra";
      macroName  = "PNLP_GPU_EVAL_OBJ_GRA";
      break;
    }
    case PnlpConFunc: {
      fileName   = "pnlpGpuEvalCon";
      className  = "PnlpGpuEvalCon";
      macroName  = "PNLP_GPU_EVAL_CON";
      break;
    }
    case PnlpConJacFunc: {
      fileName   = "pnlpGpuEvalConJac";
      className  = "PnlpGpuEvalConJac";
      macroName  = "PNLP_GPU_EVAL_CON_JAC";
      break;
    }
  }

  pFuncs = _pFuncs[funcType];

  // Create the cpp file
  string iFileName = "template/pnlpGpuEval.cpp";
  string oFileName = "generated/" + _name + "/src/" + fileName + ".cu";

  FILE *iFile = fopen(iFileName.c_str(), "r");
  FILE *oFile = fopen(oFileName.c_str(), "w");

  while (fgets(_buffer, _bufferSize, iFile)) {
    // Replace the placeholder for class name and file name in the template file
    _replaceString(_buffer, "PH_PnlpGpuEval", className.c_str());
    _replaceString(_buffer, "PH_pnlpGpuEval", fileName.c_str());

    if (strcmp(_buffer, "//PH_INIT\n") == 0) {
      fprintf(oFile, "  %s_pInMemCpu  = (double*)malloc(%d*sizeof(double));\n",
                        className.c_str(), pFuncs->getInMemSize());
      fprintf(oFile, "  %s_pOutMemCpu = (double*)malloc(%d*sizeof(double));\n\n",
                        className.c_str(), pFuncs->getOutMemSize());
      fprintf(oFile, "  #ifdef PNLP_ON_GPU\n");
      fprintf(oFile, "  cudaMalloc(&%s_pInMemGpu,  %d*sizeof(double));\n",
                        className.c_str(), pFuncs->getInMemSize());
      fprintf(oFile, "  cudaMalloc(&%s_pOutMemGpu, %d*sizeof(double));\n",
                        className.c_str(), pFuncs->getOutMemSize());
      fprintf(oFile, "  #endif // PNLP_ON_GPU\n");
    }
    else
    if (strcmp(_buffer, "//PH_EVAL\n") == 0) {
      fprintf(oFile, "  // Preamble\n");

      for (int f = 0; f < pFuncs->getNumFuncs(); f++) {
        int offset = pFuncs->getFuncInMemOffset(f);

        for (int i = 0; i < pFuncs->getFuncInDepSize(f); i++)
          fprintf(oFile, "  %s_pInMemCpu[%d+%d] = input[%d];\n",
                            className.c_str(), offset, i, pFuncs->getFuncInDep(f, i));

        for (int i = 0; i < pFuncs->getFuncInAuxSize(f); i++)
          fprintf(oFile, "  %s_pInMemCpu[%d+%d] = %f;\n",
                            className.c_str(), offset, i, pFuncs->getFuncInAux(f, i));
      }
      fprintf(oFile, "\n");

      fprintf(oFile, "  #ifdef PNLP_ON_GPU\n");
      fprintf(oFile, "  cudaMemcpy(%s_pInMemGpu, %s_pInMemCpu, %d*sizeof(double), %s);\n\n",
                        className.c_str(), className.c_str(),
                        pFuncs->getInMemSize(), "cudaMemcpyHostToDevice");

      fprintf(oFile, "  // Deploy\n");
      fprintf(oFile, "  %sKernel <<<%d, 1>>> (%s_pInMemGpu, %s_pOutMemGpu);\n\n",
                        className.c_str(), pFuncs->getNumFuncs(), className.c_str(), className.c_str());
      fprintf(oFile, "  cudaDeviceSynchronize();\n\n");

      fprintf(oFile, "  // Postamble\n");
      fprintf(oFile, "  cudaMemcpy(%s_pOutMemCpu, %s_pOutMemGpu, %d*sizeof(double), %s);\n",
                        className.c_str(), className.c_str(),
                        pFuncs->getOutMemSize(), "cudaMemcpyDeviceToHost");
      fprintf(oFile, "  #else\n");
      fprintf(oFile, "  for (int f = 0; f < %d; f++)\n", pFuncs->getNumFuncs());
      fprintf(oFile, "    %sKernel(f, %s_pInMemCpu, %s_pOutMemCpu);\n",
                          className.c_str(), className.c_str(), className.c_str());
      fprintf(oFile, "  #endif\n\n");

      fprintf(oFile, "  memset(output, 0, %d*sizeof(double));\n",
                        pFuncs->getOutSize());

      for (int f = 0; f < pFuncs->getNumFuncs(); f++) {
        int offset = pFuncs->getFuncOutMemOffset(f);

        for (int i = 0; i < pFuncs->getFuncOutDepSize(f); i++)
          fprintf(oFile, "  output[%d] += %s_pOutMemCpu[%d+%d];\n",
                            pFuncs->getFuncOutDep(f, i), className.c_str(), offset, i);
      }
    }
    else
    if (strcmp(_buffer, "//PH_KERNEL\n") == 0) {
      fprintf(oFile, "  switch (idx) {\n");
      for (int f = 0; f < pFuncs->getNumFuncs(); f++)
        fprintf(oFile, "    case %d: { %s(output+%d, input+%d); break; }\n",
                            f, pFuncs->getFuncName(f).c_str(),
                            pFuncs->getFuncOutMemOffset(f),
                            pFuncs->getFuncInMemOffset(f));
      fprintf(oFile, "    default: break;\n");
      fprintf(oFile, "  }\n");
    }
    else
      fprintf(oFile, "%s", _buffer);
  }

  fclose(iFile);
  fclose(oFile);

  // Create the header file
  iFileName = "template/pnlpGpuEval.h";
  oFileName = "generated/" + _name + "/include/" + fileName + ".h";

  iFile = fopen(iFileName.c_str(), "r");
  oFile = fopen(oFileName.c_str(), "w");

  while (fgets(_buffer, _bufferSize, iFile)) {
    string line = _buffer;

    _replaceString(_buffer, "PH_PNLP_GPU_EVAL", macroName.c_str());
    _replaceString(_buffer, "PH_PnlpGpuEval",   className.c_str());

    fprintf(oFile, "%s", _buffer);
  }

  fclose(iFile);
  fclose(oFile);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void PnlpProblem::_replaceString(char *str, const char *oldToken, const char *newToken)
{
  string line = str;

  int pos = line.find(oldToken);

  if (pos != string::npos) {
    line.replace(pos, strlen(oldToken), newToken);

    strcpy(str, line.c_str());
  }
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void PnlpProblem::_copyFile(const char *srcFile, const char *dstFile, bool force)
{
  _copyFile(srcFile, dstFile, "", "", force);
}

// -----------------------------------------------------------------------------
void PnlpProblem::_copyFile(const char *srcFile, const char *dstFile,
                            const char *oldName, const char *newName, bool force)
{
  if (!force) {
    FILE *fp = fopen(dstFile, "r");

    if (fp) {
      fclose(fp);
      return;
    }
  }

  FILE *iFile = fopen(srcFile, "r");
  FILE *oFile = fopen(dstFile, "w");

  while (fgets(_buffer, _bufferSize, iFile)) {
    if (oldName != "") {
      string line = _buffer;

      int pos = line.find(oldName);
      if (pos != string::npos) {
        line.replace(pos, strlen(oldName), newName);
        strcpy(_buffer, line.c_str());
      }
    }

    fprintf(oFile, "%s", _buffer);
  }

  fclose(iFile);
  fclose(oFile);
}

