
#include <string>

#include <cstdio>

#include "pnlpProblem.h"
#include "pnlpFuncs.h"

#include "rapidjson/filereadstream.h"

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
bool PnlpProblem::load(const char *fileName)
{
  FILE *fp = fopen(fileName, "r");

  if (!fp) {
    printf("\nError: can not open file \"%s\"\n\n", fileName);
    return false;
  }

  char buffer[1024];
  rapidjson::FileReadStream fileStream(fp, buffer, sizeof(buffer));

  _doc.ParseStream(fileStream);

  if (!_doc.IsObject()) {
    printf("\nError: parse file \"%s\" failed\n\n", fileName);
    return false;
  }

  fclose(fp);

  return true;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
bool PnlpProblem::build()
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
  _buildObjFuncs();
  _buildObjGraFuncs();
  _buildConFuncs();
  _buildConJacFuncs();

  printf("Number of variables:        %d\n", _numVars);
  printf("Number of constraints:      %d\n", _numCons);
  printf("Number of nonzero Jacobian: %d\n", _numNzJac);

  return true;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void PnlpProblem::generate()
{
  system("mkdir -p generated/nlp051/include");
  system("mkdir -p generated/nlp051/src");
  system("mkdir -p generated/nlp051/obj");
  system("mkdir -p generated/nlp051/dep");
  system("mkdir -p generated/nlp051/eval/include");
  system("mkdir -p generated/nlp051/eval/src");

  // May want to skip copying inline_nath.h
  //_copyFile("template/inline_math.h", "generated/nlp051/include/inline_math.h");
  _copyFile("template/pnlpGpu.h", "generated/nlp051/include/pnlpGpu.h");
  _copyFile("template/main.cpp", "generated/nlp051/src/main.cpp");
  _copyFile("template/ipopt.opt", "generated/nlp051/ipopt.opt");
  _copyFile("template/Makefile", "generated/nlp051/Makefile", "PH_TARGET", "nlp051");

  FILE *iFile;
  FILE *oFile;
  char buffer[1024];
  oFile = fopen("generated/nlp051/include/evalFuncs.h", "w");
  fprintf(oFile, "\n#ifndef EVAL_FUNCS_H\n#define EVAL_FUNCS_H\n\n");
  for (int i = 0; i < _pObjFuncs->getNumFuncs(); i++) {
    fprintf(oFile, "#include \"%s.hh\"\n",   _pObjFuncs->getFuncName(i).c_str());
    fprintf(oFile, "#include \"J_%s.hh\"\n", _pObjFuncs->getFuncName(i).c_str());
  }
  for (int i = 0; i < _pConFuncs->getNumFuncs(); i++) {
    fprintf(oFile, "#include \"%s.hh\"\n",   _pConFuncs->getFuncName(i).c_str());
    fprintf(oFile, "#include \"J_%s.hh\"\n", _pConFuncs->getFuncName(i).c_str());
  }
  fprintf(oFile, "\n#endif\n\n");
  fclose(oFile);

  _generateGpuCpp();

  _generateGpuEvalCpp(PnlpObjFunc);
  _generateGpuEvalCpp(PnlpObjGraFunc);
  _generateGpuEvalCpp(PnlpConFunc);
  _generateGpuEvalCpp(PnlpConJacFunc);
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
  _numNzGra = _doc["Objective"]["nnzJac"].GetInt();

  // Get gradient function's nonzero rows/columns
  for (int i = 0; i < _numNzGra; i++) {
    _nzGraRows.push_back(_doc["Objective"]["nzJacRows"][i].GetInt()-1);
    _nzGraCols.push_back(_doc["Objective"]["nzJacCols"][i].GetInt()-1);
  }

  _numNzJac = _doc["Constraint"]["nnzJac"].GetInt();

  // Get Jacobian function's nonzero rows/columns
  for (int i = 0; i < _numNzJac; i++) {
    _nzJacRows.push_back(_doc["Constraint"]["nzJacRows"][i].GetInt()-1);
    _nzJacCols.push_back(_doc["Constraint"]["nzJacCols"][i].GetInt()-1);
  }

  return true;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
bool PnlpProblem::_buildObjFuncs()
{
  _pObjFuncs = new PnlpFuncs(PnlpObjFunc, _doc);

  _pObjFuncs->load(_doc);

  return _pObjFuncs->build();
}

// -----------------------------------------------------------------------------
bool PnlpProblem::_buildObjGraFuncs()
{
  _pObjGraFuncs = new PnlpFuncs(PnlpObjGraFunc, _doc);

  _pObjGraFuncs->load(_doc);

  return _pObjGraFuncs->build();
}

// -----------------------------------------------------------------------------
bool PnlpProblem::_buildConFuncs()
{
  _pConFuncs = new PnlpFuncs(PnlpConFunc, _doc);

  _pConFuncs->load(_doc);

  return _pConFuncs->build();
}

// -----------------------------------------------------------------------------
bool PnlpProblem::_buildConJacFuncs()
{
  _pConJacFuncs = new PnlpFuncs(PnlpConJacFunc, _doc);

  _pConJacFuncs->load(_doc);

  return _pConJacFuncs->build();
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void PnlpProblem::_generateGpuCpp()
{
  FILE *iFile = fopen("template/pnlpGpu.cpp", "r");
  FILE *oFile = fopen("generated/nlp051/src/pnlpGpu.cpp", "w");

  char buffer[1024];

  while (fgets(buffer, sizeof(buffer), iFile)) {
    // Replace the placeholder in get_nlp_info
    if (strcmp(buffer, "//PH_NLP_INFO\n") == 0) {
      fprintf(oFile, "  n = %d;\n", _numVars);
      fprintf(oFile, "  m = %d;\n", _numCons);
      fprintf(oFile, "  nnz_jac_g = %d;\n", _numNzJac);
    }
    else
    // Replace the placeholder in get_bounds_info
    if (strcmp(buffer, "//PH_BOUNDS_INFO\n") == 0) {
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
    if (strcmp(buffer, "//PH_STARTING_POINT\n") == 0) {
      for (int i = 0; i < _numVars; i++)
        fprintf(oFile, "  x[%d] = %f;\n", i, _varInitValues[i]);
    }
    else
    // Replace the placeholder for Jacobian sparsity structure
    if (strcmp(buffer, "//PH_JAC_STRUCTURE\n") == 0) {
      for (int i = 0; i < _numNzJac; i++) {
        fprintf(oFile, "    iRow[%d] = %d;", i, _nzJacRows[i]);
        fprintf(oFile, "  jCol[%d] = %d;\n", i, _nzJacCols[i]);
      }
    }
    else
      fprintf(oFile, "%s", buffer);
  }

  fclose(iFile);
  fclose(oFile);
}

// -----------------------------------------------------------------------------
void PnlpProblem::_generateGpuEvalCpp(PnlpFuncType funcType)
{
  string fileName;
  string className;
  string macroName;
  string funcPrefix;
  PnlpFuncs *pFuncs;

  switch (funcType) {
    case PnlpObjFunc: {
      fileName   = "pnlpGpuEvalObj";
      className  = "PnlpGpuEvalObj";
      macroName  = "PNLP_GPU_EVAL_OBJ";
      funcPrefix = "";
      pFuncs     = _pObjFuncs;
      break;
    }
    case PnlpObjGraFunc: {
      fileName   = "pnlpGpuEvalObjGra";
      className  = "PnlpGpuEvalObjGra";
      macroName  = "PNLP_GPU_EVAL_OBJ_GRA";
      funcPrefix = "J_";
      pFuncs     = _pObjGraFuncs;
      break;
    }
    case PnlpConFunc: {
      fileName   = "pnlpGpuEvalCon";
      className  = "PnlpGpuEvalCon";
      macroName  = "PNLP_GPU_EVAL_CON";
      funcPrefix = "";
      pFuncs     = _pConFuncs;
      break;
    }
    case PnlpConJacFunc: {
      fileName   = "pnlpGpuEvalConJac";
      className  = "PnlpGpuEvalConJac";
      macroName  = "PNLP_GPU_EVAL_CON_JAC";
      funcPrefix = "J_";
      pFuncs     = _pConJacFuncs;
      break;
    }
  }

  string iFileName = "template/pnlpGpuEval.cpp";
  string oFileName = "generated/nlp051/src/" + fileName + ".cu";

  FILE *iFile = fopen(iFileName.c_str(), "r");
  FILE *oFile = fopen(oFileName.c_str(), "w");

  char buffer[1024];

  while (fgets(buffer, sizeof(buffer), iFile)) {
    string line = buffer;

    int pos = line.find("PH_PnlpGpuEval");
    if (pos != string::npos) {
      line.replace(pos, 14, className);
      strcpy(buffer, line.c_str());
    }

    // Replace the placeholder for file name with the actual name, most likely
    // only for the include .h statement
    pos = line.find("PH_pnlpGpuEval");
    if (pos != string::npos) {
      line.replace(pos, 14, fileName);
      strcpy(buffer, line.c_str());
    }

    if (strcmp(buffer, "//PH_INIT\n") == 0) {
      fprintf(oFile, "  %s_pInMemCpu  = (double*)malloc(%d*sizeof(double));\n",
                        className.c_str(), pFuncs->getInMemSize());
      fprintf(oFile, "  %s_pOutMemCpu = (double*)malloc(%d*sizeof(double));\n\n",
                        className.c_str(), pFuncs->getOutMemSize());
      fprintf(oFile, "  cudaMalloc(&%s_pInMemGpu,  %d*sizeof(double));\n",
                        className.c_str(), pFuncs->getInMemSize());
      fprintf(oFile, "  cudaMalloc(&%s_pOutMemGpu, %d*sizeof(double));\n",
                        className.c_str(), pFuncs->getOutMemSize());
    }
    else
    if (strcmp(buffer, "//PH_EVAL\n") == 0) {
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

      fprintf(oFile, "  cudaMemcpy(%s_pInMemGpu, %s_pInMemCpu, %d*sizeof(double), %s);\n\n",
                        className.c_str(), className.c_str(),
                        pFuncs->getInMemSize(), "cudaMemcpyHostToDevice");

      fprintf(oFile, "  // Deploy\n");
      fprintf(oFile, "  %sKernel <<<3, 1>>> (%s_pInMemGpu, %s_pOutMemGpu);\n\n",
                        className.c_str(), className.c_str(), className.c_str());
      fprintf(oFile, "  cudaDeviceSynchronize();\n\n");

      fprintf(oFile, "  // Postamble\n");
      fprintf(oFile, "  cudaMemcpy(%s_pOutMemCpu, %s_pOutMemGpu, %d*sizeof(double), %s);\n\n",
                        className.c_str(), className.c_str(),
                        pFuncs->getOutMemSize(), "cudaMemcpyDeviceToHost");
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
    if (strcmp(buffer, "//PH_KERNEL\n") == 0) {
      fprintf(oFile, "  switch (idx) {\n");
      for (int f = 0; f < pFuncs->getNumFuncs(); f++)
        fprintf(oFile, "    case %d: { %s%s(output+%d, input+%d); break; }\n",
                            f, funcPrefix.c_str(), pFuncs->getFuncName(f).c_str(),
                            pFuncs->getFuncOutMemOffset(f),
                            pFuncs->getFuncInMemOffset(f));
      fprintf(oFile, "    default: break;\n");
      fprintf(oFile, "  }\n");
    }
    else
      fprintf(oFile, "%s", buffer);
  }

  fclose(iFile);
  fclose(oFile);

  // Copy header file and change the class name
  iFileName = "template/pnlpGpuEval.h";
  oFileName = "generated/nlp051/include/" + fileName + ".h";

  iFile = fopen(iFileName.c_str(), "r");
  oFile = fopen(oFileName.c_str(), "w");

  while (fgets(buffer, sizeof(buffer), iFile)) {
    string line = buffer;

    int pos = line.find("PH_PNLP_GPU_EVAL");
    if (pos != string::npos) {
      line.replace(pos, 16, macroName);
      strcpy(buffer, line.c_str());
    }

    pos = line.find("PH_PnlpGpuEval");
    if (pos != string::npos) {
      line.replace(pos, 14, className);
      strcpy(buffer, line.c_str());
    }

    fprintf(oFile, "%s", buffer);
  }

  fclose(iFile);
  fclose(oFile);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void PnlpProblem::_copyFile(const char *srcFile, const char *dstFile)
{
  _copyFile(srcFile, dstFile, "", "");
}

// -----------------------------------------------------------------------------
void PnlpProblem::_copyFile(const char *srcFile, const char *dstFile,
                            const char *oldName, const char *newName)
{
  char buffer[1024];

  FILE *iFile = fopen(srcFile, "r");
  FILE *oFile = fopen(dstFile, "w");

  while (fgets(buffer, sizeof(buffer), iFile)) {
    if (oldName != "") {
      string line = buffer;

      int pos = line.find(oldName);
      if (pos != string::npos) {
        line.replace(pos, strlen(oldName), newName);
        strcpy(buffer, line.c_str());
      }
    }

    fprintf(oFile, "%s", buffer);
  }

  fclose(iFile);
  fclose(oFile);
}

