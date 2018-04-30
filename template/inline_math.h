
#ifndef INLINE_MATH_H
#define INLINE_MATH_H

#ifdef PNLP_ON_CPU
  #define SPECIFIER
#else
  #define SPECIFIER __device__
#endif

SPECIFIER inline double Power(double x, double y) { return pow(x, y); }
SPECIFIER inline double Sqrt(double x) { return sqrt(x); }

SPECIFIER inline double Abs(double x) { return fabs(x); }

SPECIFIER inline double Exp(double x) { return exp(x); }
SPECIFIER inline double Log(double x) { return log(x); }

SPECIFIER inline double Sin(double x) { return sin(x); }
SPECIFIER inline double Cos(double x) { return cos(x); }
SPECIFIER inline double Tan(double x) { return tan(x); }

SPECIFIER inline double ArcSin(double x) { return asin(x); }
SPECIFIER inline double ArcCos(double x) { return acos(x); }
SPECIFIER inline double ArcTan(double x) { return atan(x); }
SPECIFIER inline double ArcTan(double x, double y) { return atan2(y,x); }

SPECIFIER inline double Sinh(double x) { return sinh(x); }
SPECIFIER inline double Cosh(double x) { return cosh(x); }
SPECIFIER inline double Tanh(double x) { return tanh(x); }

SPECIFIER const double E = 2.71828182845904523536029;
SPECIFIER const double Pi = 3.14159265358979323846264;
SPECIFIER const double Degree = 0.01745329251994329576924;

#endif // INLINE_MATH_H

