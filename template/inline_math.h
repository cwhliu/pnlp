
#ifndef INLINE_MATH_H
#define INLINE_MATH_H

__device__ inline double Power(double x, double y) { return pow(x, y); }
__device__ inline double Sqrt(double x) { return sqrt(x); }

__device__ inline double Abs(double x) { return fabs(x); }

__device__ inline double Exp(double x) { return exp(x); }
__device__ inline double Log(double x) { return log(x); }

__device__ inline double Sin(double x) { return sin(x); }
__device__ inline double Cos(double x) { return cos(x); }
__device__ inline double Tan(double x) { return tan(x); }

__device__ inline double ArcSin(double x) { return asin(x); }
__device__ inline double ArcCos(double x) { return acos(x); }
__device__ inline double ArcTan(double x) { return atan(x); }
__device__ inline double ArcTan(double x, double y) { return atan2(y,x); }

__device__ inline double Sinh(double x) { return sinh(x); }
__device__ inline double Cosh(double x) { return cosh(x); }
__device__ inline double Tanh(double x) { return tanh(x); }

__device__ const double E = 2.71828182845904523536029;
__device__ const double Pi = 3.14159265358979323846264;
__device__ const double Degree = 0.01745329251994329576924;

#endif // INLINE_MATH_H

