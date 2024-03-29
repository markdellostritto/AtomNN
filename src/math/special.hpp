#pragma once
#ifndef MATH_SPECIAL_HPP
#define MATH_SPECIAL_HPP

// c libaries
#include <cstdio>
#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
#include <cmath>
#elif (defined __ICC || defined __INTEL_COMPILER)
#include <mathimf.h> //intel math library
#else
#include <cmath>
#endif
// c++ libraries
#include <iosfwd>
#include <vector>
#include <array>
// ann - math
#include "src/math/const.hpp"

#ifndef DEBUG_MATH_SPECIAL
#define DEBUG_MATH_SPECIAL 0
#endif 

namespace math{

namespace special{
	
	static const double prec=1E-8;
	
	//**************************************************************
	// comparison functions
	//**************************************************************
	
	template <class T> inline T max(T x1, T x2)noexcept{return (x1>x2)?x1:x2;}
	template <class T> inline T min(T x1, T x2)noexcept{return (x1>x2)?x2:x1;}
	
	//**************************************************************
	// modulus functions
	//**************************************************************
	
	template <class T> inline T mod(T n, T z)noexcept{return (n%z+z)%z;}
	template <> inline int mod<int>(int n, int z)noexcept{return (n%z+z)%z;}
	template <> inline unsigned int mod<unsigned int>(unsigned int n, unsigned int z)noexcept{return (n%z+z)%z;}
	template <> inline float mod<float>(float n, float z)noexcept{return fmod(fmod(n,z)+z,z);}
	template <> inline double mod<double>(double n, double z)noexcept{return fmod(fmod(n,z)+z,z);}
	template <class T> inline T mod(T n, T lLim, T uLim)noexcept{return mod<T>(n-lLim,uLim-lLim)+lLim;}
	
	//**************************************************************
	// sign function
	//**************************************************************

	template <typename T> int sgn(T val){return (T(0) < val) - (val < T(0));}
	
	//**************************************************************
	//trig (fdlibm)
	//**************************************************************
	
	static const double cos_const[6]={
		4.16666666666666019037e-02,
		-1.38888888888741095749e-03,
		2.48015872894767294178e-05,
		-2.75573143513906633035e-07,
		2.08757232129817482790e-09,
		-1.13596475577881948265e-11
	};
	//cosine function
	double cos(double x)noexcept;
	
	static const double sin_const[6]={
		-1.66666666666666324348e-01,
		8.33333333332248946124e-03,
		-1.98412698298579493134e-04,
		2.75573137070700676789e-06,
		-2.50507602534068634195e-08,
		1.58969099521155010221e-10
	};
	//sine function
	double sin(double x)noexcept;
	
	//**************************************************************
	//Hypberbolic Functions
	//**************************************************************
	
	double sinh(double x);
	double cosh(double x);
	double tanh(double x);
	double csch(double x);
	double sech(double x);
	double coth(double x);
	void tanhsech(double x, double& ftanh, double& fsech);
	
	//**************************************************************
	//Power
	//**************************************************************
	
	double powint(double x, int p);
	
	double sqrta(double x);
	
	//**************************************************************
	//Logarithm
	//**************************************************************
	
	double logp1(double x)noexcept;
	
	//**************************************************************
	//Sigmoid
	//**************************************************************
	
	double sigmoid(double x);
	
	//**************************************************************
	//Softplus
	//**************************************************************
	
	double softplus(double x)noexcept;
	
	double exp2_x86(double x);
	double fmexp(double x);

	//**************************************************************
	//Gamma Function
	//**************************************************************
	
	static const double gammac[15]={
		0.99999999999999709182,
		57.156235665862923517,
		-59.597960355475491248,
		14.136097974741747174,
		-0.49191381609762019978,
		0.33994649984811888699e-4,
		0.46523628927048575665e-4,
		-0.98374475304879564677e-4,
		0.15808870322491248884e-3,
		-0.21026444172410488319e-3,
		0.21743961811521264320e-3,
		-0.16431810653676389022e-3,
		0.84418223983852743293e-4,
		-0.26190838401581408670e-4,
		0.36899182659531622704e-5
	};
	double lgamma(double x);
	double tgamma(double x);
	
	//**************************************************************
	//Beta Function
	//**************************************************************
	
	double beta(double z, double w);
	
	//**************************************************************
	//Kummer's (confluent hypergeometric) function 
	//**************************************************************
	double M(double a, double b, double z, double prec=1e-8);

}

namespace poly{
	
	//**************************************************************
	//Legendre Poylnomials
	//**************************************************************
	double legendre(int n, double x);
	std::vector<double>& legendre_c(int n, std::vector<double>& c);
	
	//**************************************************************
	//Chebyshev Polynomials
	//**************************************************************
	double chebyshev1l(int n, double x);//chebyshev polynomial of the first kind
	double chebyshev2l(int n, double x);//chebyshev polynomial of the second kind
	std::vector<double>& chebyshev1_c(int n, double x, std::vector<double>& r);//polynomial coefficients
	std::vector<double>& chebyshev2_c(int n, double x, std::vector<double>& r);//polynomial coefficients
	std::vector<double>& chebyshev1_r(int n, std::vector<double>& r);//polynomial roots
	std::vector<double>& chebyshev2_r(int n, std::vector<double>& r);//polynomial roots
	
	//**************************************************************
	//Jacobi Polynomials
	//**************************************************************
	double jacobi(int n, double a, double b, double x);
	std::vector<double>& jacobi(int n, double a, double b, std::vector<double>& c);
	
	//**************************************************************
	//Laguerre Polynomials
	//**************************************************************
	double laguerre(int n, double x);
	std::vector<double>& laguerre_c(int n, std::vector<double>& c);
}

namespace pdist{
	
	//******************************************************
	// Distribution - Base
	//******************************************************
	
	class Dist{
	public:
		virtual ~Dist(){}
		virtual double p(double x)=0;
		virtual double cdf(double x)=0;
		virtual double icdf(double x)=0;
	};
	
	//******************************************************
	// Distribution - Exponential
	//******************************************************
	
	class Exp{
	private:
		double beta_;
	public:
		Exp(double beta):beta_(beta){}
		double p(double x);
		double cdf(double x);
		double icdf(double x);
	};
	
	//******************************************************
	// Distribution - Normal
	//******************************************************
	
	class Normal{
	private:
		static const double Rad2PI;
		double mu_,sigma_;
	public:
		Normal(double mu, double sigma):mu_(mu),sigma_(sigma){}
		double p(double x);
		double cdf(double x);
		double icdf(double x);
	};
	
	//******************************************************
	// Distribution - Logistic
	//******************************************************
	
	class Logistic{
	private:
		static const double a_;
		double mu_,sigma_;
	public:
		Logistic(){}
		double p(double x);
		double cdf(double x);
		double icdf(double x);
	};
	
	//******************************************************
	// Distribution - Cauchy
	//******************************************************
	
	class Cauchy{
	private:
		double mu_,sigma_;
	public:
		Cauchy(double mu, double sigma):mu_(mu),sigma_(sigma){}
		double p(double x);
		double cdf(double x);
		double icdf(double x);
	};
	
	//******************************************************
	// Distribution - LogNormal
	//******************************************************
	
	class LogNormal{
	private:
		static const double rad2Pi;
		double mu_,sigma_;
	public:
		LogNormal(double mu, double sigma):mu_(mu),sigma_(sigma){}
		double p(double x);
		double cdf(double x);
		double icdf(double x);
	};
	
	//******************************************************
	// Distribution - Gamma
	//******************************************************
	
	class Gamma{
	private:
		double alpha_,beta_;
	public:
		Gamma(double alpha, double beta):alpha_(alpha),beta_(beta){}
		double p(double x);
		double cdf(double x);
		double icdf(double x);
	};
	
}

}

#endif