// c++ libraries
#include <ostream>
#include <stdexcept>
// math - special
#include "src/math/special.hpp"

namespace math{

namespace special{
	
	//**************************************************************
	//trig (fdlibm)
	//**************************************************************
	
	//cosine function
	double cos(double x)noexcept{
		x*=x;
		return 1.0+x*(-0.5+x*(cos_const[0]+x*(cos_const[1]+x*(cos_const[2]+x*(cos_const[3]+x*(cos_const[4]+x*cos_const[5]))))));
	}
	
	//sine function
	double sin(double x)noexcept{
		const double r=x*x;
		return x*(1.0+r*(sin_const[0]+r*(sin_const[1]+r*(sin_const[2]+r*(sin_const[3]+r*(sin_const[4]+r*sin_const[5]))))));
	}
	
	//**************************************************************
	//Hypberbolic Function
	//**************************************************************
	
	double sinh(double x){
		if(x>=0){
			const double expf=exp(-x);
			return (1.0-expf*expf)/(2.0*expf);
		} else {
			const double expf=exp(x);
			return (expf*expf-1.0)/(2.0*expf);
		}
	}
	
	double cosh(double x){
		if(x>=0){
			const double expf=exp(-x);
			return (1.0+expf*expf)/(2.0*expf);
		} else {
			const double expf=exp(x);
			return (expf*expf+1.0)/(2.0*expf);
		}
	}
	
	double tanh(double x){
		if(x>=0){
			const double expf=exp(-2.0*x);
			return (1.0-expf)/(1.0+expf);
		} else {
			const double expf=exp(2.0*x);
			return (expf-1.0)/(expf+1.0);
		}
	}
	
	double csch(double x){
		if(x>=0){
			const double expf=exp(-x);
			return 2.0*expf/(1.0-expf*expf);
		} else {
			const double expf=exp(x);
			return 2.0*expf/(expf*expf-1.0);
		}
	}
	
	double sech(double x){
		if(x>=0){
			const double expf=exp(-x);
			return 2.0*expf/(1.0+expf*expf);
		} else {
			const double expf=exp(x);
			return 2.0*expf/(expf*expf+1.0);
		}
	}
	
	double coth(double x){
		if(x>=0){
			const double expf=exp(-2.0*x);
			return (1.0+expf)/(1.0-expf);
		} else {
			const double expf=exp(2.0*x);
			return (expf+1.0)/(expf-1.0);
		}
	}
	
	void tanhsech(double x, double& ftanh, double& fsech){
		if(x>=0){
			const double fexp=exp(-x);
			const double fexp2=fexp*fexp;
			const double den=1.0/(1.0+fexp2);
			ftanh=(1.0-fexp2)*den;
			fsech=2.0*fexp*den;
		} else {
			const double fexp=exp(x);
			const double fexp2=fexp*fexp;
			const double den=1.0/(1.0+fexp2);
			ftanh=(fexp2-1.0)*den;
			fsech=2.0*fexp*den;
		}
	}
	
	//**************************************************************
	//Power
	//**************************************************************
	
	double powint(double x, const int n){
		double yy, ww;
		if (n == 0) return 1.0;
		if (x == 0.0) return 0.0;
		int nn = (n > 0) ? n : -n;
		ww = x;
		for (yy = 1.0; nn != 0; nn >>= 1, ww *= ww)
		if (nn & 1) yy *= ww;
		return (n > 0) ? yy : 1.0 / yy;
	}
	
	
	double sqrta(double x){
		return (256.0+x*(1792.0+x*(1120.0+x*(112.0+x))))/(1024.0+x*(1792.0+x*(448.0+x*16)));
	}
	
	//**************************************************************
	//Logarithm
	//**************************************************************
	
	double logp1(double x)noexcept{
		const double y=x/(x+2.0);
		const double y2=y*y;
		return 2.0*y*(1.0+y2*(1.0/3.0+y2*(1.0/5.0+y2*(1.0/7.0+y2*1.0/9.0))));
	}
	
	//**************************************************************
	//Sigmoid
	//**************************************************************
	
	double sigmoid(double x){
		if(x>=0){
			return 1.0/(1.0+exp(-x));
		} else {
			const double expf=exp(x);
			return expf/(expf+1.0);
		}
	}
	
	//**************************************************************
	//Softplus
	//**************************************************************
	
	double softplus(double x)noexcept{
		if(x>=0.0) return x+logp1(exp(-x));
		else return logp1(exp(x));
	}
	
	//**************************************************************
	//Exponential
	//**************************************************************
	
	/* optimizer friendly implementation of exp2(x).
	*
	* strategy:
	*
	* split argument into an integer part and a fraction:
	* ipart = floor(x+0.5);
	* fpart = x - ipart;
	*
	* compute exp2(ipart) from setting the ieee754 exponent
	* compute exp2(fpart) using a pade' approximation for x in [-0.5;0.5[
	*
	* the result becomes: exp2(x) = exp2(ipart) * exp2(fpart)
	*/

	/* IEEE 754 double precision floating point data manipulation */
	typedef union {
		double   f;
		uint64_t u;
		struct {int32_t  i0,i1;} s;
	}  udi_t;

	static const double fm_exp2_q[] = {
	/*  1.00000000000000000000e0, */
		2.33184211722314911771e2,
		4.36821166879210612817e3
	};
	static const double fm_exp2_p[] = {
		2.30933477057345225087e-2,
		2.02020656693165307700e1,
		1.51390680115615096133e3
	};

	/* double precision constants */
	#define FM_DOUBLE_LOG2OFE  1.4426950408889634074
	
	double exp2_x86(double x){
	#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
		double   ipart, fpart, px, qx;
		udi_t    epart;

		ipart = floor(x+0.5);
		fpart = x - ipart;
		epart.s.i0 = 0;
		epart.s.i1 = (((int) ipart) + 1023) << 20;

		x = fpart*fpart;

		px =        fm_exp2_p[0];
		px = px*x + fm_exp2_p[1];
		qx =    x + fm_exp2_q[0];
		px = px*x + fm_exp2_p[2];
		qx = qx*x + fm_exp2_q[1];

		px = px * fpart;

		x = 1.0 + 2.0*(px/(qx-px));
		return epart.f*x;
	#else
		return pow(2.0, x);
	#endif
	}
	
	double fmexp(double x)
	{
	#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
	    if (x < -1022.0/FM_DOUBLE_LOG2OFE) return 0;
	    if (x > 1023.0/FM_DOUBLE_LOG2OFE) return INFINITY;
	    return exp2_x86(FM_DOUBLE_LOG2OFE * x);
	#else
	    return ::exp(x);
	#endif
	}


	//**************************************************************
	//Gamma Function
	//**************************************************************
	
	double lgamma(double x){
		const double r2pi=2.5066282746310005;
		const double g=671.0/128.0;
		double s=gammac[0];
		for(int i=1; i<15; ++i) s+=gammac[i]/(x+i);
		return (x+0.5)*log(x+g)-(x+g)+log(r2pi*s/x);
	}
	double tgamma(double x){return exp(lgamma(x));}
	
	//**************************************************************
	//Beta Function
	//**************************************************************
	
	double beta(double z, double w){return exp(lgamma(z)+lgamma(w)-lgamma(z+w));}
	
	//**************************************************************
	//Kummer's (confluent hypergeometric) function 
	//**************************************************************
	
	double M(double a, double b, double z, double prec){
		const int nMax=1e8;
		double fac=1,result=1;
		for(int n=1; n<=nMax; ++n){
			fac*=a*z/(n*b);
			result+=fac;
			++a; ++b;
			if(fabs(fac/result)*100<prec) break;
		}
		return result;
	}

}

namespace poly{
	
	//**************************************************************
	//Legendre Polynomials
	//**************************************************************
	
	double legendre(int n, double x){
		if(n<0) throw std::runtime_error("legendre(int,double): invalid order");
		else if(n==0) return 1;
		else {
			double rm2=1.0,rm1=x,r=x;
			for(int i=2; i<=n; ++i){
				r=((2.0*n-1.0)*x*rm1-(n-1.0)*rm2)/i;
				rm2=rm1; rm1=r;
			}
			return r;
		}
	}
	
	std::vector<double>& legendre_c(int n, std::vector<double>& c){
		if(n==0) c.resize(n+1,1.0);
		else if(n==1){c.resize(n+1,0.0); c[1]=1.0;}
		else {
			c.resize(n+1,0.0);
			std::vector<double> ct1(n+1,0.0),ct2(n+1,0.0);
			ct1[0]=1.0; ct2[1]=1.0;
			for(int m=2; m<=n; ++m){
				for(int l=m; l>0; --l) c[l]=(2.0*m-1.0)/m*ct2[l-1];
				c[0]=0.0;
				for(int l=m; l>=0; --l) c[l]-=(m-1.0)/m*ct1[l];
				ct1=ct2; ct2=c;
			}
		}
		return c;
	}
	
	//**************************************************************
	//Chebyshev Polynomials
	//**************************************************************
	
	double chebyshev1(int n, double x){
		if(n<0) throw std::runtime_error("chebyshev1(int,double): invalid order");
		else if(n==0) return 1;
		else {
			double rm2=1.0,rm1=x,r=x;
			for(int i=2; i<=n; ++i){
				r=2*x*rm1-rm2;
				rm2=rm1; rm1=r;
			}
			return r;
		}
	}
	
	double chebyshev2(int n, double x){
		if(n<0) throw std::runtime_error("chebyshev2(int,double): invalid order");
		else if(n==0) return 1;
		else {
			double rm2=1,rm1=2.0*x,r=2.0*x;
			for(int i=2; i<=n; ++i){
				r=2.0*x*rm1-rm2;
				rm2=rm1; rm1=r;
			}
			return r;
		}
	}
	
	std::vector<double>& chebyshev1_c(int n, double x, std::vector<double>& r){
		if(r.size()!=n+1) throw std::invalid_argument("Invalid vector size.");
		r[0]=1;
		if(n>=1){
			r[1]=x;
			for(int i=2; i<=n; ++i){
				r[i]=2*x*r[i-1]-r[i-2];
			}
		}
		return r;
	}
	
	std::vector<double>& chebyshev2_c(int n, double x, std::vector<double>& r){
		if(r.size()!=n+1) throw std::invalid_argument("Invalid vector size.");
		r[0]=1;
		if(n>=1){
			r[1]=2*x;
			for(int i=2; i<=n; ++i){
				r[i]=2*x*r[i-1]-r[i-2];
			}
		}
		return r;
	}
	
	std::vector<double>& chebyshev1_r(int n, std::vector<double>& r){
		r.resize(n);
		for(int i=0; i<n; i++) r[i]=cos((2.0*i+1.0)/(2.0*n)*constant::PI);
		return r;
	}
	
	std::vector<double>& chebyshev2_r(int n, std::vector<double>& r){
		r.resize(n);
		for(int i=0; i<n; i++) r[i]=cos((i+1.0)/(n+1.0)*constant::PI);
		return r;
	}
	
	//**************************************************************
	//Jacobi Polynomials
	//**************************************************************
	
	double jacobi(int n, double a, double b, double x){
		if(n==0) return 1;
		else if(n==1) return 0.5*(2*(a+1)+(a+b+2)*(x-1));
		else return 
			(2*n+a+b-1)*((2*n+a+b)*(2*n+a+b-2)*x+a*a-b*b)/(2*n*(n+a+b)*(2*n+a+b-2))*jacobi(n-1,a,b,x)
			-(n+a-1)*(n+b-1)*(2*n+a+b)/(n*(n+a+b)*(2*n+a+b-2))*jacobi(n-2,a,b,x);
	}
	
	std::vector<double>& jacobi(int n, double a, double b, std::vector<double>& c){
		if(n==0) c.resize(1,1);
		else if(n==1){
			c.resize(2,0);
			c[0]=0.5*(a-b);
			c[1]=0.5*(a+b+2);
		} else {
			c.resize(n+1,0.0);
			std::vector<double> ct1(n+1,0.0),ct2(n+1,0.0);
			ct1[0]=1.0;
			ct2[0]=0.5*(a-b);
			ct2[1]=0.5*(a+b+2);
			for(int m=2; m<=n; ++m){
				c[0]=0.0;
				for(int l=m; l>0; --l) c[l]=(2*m+a+b-1)*(2*m+a+b)/(2*m*(m+a+b))*ct2[l-1];
				for(int l=m; l>=0; --l) c[l]+=(2*m+a+b-1)*(a*a-b*b)/(2*m*(m+a+b)*(2*m+a+b-2))*ct2[l];
				for(int l=m; l>=0; --l) c[l]-=(m+a-1)*(m+b-1)*(2*m+a+b)/(m*(m+a+b)*(2*m+a+b-2))*ct1[l];
				ct1=ct2; ct2=c;
			}
		}
		return c;
	}
	
	//**************************************************************
	//Laguerre Polynomials
	//**************************************************************
	
	double laguerre(int n, double x){
		if(n<0) throw std::runtime_error("chebyshev2(int,double): invalid order");
		else if(n==0) return 1;
		else {
			double rm2=1.0,rm1=1.0-x,r=1.0-x;
			for(int i=2; i<=n; ++i){
				r=((2.0*i-1.0-x)*rm1-(i-1.0)*rm2)/i;
				rm2=rm1; rm1=r;
			}
			return r;
		}
	}
	
	std::vector<double>& laguerre(int n, std::vector<double>& c){
		if(n==0) c.resize(1,1);
		else if(n==1){
			c.resize(2,1);
			c[1]=-1;
		} else {
			c.resize(n+1,0);
			std::vector<double> ct1(n+1,0.0),ct2(n+1,0.0);
			ct1[0]=1;
			ct2[0]=1; ct2[1]=-1;
			for(int m=2; m<=n; ++m){
				c[0]=0;
				for(int l=m; l>0; --l) c[l]=-1.0/m*ct2[l-1];
				for(int l=m; l>=0; --l) c[l]+=(2.0*m-1.0)/m*ct2[l];
				for(int l=m; l>=0; --l) c[l]-=(m-1.0)/m*ct1[l];
				ct1=ct2; ct2=c;
			}
		}
		return c;
	}

}

namespace pdist{
	
	//******************************************************
	// Distribution - Exp
	//******************************************************
	
	double Exp::p(double x){
		return beta_*exp(-beta_*x);
	}
	double Exp::cdf(double x){
		return 1.0-exp(-beta_*x);
	}
	double Exp::icdf(double x){
		return -log(1.0-x)/beta_;
	}
	
	//******************************************************
	// Distribution - Normal
	//******************************************************
	
	const double Normal::Rad2PI=sqrt(2.0*constant::PI);
	double Normal::p(double x){
		return exp(-(x-mu_)*(x-mu_)/(2.0*sigma_*sigma_))/(sigma_*Rad2PI);
	}
	double Normal::cdf(double x){
		return 0.5*erfc(-1.0/constant::Rad2*(x-mu_)/sigma_);
	}
	double Normal::icdf(double x){
		//return mu_-constant::Rad2*sigma_*1.0/std::erfc(2.0*x);
		return 0;
	}
	
	//******************************************************
	// Distribution - Logistic
	//******************************************************
	
	const double Logistic::a_=constant::PI/constant::Rad3;
	double Logistic::p(double x){
		const double expf=(x>=0)?exp(-x):exp(+x);
		return a_/sigma_*x/((1.0+x)*(1.0+x));
	}
	double Logistic::cdf(double x){
		return 0.5+1.0/constant::RadPI*atan((x-mu_)/sigma_);
	}
	double Logistic::icdf(double x){
		return mu_+sigma_*tan(constant::PI*(x-0.5));
	}
	
	//******************************************************
	// Distribution - Cauchy
	//******************************************************
	
	double Cauchy::p(double x){
		return 1.0/(sigma_*constant::PI*(1.0+(x-mu_)*(x-mu_)/sigma_*sigma_));
	}
	double Cauchy::cdf(double x){
		return 0.5+1.0/constant::RadPI*atan((x-mu_)/sigma_);
	}
	double Cauchy::icdf(double x){
		return mu_+sigma_*tan(constant::PI*(x-0.5));
	}
	
	//******************************************************
	// Distribution - LogNormal
	//******************************************************
	
	const double LogNormal::rad2Pi=constant::Rad2*constant::RadPI;
	double LogNormal::p(double x){
		const double l=log(x);
		return 1.0/(rad2Pi*sigma_*x)*exp(-(l-mu_)*(l-mu_)/(2.0*sigma_*sigma_));
	}
	double LogNormal::cdf(double x){
		return 0.5*erfc(-(log(x)-mu_)/(constant::Rad2*sigma_));
	}
	double LogNormal::icdf(double x){
		//return std::exp(mu_-std::Rad2*sigma_*erfci(2.0*x));
		return 0;
	}
	
	//******************************************************
	// Distribution - Gamma
	//******************************************************
	
	double Gamma::p(double x){
		return pow(beta_,alpha_)/tgamma(alpha_)*pow(x,alpha_-1.0)*exp(-beta_*x);
	}
	double Gamma::cdf(double x){
		return 0;
	}
	double Gamma::icdf(double x){
		return 0;
	}

}

}