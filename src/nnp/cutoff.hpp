#pragma once
#ifndef CUTOFF_HPP
#define CUTOFF_HPP

// c libraries
#include <cstdio>
#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
#include <cmath>
#elif (defined __ICC || defined __INTEL_COMPILER)
#include <mathimf.h> //intel math library
#endif
// c++ libraries
#include <iosfwd> 
// ann - math
#include "src/math/const.hpp"

namespace cutoff{

//==== using statements ====

using math::constant::PI;

//************************************************************
// NORMALIZATION SCHEMES
//************************************************************

struct Norm{
public:
	enum Type{
		UNKNOWN=-1,
		UNIT=0,
		VOL=1
	};
	//constructor
	Norm():t_(Type::UNKNOWN){}
	Norm(Type t):t_(t){}
	//operators
	operator Type()const{return t_;}
	//member functions
	static Norm read(const char* str);
	static const char* name(const Norm& norm);
private:
	Type t_;
	//prevent automatic conversion for other built-in types
	//template<typename T> operator T() const;
};
std::ostream& operator<<(std::ostream& out, const Norm& norm);

//************************************************************
// CUTOFF NAMES
//************************************************************

//cutoff names
class Name{
public:
	enum Type{
		UNKNOWN=-1,
		COS=0,
		TANH=1
	};
	//constructor
	Name():t_(Type::UNKNOWN){}
	Name(Type t):t_(t){}
	//operators
	operator Type()const{return t_;}
	//member functions
	static Name read(const char* str);
	static const char* name(const Name& name);
private:
	Type t_;
	//prevent automatic conversion for other built-in types
	//template<typename T> operator T() const;
};
std::ostream& operator<<(std::ostream& out, const Name& name);

//************************************************************
// CUTOFF FUNCTIONS
//************************************************************

//==== Func ====

class Func{
protected:
	Name name_;
	double rc_,rci_;
public:
	//constructors/destructors
	Func():rc_(0),rci_(0),name_(Name::UNKNOWN){}
	Func(double rc):rc_(rc),rci_(1.0/rc),name_(Name::UNKNOWN){}
	virtual ~Func(){}
	//access
	const Name& name()const{return name_;}
	const double& rc()const{return rc_;}
	const double& rci()const{return rci_;}
	//member functions
	virtual double val(double r)const=0;
	virtual double grad(double r)const=0;
	virtual void compute(double r, double& v, double& g)const=0;
};

//==== Cos ====

class Cos final: public Func{
private:
	double prci_;
public:
	//constructors/destructors
	Cos():Func(){name_=Name::COS;}
	Cos(double rc):Func(rc),prci_(rci_*PI){name_=Name::COS;}
	virtual ~Cos(){}
	//operators
	friend std::ostream& operator<<(std::ostream& out, const Cos& obj);
	//member functions
	double val(double r)const;
	double grad(double r)const;
	void compute(double r, double& v, double& g)const;
};

//==== Tanh ====

class Tanh final: public Func{
public:
	//constructors/destructors
	Tanh():Func(){name_=Name::TANH;}
	Tanh(double rc):Func(rc){name_=Name::TANH;}
	virtual ~Tanh(){}
	//operators
	friend std::ostream& operator<<(std::ostream& out, const Tanh& obj);
	//member functions
	double val(double r)const;
	double grad(double r)const;
	void compute(double r, double& v, double& g)const;
};

}

#endif
