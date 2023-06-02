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
		UNKNOWN,
		UNIT,
		VOL
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
		UNKNOWN,
		STEP,
		COS,
		TANH
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

}

#endif
