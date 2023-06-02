#pragma once
#ifndef DECAY_HPP
#define DECAY_HPP

// c++
#include <iosfwd>
#include <memory>
// str
#include "src/str/token.hpp"
// opt
#include "src/opt/objective.hpp"

#ifndef OPT_DECAY_PRINT_FUNC
#define OPT_DECAY_PRINT_FUNC 0
#endif

//***************************************************
// decay method
//***************************************************

namespace opt{
namespace decay{

//***************************************************
// decay name
//***************************************************

class Name{
public:
	enum Type{
		UNKNOWN=0,
		CONST=1,
		EXP=2,
		SQRT=3,
		INV=4
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

//***************************************************
// decay function - base
//***************************************************

class Base{
private:
	Name name_;
public:
	//==== constructors/destructors ====
	Base():name_(Name::UNKNOWN){}
	Base(Name name):name_(name){}
	virtual ~Base(){}
	
	//==== access ====
	const Name& name()const{return name_;}
	
	//==== virtual functions ====
	virtual double step(const Objective& obj)=0;
	virtual void read(Token& token){}
};

//***************************************************
// decay function - const
//***************************************************

class Const: public Base{
public:
	//==== constructors/destructors ====
	Const():Base(Name::CONST){}
	Const(double a):Base(Name::CONST){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Const& obj);
	
	//==== member functions ====
	double step(const Objective& obj);
	void read(Token& token){}
};

//***************************************************
// decay function - exp
//***************************************************

class Exp: public Base{
private:
	double alpha_;
public:
	//==== constructors/destructors ====
	Exp():Base(Name::EXP),alpha_(0){}
	Exp(double a):Base(Name::EXP),alpha_(a){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Exp& obj);
	
	//==== access ====
	double& alpha(){return alpha_;}
	const double& alpha()const{return alpha_;}
	
	//==== member functions ====
	double step(const Objective& obj);
	void read(Token& token);
};

//***************************************************
// decay function - sqrt
//***************************************************

class Sqrt: public Base{
private:
	double alpha_;
public:
	//==== constructors/destructors ====
	Sqrt():Base(Name::SQRT),alpha_(0){}
	Sqrt(double a):Base(Name::SQRT),alpha_(a){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Sqrt& obj);
	
	//==== access ====
	double& alpha(){return alpha_;}
	const double& alpha()const{return alpha_;}
	
	//==== member functions ====
	double step(const Objective& obj);
	void read(Token& token);
};

//***************************************************
// decay function - inv
//***************************************************

class Inv: public Base{
private:
	double alpha_;
public:
	//==== constructors/destructors ====
	Inv():Base(Name::INV),alpha_(0){}
	Inv(double a):Base(Name::INV),alpha_(a){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Inv& obj);
	
	//==== access ====
	double& alpha(){return alpha_;}
	const double& alpha()const{return alpha_;}
	
	//==== member functions ====
	double step(const Objective& obj);
	void read(Token& token);
};

//***************************************************
// factory
//***************************************************

std::ostream& operator<<(std::ostream& out, const std::shared_ptr<Base>& obj);
std::shared_ptr<Base>& make(std::shared_ptr<Base>& obj, Name name);
std::shared_ptr<Base>& read(std::shared_ptr<Base>& obj, Token& token);

}
}

//**********************************************
// serialization
//**********************************************

namespace serialize{

//**********************************************
// byte measures
//**********************************************

template <> int nbytes(const std::shared_ptr<opt::decay::Base>& obj);
template <> int nbytes(const opt::decay::Const& obj);
template <> int nbytes(const opt::decay::Exp& obj);
template <> int nbytes(const opt::decay::Sqrt& obj);
template <> int nbytes(const opt::decay::Inv& obj);

//**********************************************
// packing
//**********************************************

template <> int pack(const std::shared_ptr<opt::decay::Base>& obj, char* arr);
template <> int pack(const opt::decay::Const& obj, char* arr);
template <> int pack(const opt::decay::Exp& obj, char* arr);
template <> int pack(const opt::decay::Sqrt& obj, char* arr);
template <> int pack(const opt::decay::Inv& obj, char* arr);

//**********************************************
// unpacking
//**********************************************

template <> int unpack(std::shared_ptr<opt::decay::Base>& obj, const char* arr);
template <> int unpack(opt::decay::Const& obj, const char* arr);
template <> int unpack(opt::decay::Exp& obj, const char* arr);
template <> int unpack(opt::decay::Sqrt& obj, const char* arr);
template <> int unpack(opt::decay::Inv& obj, const char* arr);
	
}

#endif