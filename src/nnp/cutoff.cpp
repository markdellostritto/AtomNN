// c libraries
#include <cstring>
// c++ libraries
#include <ostream> 
// ann - cutoff
#include "src/nnp/cutoff.hpp"

namespace cutoff{

//************************************************************
// NORMALIZATION SCHEMES
//************************************************************

Norm Norm::read(const char* str){
	if(std::strcmp(str,"UNIT")==0) return Norm::UNIT;
	else if(std::strcmp(str,"VOL")==0) return Norm::VOL;
	else return Norm::UNKNOWN;
}

const char* Norm::name(const Norm& norm){
	switch(norm){
		case Norm::UNIT: return "UNIT";
		case Norm::VOL: return "VOL";
		default: return "UNKNOWN";
	}
}

std::ostream& operator<<(std::ostream& out, const Norm& norm){
	switch(norm){
		case Norm::UNIT: out<<"UNIT"; break;
		case Norm::VOL: out<<"VOL"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}
	
//************************************************************
// CUTOFF NAMES
//************************************************************

Name Name::read(const char* str){
	if(std::strcmp(str,"COS")==0) return Name::COS;
	else if(std::strcmp(str,"TANH")==0) return Name::TANH;
	else return Name::UNKNOWN;
}

const char* Name::name(const Name& name){
	switch(name){
		case Name::COS: return "COS";
		case Name::TANH: return "TANH";
		default: return "UNKNOWN";
	}
}

std::ostream& operator<<(std::ostream& out, const Name& name){
	switch(name){
		case Name::COS: out<<"COS"; break;
		case Name::TANH: out<<"TANH"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

//************************************************************
// CUTOFF FUNCTIONS
//************************************************************
	
//==== Cos ====

//operators

std::ostream& operator<<(std::ostream& out, const Cos& obj){
	return out<<"COS rc "<<obj.rc_<<" rci "<<obj.rci_;
}

//member functions

/**
* compute the value of the cutoff function
* @param r - interatomic distance
* @return the value of the cutoff function
*/
double Cos::val(double r)const{
	return (r>rc_)?0:0.5*(cos(r*prci_)+1.0);
}

/**
* compute the gradient of the cutoff function
* @param r - interatomic distance
* @return the gradient of the cutoff function
*/
double Cos::grad(double r)const{
	return (r>rc_)?0:-0.5*prci_*sin(r*prci_);
}

/**
* compute the value and gradient of the cutoff function
* @param r - interatomic distance
* @param v - stores value
* @param g - stores gradient
*/
void Cos::compute(double r, double& v, double& g)const{
	if(r<rc_){
		v=0.5*(cos(r*prci_)+1.0);
		g=-0.5*prci_*sin(r*prci_);
	} else {
		v=0;
		g=0;
	}
}

//==== Tanh ====

//operators

std::ostream& operator<<(std::ostream& out, const Tanh& obj){
	return out<<"TANH rc "<<obj.rc_<<" rci "<<obj.rci_;
}

//member functions

/**
* compute the value of the cutoff function
* @param r - interatomic distance
* @return the value of the cutoff function
*/
double Tanh::val(double r)const{
	const double f=(r>rc_)?0:tanh(1.0-r*rci_); return f*f*f;
}

/**
* compute the gradient of the cutoff function
* @param r - interatomic distance
* @return the gradient of the cutoff function
*/
double Tanh::grad(double r)const{
	if(r<rc_){
		const double f=tanh(1.0-r*rci_);
		return -3.0*f*f*(1.0-f*f)*rci_;
	} else return 0;
}

/**
* compute the value and gradient of the cutoff function
* @param r - interatomic distance
* @param v - stores value
* @param g - stores gradient
*/
void Tanh::compute(double r, double& v, double& g)const{
	if(r<rc_){
		const double f=tanh(1.0-r*rci_);
		v=f*f*f;
		g=-3.0*f*f*(1.0-f*f)*rci_;
	} else {
		v=0;
		g=0;
	}
}

}
