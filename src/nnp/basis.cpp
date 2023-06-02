// c libraries
#include <cstring>
#include <cstdio>
// c++ libraries
#include <iostream>
#include <vector>
// ann - str
#include "src/str/string.hpp"
// ann - basis - radial
#include "src/nnp/basis.hpp"

//==== using statements ====

using math::constant::PI;

//==== constructors/destructors ====

/**
* default constructor
*/
Basis::Basis(){
	if(BASIS_PRINT_FUNC>0) std::cout<<"Basis():\n";
	clear();
}

/**
* constructor
* @param rc - cutoff radius
* @param cutname - name of cutoff function
* @param cutnorm - name of cutoff normalization scheme
* @param size - number of symmetry functions
*/
Basis::Basis(double rc, cutoff::Name cutname, cutoff::Norm cutnorm, int size=0){
	if(BASIS_PRINT_FUNC>0) std::cout<<"Basis(double,cutoff::Name,cutoff::Norm,int):\n";
	if(rc<=0) throw std::invalid_argument("Basis(double,cutoff::Name,cutoff::Norm,int): Invalid cutoff radius.\n");
	//set cutoff
	cutname_=cutname;
	cutnorm_=cutnorm;
	rc_=rc;//cutoff value
	rci_=1.0/rc_;
	pirci_=PI/rc_;
	//resize
	resize(size);
	//init
	norm_=norm(cutnorm_,rc_);//normalization scheme
}

/**
* destructor
*/
Basis::~Basis(){
	clear();
}

//==== member functions ====

/**
* clear basis
*/
void Basis::clear(){
	if(BASIS_PRINT_FUNC>0) std::cout<<"Basis::clear():\n";
	cutname_=cutoff::Name::UNKNOWN;
	cutnorm_=cutoff::Norm::UNKNOWN;
	rc_=0;
	rci_=0;
	pirci_=0;
	norm_=0;
	size_=0;
}

void Basis::resize(int size){
	if(BASIS_PRINT_FUNC>0) std::cout<<"Basis::resize(int):\n";
	if(size<0) throw std::invalid_argument("Basis::resize(int): invalid number of functions.");
	size_=size;
	if(size_>0){
		symm_.resize(size_);
	}
}

double Basis::cut_func(double dr)const{
	double val=0;
	switch(cutname_){
		case cutoff::Name::STEP:{
			val=(dr>rc_)?0.0:1.0;
		} break;
		case cutoff::Name::COS:{
			val=(dr>rc_)?0:0.5*(cos(dr*pirci_)+1.0);
		} break;
		case cutoff::Name::TANH:{
			const double f=(dr>rc_)?0:tanh(1.0-dr*rci_);
			val=f*f*f;
		} break;
		default:
			throw std::invalid_argument("Basis:cutf(double): invalid cutoff type.");
		break;
	}
	return val;
}

double Basis::cut_grad(double dr)const{
	double val=0;
	switch(cutname_){
		case cutoff::Name::STEP:{
			val=0.0;
		} break;
		case cutoff::Name::COS:{
			val=(dr>rc_)?0:-0.5*pirci_*sin(dr*pirci_);
		} break;
		case cutoff::Name::TANH:{
			if(dr<rc_){
				const double f=tanh(1.0-dr*rci_);
				val=-3.0*f*f*(1.0-f*f)*rci_;
			} else val=0;
		} break;
		default:
			throw std::invalid_argument("Basis:cutf(double): invalid cutoff type.");
		break;
	}
	return val;
}

void Basis::cut_comp(double dr, double& v, double& g)const{
	switch(cutname_){
		case cutoff::Name::STEP:{
			v=(dr>rc_)?0.0:1.0;
			g=0.0;
		} break;
		case cutoff::Name::COS:{
			if(dr<rc_){
				const double arg=dr*pirci_;
				v=0.5*(cos(arg)+1.0);
				g=-0.5*pirci_*sin(arg);
			} else {
				v=0;
				g=0;
			}
		} break;
		case cutoff::Name::TANH:{
			if(dr<rc_){
				const double t=tanh(1.0-dr*rci_);
				v=t*t*t;
				g=-3.0*t*t*(1.0-t*t)*rci_;
			} else {
				v=0;
				g=0;
			}
		} break;
		default:
			throw std::invalid_argument("Basis:cutf(double): invalid cutoff type.");
		break;
	}
}

//==== static functions ====

/**
* compute the normalization constant
* @param rc - the cutoff radius
*/
double Basis::norm(cutoff::Norm cutnorm, double rc){
	double tmp=0;
	switch(cutnorm){
		case cutoff::Norm::UNIT: tmp=1.0; break;
		case cutoff::Norm::VOL: tmp=1.0/(0.5*4.0/3.0*(PI*PI-6.0)/PI*rc*rc*rc); break;
		default: throw std::invalid_argument("Basis::norm(NormT::type,double): invalid normalization scheme.");
	}
	return tmp;
}
