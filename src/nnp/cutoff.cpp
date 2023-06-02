// c libraries
#include <cstring>
// c++ libraries
#include <ostream> 
#include <stdexcept> 
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
	if(std::strcmp(str,"STEP")==0) return Name::STEP;
	else if(std::strcmp(str,"COS")==0) return Name::COS;
	else if(std::strcmp(str,"TANH")==0) return Name::TANH;
	else return Name::UNKNOWN;
}

const char* Name::name(const Name& name){
	switch(name){
		case Name::STEP: return "STEP";
		case Name::COS: return "COS";
		case Name::TANH: return "TANH";
		default: return "UNKNOWN";
	}
}

std::ostream& operator<<(std::ostream& out, const Name& name){
	switch(name){
		case Name::STEP: out<<"STEP"; break;
		case Name::COS: out<<"COS"; break;
		case Name::TANH: out<<"TANH"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

}
