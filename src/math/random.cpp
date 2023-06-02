//c libraries
#include <cstring>
#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
#include <cmath>
#elif (defined __ICC || defined __INTEL_COMPILER)
#include <mathimf.h> //intel math library
#endif
//c++ libraries
#include <iostream>
//ann - random
#include "src/math/random.hpp"

namespace rng{

namespace dist{
	
	//******************************************************
	// Distribution - Name
	//******************************************************

	std::ostream& operator<<(std::ostream& out, const Name& n){
		switch(n){
			case Name::UNIFORM: out<<"UNIFORM"; break;
			case Name::EXP: out<<"EXP"; break;
			case Name::NORMAL: out<<"NORMAL"; break;
			case Name::RAYLEIGH: out<<"RAYLEIGH"; break;
			case Name::LOGISTIC: out<<"LOGISTIC"; break;
			case Name::CAUCHY: out<<"CAUCHY"; break;
			default: out<<"UNKNOWN"; break;
		}
		return out;
	}

	const char* Name::name(const Name& n){
		switch(n){
			case Name::UNIFORM: return "UNIFORM";
			case Name::EXP: return "EXP";
			case Name::NORMAL: return "NORMAL";
			case Name::RAYLEIGH: return "RAYLEIGH";
			case Name::LOGISTIC: return "LOGISTIC";
			case Name::CAUCHY: return "CAUCHY";
			default: return "UNKNOWN";
		}
	}

	Name Name::read(const char* str){
		if(std::strcmp(str,"UNIFORM")==0) return Name::UNIFORM;
		else if(std::strcmp(str,"EXP")==0) return Name::EXP;
		else if(std::strcmp(str,"NORMAL")==0) return Name::NORMAL;
		else if(std::strcmp(str,"RAYLEIGH")==0) return Name::RAYLEIGH;
		else if(std::strcmp(str,"LOGISTIC")==0) return Name::LOGISTIC;
		else if(std::strcmp(str,"CAUCHY")==0) return Name::CAUCHY;
		else return Name::UNKNOWN;
	}

} //end namespace dist
	
} //end namespace rng