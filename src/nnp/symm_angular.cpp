// c libraries
#include <cstring>
// c++ libraries
#include <ostream>
// ann - math
#include "src/math/const.hpp"
// ann - symm - angular
#include "src/nnp/symm_angular.hpp"

//*****************************************
// PhiAN - angular function names
//*****************************************

PhiAN PhiAN::read(const char* str){
	if(std::strcmp(str,"G3")==0) return PhiAN::G3;
	else if(std::strcmp(str,"G4")==0) return PhiAN::G4;
	else return PhiAN::UNKNOWN;
}

const char* PhiAN::name(const PhiAN& phiAN){
	switch(phiAN){
		case PhiAN::G3: return "G3";
		case PhiAN::G4: return "G4";
		default: return "UNKNOWN";
	}
}

std::ostream& operator<<(std::ostream& out, const PhiAN& phiAN){
	switch(phiAN){
		case PhiAN::G3: out<<"G3"; break;
		case PhiAN::G4: out<<"G4"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}
//*****************************************
// PhiA - angular function base class
//*****************************************

//==== operators ====

bool operator==(const PhiA& phia1, const PhiA& phia2){
	return true;
}

std::ostream& operator<<(std::ostream& out, const PhiA& f){
	return out;
}

//*****************************************
// PhiA - serialization
//*****************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const PhiA& obj){
		return 0;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const PhiA& obj, char* arr){
		return 0;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(PhiA& obj, const char* arr){
		return 0;
	}
	
}