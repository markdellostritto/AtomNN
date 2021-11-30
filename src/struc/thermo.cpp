//c libraries
#include <stdexcept>
#include <cstring>
//c++ libraries
#include <iostream>
// ann - structure
#include "src/struc/thermo.hpp"

//**********************************************************************************************
//Thermo
//**********************************************************************************************

//==== operators ====

std::ostream& operator<<(std::ostream& out, const Thermo& obj){
	out<<"energy = "<<obj.energy_<<"\n";
	out<<"ewald  = "<<obj.ewald_<<"\n";
	out<<"temp   = "<<obj.temp_<<"\n";
	out<<"press  = "<<obj.press_;
	return out;
}

//==== member functions ====

void Thermo::clear(){
	energy_=0;
	ewald_=0;
	temp_=0;
	press_=0;
}

//**********************************************************************************************
// serialization
//**********************************************************************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const Thermo& obj){
		int size=0;
		size+=sizeof(obj.energy());
		size+=sizeof(obj.ewald());
		size+=sizeof(obj.temp());
		size+=sizeof(obj.press());
		return size;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const Thermo& obj, char* arr){
		int pos=0;
		std::memcpy(arr+pos,&obj.energy(),sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&obj.ewald(),sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&obj.temp(),sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&obj.press(),sizeof(double)); pos+=sizeof(double);
		return pos;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(Thermo& obj, const char* arr){
		int pos=0;
		std::memcpy(&obj.energy(),arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.ewald(),arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.temp(),arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.press(),arr+pos,sizeof(double)); pos+=sizeof(double);
		return pos;
	}
		
}
