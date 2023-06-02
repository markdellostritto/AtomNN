//c libraries
#include <stdexcept>
#include <cstring>
//c++ libraries
#include <iostream>
// structure
#include "src/struc/state.hpp"

//**********************************************************************************************
//State
//**********************************************************************************************

//==== operators ====

std::ostream& operator<<(std::ostream& out, const State& obj){
	out<<"energy = "<<obj.energy_<<"\n";
	out<<"ewald  = "<<obj.ewald_<<"\n";
	out<<"temp   = "<<obj.temp_<<"\n";
	out<<"press  = "<<obj.press_<<"\n";
	out<<"qtot   = "<<obj.qtot_;
	return out;
}

//==== member functions ====

void State::clear(){
	energy_=0;
	ewald_=0;
	temp_=0;
	press_=0;
	qtot_=0;
	ke_=0;
	pe_=0;
	t_=0;
	dt_=0;
}

//**********************************************************************************************
// serialization
//**********************************************************************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const State& obj){
		int size=0;
		size+=sizeof(obj.energy());
		size+=sizeof(obj.ewald());
		size+=sizeof(obj.temp());
		size+=sizeof(obj.press());
		size+=sizeof(obj.qtot());
		size+=sizeof(obj.pe());
		size+=sizeof(obj.ke());
		size+=sizeof(obj.dt());
		size+=sizeof(obj.t());
		return size;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const State& obj, char* arr){
		int pos=0;
		std::memcpy(arr+pos,&obj.energy(),sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&obj.ewald(),sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&obj.temp(),sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&obj.press(),sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&obj.qtot(),sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&obj.pe(),sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&obj.ke(),sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&obj.dt(),sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&obj.t(),sizeof(int)); pos+=sizeof(int);
		return pos;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(State& obj, const char* arr){
		int pos=0;
		std::memcpy(&obj.energy(),arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.ewald(),arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.temp(),arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.press(),arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.qtot(),arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.pe(),arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.ke(),arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.dt(),arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.t(),arr+pos,sizeof(int)); pos+=sizeof(int);
		return pos;
	}
	
}
