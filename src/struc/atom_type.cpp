//c libraries
#include <cstring>
//c++ libraries
#include <iostream>
// DAME - structure
#include "src/struc/atom_type.hpp"

//**********************************************************************************************
//AtomType
//**********************************************************************************************

std::ostream& operator<<(std::ostream& out, const AtomType& atomT){
	//basic properties
	if(atomT.name)		out<<"name ";
	if(atomT.an)		out<<"an ";
	if(atomT.type)		out<<"type ";
	if(atomT.index)	out<<"index ";
	//serial properties
	if(atomT.mass)		out<<"mass ";
	if(atomT.charge)	out<<"charge ";
	if(atomT.spin)		out<<"spin ";
	//vector properties
	if(atomT.posn)		out<<"posn ";
	if(atomT.vel)		out<<"vel ";
	if(atomT.force)	out<<"force ";
	//nnp
	if(atomT.symm)		out<<"symm ";
	return out;
}

void AtomType::defaults(){
	//basic properties
	name  =false;
	an    =false;
	type  =false;
	index =false;
	//serial properties
	mass	 =false;
	charge=false;
	spin  =false;
	//vector properties
	posn  =false;
	vel   =false;
	force =false;
	//nnp
	symm  =false;
}

//**********************************************************************************************
// serialization
//**********************************************************************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const AtomType& obj){
		int size=0;
		//basic properties
		size+=sizeof(obj.name);
		size+=sizeof(obj.an);
		size+=sizeof(obj.type);
		size+=sizeof(obj.index);
		//serial properties
		size+=sizeof(obj.mass);
		size+=sizeof(obj.charge);
		size+=sizeof(obj.spin);
		//vector properties
		size+=sizeof(obj.posn);
		size+=sizeof(obj.vel);
		size+=sizeof(obj.force);
		//nnp
		size+=sizeof(obj.symm);
		return size;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const AtomType& obj, char* arr){
		int pos=0;
		//basic properties
		std::memcpy(arr+pos,&obj.name,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(arr+pos,&obj.an,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(arr+pos,&obj.type,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(arr+pos,&obj.index,sizeof(bool)); pos+=sizeof(bool);
		//serial properties
		std::memcpy(arr+pos,&obj.mass,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(arr+pos,&obj.charge,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(arr+pos,&obj.spin,sizeof(bool)); pos+=sizeof(bool);
		//vector properties
		std::memcpy(arr+pos,&obj.posn,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(arr+pos,&obj.vel,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(arr+pos,&obj.force,sizeof(bool)); pos+=sizeof(bool);
		//nnp
		std::memcpy(arr+pos,&obj.symm,sizeof(bool)); pos+=sizeof(bool);
		return pos;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(AtomType& obj, const char* arr){
		int pos=0;
		//basic properties
		std::memcpy(&obj.name,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(&obj.an,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(&obj.type,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(&obj.index,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		//serial properties
		std::memcpy(&obj.mass,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(&obj.charge,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(&obj.spin,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		//vector properties
		std::memcpy(&obj.posn,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(&obj.vel,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(&obj.force,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		//nnp
		std::memcpy(&obj.symm,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		return pos;
	}
	
}
