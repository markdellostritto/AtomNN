#pragma once
#ifndef THERMO_HPP
#define THERMO_HPP

// c++ libraries
#include <iosfwd>
// DAME - mem
#include "src/mem/serialize.hpp"

//**********************************************************************************************
//Thermo
//**********************************************************************************************

class Thermo{
protected:
	double energy_;//energy
	double ewald_;//ewald
	double temp_;//temperature
	double press_;//pressure
public:
	//==== constructors/destructors ====
	Thermo():energy_(0.0),ewald_(0.0),temp_(0.0),press_(0.0){}
	~Thermo(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Thermo& obj);
	
	//==== access ====
	double& energy(){return energy_;}
	const double& energy()const{return energy_;}
	double& ewald(){return energy_;}
	const double& ewald()const{return energy_;}
	double& temp(){return temp_;}
	const double& temp()const{return temp_;}
	double& press(){return press_;}
	const double& press()const{return press_;}
	
	//==== member functions ====
	void clear();
};

//**********************************************************************************************
// serialization
//**********************************************************************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const Thermo& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const Thermo& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(Thermo& obj, const char* arr);
	
}

#endif