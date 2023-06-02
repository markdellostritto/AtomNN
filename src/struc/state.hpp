#pragma once
#ifndef STATE_HPP
#define STATE_HPP

// c++ libraries
#include <iosfwd>
// mem
#include "src/mem/serialize.hpp"

//**********************************************************************************************
//State
//**********************************************************************************************

class State{
protected:
	double energy_;//energy
	double ewald_;//ewald
	double pe_;//kinetic energy
	double ke_;//potential energy
	double temp_;//temperature
	double press_;//pressure
	double qtot_;//total charge
	double dt_;//timestep
	int t_;//time
public:
	//==== constructors/destructors ====
	State():energy_(0.0),ewald_(0.0),temp_(0.0),press_(0.0),qtot_(0.0),ke_(0.0),pe_(0.0),t_(0),dt_(0){}
	~State(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const State& obj);
	
	//==== access ====
	double& energy(){return energy_;}
	const double& energy()const{return energy_;}
	double& ewald(){return ewald_;}
	const double& ewald()const{return ewald_;}
	double& temp(){return temp_;}
	const double& temp()const{return temp_;}
	double& T(){return temp_;}
	const double& T()const{return temp_;}
	double& press(){return press_;}
	const double& press()const{return press_;}
	double& P(){return press_;}
	const double& P()const{return press_;}
	double& qtot(){return qtot_;}
	const double& qtot()const{return qtot_;}
	double& pe(){return pe_;}
	const double& pe()const{return pe_;}
	double& ke(){return ke_;}
	const double& ke()const{return ke_;}
	double& dt(){return dt_;}
	const double& dt()const{return dt_;}
	int& t(){return t_;}
	const int& t()const{return t_;}
	
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
	
	template <> int nbytes(const State& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const State& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(State& obj, const char* arr);
	
}

#endif