#pragma once
#ifndef ATOM_HPP
#define ATOM_HPP

//c++ libraries
#include <iosfwd>
#include <string>
// ann - serialize
#include "src/mem/serialize.hpp"

#ifndef ATOM_PRINT_FUNC
#define ATOM_PRINT_FUNC 0
#endif

//************************************************************
// ATOM
//************************************************************

class Atom{
private:
	double mass_,energy_,charge_;
	int id_;
	std::string name_;
public:
	//constructors/destructors
	Atom(){clear();}
	~Atom(){}
	//access
	double& mass(){return mass_;}
	const double& mass()const{return mass_;}
	double& energy(){return energy_;}
	const double& energy()const{return energy_;}
	double& charge(){return charge_;}
	const double& charge()const{return charge_;}
	int& id(){return id_;}
	const int& id()const{return id_;}
	std::string& name(){return name_;}
	const std::string& name()const{return name_;}
	//member functions
	void clear();
	//static functions
	static Atom& read(const char* str, Atom& atom);
	static void print(FILE* out, const Atom& atom);
	//operators
	friend std::ostream& operator<<(std::ostream& out, const Atom& atom);
};
bool operator==(const Atom& atom1, const Atom& atom2);
inline bool operator!=(const Atom& atom1, const Atom& atom2){return !(atom1==atom2);}

namespace serialize{

//**********************************************
// byte measures
//**********************************************

template <> int nbytes(const Atom& obj);

//**********************************************
// packing
//**********************************************

template <> int pack(const Atom& obj, char* arr);

//**********************************************
// unpacking
//**********************************************

template <> int unpack(Atom& obj, const char* arr);

}

#endif
