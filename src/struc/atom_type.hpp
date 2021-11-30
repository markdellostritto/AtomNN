#pragma once
#ifndef ATOM_TYPE_HPP
#define ATOM_TYPE_HPP

// c++ libraries
#include <iosfwd>
// ann - mem
#include "src/mem/serialize.hpp"

//**********************************************************************************************
//AtomType
//**********************************************************************************************

struct AtomType{
	//==== data ====
	//coordinates
	bool frac;
	//basic properties
	bool name;
	bool an;
	bool type;
	bool index;
	//serial properties
	bool mass;
	bool charge;
	bool spin;
	//vector properties
	bool posn;
	bool vel;
	bool force;
	//nnp
	bool symm;
	//==== constructors/destructors ====
	AtomType(){defaults();}
	~AtomType(){}
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const AtomType& atomT);
	//==== member functions ====
	void defaults();
	void clear(){defaults();}
};

//**********************************************************************************************
// serialization
//**********************************************************************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const AtomType& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const AtomType& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(AtomType& obj, const char* arr);
	
}

#endif