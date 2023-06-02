#pragma once
#ifndef ATOM_TYPE_HPP
#define ATOM_TYPE_HPP

// c++ libraries
#include <iosfwd>
// mem
#include "src/mem/serialize.hpp"
// str
#include "src/str/token.hpp"

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
	bool radius;
	bool chi;
	bool eta;
	bool c6;
	bool js;
	//vector properties
	bool posn;
	bool vel;
	bool force;
	bool spin;
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
	static AtomType read(Token& token);
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