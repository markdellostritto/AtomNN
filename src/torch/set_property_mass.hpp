#pragma once
#ifndef SET_PROPERTY_MASS_HPP
#define SET_PROPERTY_MASS_HPP

//c++
#include <iosfwd>
#include <vector>
//string
#include "src/str/token.hpp"
//structure
#include "src/struc/structure.hpp"
//torch
#include "src/torch/set_property.hpp"

namespace property{
	
class Mass: public Base{
private:
	double mass_;
public:
	//==== constructors/destructors ====
	Mass():Base(Name::MASS){}
	~Mass(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Mass& mass);
	
	//==== access ====
	double& mass(){return mass_;}
	const double& mass()const{return mass_;}
	
	//==== member functions ====
	void read(Token& token);
	void set(Structure& struc);
};
	
}

#endif