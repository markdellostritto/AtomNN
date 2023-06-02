#pragma once
#ifndef STOCHASTIC_HPP
#define STOCHASTIC_HPP

// string
#include "src/str/token.hpp"
// torch
#include "src/torch/engine.hpp"
#include "src/torch/monte_carlo.hpp"

class Stochastic{
private:
	double dr_;
	double dv_;
	Metropolis met_;
public:
	//==== constructors/destructors ====
	Stochastic():dr_(0),dv_(0){}
	~Stochastic(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Stochastic& s);
	
	//==== access ====
	double& dr(){return dr_;}
	const double& dr()const{return dr_;}
	double& dv(){return dv_;}
	const double& dv()const{return dv_;}
	Metropolis& met(){return met_;}
	const Metropolis& met()const{return met_;}
	
	//==== member functions ====
	void read(Token& token);
	
	//==== member functions ====
	Structure& step(Structure& struc, Engine& engine);
};

#endif