#pragma once
#ifndef MOLGRAPH_HPP
#define MOLGRAPH_HPP

// c++ libraries
#include <ostream>

namespace molgraph{

class Atom{
private:
	std::string name_;
	int an_;
	int type_;
	std::vector<int> bonds_;
public:
	//==== constructors/destuctors ====
	Atom(){}
	Atom(const std::string& name):name_(name){}
	~Atom();
	
	//==== access ====
	std::string& name(){return name_;}
	const std::string& name()const{return name_;}
	int& type(){return type_;}
	const int& type()const{return type_;}
	int& an(){return an_;}
	const int& an()const{return an_;}
	std::vector<int>& bonds(){return bonds_;}
	const std::vector<int>& bonds()const{return bonds_;}
};

class Molecule{
private:
	std::vector<Atom> atoms_;
public:
	Molecule(){}
	~Molecule(){}
	
	static Molecule& build(const Structure& struc, Molecule& molecule);
};
	
};