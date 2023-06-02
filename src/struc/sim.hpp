#pragma once
#ifndef SIM_HPP
#define SIM_HPP

//no bounds checking in Eigen
#define EIGEN_NO_DEBUG

//c++ libraries
#include <iosfwd>
//Eigen
#include <Eigen/Dense>
// ann - cell
#include "src/struc/structure.hpp"
// ann - serialize
#include "src/mem/serialize.hpp"
// ann - string
#include "src/str/string.hpp"

#ifndef SIM_PRINT_FUNC
#define SIM_PRINT_FUNC 0
#endif

//**********************************************************************************************
//Interval
//**********************************************************************************************

class Interval{
private:
	int beg_,end_,stride_;
public:
	//==== constructors/destructors ====
	Interval(int b,int e,int s):beg_(b),end_(e),stride_(s){}
	Interval():beg_(-1),end_(-1),stride_(-1){}
	~Interval(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Interval& i);
	
	//==== access ====
	int& beg(){return beg_;}
	const int& beg()const{return beg_;}
	int& end(){return end_;}
	const int& end()const{return end_;}
	int& stride(){return stride_;}
	const int& stride()const{return stride_;}
	const int len()const{return end_-beg_+1;}
	const int nsteps()const{return (end_-beg_+1)/stride_;}
	
	//==== member functions ====
	static Interval& read(const char* str, Interval& interval);
	static Interval split(const Interval& interval, int rank, int nproc);
};

//**********************************************
// Simulation
//**********************************************

class Simulation{
private:
	std::string name_;
	double timestep_;
	int timesteps_;
	AtomType atomT_;
	std::vector<Structure> frames_;
public:
	//==== constructors/destructors ====
	Simulation(){defaults();}
	~Simulation(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Simulation& sim);
	
	//==== access ====
	std::string& name(){return name_;}
	const std::string& name()const{return name_;}
	double& timestep(){return timestep_;}
	const double& timestep()const{return timestep_;}
	int& timesteps(){return timesteps_;}
	const int& timesteps()const{return timesteps_;}
	AtomType& atomT(){return atomT_;}
	const AtomType& atomT()const{return atomT_;}
	Structure& frame(int i){return frames_[i];}
	const Structure& frame(int i)const{return frames_[i];}
	
	//==== member functions ====
	void defaults();
	void clear();
	void resize(int ts, int nAtoms, const AtomType& atomT);
	void resize(int ts);
};


//**********************************************************************************************
// serialization
//**********************************************************************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const Interval& obj);
	template <> int nbytes(const Simulation& sim);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const Interval& obj, char* arr);
	template <> int pack(const Simulation& sim, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(Interval& obj, const char* arr);
	template <> int unpack(Simulation& sim, const char* arr);
	
}

#endif