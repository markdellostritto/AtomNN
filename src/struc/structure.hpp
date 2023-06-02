#pragma once
#ifndef STRUCTURE_HPP
#define STRUCTURE_HPP

//no bounds checking in Eigen
#define EIGEN_NO_DEBUG

//c++ libraries
#include <iosfwd>
//Eigen
#include <Eigen/Dense>
// ann - structure
#include "src/struc/cell.hpp"
#include "src/struc/state.hpp"
#include "src/struc/atom_type.hpp"
// ann - serialize
#include "src/mem/serialize.hpp"

#ifndef STRUC_PRINT_FUNC
#define STRUC_PRINT_FUNC 0
#endif

#ifndef STRUC_PRINT_STATUS
#define STRUC_PRINT_STATUS 0
#endif

#ifndef STRUC_PRINT_DATA
#define STRUC_PRINT_DATA 0
#endif

typedef Eigen::Matrix<double,3,1> Vec3d;
typedef Eigen::Matrix<double,Eigen::Dynamic,1> VecXd;

//**********************************************************************************************
//AtomData
//**********************************************************************************************

class AtomData{
protected:
	//atom type
	AtomType atomType_;
	//number of atoms
	int nAtoms_;
	//basic properties
	std::vector<std::string> name_;//name
	std::vector<int>	an_;//atomic_number
	std::vector<int>	type_;//type
	std::vector<int>	index_;//index
	//serial properties
	std::vector<double>	mass_;//mass
	std::vector<double>	charge_;//charge
	std::vector<double> radius_;//radius
	std::vector<double>	chi_;//chi
	std::vector<double>	eta_;//eta
	std::vector<double>	c6_;//c6 - london disperion coefficient
	std::vector<double>	js_;//js - spin interaction coefficient
	//vector properties
	std::vector<Vec3d>	posn_;//position
	std::vector<Vec3d>	vel_;//velocity
	std::vector<Vec3d>	force_;//force
	std::vector<Vec3d>	spin_;//spin
	//nnp
	std::vector<VecXd>	symm_;//symmetry function
public:
	//==== constructors/destructors ====
	AtomData(){}
	~AtomData(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const AtomData& ad);
	
	//==== access - global ====
	const AtomType& atomType()const{return atomType_;}
	const int& nAtoms()const{return nAtoms_;}
	
	//==== access - vectors ====
	//basic properties
	std::vector<std::string>& name(){return name_;}
	const std::vector<std::string>& name()const{return name_;}
	std::vector<int>& an(){return an_;}
	const std::vector<int>& an()const{return an_;}
	std::vector<int>& type(){return type_;}
	const std::vector<int>& type()const{return type_;}
	std::vector<int>& index(){return index_;}
	const std::vector<int>& index()const{return index_;}
	//serial properties
	std::vector<double>& mass(){return mass_;}
	const std::vector<double>& mass()const{return mass_;}
	std::vector<double>& charge(){return charge_;}
	const std::vector<double>& charge()const{return charge_;}
	std::vector<double>& radius(){return radius_;}
	const std::vector<double>& radius()const{return radius_;}
	std::vector<double>& chi(){return chi_;}
	const std::vector<double>& chi()const{return chi_;}
	std::vector<double>& eta(){return eta_;}
	const std::vector<double>& eta()const{return eta_;}
	std::vector<double>& c6(){return c6_;}
	const std::vector<double>& c6()const{return c6_;}
	std::vector<double>& js(){return js_;}
	const std::vector<double>& js()const{return js_;}
	//vector properties
	std::vector<Vec3d>& posn(){return posn_;}
	const std::vector<Vec3d>& posn()const{return posn_;}
	std::vector<Vec3d>& vel(){return vel_;}
	const std::vector<Vec3d>& vel()const{return vel_;}
	std::vector<Vec3d>& force(){return force_;}
	const std::vector<Vec3d>& force()const{return force_;}
	std::vector<Vec3d>& spin(){return spin_;}
	const std::vector<Vec3d>& spin()const{return spin_;}
	//nnp
	std::vector<VecXd>& symm(){return symm_;}
	const std::vector<VecXd>& symm()const{return symm_;}
	
	//==== access - atoms ====
	//basic properties
	std::string& name(int i){return name_[i];}
	const std::string& name(int i)const{return name_[i];}
	int& an(int i){return an_[i];}
	const int& an(int i)const{return an_[i];}
	int& type(int i){return type_[i];}
	const int& type(int i)const{return type_[i];}
	int& index(int i){return index_[i];}
	const int& index(int i)const{return index_[i];}
	//serial properties
	double& mass(int i){return mass_[i];}
	const double& mass(int i)const{return mass_[i];}
	double& charge(int i){return charge_[i];}
	const double& charge(int i)const{return charge_[i];}
	double& radius(int i){return radius_[i];}
	const double& radius(int i)const{return radius_[i];}
	double& chi(int i){return chi_[i];}
	const double& chi(int i)const{return chi_[i];}
	double& eta(int i){return eta_[i];}
	const double& eta(int i)const{return eta_[i];}
	double& c6(int i){return c6_[i];}
	const double& c6(int i)const{return c6_[i];}
	double& js(int i){return js_[i];}
	const double& js(int i)const{return js_[i];}
	//vector properties
	Vec3d& posn(int i){return posn_[i];}
	const Vec3d& posn(int i)const{return posn_[i];}
	Vec3d& vel(int i){return vel_[i];}
	const Vec3d& vel(int i)const{return vel_[i];}
	Vec3d& force(int i){return force_[i];}
	const Vec3d& force(int i)const{return force_[i];}
	Vec3d& spin(int i){return spin_[i];}
	const Vec3d& spin(int i)const{return spin_[i];}
	//nnp
	VecXd& symm(int i){return symm_[i];}
	const VecXd& symm(int i)const{return symm_[i];}
	
	//==== member functions ====
	void clear();
	void resize(int nAtoms, const AtomType& atomT);
};

//**********************************************************************************************
//Structure
//**********************************************************************************************

class Structure: public Cell, public State, public AtomData{
public:
	//==== constructors/destructors ====
	Structure(){}
	Structure(int nAtoms, const AtomType& atomT){resize(nAtoms,atomT);}
	~Structure(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Structure& sim);
	
	//==== member functions ====
	void clear();
	
	//==== static functions ====
	static void write_binary(const Structure& struc, const char* file);
	static void read_binary(Structure& struc, const char* file);
	static Structure& super(const Structure& struc, Structure& superc, const Eigen::Vector3i nlat);
};

//**********************************************************************************************
//AtomSpecies
//**********************************************************************************************

class AtomSpecies{
protected:
	//==== atomic data ====
	int nSpecies_;
	std::vector<std::string> species_;//the names of each species
	std::vector<int> nAtoms_;//the number of atoms of each species
	std::vector<int> offsets_;//the offsets for each species
public:
	//==== constructors/destructors ====
	AtomSpecies(){defaults();}
	AtomSpecies(const std::vector<std::string>& names, const std::vector<int>& nAtoms){resize(names,nAtoms);}
	AtomSpecies(const Structure& struc){resize(struc);}
	~AtomSpecies(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const AtomSpecies& as);
	
	//==== access - number ====
	int nSpecies()const{return nSpecies_;}
	int nTot()const;
	
	//==== access - species ====
	std::vector<std::string>& species(){return species_;}
	const std::vector<std::string>& species()const{return species_;}
	std::string& species(int i){return species_[i];}
	const std::string& species(int i)const{return species_[i];}
	
	//==== access - numbers ====
	std::vector<int>& nAtoms(){return nAtoms_;}
	const std::vector<int>& nAtoms()const{return nAtoms_;}
	int& nAtoms(int i){return nAtoms_[i];}
	const int& nAtoms(int i)const{return nAtoms_[i];}
	
	//==== access - offsets ====
	std::vector<int>& offsets(){return offsets_;}
	const std::vector<int>& offsets()const{return offsets_;}
	int& offsets(int i){return offsets_[i];}
	const int& offsets(int i)const{return offsets_[i];}
	
	//==== static functions ====
	static int index_species(const std::string& str, const std::vector<std::string>& names);
	static int index_species(const char* str, const std::vector<std::string>& names);
	static std::vector<int>& read_atoms(const AtomSpecies& as, const char* str, std::vector<int>& ids);
	static int read_natoms(const char* str);
	static std::vector<int>& read_indices(const char* str, std::vector<int>& indices);
	static std::vector<std::string>& read_names(const char* str, std::vector<std::string>& names);
	static void set_species(const AtomSpecies& as, Structure& struc);
	
	//==== member functions ====
	void clear(){defaults();}
	void defaults();
	void resize(const std::vector<std::string>& names, const std::vector<int>& nAtoms);
	void resize(const Structure& struc);
	int index_species(const std::string& str)const{return index_species(str,species_);}
	int index_species(const char* str)const{return index_species(str,species_);}
	int index(int si, int ai)const{return offsets_[si]+ai;}
	
};

//**********************************************************************************************
// serialization
//**********************************************************************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const AtomData& obj);
	template <> int nbytes(const Structure& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const AtomData& obj, char* arr);
	template <> int pack(const Structure& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(AtomData& obj, const char* arr);
	template <> int unpack(Structure& obj, const char* arr);
	
}

#endif
