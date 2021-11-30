#pragma once
#ifndef NNP_HPP
#define NNP_HPP

// c++ libraries
#include <iosfwd>
// ann - math
#include "src/math/lmat.hpp"
// ann - structure
#include "src/struc/structure_fwd.hpp"
#include "src/struc/neighbor.hpp"
// ann - ml
#include "src/ml/nn.hpp"
// ann - mem
#include "src/mem/map.hpp"
// ann - string
#include "src/str/string.hpp"
// basis functions
#include "src/nnp/basis_radial.hpp"
#include "src/nnp/basis_angular.hpp"
// atom
#include "src/nnp/atom.hpp"

//***********************************************************************
// COMPILER DIRECTIVES
//***********************************************************************

#ifndef NNP_PRINT_FUNC
#define NNP_PRINT_FUNC 0
#endif

#ifndef NNP_PRINT_STATUS
#define NNP_PRINT_STATUS 0
#endif

#ifndef NNP_PRINT_DATA
#define NNP_PRINT_DATA 0
#endif

//************************************************************
// TERMINOLOGY:
// symmetry function - measure of local symmetry around a given atom
// basis - group of symmetry functions associated with a given interaction
// Each species has its own basis for each pair interaction and each
// triple interaction (i.e. H has (H)-H, (H)-O, (H)-O-O, (H)-H-H, (H)-H-O/(H)-O-H)
// Since the atomic neural networks are different, the basis for pair/triple
// interactions on different species of center atoms can be the same, reducing
// the number of necessary parameters.  Note however, that they can be different.
//************************************************************

//************************************************************
// NEURAL NETWORK HAMILTONIAN (NNH)
//************************************************************

/*
PRIVATE:
	int nspecies_ - the total number of species in a neural network potential
	Atom atom_ - the central atom of the NNH
	NN::ANN nn_ - the neural network which determines the energy of the central atom
	basisR_ - the radial basis functions
		there is a radial basis for each species in the simulation (nspecies)
		each basis for each species contains a set of radial symmetry functions
		each symmetry function then corresponds to a unique input to the neural network
	basisA_ - the angular basis functions
		there is an angular basis for each unique pair of species in the simulation (nspecies x (nspecies-1)/2)
		each basis for each pair contains a set of angular symmetry functions
		each symmetry function then corresponds to a unique input to the neural network
	nInput_ - the total number of inputs to the network
		defined as the total number of radial and angular symmetry functions in each basis
		the inputs are arranged with the radial inputs preceding the angular inputs
	nInputR_ - the total number of radial inputs to the network
		defined as the total number of radial symmetry functions in each basis
	nInputA_ - the total number of angular inputs to the network
		defined as the total number of angular symmetry functions in each basis
	offsetR_ - the offset of each radial input
		all symmetry functions must be serialized into a single vector - the input to the neural network
		each radial symmetry function thus has an offset defined as the total number of symmetry functions
		in all bases preceding the current basis
	offsetA_ - the offset of each angular input
		all symmetry functions must be serialized into a single vector - the input to the neural network
		each angular symmetry function thus has an offset defined as the total number of symmetry functions
		in all basis pairs preceding the current basis pair
		note that the offset is from the beginning of the angular section of the inputs
		thus, ths offset from the beginning of the input vector is nInputR_ + offsetA_(i,j)
NOTES:
	This class is not meant to be used independently, only as a part of the class NNP.
	This class alone does not have enough data to define a neural network potential.
	Rather, this class accumulates all the data associated with the inputs and nueral network for a given atom.
	Thus, if one has a valid symmetry function defining the the local symmetry around a given atom of the
	correct species, this class can be used to compute the energy.
	However, the class NNP is required to define all species, define all neural network potentials, and to
	compute the symmetry functions, total energies, and forces for a given atomic configuration.
*/
class NNH{
private:
	//network configuration
	int nInput_;//number of radial + angular symmetry functions
	int nInputR_;//number of radial symmetry functions
	int nInputA_;//number of angular symmetry functions
	
	//hamiltonian
	int nspecies_;//the total number of species
	Atom atom_;//atom - name, mass, energy, charge
	NN::ANN nn_;//neural network hamiltonian
	NN::DOutDVal dOutDVal_;//gradient of the output w.r.t. node values
	
	//basis for pair/triple interactions
	std::vector<BasisR> basisR_;//radial basis functions (nspecies_)
	std::vector<int> offsetR_;//offset for the given radial basis (nspecies_)
	LMat<BasisA> basisA_;//angular basis functions (nspecies x (nspecies+1)/2)
	LMat<int> offsetA_;//offset for the given radial basis (nspecies x (nspecies+1)/2)
public:
	//==== constructors/destructors ====
	NNH(){defaults();}
	~NNH(){}
	
	//operators
	friend std::ostream& operator<<(std::ostream& out, const NNH& nmh);
	
	//==== access ====
	//hamiltonian
		const int& nspecies()const{return nspecies_;}
		Atom& atom(){return atom_;}
		const Atom& atom()const{return atom_;}
		NN::ANN& nn(){return nn_;}
		const NN::ANN& nn()const{return nn_;}
		NN::DOutDVal& dOutDVal(){return dOutDVal_;}
		const NN::DOutDVal& dOutDVal()const{return dOutDVal_;}
	//basis for pair/triple interactions
		BasisR& basisR(int i){return basisR_[i];}
		const BasisR& basisR(int i)const{return basisR_[i];}
		BasisA& basisA(int i, int j){return basisA_(i,j);}
		const BasisA& basisA(int i, int j)const{return basisA_(i,j);}
	//network configuration
		const int& nInput()const{return nInput_;}
		const int& nInputR()const{return nInputR_;}
		const int& nInputA()const{return nInputA_;}
		const int& offsetR(int i)const{return offsetR_[i];}
		const int& offsetA(int i, int j)const{return offsetA_(i,j);}
	
	//==== member functions ====
	//misc
		void defaults();//set defaults
		void clear(){defaults();}//clear the potential
	//resizing
		void resize(int nspecies);//resize
		void init_input();//initialize the inputs
	//output
		double energy(const Eigen::VectorXd& symm);//compute energy of atom
};

//************************************************************
// NEURAL NETWORK POTENTIAL (NNP)
//************************************************************

/*
PRIVATE:
	double rc_ - global cutoff
		this cutoff is used for determining which atoms will be included in the calculations of the symmetry functions and forces
		different cutoffs can then be used for different symmetry functions
	int nspecies_ - the total number of atomic species
	Map<int,int> map_ - map assigning atom ids to the index of a given atom in the NNP
		note that the atom id is a unique integer generated from the atom name
		the index is the position of the atom in the list of NNHs (nnh_)
		thus, this maps assigns the correct index in "nnh_" to each atom id, and thus each atom name
*/
class NNP{
private:
	double rc_;//global cutoff
	int nspecies_;//number of types of atoms
	Map<int,int> map_;//map atom ids to nnpot index
	std::vector<NNH> nnh_;//the hamiltonians for each species
public:
	//==== constructors/destructors ====
	NNP(){defaults();}
	~NNP(){}
	
	//==== access ====
	//species
		int nspecies()const{return nspecies_;}
		int index(const char* name)const{return map_[string::hash(name)];}
		int index(const std::string& name)const{return map_[string::hash(name)];}
		Map<int,int>& map(){return map_;}
		const Map<int,int>& map()const{return map_;}
		NNH& nnh(int i){return nnh_[i];}
		const NNH& nnh(int i)const{return nnh_[i];}
	//global cutoff
		double& rc(){return rc_;}
		const double& rc()const{return rc_;}
		
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const NNP& nnpot);
	friend FILE* operator<<(FILE* out, const NNP& nnpot);
	
	//==== member functions ====
	//auxilliary
		void defaults();//set defaults
		void clear(){defaults();};//clear the potential
	//resizing
		void resize(const std::vector<Atom>& species);
		
	//==== static functions ====
	//read/write basis
		static void read_basis(const char* file, NNP& nnpot, const char* atomName);//read basis for atomName
		static void read_basis(FILE* reader, NNP& nnpot, const char* atomName);//read basis for atomName
	//read/write nnp
		static void write(const char* file, const NNP& nnpot);//write NNP to "file"
		static void read(const char* file, NNP& nnpot);//read NNP from "file"
		static void write(FILE* writer, const NNP& nnpot);//write NNP to "writer"
		static void read(FILE* reader, NNP& nnpot);//read NNP from "reader"
	//calculation
		static void init(const NNP& nnp, Structure& struc);//assign vector of all species in the simulations
		static void symm(NNP& nnp, Structure& struc, const NeighborList& nlist);//compute symmetry functions
		static double energy(NNP& nnp, Structure& struc);//compute energy
		static void force(NNP& nnp, Structure& struc, const NeighborList& nlist);//compute forces
		static void charge(NNP& nnp, Structure& struc);//compute charge
};

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const NNH& obj);
	template <> int nbytes(const NNP& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const NNH& obj, char* arr);
	template <> int pack(const NNP& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(NNH& obj, const char* arr);
	template <> int unpack(NNP& obj, const char* arr);
	
}

#endif
