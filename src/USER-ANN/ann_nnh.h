#pragma once
#ifndef ANN_NN_POT_HPP
#define ANN_NN_POT_HPP

// c++ libraries
#include <iosfwd>
// ann - lower triangular matrix
#include "ann_lmat.h"
// neural networks
#include "ann_nn.h"
// map
#include "ann_map.h"
// basis functions
#include "ann_basis_radial.h"
#include "ann_basis_angular.h"
// atom
#include "ann_atom.h"

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
	AtomANN atom_ - the central atom of the NNH
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
	AtomANN atom_;//atom - name, mass, energy, charge
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
		AtomANN& atom(){return atom_;}
		const AtomANN& atom()const{return atom_;}
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

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const NNH& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const NNH& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(NNH& obj, const char* arr);
	
}

#endif
