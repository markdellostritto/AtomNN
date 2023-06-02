#pragma once
#ifndef NNH_HPP
#define NNH_HPP

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
#include "src/nnp/type.hpp"

//***********************************************************************
// COMPILER DIRECTIVES
//***********************************************************************

#ifndef NNH_PRINT_FUNC
#define NNH_PRINT_FUNC 0
#endif

#ifndef NNH_PRINT_STATUS
#define NNH_PRINT_STATUS 0
#endif

#ifndef NNH_PRINT_DATA
#define NNH_PRINT_DATA 0
#endif

//************************************************************
// NEURAL NETWORK HAMILTONIAN (NNH)
//************************************************************

/**
* Class defining a Neural Network Hamiltonian
* This class contains all the data and methods necessary to define a Hamiltonian yielding the
* potential energy surface for a specific atom type given a set of inputs to the neural network, 
* the output of which yields the atomic energy for each set of inputs.
*/
class NNH{
private:
	//network configuration
	int nInput_;//number of radial + angular symmetry functions
	int nInputR_;//number of radial symmetry functions
	int nInputA_;//number of angular symmetry functions
	
	//hamiltonian
	int ntypes_;//the total number of types
	Type type_;//type associated with NNH
	NN::ANN nn_;//neural network hamiltonian
	NN::DODZ dOdZ_;//gradient of the output w.r.t. node values
	
	//basis for pair/triple interactions
	std::vector<BasisR> basisR_;//radial basis functions (ntypes_)
	std::vector<int> offsetR_;//offset for the given radial basis (ntypes_)
	LMat<BasisA> basisA_;//angular basis functions (ntypes x (ntypes+1)/2)
	LMat<int> offsetA_;//offset for the given radial basis (ntypes x (ntypes+1)/2)
public:
	//==== constructors/destructors ====
	NNH(){defaults();}
	~NNH(){}
	
	//operators
	friend std::ostream& operator<<(std::ostream& out, const NNH& nmh);
	
	//==== access ====
	//hamiltonian
		const int& ntypes()const{return ntypes_;}
		Type& type(){return type_;}
		const Type& type()const{return type_;}
		NN::ANN& nn(){return nn_;}
		const NN::ANN& nn()const{return nn_;}
		NN::DODZ& dOdZ(){return dOdZ_;}
		const NN::DODZ& dOdZ()const{return dOdZ_;}
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
		void resize(int ntypes);//resize
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