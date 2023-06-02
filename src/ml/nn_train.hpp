#pragma once
#ifndef NN_TRAIN_HPP
#define NN_TRAIN_HPP

// mpi
#include <mpi.h>
// c++ libraries
#include <iosfwd>
#include <random>
// c libraries
#include <memory>
// typedefs
#include "src/util/typedef.hpp"
// ml
#include "src/ml/nn.hpp"
#include "src/ml/batch.hpp"
#include "src/ml/data.hpp"
// opt
#include "src/opt/loss.hpp"
#include "src/opt/stop.hpp"
#include "src/opt/decay.hpp"
#include "src/opt/algo.hpp"
#include "src/opt/objective.hpp"
// ann - random
#include "src/math/random.hpp"

#ifndef NN_TRAIN_PRINT_FUNC
#define NN_TRAIN_PRINT_FUNC 0
#endif

#ifndef NN_TRAIN_PRINT_STATUS
#define NN_TRAIN_PRINT_STATUS 0
#endif

#ifndef NN_TRAIN_PRINT_DATA
#define NN_TRAIN_PRINT_DATA 0
#endif

//***********************************************************************
// NN Optimization
//***********************************************************************

class NNOpt{
private:
	//neural network
		NN::Cost cost_;//cost function
		NN::ANN nn_;//neural network
	//optimization
		int seed_;
		std::mt19937 rngen_;
		opt::Objective obj_; //optimization objective
		std::shared_ptr<opt::algo::Base> algo_; //optimization algorithm
		std::shared_ptr<opt::decay::Base> decay_;//step decay
		Batch batch_; //batch
		double err_train_; //error - training
		double err_val_; //error - validation
	//conditioning
		bool preCond_;  //whether to pre-condition  the inputs
		bool postCond_; //whether to post-condition the inputs
	//file i/o
		bool restart_;//whether restarting
		std::string file_error_;//file storing the error
		std::string file_restart_;//restart file
public:
	//==== constructors/destructors ====
	NNOpt(){defaults();}
	~NNOpt(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const NNOpt& nnopt);
	
	//==== access ====
	//neural network
		NN::ANN& nn(){return nn_;}
		const NN::ANN& nn()const{return nn_;}
	//optimization
		int& seed(){return seed_;}
		const int& seed()const{return seed_;}
		opt::Objective& obj(){return obj_;}
		const opt::Objective& obj()const{return obj_;}
		std::shared_ptr<opt::algo::Base>& algo(){return algo_;}
		const std::shared_ptr<opt::algo::Base>& algo()const{return algo_;}
		std::shared_ptr<opt::decay::Base>& decay(){return decay_;}
		const std::shared_ptr<opt::decay::Base>& decay()const{return decay_;}
		Batch& batch(){return batch_;}
		const Batch& batch()const{return batch_;}
		double err_train()const{return err_train_;}
		double err_val()const{return err_val_;}
	//conditioning
		bool& preCond(){return preCond_;}
		const bool& preCond()const{return preCond_;}
		bool& postCond(){return postCond_;}
		const bool& postCond()const{return postCond_;}
	//file i/o
		bool& restart(){return restart_;}
		const bool& restart()const{return restart_;}
		std::string& file_error(){return file_error_;}
		const std::string& file_error()const{return file_error_;}
		std::string& file_restart(){return file_restart_;}
		const std::string& file_restart()const{return file_restart_;}
	
	//==== member functions ====
	//training
	void train(int nbatchl, const MLData& data_train, const MLData& data_val);
	//error
	double error(const MLData& data_train, const MLData& data_val);
	//misc
	void defaults();
	void clear();
};

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************

	template <> int nbytes(const NNOpt& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const NNOpt& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(NNOpt& obj, const char* arr);
	
}

#endif