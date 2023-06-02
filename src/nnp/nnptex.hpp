// c++ libraries
#include <memory>
// structure
#include "src/struc/structure_fwd.hpp"
// ml
#include "src/ml/nn.hpp"
#include "src/ml/batch.hpp"
// nnp
#include "src/nnp/nnp.hpp"
// torch
#include "src/torch/qtpie.hpp"
// optimization
#include "src/opt/objective.hpp"
#include "src/opt/algo.hpp"
#include "src/opt/decay.hpp"

#define EIGEN_NO_DEBUG

#ifndef NNPTEX_PRINT_FUNC
#define NNPTEX_PRINT_FUNC 0
#endif

#ifndef NNPTEX_PRINT_STATUS
#define NNPTEX_PRINT_STATUS 0
#endif

#ifndef NNPTEX_PRINT_DATA
#define NNPTEX_PRINT_DATA 0
#endif

//************************************************************
// Mode
//************************************************************

class Mode{
public:
	enum Type{
		TRAIN,
		TEST,
		SYMM,
		UNKNOWN
	};
	//constructor
	Mode():t_(Type::UNKNOWN){}
	Mode(Type t):t_(t){}
	//operators
	operator Type()const{return t_;}
	//member functions
	static Mode read(const char* str);
	static const char* name(const Mode& mode);
private:
	Type t_;
	//prevent automatic conversion for other built-in types
	//template<typename T> operator T() const;
};
std::ostream& operator<<(std::ostream& out, const Mode& mode);

//************************************************************
// NNPTEX - Neural Network Potential - Optimization
//************************************************************

class NNPTEX{
public:
	//nnp
		int nTypes_;//number of unique atomic species
		std::vector<Eigen::VectorXd> pElement_; //parameters - subset for each element (nElements)
		std::vector<Eigen::VectorXd> gElement_; //gradients - subset for each element (nElements)
		std::vector<Eigen::VectorXd> grad_;     //gradients - local to each processor
		NNP nnp_;//the neural network potential
		std::vector<NN::Cost> cost_;//gradient of the cost functions
	//input/output
		std::string file_params_;   //file - stores parameters
		std::string file_error_;   //file - stores error
		std::string file_ann_;     //file - stores ann
		std::string file_restart_; //file - stores restart info
	//flags
		bool restart_; //flag - whether to restart
		bool preCond_; //flag - whether to pre-condition the inputs
		bool force_;   //flag - whether to compute the force
		bool symm_;    //flag - whether to compute the symmetry functions
		bool norm_;    //flag - whether normalize the energies
		bool wparams_; //flag - whether to write the parameters to file
	//charge
		QTPIE qtpie_;
		std::vector<Eigen::VectorXd> bt_;//solution vector - training
		std::vector<Eigen::VectorXd> xt_;//variable vector - training
		std::vector<Eigen::VectorXd> st_;//sum over row of overlap matrix
		std::vector<Eigen::MatrixXd> St_;//overlap matrices
		std::vector<Eigen::MatrixXd> AIt_;//inverse of qtpie matrix - training
		std::vector<Eigen::VectorXd> bv_;//solution vector - training
		std::vector<Eigen::VectorXd> xv_;//variable vector - training
		std::vector<Eigen::VectorXd> sv_;//sum over row of overlap matrix
		std::vector<Eigen::MatrixXd> Sv_;//overlap matrices
		std::vector<Eigen::MatrixXd> AIv_;//inverse of qtpie matrix - training
	//optimization
		std::vector<thread::Dist> dist_atomt;
		std::vector<thread::Dist> dist_atomv;
		std::mt19937 rngen_;//random number generator
		Batch batch_; //batch
		opt::Objective obj_;//objective
		std::shared_ptr<opt::algo::Base> algo_; //optimization algorithm
		std::shared_ptr<opt::decay::Base> decay_;//step decay
		Eigen::VectorXd identity_;//identity vector
	//error
		double error_scale_; //global error scaling
		double error_train_; //error - training
		double error_val_;   //error - validation
	//constructors/destructors
		NNPTEX();
		~NNPTEX(){}
	//operators
		friend std::ostream& operator<<(std::ostream& out, const NNPTEX& nnPotOpt);
public:
	//==== utility ====
	void defaults();//default variable values
	void clear();//reset object
	
	//==== reading/writing restart ====
	void write_restart(const char* file);//write restart file
	void read_restart(const char* file);//read restart file
	
	//==== error ====
	void train(int batchSize, std::vector<Structure>& struc_train, std::vector<Structure>& struc_val);//train potential
	double error(const Eigen::VectorXd& x, std::vector<Structure>& struc_train, std::vector<Structure>& struc_val);//compute error for a potential
	
	//==== static functions ====
	static void read(const char* file, NNPTEX& nnpte);
	static void read(FILE* reader, NNPTEX& nnpte);
	
	//==== access ====
	//elements
		int nTypes()const{return nTypes_;}
	//files
		std::string& file_params(){return file_params_;}
		const std::string& file_params()const{return file_params_;}
		std::string& file_error(){return file_error_;}
		const std::string& file_error()const{return file_error_;}
		std::string& file_ann(){return file_ann_;}
		const std::string& file_ann()const{return file_ann_;}
		std::string& file_restart(){return file_restart_;}
		const std::string& file_restart()const{return file_restart_;}
	//flags
		bool& restart(){return restart_;}
		const bool& restart()const{return restart_;}
		bool& preCond(){return preCond_;}
		const bool& preCond()const{return preCond_;}
		bool& force(){return force_;}
		const bool& force()const{return force_;}
		bool& symm(){return symm_;}
		const bool& symm()const{return symm_;}
		bool& norm(){return norm_;}
		const bool& norm()const{return norm_;}
		bool& wparams(){return wparams_;}
		const bool& wparams()const{return wparams_;}
	//charge
		QTPIE& qtpie(){return qtpie_;}
		const QTPIE& qtpie()const{return qtpie_;}
	//nnp
		NNP& nnp(){return nnp_;}
		const NNP& nnp()const{return nnp_;}
	//optimization
		std::mt19937& rngen(){return rngen_;}
		const std::mt19937& rngen()const{return rngen_;}
		const Batch& batch()const{return batch_;}
		opt::Objective& obj(){return obj_;}
		const opt::Objective& obj()const{return obj_;}
		std::shared_ptr<opt::algo::Base>& algo(){return algo_;}
		const std::shared_ptr<opt::algo::Base>& algo()const{return algo_;}
		std::shared_ptr<opt::decay::Base>& decay(){return decay_;}
		const std::shared_ptr<opt::decay::Base>& decay()const{return decay_;}
	//error
		double& error_scale(){return error_scale_;}
		const double& error_scale()const{return error_scale_;}
		double& error_train(){return error_train_;}
		const double& error_train()const{return error_train_;}
		double& error_val(){return error_val_;}
		const double& error_val()const{return error_val_;}
	
};

namespace serialize{
		
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const NNPTEX& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const NNPTEX& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(NNPTEX& obj, const char* arr);
	
}
