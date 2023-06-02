// c++ libraries
#include <memory>
// structure
#include "src/struc/structure_fwd.hpp"
// ml
#include "src/ml/nn.hpp"
#include "src/ml/batch.hpp"
// nnp
#include "src/nnp/nnp.hpp"
// optimization
#include "src/opt/objective.hpp"
#include "src/opt/algo.hpp"
#include "src/opt/decay.hpp"

#define EIGEN_NO_DEBUG

#ifndef NNPTEF_PRINT_FUNC
#define NNPTEF_PRINT_FUNC 0
#endif

#ifndef NNPTEF_PRINT_STATUS
#define NNPTEF_PRINT_STATUS 0
#endif

#ifndef NNPTEF_PRINT_DATA
#define NNPTEF_PRINT_DATA 0
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
// NNPTEF - Neural Network Potential - Optimization
//************************************************************

class NNPTEF{
public:
	//nnp
		int nTypes_;//number of unique atomic species
		std::vector<Eigen::VectorXd> pElement_; //parameters - subset for each element (nElements)
		std::vector<Eigen::VectorXd> gElement_; //gradients - subset for each element (nElements)
		std::vector<Eigen::VectorXd> grad_;     //gradients - local to each processor
		std::vector<Eigen::VectorXd> gradIJ_;     //gradients - local to each processor
		std::vector<Eigen::VectorXd> gradIK_;     //gradients - local to each processor
		std::vector<Eigen::VectorXd> gradJK_;     //gradients - local to each processor
		NNP nnp_;//the neural network potential
		std::vector<NN::Cost> cost_;
		std::vector<NN::DODP> dOdP_;
		std::vector<NN::D2ODZDI> d2OdZdI_;
	//input/output
		std::string file_params_;   //file - stores parameters
		std::string file_error_;   //file - stores error
		std::string file_ann_;     //file - stores ann
		std::string file_restart_; //file - stores restart info
	//flags
		bool restart_; //flag - whether to restart
		bool preCond_; //flag - whether to pre-condition the inputs
		bool wparams_; //flag - whether to write the parameters to file
	//optimization
		double beta_;
		double zeta_;
		std::vector<thread::Dist> dist_atomt;
		std::vector<thread::Dist> dist_atomv;
		std::mt19937 rngen_;//random number generator
		std::uniform_real_distribution<double> uniform_;
		Batch batch_; //batch
		opt::Objective obj_;//objective
		std::shared_ptr<opt::algo::Base> algo_; //optimization algorithm
		std::shared_ptr<opt::decay::Base> decay_;//step decay
		Eigen::VectorXd identity_;//identity vector
	//structures
		std::vector<Structure> struc_train_ref_;
		std::vector<Structure> struc_train_nnp_;
		std::vector<Structure> struc_val_ref_;
		std::vector<Structure> struc_val_nnp_;
	//error
		double error_[4];
	//constructors/destructors
		NNPTEF();
		~NNPTEF(){}
	//operators
		friend std::ostream& operator<<(std::ostream& out, const NNPTEF& nnPotOpt);
public:
	//==== utility ====
	void defaults();//default variable values
	void clear();//reset object
	
	//==== reading/writing restart ====
	void write_restart(const char* file);//write restart file
	void read_restart(const char* file);//read restart file
	
	//==== error ====
	void train(int batchSize);//train potential
	void error(const Eigen::VectorXd& x);//compute error for a potential
	void error_energy(const Eigen::VectorXd& x);//compute error for a potential
	void error_force(const Eigen::VectorXd& x);//compute error for a potential
	
	//==== static functions ====
	static void read(const char* file, NNPTEF& nnpte);
	static void read(FILE* reader, NNPTEF& nnpte);
	
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
		bool& wparams(){return wparams_;}
		const bool& wparams()const{return wparams_;}
	//nnp
		NNP& nnp(){return nnp_;}
		const NNP& nnp()const{return nnp_;}
	//optimization
		double& beta(){return beta_;}
		const double& beta()const{return beta_;}
		double& zeta(){return zeta_;}
		const double& zeta()const{return zeta_;}
		std::mt19937& rngen(){return rngen_;}
		const std::mt19937& rngen()const{return rngen_;}
		const Batch& batch()const{return batch_;}
		opt::Objective& obj(){return obj_;}
		const opt::Objective& obj()const{return obj_;}
		std::shared_ptr<opt::algo::Base>& algo(){return algo_;}
		const std::shared_ptr<opt::algo::Base>& algo()const{return algo_;}
		std::shared_ptr<opt::decay::Base>& decay(){return decay_;}
		const std::shared_ptr<opt::decay::Base>& decay()const{return decay_;}
	//structures
		std::vector<NeighborList> nlist_train_;
		std::vector<NeighborList> nlist_val_;
		std::vector<Structure>& struc_train_ref(){return struc_train_ref_;}
		const std::vector<Structure>& struc_train_ref()const{return struc_train_ref_;}
		std::vector<Structure>& struc_val_ref(){return struc_val_ref_;}
		const std::vector<Structure>& struc_val_ref()const{return struc_val_ref_;}
};

namespace serialize{
		
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const NNPTEF& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const NNPTEF& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(NNPTEF& obj, const char* arr);
	
}
