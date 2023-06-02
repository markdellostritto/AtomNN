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
#include "src/torch/pot_spin_ex.hpp"
// optimization
#include "src/opt/optimize.hpp"

#define EIGEN_NO_DEBUG

#ifndef NNPTES_PRINT_FUNC
#define NNPTES_PRINT_FUNC 0
#endif

#ifndef NNPTES_PRINT_STATUS
#define NNPTES_PRINT_STATUS 0
#endif

#ifndef NNPTES_PRINT_DATA
#define NNPTES_PRINT_DATA 0
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
// NNPTES - Neural Network Potential - Optimization
//************************************************************

class NNPTES{
public:
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
		PotSpinEx pot_;//spin interaction potential
		std::vector<Eigen::MatrixXd> Jt_;//spin interaction matrix - training
		std::vector<Eigen::MatrixXd> Jv_;//spin interaction matrix - training
	//nnp
		int nElements_;//number of unique atomic species
		std::vector<Eigen::VectorXd> pElement_; //parameters - subset for each element (nElements)
		std::vector<Eigen::VectorXd> gElement_; //gradients - subset for each element (nElements)
		NNP nnp_;//the neural network potential
		std::vector<NN::Cost> cost_;//gradient of the cost functions
	//optimization
		std::vector<thread::Dist> dist_atomt;
		std::vector<thread::Dist> dist_atomv;
		std::mt19937 rngen_;//random number generator
		Batch batch_; //batch
		Opt::Loss loss_;//loss function
		Opt::Data data_;//optimization - data
		std::shared_ptr<Opt::Model> model_;//optimization - model
		Eigen::VectorXd identity_;//identity vector
	//error
		double error_scale_; //global error scaling
		double error_train_; //error - training
		double error_val_;   //error - validation
	//constructors/destructors
		NNPTES();
		~NNPTES(){}
	//operators
		friend std::ostream& operator<<(std::ostream& out, const NNPTES& nnptes);
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
	static void read(const char* file, NNPTES& nnptes);
	static void read(FILE* reader, NNPTES& nnptes);
	
	//==== access ====
	//elements
		int nElements()const{return nElements_;}
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
	//pot
		PotSpinEx& pot(){return pot_;}
		const PotSpinEx& pot()const{return pot_;}
	//nnp
		NNP& nnp(){return nnp_;}
		const NNP& nnp()const{return nnp_;}
	//optimization
		std::mt19937& rngen(){return rngen_;}
		const std::mt19937& rngen()const{return rngen_;}
		const Batch& batch()const{return batch_;}
		Opt::Loss& loss(){return loss_;}
		const Opt::Loss& loss()const{return loss_;}
		const Opt::Data& data()const{return data_;}
		std::shared_ptr<Opt::Model>& model(){return model_;}
		const std::shared_ptr<Opt::Model>& model()const{return model_;}
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
	
	template <> int nbytes(const NNPTES& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const NNPTES& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(NNPTES& obj, const char* arr);
	
}
