// c++ libraries
#include <memory>
// ann - structure
#include "src/struc/structure_fwd.hpp"
// ann - ml
#include "src/ml/nn.hpp"
#include "src/ml/batch.hpp"
// ann - nnp
#include "src/nnp/nnp.hpp"
// ann - optimization
#include "src/opt/optimize.hpp"

#define EIGEN_NO_DEBUG

#ifndef NNP_TRAIN_PRINT_FUNC
#define NNP_TRAIN_PRINT_FUNC 0
#endif

#ifndef NNP_TRAIN_PRINT_STATUS
#define NNP_TRAIN_PRINT_STATUS 0
#endif

#ifndef NNP_TRAIN_PRINT_DATA
#define NNP_TRAIN_PRINT_DATA 0
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
// NNPTrain - Neural Network Potential - Optimization
//************************************************************

class NNPTrain{
public:
	//elements
		int nElements_;//number of unique atomic species
		std::vector<Eigen::VectorXd> pElement_; //parameters - subset for each element (nElements)
		std::vector<Eigen::VectorXd> gElement_; //gradients - subset for each element (nElements)
		std::vector<Eigen::VectorXd> gLocal_;   //gradients - local to each processor
		std::vector<Eigen::VectorXd> gTotal_;   //gradients - total for each structure
	//input/output
		std::string file_error_;   //file - stores error
		std::string file_ann_;     //file - stores ann
		std::string file_restart_; //file - stores restart info
	//flags
		bool restart_; //flag - whether to restart
		bool preCond_; //flag - whether to pre-condition the inputs
		bool force_;   //flag - whether to compute the force
		bool symm_;    //flag - whether to compute the symmetry functions
		bool norm_;    //flag - whether normalize the energies
		bool charge_;  //flag - whether the atoms are charged
	//nnp
		int nParams_;//total number of parameters
		NNP nnp_;//the neural network potential
		std::vector<NN::Cost> cost_;//gradient of the cost function
	//optimization
		std::vector<parallel::Dist> dist_atomt;
		std::vector<parallel::Dist> dist_atomv;
		std::mt19937 rngen_;//random number generator
		Batch batch_; //batch
		Opt::Loss loss_;//loss function
		Opt::Data data_;//optimization - data
		std::shared_ptr<Opt::Model> model_;//optimization - model
		Eigen::VectorXd identity_;//identity vector
		double w_,w2_;
	//error
		double error_scale_; //global error scaling
		double error_train_; //error - training
		double error_val_;   //error - validation
	//constructors/destructors
		NNPTrain();
		~NNPTrain(){}
	//operators
		friend std::ostream& operator<<(std::ostream& out, const NNPTrain& nnPotOpt);
public:
	//==== utility ====
	void defaults();//default variable values
	void clear();//reset object
	
	//==== reading/writing restart ====
	void write_restart(const char* file);//write restart file
	void read_restart(const char* file);//read restart file
	
	//==== error ====
	void train(int batchSize, const std::vector<Structure>& struc_train, const std::vector<Structure>& struc_val);//train potential
	double error_energy(const Eigen::VectorXd& x, const std::vector<Structure>& struc_train, const std::vector<Structure>& struc_val);//compute error for a potential
	
	//==== access ====
	//elements
		int nElements()const{return nElements_;}
	//files
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
		bool& charge(){return charge_;}
		const bool& charge()const{return charge_;}
	//nnp
		int nParams()const{return nParams_;}
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
		double& w(){return w_;}
		const double& w()const{return w_;}
		double& w2(){return w2_;}
		const double& w2()const{return w2_;}
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
	
	template <> int nbytes(const NNPTrain& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const NNPTrain& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(NNPTrain& obj, const char* arr);
	
}
