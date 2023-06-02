#pragma once
#ifndef NN_HPP
#define NN_HPP

#define EIGEN_NO_DEBUG

// c++ libraries
#include <iosfwd>
// eigen
#include <Eigen/Dense>
// ann - math
#include "ann_random.h"
// ann - mem
#include "ann_serialize.h"
// ann - typedef
#include "ann_typedef.h"

namespace NN{
	
//***********************************************************************
// COMPILER DIRECTIVES
//***********************************************************************

#ifndef NN_PRINT_FUNC
#define NN_PRINT_FUNC 0
#endif

#ifndef NN_PRINT_STATUS
#define NN_PRINT_STATUS 0
#endif

#ifndef NN_PRINT_DATA
#define NN_PRINT_DATA 0
#endif

//***********************************************************************
// FORWARD DECLARATIONS
//***********************************************************************

class ANN;
class ANNInit;
class Cost;
class DOutDVal;
class DOutDP;
class D2OutDPDVal;

//***********************************************************************
// INITIALIZATION METHOD
//***********************************************************************

class Init{
public:
	//enum
	enum Type{
		UNKNOWN=0,
		RAND=1,
		XAVIER=2,
		HE=3,
		MEAN=4
	};
	//constructor
	Init():t_(Type::UNKNOWN){}
	Init(Type t):t_(t){}
	//operators
	operator Type()const{return t_;}
	//member functions
	static Init read(const char* str);
	static const char* name(const Init& init);
private:
	Type t_;
	//prevent automatic conversion for other built-in types
	//template<typename T> operator T() const;
};
std::ostream& operator<<(std::ostream& out, const Init& init);

//***********************************************************************
// TRANSFER FUNCTIONS 
//***********************************************************************

class Transfer{
public:
	//type
	enum Type{
		UNKNOWN=0,
		LINEAR=1,
		SIGMOID=2,
		TANH=3,
		ISRU=4,
		ARCTAN=5,
		SOFTSIGN=6,
		RELU=7,
		SOFTPLUS=8,
		ELU=9,
		GELU=10,
		SWISH=11,
		MISH=12,
		TANHRE=13
	};
	//constructor
	Transfer():t_(Type::UNKNOWN){}
	Transfer(Type t):t_(t){}
	//operators
	operator Type()const{return t_;}
	//member functions
	static Transfer read(const char* str);
	static const char* name(const Transfer& tf);
	//function
	static void tf_lin(VecXd& f, VecXd& d);
	static void tf_sigmoid(VecXd& f, VecXd& d);
	static void tf_tanh(VecXd& f, VecXd& d);
	static void tf_isru(VecXd& f, VecXd& d);
	static void tf_arctan(VecXd& f, VecXd& d);
	static void tf_softsign(VecXd& f, VecXd& d);
	static void tf_relu(VecXd& f, VecXd& d);
	static void tf_softplus(VecXd& f, VecXd& d);
	static void tf_elu(VecXd& f, VecXd& d);
	static void tf_gelu(VecXd& f, VecXd& d);
	static void tf_swish(VecXd& f, VecXd& d);
	static void tf_mish(VecXd& f, VecXd& d);
	static void tf_tanhre(VecXd& f, VecXd& d);
private:
	Type t_;
	//prevent automatic conversion for other built-in types
	//template<typename T> operator T() const;
};
std::ostream& operator<<(std::ostream& out, const Transfer& tf);

//***********************************************************************
// ANN
//***********************************************************************

/*
DEFINITIONS:
	ensemble - total set of all data (e.g. training "ensemble")
	element - single datum from ensemble
	c - "c" donotes the cost function, e.g. the gradient of the cost function w.r.t. the value of a node is dc/da
	z - "z" is the input to each node, e.g. the gradient of a node w.r.t. to its input is da/dz
	a - "a" is the value of a node, e.g. the gradient of a node w.r.t. to its input is da/dz
	o - "o" is the output of the network (i.e. out_), e.g. the gradient of the output w.r.t. the input is do/di
	i - "i" is the input of the network (i.e. in_), e.g. the gradient of the output w.r.t. the input is do/di
PRIVATE:
	VecXd in_ - raw input data for a single element of the ensemble (e.g. training set)
	VecXd inw_ - weight used to scale the input data
	VecXd inb_ - bias used to shift the input data
	VecXd out_ - raw output data given a single input element
	VecXd outw_ - weight used to scale the output data
	VecXd outb_ - bias used to shift the output data
	int nlayer_ - 
		total number of hidden layers
		best thought of as the number of "connections" between layers
		nlayer_ must be greater than zero for an initialized network
		this is true even for a network with zero "hidden" layers
		an uninitialized network has nlayer_ = 0
		if we have just the input and output: nlayer_ = 1
			one set of weights,biases connecting input/output
		if we have one hidden layer: nlayer_ = 2
			two sets of weights,biases connecting input/layer0/output
		if we have two hidden layers: nlayer_ = 3
			three sets of weights,biases connecting input/layer0/layer1/output
		et cetera
	std::vector<VecXd> node_ - 
		all nodes, including the input, output, and hidden layers
		the raw input and output (in_,out_) are separate from "node_"
		this is because the raw input/output may be shifted/scaled before being used
		thus, while in_/out_ are the "raw" input/output,
		the front/back of "node_" can be thought of the "scaled" input/output
		note that scaling is not necessary, but made optional with the use of in_/out_
		has a size of "nlayer_+1", as there are "nlayer_" connections between "nlayer_+1" nodes
	std::vector<VecXd> bias_ - 
		the bias of each layer, best thought of as the bias "between" layers n,n+1
		bias_[n] must have the size node_[n+1] - we add this bias when going from node_[n] to node_[n+1]
		has a size of "nlayer_", as there are "nlayer_" connections between "nlayer_+1" nodes
	std::vector<MatXd> edge_ -
		the weights of each layer, best though of as transforming from layers n to n+1
		edge_[n] must have the size (node_[n+1],node_[n]) - matrix multiplying (node_[n]) to get (node_[n+1])
		has a size of "nlayer_", as there are "nlayer_" connections between "nlayer_+1" nodes
	std::vector<VecXd> dadz_ - 
		the gradient of the value of a node (a) w.r.t. the input of the node (z) - da/dz
		practically, the gradient of the transfer function of each layer
		best thought of as the gradient associated with function transferring "between" layers n,n+1
		thus, dadz_[n] must have the size node_[n+1]
		has a size of "nlayer_", as there are "nlayer_" connections between "nlayer_+1" nodes
	tf_ -
		the type of the transfer function
		note the transfer function for the last layer is always linear
	tfp_ - 
		(Transfer Function, Function Derivative, Vector)
		the transfer function for each layer, operates on entire vector at once
		computes both function and derivative simultaneously
*/
class ANN{
private:
	//typedefs
		typedef void (*TFP)(VecXd&,VecXd&);
	//layers
		int nlayer_;//number of layers (weights,biases)
	//transfer functions
		Transfer tf_;//transfer function type
		std::vector<TFP> tfp_;//transfer function - input for indexed layer (nlayer_)
	//input/output
		VecXd in_;//input layer
		VecXd out_;//output layer
		VecXd inw_,inb_;//input weight, bias
		VecXd outw_,outb_;//output weight, bias
	//gradients - nodes
		std::vector<VecXd> dadz_;//node derivative - not including input layer (nlayer_)
	//node weights and biases
		std::vector<VecXd> node_;//nodes (nlayer_+1)
		std::vector<VecXd> bias_;//bias (nlayer_)
		std::vector<MatXd> edge_;//edges (nlayer_)
public:
	//==== constructors/destructors ====
	ANN(){defaults();}
	~ANN(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const ANN& n);
	friend FILE* operator<<(FILE* out, const ANN& n);
	friend VecXd& operator>>(const ANN& nn, VecXd& v);
	friend ANN& operator<<(ANN& nn, const VecXd& v);
	
	//==== access ====
	//network dimensions
		int nlayer()const{return nlayer_;}
	//nodes
		VecXd& in(){return in_;}
		const VecXd& in()const{return in_;}
		VecXd& out(){return out_;}
		const VecXd& out()const{return out_;}
		VecXd& node(int n){return node_[n];}
		const VecXd& node(int n)const{return node_[n];}
		int nNodes(int n)const{return node_[n].size();}
	//scaling
		VecXd& inw(){return inw_;}
		const VecXd& inw()const{return inw_;}
		VecXd& inb(){return inb_;}
		const VecXd& inb()const{return inb_;}
		VecXd& outw(){return outw_;}
		const VecXd& outw()const{return outw_;}
		VecXd& outb(){return outb_;}
		const VecXd& outb()const{return outb_;}
	//bias
		VecXd& bias(int l){return bias_[l];}
		const VecXd& bias(int l)const{return bias_[l];}
	//edge
		MatXd& edge(int l){return edge_[l];}
		const MatXd& edge(int l)const{return edge_[l];}
	//size
		int nIn()const{return in_.size();}
		int nOut()const{return out_.size();}
	//gradients - nodes
		VecXd& dadz(int n){return dadz_[n];}
		const VecXd& dadz(int n)const{return dadz_[n];}
	//transfer functions
		Transfer& tf(){return tf_;}
		const Transfer& tf()const{return tf_;}
		TFP tfp(int l){return tfp_[l];}
		const TFP tfp(int l)const{return tfp_[l];}
		
	//==== member functions ====
	//clearing/initialization
		void defaults();
		void clear();
	//info
		int size()const;
		int nBias()const;
		int nWeight()const;
	//resizing
		void resize(const ANNInit& init, int nInput, int nOutput);
		void resize(const ANNInit& init, int nInput, const std::vector<int>& nNodes, int nOutput);
		void resize(const ANNInit& init, const std::vector<int>& nNodes);
	//error
		double error_lambda()const;
		VecXd& grad_lambda(VecXd& grad)const;
	//execution
		const VecXd& execute();
		const VecXd& execute(const VecXd& in){in_=in;return execute();}
		
	//==== static functions ====
	static void write(FILE* writer, const ANN& nn);
	static void write(const char*, const ANN& nn);
	static void read(FILE* writer, ANN& nn);
	static void read(const char*, ANN& nn);
};

bool operator==(const ANN& n1, const ANN& n2);
inline bool operator!=(const ANN& n1, const ANN& n2){return !(n1==n2);}

//***********************************************************************
// ANNInit
//***********************************************************************

class ANNInit{
private:
	double bInit_;//initial value - bias
	double wInit_;//initial value - weight
	double sigma_;//distribution size parameter
	rng::dist::Name dist_;//distribution type
	Init init_;//initialization scheme
	int seed_;//random seed	
public:
	//==== constructors/destructors ====
	ANNInit(){defaults();}
	~ANNInit(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const ANNInit& init);
	
	//==== access ====
	double& bInit(){return bInit_;}
	const double& bInit()const{return bInit_;}
	double& wInit(){return wInit_;}
	const double& wInit()const{return wInit_;}
	double& sigma(){return sigma_;}
	const double& sigma()const{return sigma_;}
	rng::dist::Name& dist(){return dist_;}
	const rng::dist::Name& dist()const{return dist_;}
	Init& init(){return init_;}
	const Init& init()const{return init_;}
	int& seed(){return seed_;}
	const int& seed()const{return seed_;}
	
	//==== member functions ====
	void defaults();
	void clear(){defaults();}
};

//***********************************************************************
// Cost
//***********************************************************************

/*
dcdz_ - 
	the gradient of the cost function (c) w.r.t. the node inputs (z) - dc/dz
*/
class Cost{
private:
	std::vector<VecXd> dcdz_;//derivative of cost function w.r.t. node inputs (nlayer_)
	VecXd grad_;//gradient of the cost function with respect to each parameter (bias + weight)
public:
	//==== constructors/destructors ====
	Cost(){}
	Cost(const ANN& nn){resize(nn);}
	~Cost(){}
	
	//==== access ====
	std::vector<VecXd>& dcdz(){return dcdz_;}
	const std::vector<VecXd>& dcdz()const{return dcdz_;}
	VecXd& grad(){return grad_;}
	const VecXd& grad()const{return grad_;}
	
	//==== member functions ====
	void clear();
	void resize(const ANN& nn);
	const VecXd& grad(const ANN& nn, const VecXd& dcdo);
};

//***********************************************************************
// DOutDVal
//***********************************************************************

/*
doda_ -
	the derivative of the output (o) w.r.t. the value of all nodes (a)
	has a size of "nlayer_+1" as we need to compute the gradient w.r.t. all nodes
	thus, doda_[n] must of the size node_[n]
	this includes the hidden layers as well as the input/ouput layers
	note these are the scaled inputs/outputs
dodi_ -
	the derivative of the output w.r.t. the raw input
	this is the first element of doda_ multiplied by the input scaling
*/
class DOutDVal{
private:
	MatXd dodi_;//derivative of out_ w.r.t. to in_ (out_.size() x in_.size())
	std::vector<MatXd> doda_;//derivative of out_ w.r.t. to the value "a" of all nodes (nlayer_+1)
public:
	//==== constructors/destructors ====
	DOutDVal(){}
	DOutDVal(const ANN& nn){resize(nn);}
	~DOutDVal(){}
	
	//==== access ====
	MatXd& dodi(){return dodi_;}
	const MatXd& dodi()const{return dodi_;}
	MatXd& doda(int n){return doda_[n];}
	const MatXd& doda(int n)const{return doda_[n];}
	std::vector<MatXd>& doda(){return doda_;}
	const std::vector<MatXd>& doda()const{return doda_;}
	
	//==== member functions ====
	void clear();
	void resize(const ANN& nn);
	void grad(const ANN& nn);
};

//***********************************************************************
// DOutDP
//***********************************************************************

class DOutDP{
private:
	std::vector<MatXd> dodz_;//derivative of output w.r.t. node inputs (nlayer_)
	std::vector<std::vector<VecXd> > dodb_;//derivative of output w.r.t. biases
	std::vector<std::vector<MatXd> > dodw_;//derivative of output w.r.t. weights
public:
	//==== constructors/destructors ====
	DOutDP(){}
	DOutDP(const ANN& nn){resize(nn);}
	~DOutDP(){}
	
	//==== access ====
	MatXd& dodz(int n){return dodz_[n];}
	const MatXd& dodz(int n)const{return dodz_[n];}
	std::vector<MatXd>& dodz(){return dodz_;}
	const std::vector<MatXd>& dodz()const{return dodz_;}
	MatXd& dodb(int n){return dodz_[n];}
	const MatXd& dodb(int n)const{return dodz_[n];}
	std::vector<std::vector<VecXd> >& dodb(){return dodb_;}
	const std::vector<std::vector<VecXd> >& dodb()const{return dodb_;}
	std::vector<std::vector<MatXd> >& dodw(){return dodw_;}
	const std::vector<std::vector<MatXd> >& dodw()const{return dodw_;}
	
	//==== member functions ====
	void clear();
	void resize(const ANN& nn);
	void grad(const ANN& nn);
};

//***********************************************************************
// D2OutDPDVal
//***********************************************************************

class D2OutDPDVal{
private:
	ANN nnc_;
	DOutDVal dOutDVal_;
	std::vector<MatXd> d2odpda_;
	MatXd pt1_,pt2_;
public:
	//==== constructors/destructors ====
	D2OutDPDVal(){}
	D2OutDPDVal(const ANN& nn){resize(nn);}
	~D2OutDPDVal(){}
	
	//==== access ====
	std::vector<MatXd>& d2odpda(){return d2odpda_;}
	const std::vector<MatXd>& d2odpda()const{return d2odpda_;}
	MatXd& d2odpda(int i){return d2odpda_[i];}
	const MatXd& d2odpda(int i)const{return d2odpda_[i];}
	
	//==== member functions ====
	void clear();
	void resize(const ANN& nn);
	void grad(const ANN& nn);
};

}

//**********************************************
// serialization
//**********************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const NN::ANNInit& obj);
	template <> int nbytes(const NN::ANN& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const NN::ANNInit& obj, char* arr);
	template <> int pack(const NN::ANN& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(NN::ANNInit& obj, const char* arr);
	template <> int unpack(NN::ANN& obj, const char* arr);
	
}

#endif