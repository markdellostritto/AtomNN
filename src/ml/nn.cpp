// c libraries
#include <cstdio>
#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
#include <cmath>
#elif (defined __ICC || defined __INTEL_COMPILER)
#include <mathimf.h> //intel math library
#else
#include <cmath>
#endif
#include <ctime>
// c++ libraries
#include <iostream>
#include <random>
#include <chrono>
// ann - math 
#include "src/math/special.hpp"
// ann - str
#include "src/str/string.hpp"
#include "src/str/print.hpp"
// ann - nn
#include "src/ml/nn.hpp"

namespace NN{

//***********************************************************************
// INITIALIZATION METHOD
//***********************************************************************

std::ostream& operator<<(std::ostream& out, const Init& init){
	switch(init){
		case Init::RAND: out<<"RAND"; break;
		case Init::XAVIER: out<<"XAVIER"; break;
		case Init::HE: out<<"HE"; break;
		case Init::MEAN: out<<"MEAN"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

const char* Init::name(const Init& init){
	switch(init){
		case Init::RAND: return "RAND";
		case Init::XAVIER: return "XAVIER";
		case Init::HE: return "HE";
		case Init::MEAN: return "MEAN";
		default: return "UNKNOWN";
	}
}

Init Init::read(const char* str){
	if(std::strcmp(str,"RAND")==0) return Init::RAND;
	else if(std::strcmp(str,"XAVIER")==0) return Init::XAVIER;
	else if(std::strcmp(str,"HE")==0) return Init::HE;
	else if(std::strcmp(str,"MEAN")==0) return Init::MEAN;
	else return Init::UNKNOWN;
}

//***********************************************************************
// TRANSFER FUNCTIONS
//***********************************************************************

//==== type ====

std::ostream& operator<<(std::ostream& out, const Transfer& tf){
	switch(tf){
		case Transfer::LINEAR: out<<"LINEAR"; break;
		case Transfer::SIGMOID: out<<"SIGMOID"; break;
		case Transfer::TANH: out<<"TANH"; break;
		case Transfer::ISRU: out<<"ISRU"; break;
		case Transfer::ARCTAN: out<<"ARCTAN"; break;
		case Transfer::SOFTSIGN: out<<"SOFTSIGN"; break;
		case Transfer::RELU: out<<"RELU"; break;
		case Transfer::SOFTPLUS: out<<"SOFTPLUS"; break;
		case Transfer::ELU: out<<"ELU"; break;
		case Transfer::GELU: out<<"GELU"; break;
		case Transfer::SWISH: out<<"SWISH"; break;
		case Transfer::MISH: out<<"MISH"; break;
		case Transfer::TANHRE: out<<"TANHRE"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

const char* Transfer::name(const Transfer& tf){
	switch(tf){
		case Transfer::LINEAR: return "LINEAR";
		case Transfer::SIGMOID: return "SIGMOID";
		case Transfer::TANH: return "TANH";
		case Transfer::ISRU: return "ISRU";
		case Transfer::ARCTAN: return "ARCTAN";
		case Transfer::SOFTSIGN: return "SOFTSIGN";
		case Transfer::RELU: return "RELU";
		case Transfer::SOFTPLUS: return "SOFTPLUS";
		case Transfer::ELU: return "ELU";
		case Transfer::GELU: return "GELU";
		case Transfer::SWISH: return "SWISH";
		case Transfer::MISH: return "MISH";
		case Transfer::TANHRE: return "TANHRE";
		default: return "UNKNOWN";
	}
}

Transfer Transfer::read(const char* str){
	if(std::strcmp(str,"LINEAR")==0) return Transfer::LINEAR;
	else if(std::strcmp(str,"SIGMOID")==0) return Transfer::SIGMOID;
	else if(std::strcmp(str,"TANH")==0) return Transfer::TANH;
	else if(std::strcmp(str,"ISRU")==0) return Transfer::ISRU;
	else if(std::strcmp(str,"ARCTAN")==0) return Transfer::ARCTAN;
	else if(std::strcmp(str,"SOFTSIGN")==0) return Transfer::SOFTSIGN;
	else if(std::strcmp(str,"RELU")==0) return Transfer::RELU;
	else if(std::strcmp(str,"SOFTPLUS")==0) return Transfer::SOFTPLUS;
	else if(std::strcmp(str,"ELU")==0) return Transfer::ELU;
	else if(std::strcmp(str,"GELU")==0) return Transfer::GELU;
	else if(std::strcmp(str,"SWISH")==0) return Transfer::SWISH;
	else if(std::strcmp(str,"MISH")==0) return Transfer::MISH;
	else if(std::strcmp(str,"TANHRE")==0) return Transfer::TANHRE;
	else return Transfer::UNKNOWN;
}

//==== functions ====

void Transfer::tf_lin(VecXd& f, VecXd& d){
	for(int i=0; i<d.size(); ++i) d[i]=1.0;
}

void Transfer::tf_sigmoid(VecXd& f, VecXd& d){
	const int size=f.size();
	for(int i=0; i<size; ++i){
		if(f[i]>=0){
			const double expf=exp(-f[i]);
			f[i]=1.0/(1.0+expf);
			d[i]=expf/((1.0+expf)*(1.0+expf));
		} else {
			const double expf=exp(f[i]);
			f[i]=expf/(expf+1.0);
			d[i]=expf/((1.0+expf)*(1.0+expf));
		}
	}	
}

void Transfer::tf_tanh(VecXd& f, VecXd& d){
	const int size=f.size();
	for(int i=0; i<size; ++i) f[i]=tanh(f[i]);
	for(int i=0; i<size; ++i) d[i]=1.0-f[i]*f[i];
}

void Transfer::tf_isru(VecXd& f, VecXd& d){
	const int size=f.size();
	for(int i=0; i<size; ++i){
		const double isr=1.0/sqrt(1.0+f[i]*f[i]);
		f[i]=f[i]*isr;
		d[i]=isr*isr*isr;
	}
}

void Transfer::tf_arctan(VecXd& f, VecXd& d){
	const int size=f.size();
	for(int i=0; i<size; ++i){
		d[i]=(2.0/math::constant::PI)/(1.0+f[i]*f[i]);
		f[i]=(2.0/math::constant::PI)*atan(f[i]);
	}
}

void Transfer::tf_softsign(VecXd& f, VecXd& d){
	const int size=f.size();
	for(int i=0; i<size; ++i){
		const double inv=1.0/(1.0+fabs(f[i]));
		f[i]=f[i]*inv;
		d[i]=inv*inv;
	}
}

void Transfer::tf_relu(VecXd& f, VecXd& d){
	const int size=f.size();
	for(int i=0; i<size; ++i){
		if(f[i]>0.0){
			d[i]=1.0;
		} else {
			f[i]=0.0;
			d[i]=0.0;
		}
	}
}

void Transfer::tf_softplus(VecXd& f, VecXd& d){
	const int size=f.size();
	for(int i=0; i<size; ++i){
		if(f[i]>=1.0){
			//f(x)=x+ln(1+exp(-x))-ln(2)
			const double expf=exp(-f[i]);
			f[i]+=math::special::logp1(expf)-math::constant::LOG2;
			d[i]=1.0/(1.0+expf);
		} else {
			//f(x)=ln(1+exp(x))-ln(2)
			const double expf=exp(f[i]);
			f[i]=math::special::logp1(expf)-math::constant::LOG2;
			d[i]=expf/(expf+1.0);
		}
	}
}

void Transfer::tf_elu(VecXd& f, VecXd& d){
	const int size=f.size();
	for(int i=0; i<size; ++i){
		if(f[i]>0.0){
			d[i]=1.0;
		} else {
			const double expf=exp(f[i]);
			f[i]=expf-1.0;
			d[i]=expf;
		}
	}
}

void Transfer::tf_gelu(VecXd& f, VecXd& d){
	const int size=f.size();
	const double rad2pii=1.0/(math::constant::Rad2*math::constant::RadPI);
	for(int i=0; i<size; ++i){
		const double erff=0.5*(1.0+erf(f[i]/math::constant::Rad2));
		d[i]=erff+f[i]*exp(-0.5*f[i]*f[i])*rad2pii;
		f[i]*=erff;
	}
}

void Transfer::tf_swish(VecXd& f, VecXd& d){
	const int size=f.size();
	for(int i=0; i<size; ++i){
		if(f[i]>0.0){
			const double expf=exp(-f[i]);
			const double den=1.0/(1.0+expf);
			d[i]=(1.0+expf*(1.0+f[i]))*den*den;
			f[i]*=den;
		} else {
			const double expf=exp(f[i]);
			const double den=1.0/(1.0+expf);
			d[i]=expf*(f[i]+1.0+expf)*den*den;
			f[i]*=expf*den;
		}
	}
}

void Transfer::tf_mish(VecXd& f, VecXd& d){
	const int size=f.size();
	for(int i=0; i<size; ++i){
		if(f[i]>0.0){
			const double expf=exp(-f[i]);
			const double den=1.0/(2.0*expf*(expf+1.0)+1.0);
			d[i]=(1.0+expf*(4.0+expf*(4.0*f[i]+6.0+expf*4.0*(f[i]+1.0))))*den*den;
			f[i]*=(2.0*expf+1.0)*den;
		} else {
			const double expf=exp(f[i]);
			const double den=1.0/(expf*(expf+2.0)+2.0);
			d[i]=expf*(expf*(expf*(expf+4.0)+4.0*f[i]+6.0)+4.0*(f[i]+1.0))*den*den;
			f[i]*=expf*(expf+2.0)*den;
		}
	}
}

void Transfer::tf_tanhre(VecXd& f, VecXd& d){
	const int size=f.size();
	for(int i=0; i<size; ++i){
		if(f[i]>0){
			d[i]=1.0;
		} else {
			f[i]=tanh(f[i]);
			d[i]=1.0-f[i]*f[i];
		}
	}
}

//***********************************************************************
// ANN
//***********************************************************************

//==== operators ====

/**
* print network to screen
* @param out - output stream
* @param nn - neural network
* @return output stream
*/
std::ostream& operator<<(std::ostream& out, const ANN& nn){
	char* str=new char[print::len_buf];
	out<<print::buf(str)<<"\n";
	out<<print::title("ANN",str)<<"\n";
	out<<"nn   = "; for(int n=0; n<nn.node_.size(); ++n) out<<nn.node_[n].size()<<" "; out<<"\n";
	out<<"size = "<<nn.size()<<"\n";
	out<<"tf   = "<<nn.tf_<<"\n";
	out<<print::title("ANN",str)<<"\n";
	out<<print::buf(str);
	delete[] str;
	return out;
}

/**
* pack network parameters into serial array
* @param nn - neural network
* @param v - vector storing nn parameters
* @return v - vector storing nn parameters
*/
VecXd& operator>>(const ANN& nn, VecXd& v){
	if(NN_PRINT_FUNC>0) std::cout<<"operator>>(const ANN&, VecXd&):\n";
	int count=0;
	v=VecXd::Zero(nn.size());
	for(int l=0; l<nn.nlayer(); ++l){
		//for(int n=0; n<nn.bias(l).size(); ++n) v[count++]=nn.bias(l)(n);
		std::memcpy(v.data()+count,nn.bias(l).data(),nn.bias(l).size()*sizeof(double));
		count+=nn.bias(l).size();
	}
	for(int l=0; l<nn.nlayer(); ++l){
		//for(int n=0; n<nn.edge(l).size(); ++n) v[count++]=nn.edge(l)(n);
		std::memcpy(v.data()+count,nn.edge(l).data(),nn.edge(l).size()*sizeof(double));
		count+=nn.edge(l).size();
	}
	return v;
}

/**
* unpack network parameters from serial array
* @param nn - neural network
* @param v - vector storing nn parameters
* @return nn - neural network
*/
ANN& operator<<(ANN& nn, const VecXd& v){
	if(NN_PRINT_FUNC>0) std::cout<<"operator<<(ANN&,const VecXd&):\n";
	if(nn.size()!=v.size()) throw std::invalid_argument("Invalid size: vector and network mismatch.");
	int count=0;
	for(int l=0; l<nn.nlayer(); ++l){
		//for(int n=0; n<nn.bias(l).size(); ++n) nn.bias(l)(n)=v[count++];
		std::memcpy(nn.bias(l).data(),v.data()+count,nn.bias(l).size()*sizeof(double));
		count+=nn.bias(l).size();
	}
	for(int l=0; l<nn.nlayer(); ++l){
		//for(int n=0; n<nn.edge(l).size(); ++n) nn.edge(l)(n)=v[count++];
		std::memcpy(nn.edge(l).data(),v.data()+count,nn.edge(l).size()*sizeof(double));
		count+=nn.edge(l).size();
	}
	return nn;
}

//==== member functions ====

/**
* set the default values
*/
void ANN::defaults(){
	if(NN_PRINT_FUNC>0) std::cout<<"ANN::defaults():\n";
	//layers
		nlayer_=0;
	//input/output
		in_.resize(0);
		out_.resize(0);
		inw_.resize(0);
		inb_.resize(0);
		outw_.resize(0);
		outb_.resize(0);
	//node weights and biases
		node_.clear();
		bias_.clear();
		edge_.clear();
	//gradients - nodes
		dadz_.clear();
	//transfer functions
		tf_=Transfer::UNKNOWN;
		tfp_.clear();
}

/**
* clear all values
* note that parameters like tf are unchanged
*/
void ANN::clear(){
	if(NN_PRINT_FUNC>0) std::cout<<"ANN::clear():\n";
	//layers
		nlayer_=-1;
	//input/output
		in_.resize(0);
		out_.resize(0);
		inw_.resize(0);
		inb_.resize(0);
		outw_.resize(0);
		outb_.resize(0);
	//node weights and biases
		node_.clear();
		bias_.clear();
		edge_.clear();
	//gradients - nodes
		dadz_.clear();
	//transfer functions
		tfp_.clear();
}

/**
* compute and return the size of the network - the number of adjustable parameters
* @return the size of the network - the number of adjustable parameters
*/
int ANN::size()const{
	if(NN_PRINT_FUNC>0) std::cout<<"ANN::size():\n";
	int s=0;
	for(int n=0; n<nlayer_; ++n) s+=bias_[n].size();
	for(int n=0; n<nlayer_; ++n) s+=edge_[n].size();
	return s;
}

/**
* compute and return the number of bias parameters 
* @return the number of bias parameters 
*/
int ANN::nBias()const{
	if(NN_PRINT_FUNC>0) std::cout<<"ANN::nBias():\n";
	int s=0;
	for(int n=0; n<nlayer_; ++n) s+=bias_[n].size();
	return s;
}

/**
* compute and return the number of weight parameters 
* @return the number of weight parameters 
*/
int ANN::nWeight()const{
	if(NN_PRINT_FUNC>0) std::cout<<"ANN::nWeight():\n";
	int s=0;
	for(int n=0; n<nlayer_; ++n) s+=edge_[n].size();
	return s;
}

/**
* resize the network - no hidden layers
* @param init - object containing requisite initialization parameters
* @param nIn - number of inputs of the newtork
* @param nOut - the number of outputs of the network
*/
void ANN::resize(const ANNInit& init, int nIn, int nOut){
	if(NN_PRINT_FUNC>0) std::cout<<"ANN::resize(const ANNInit&,int,int):\n";
	if(nIn<=0) throw std::invalid_argument("ANN::resize(int,int): Invalid output size.");
	if(nOut<=0) throw std::invalid_argument("ANN::resize(int,int): Invalid output size.");
	std::vector<int> nn(2);
	nn[0]=nIn; nn[1]=nOut;
	resize(init,nn);
}

/**
* resize the network - given separate hidden layers and output layer
* @param init - object containing requisite initialization parameters
* @param nIn - number of inputs of the newtork
* @param nOut - the number of outputs of the network
* @param nNodes - the number of nodes in each hidden layer
*/
void ANN::resize(const ANNInit& init, int nIn, const std::vector<int>& nNodes, int nOut){
	if(NN_PRINT_FUNC>0) std::cout<<"ANN::resize(const ANNInit&,int,const std::vector<int>&,int):\n";
	if(nOut<=0) throw std::invalid_argument("ANN::resize(const ANNInit&,int,const std::vector<int>&,int): Invalid output size.");
	std::vector<int> nn(nNodes.size()+2);
	nn.front()=nIn;
	for(int n=0; n<nNodes.size(); ++n) nn[n+1]=nNodes[n];
	nn.back()=nOut;
	resize(init,nn);
}

/**
* resize the network - given combined hidden layers and output layer
* @param init - object containing requisite initialization parameters
* @param nNodes - the number of nodes in each layer of the network
*/
void ANN::resize(const ANNInit& init, const std::vector<int>& nNodes){
	if(NN_PRINT_FUNC>0) std::cout<<"ANN::resize(const ANNInit&,const std::vector<int>&):\n";
	//initialize the random number generator
		if(init.sigma()<=0) throw std::invalid_argument("ANN::resize(const ANNInit&,const std::vector<int>&): Invalid initialization deviation");
		std::mt19937 rngen(init.seed()<0?std::chrono::system_clock::now().time_since_epoch().count():init.seed());
		std::uniform_real_distribution<double> uniform(-1.0,1.0);
	//clear the network
		clear();
	//check parameters
		for(int n=0; n<nNodes.size(); ++n){
			if(nNodes[n]<=0) throw std::invalid_argument("ANN::resize(const std::vector<int>&): Invalid layer size.");
		}
	//input/output
		in_=VecXd::Zero(nNodes.front());
		out_=VecXd::Zero(nNodes.back());
	//pre/post conditioning
		inw_=VecXd::Constant(in_.size(),1);
		inb_=VecXd::Constant(in_.size(),0);
		outw_=VecXd::Constant(out_.size(),1);
		outb_=VecXd::Constant(out_.size(),0);
	//number of layers
		nlayer_=nNodes.size()-1;//number of weights, i.e. connections b/w layers, thus 1 less than size of nNodes
		if(nlayer_<1) throw std::invalid_argument("ANN::resize(const std::vector<int>&): Invalid number of layers.");
	//gradients - nodes
		dadz_.resize(nlayer_);
		for(int n=0; n<nlayer_; ++n){
			dadz_[n]=VecXd::Zero(nNodes[n+1]);
		}
	//nodes
		node_.resize(nlayer_+1);
		for(int n=0; n<nlayer_+1; ++n){
			node_[n]=VecXd::Zero(nNodes[n]);
		}
	//bias
		bias_.resize(nlayer_);
		for(int n=0; n<nlayer_; ++n){
			bias_[n]=VecXd::Zero(nNodes[n+1]);
		}
		for(int n=0; n<nlayer_; ++n){
			for(int m=0; m<bias_[n].size(); ++m){
				bias_[n][m]=uniform(rngen)*init.bInit();
			}
		}
	//edges
		edge_.resize(nlayer_);
		//edge(n) * layer(n) -> layer(n+1), thus size(edge) = (layer(n+1) rows * layer(n) cols)
		for(int n=0; n<nlayer_; ++n){
			edge_[n]=MatXd::Zero(nNodes[n+1],nNodes[n]);
		}
		if(init.dist()==rng::dist::Name::NORMAL){
			std::normal_distribution<double> dist(0.0,init.sigma());
			for(int n=0; n<nlayer_; ++n){
				for(int m=0; m<edge_[n].size(); ++m){
					edge_[n].data()[m]=dist(rngen);
				}
			}
		} else if(init.dist()==rng::dist::Name::EXP){
			std::exponential_distribution<double> dist(init.sigma());
			for(int n=0; n<nlayer_; ++n){
				for(int m=0; m<edge_[n].size(); ++m){
					edge_[n].data()[m]=dist(rngen);
				}
			}
		} else throw std::invalid_argument("ANN::resize(const ANNInit&,const std::vector<int>&): Invalid probability distribution.");
		switch(init.init()){
			case Init::RAND:   for(int n=0; n<nlayer_; ++n) edge_[n]*=init.wInit(); break;
			case Init::XAVIER: for(int n=0; n<nlayer_; ++n) edge_[n]*=init.wInit()*std::sqrt(1.0/nNodes[n]); break;
			case Init::HE:     for(int n=0; n<nlayer_; ++n) edge_[n]*=init.wInit()*std::sqrt(2.0/nNodes[n]); break;
			case Init::MEAN:   for(int n=0; n<nlayer_; ++n) edge_[n]*=init.wInit()*std::sqrt(2.0/(nNodes[n+1]+nNodes[n])); break;
			default: throw std::invalid_argument("ANN::resize(const ANNInit&,const std::vector<int>&): Invalid initialization scheme."); break;
		}
	//transfer functions
		switch(tf_){
			case Transfer::LINEAR:   tfp_.resize(nlayer_,Transfer::tf_lin); break;
			case Transfer::SIGMOID:  tfp_.resize(nlayer_,Transfer::tf_sigmoid); break;
			case Transfer::TANH:     tfp_.resize(nlayer_,Transfer::tf_tanh); break;
			case Transfer::ISRU:     tfp_.resize(nlayer_,Transfer::tf_isru); break;
			case Transfer::ARCTAN:   tfp_.resize(nlayer_,Transfer::tf_arctan); break;
			case Transfer::SOFTSIGN: tfp_.resize(nlayer_,Transfer::tf_softsign); break;
			case Transfer::SOFTPLUS: tfp_.resize(nlayer_,Transfer::tf_softplus); break;
			case Transfer::RELU:     tfp_.resize(nlayer_,Transfer::tf_relu); break;
			case Transfer::ELU:      tfp_.resize(nlayer_,Transfer::tf_elu); break;
			case Transfer::GELU:     tfp_.resize(nlayer_,Transfer::tf_gelu); break;
			case Transfer::SWISH:    tfp_.resize(nlayer_,Transfer::tf_swish); break;
			case Transfer::MISH:     tfp_.resize(nlayer_,Transfer::tf_mish); break;
			case Transfer::TANHRE:   tfp_.resize(nlayer_,Transfer::tf_tanhre); break;
			default: throw std::invalid_argument("ANN::resize(const ANNInit&,const std::vector<int>&): Invalid transfer function."); break;
		}
		tfp_.back()=Transfer::tf_lin;//final layer is typically linear
}

/**
* compute the regularization error
* @return the regularization error - 1/2 the sum of the squares of the weights
*/
double ANN::error_lambda()const{
	if(NN_PRINT_FUNC>0) std::cout<<"ANN::error_lambda():\n";
	double err=0;
	for(int l=0; l<nlayer_; ++l){
		err+=0.5*edge_[l].squaredNorm();//lambda error - quadratic
	}
	//return error
	return err;
}

/**
* compute the regularization gradient
* @param grad - stores the regularization gradient w.r.t. each parameter of the network
* @return grad - the regularization gradient w.r.t. each parameter of the network
*/
VecXd& ANN::grad_lambda(VecXd& grad)const{
	if(NN_PRINT_FUNC>0) std::cout<<"ANN::grad_lambda(VecXd&):\n";
	int count=0;
	//gradient w.r.t bias
	for(int l=0; l<nlayer_; ++l){
		for(int n=0; n<bias_[l].size(); ++n){
			grad[count++]=0.0;
		}
	}
	//gradient w.r.t. edges
	for(int l=0; l<nlayer_; ++l){
		for(int m=0; m<edge_[l].cols(); ++m){
			for(int n=0; n<edge_[l].rows(); ++n){
				grad[count++]=edge_[l](n,m);//edge(l,n,m) - quadratic
			}
		}
	}
	//return the gradient
	return grad;
}

/**
* execute the network
* @return out_ - the output of the network
*/
const VecXd& ANN::execute(){
	if(NN_PRINT_FUNC>0) std::cout<<"ANN::execute():\n";
	//scale the input
	node_.front().noalias()=inw_.cwiseProduct(in_+inb_);
	//hidden layers
	for(int l=0; l<nlayer_; ++l){
		node_[l+1]=bias_[l];
		node_[l+1].noalias()+=edge_[l]*node_[l];
		(*tfp_[l])(node_[l+1],dadz_[l]);
	}
	//scale the output
	out_=outb_;
	out_.noalias()+=node_.back().cwiseProduct(outw_);
	//return the output
	return out_;
}

//==== static functions ====

/**
* write the network to file
* @param file - the file name where the network is to be written
* @param nn - the neural network to be written
*/
void ANN::write(const char* file, const ANN& nn){
	if(NN_PRINT_FUNC>0) std::cout<<"ANN::write(const char*,const ANN&):\n";
	//local variables
	FILE* writer=NULL;
	//open the file
	writer=std::fopen(file,"w");
	if(writer!=NULL){
		ANN::write(writer,nn);
		std::fclose(writer);
		writer=NULL;
	} else throw std::runtime_error(std::string("ERROR: Could not open \"")+std::string(file)+std::string("\" for writing.\n"));
}

/**
* write the network to file
* @param writer - file pointer
* @param nn - the neural network to be written
*/
void ANN::write(FILE* writer, const ANN& nn){
	if(NN_PRINT_FUNC>0) std::cout<<"ANN::write(FILE*,const ANN&):\n";
	//print the configuration
	fprintf(writer,"nn ");
	for(int i=0; i<nn.nlayer()+1; ++i) fprintf(writer,"%i ",nn.nNodes(i));
	fprintf(writer,"\n");
	//print the transfer function
	fprintf(writer,"t_func %s\n",Transfer::name(nn.tf()));
	//print the scaling layers
	fprintf(writer,"inw ");
	for(int i=0; i<nn.nIn(); ++i) fprintf(writer,"%.15f ",nn.inw()[i]);
	fprintf(writer,"\n");
	fprintf(writer,"outw ");
	for(int i=0; i<nn.nOut(); ++i) fprintf(writer,"%.15f ",nn.outw()[i]);
	fprintf(writer,"\n");
	//print the biasing layers
	fprintf(writer,"inb ");
	for(int i=0; i<nn.nIn(); ++i) fprintf(writer,"%.15f ",nn.inb()[i]);
	fprintf(writer,"\n");
	fprintf(writer,"outb ");
	for(int i=0; i<nn.nOut(); ++i) fprintf(writer,"%.15f ",nn.outb()[i]);
	fprintf(writer,"\n");
	//print the biases
	for(int n=0; n<nn.nlayer(); ++n){
		fprintf(writer,"bias[%i] ",n+1);
		for(int i=0; i<nn.bias(n).size(); ++i){
			fprintf(writer,"%.15f ",nn.bias(n)[i]);
		}
		fprintf(writer,"\n");
	}
	//print the edge weights
	for(int n=0; n<nn.nlayer(); ++n){
		fprintf(writer,"weight[%i,%i] ",n,n+1);
		for(int i=0; i<nn.edge(n).rows(); ++i){
			for(int j=0; j<nn.edge(n).cols(); ++j){
				fprintf(writer,"%.15f ",nn.edge(n)(i,j));
			}
		}
		fprintf(writer,"\n");
	}
}

/**
* read the network from file
* @param file - the file name where the network is to be read
* @param nn - the neural network to be read
*/
void ANN::read(const char* file, ANN& nn){
	if(NN_PRINT_FUNC>0) std::cout<<"ANN::read(const char*,ANN&):\n";
	//local variables
	FILE* reader=NULL;
	//open the file
	reader=std::fopen(file,"r");
	if(reader!=NULL){
		ANN::read(reader,nn);
		std::fclose(reader);
		reader=NULL;
	} else throw std::runtime_error(std::string("ERROR: Could not open \"")+std::string(file)+std::string("\" for reading.\n"));
}

/**
* read the network from file
* @param reader - file pointer
* @param nn - the neural network to be read
*/
void ANN::read(FILE* reader, ANN& nn){
	if(NN_PRINT_FUNC>0) std::cout<<"ANN::read(FILE*,ANN&):\n";
	//==== local variables ====
	const int MAX=5000;
	const int N_DIGITS=32;//max number of digits in number
	int b_max=0;//max number of biases for a given layer
	int w_max=0;//max number of weights for a given layer
	char* input=new char[MAX];
	char* b_str=NULL;//bias string
	char* w_str=NULL;//weight string
	char* i_str=NULL;//input string
	char* o_str=NULL;//output string
	std::vector<int> nodes;
	std::vector<std::string> strlist;
	ANNInit init;
	//==== clear the network ====
	if(NN_PRINT_STATUS>0) std::cout<<"clearing the network\n";
	nn.clear();
	//==== load the configuration ====
	if(NN_PRINT_STATUS>0) std::cout<<"reading configuration\n";
	string::split(fgets(input,MAX,reader),string::WS,strlist);
	if(strlist.size()<2) throw std::invalid_argument("ANN::read(FILE*,ANN&): Invalid network configuration.");
	const int nlayer=strlist.size()-2;//"nn" nIn nh0 nh1 nh2 ... nOut
	nodes.resize(nlayer+1);
	for(int i=1; i<strlist.size(); ++i) nodes[i-1]=std::atoi(strlist[i].c_str());
	if(NN_PRINT_DATA>0){for(int i=0; i<nodes.size(); ++i) std::cout<<nodes[i]<<" "; std::cout<<"\n";}
	//==== set the transfer function ====
	if(NN_PRINT_STATUS>0) std::cout<<"reading transfer function\n";
	string::split(fgets(input,MAX,reader),string::WS,strlist);
	nn.tf()=Transfer::read(strlist[1].c_str());
	if(nn.tf()==Transfer::UNKNOWN) throw std::invalid_argument("ANN::read(FILE*,ANN&): Invalid transfer function.");
	//==== resize the nueral newtork ====
	if(NN_PRINT_STATUS>0) std::cout<<"resizing neural network\n";
	if(NN_PRINT_STATUS>0) {for(int i=0; i<nodes.size(); ++i) std::cout<<nodes[i]<<" "; std::cout<<"\n";}
	nn.resize(init,nodes);
	if(NN_PRINT_STATUS>1) std::cout<<"nn = "<<nn<<"\n";
	w_max=nn.nNodes(0)*nn.nIn();
	for(int i=0; i<nn.nlayer(); ++i) b_max=(b_max>nn.nNodes(i))?b_max:nn.nNodes(i);
	for(int i=1; i<nn.nlayer(); ++i) w_max=(w_max>nn.nNodes(i)*nn.nNodes(i-1))?w_max:nn.nNodes(i)*nn.nNodes(i-1);
	if(NN_PRINT_DATA>0) std::cout<<"b_max "<<b_max<<" w_max "<<w_max<<"\n";
	b_str=new char[b_max*N_DIGITS];
	w_str=new char[w_max*N_DIGITS];
	i_str=new char[nn.nIn()*N_DIGITS];
	o_str=new char[nn.nOut()*N_DIGITS];
	//==== read the scaling layers ====
	if(NN_PRINT_STATUS>0) std::cout<<"reading scaling layers\n";
	string::split(fgets(i_str,nn.nIn()*N_DIGITS,reader),string::WS,strlist);
	for(int j=1; j<strlist.size(); ++j) nn.inw()[j-1]=std::atof(strlist[j].c_str());
	string::split(fgets(o_str,nn.nOut()*N_DIGITS,reader),string::WS,strlist);
	for(int j=1; j<strlist.size(); ++j) nn.outw()[j-1]=std::atof(strlist[j].c_str());
	//==== read the biasing layers ====
	if(NN_PRINT_STATUS>0) std::cout<<"reading biasing layers\n";
	string::split(fgets(i_str,nn.nIn()*N_DIGITS,reader),string::WS,strlist);
	for(int j=1; j<strlist.size(); ++j) nn.inb()[j-1]=std::atof(strlist[j].c_str());
	string::split(fgets(o_str,nn.nOut()*N_DIGITS,reader),string::WS,strlist);
	for(int j=1; j<strlist.size(); ++j) nn.outb()[j-1]=std::atof(strlist[j].c_str());
	//==== read in the biases ====
	if(NN_PRINT_STATUS>0) std::cout<<"reading biases\n";
	for(int n=0; n<nn.nlayer(); ++n){
		string::split(fgets(b_str,b_max*N_DIGITS,reader),string::WS,strlist);
		for(int i=0; i<nn.bias(n).size(); ++i){
			nn.bias(n)[i]=std::atof(strlist[i+1].c_str());
		}
	}
	//==== read in the edge weights ====
	if(NN_PRINT_STATUS>0) std::cout<<"reading weights\n";
	for(int n=0; n<nn.nlayer(); ++n){
		string::split(fgets(w_str,w_max*N_DIGITS,reader),string::WS,strlist);
		int count=0;
		for(int i=0; i<nn.edge(n).rows(); ++i){
			for(int j=0; j<nn.edge(n).cols(); ++j){
				nn.edge(n)(i,j)=std::atof(strlist[++count].c_str());
			}
		}
	}
	//==== free local variables ====
	if(input!=NULL) delete[] input;
	if(b_str!=NULL) delete[] b_str;
	if(w_str!=NULL) delete[] w_str;
	if(i_str!=NULL) delete[] i_str;
	if(o_str!=NULL) delete[] o_str;
}

//==== operators ====

/**
* check equality of two networks
* @param n1 - neural network - first
* @param n2 - neural network - second
* @return equality of n1 and n2
*/
bool operator==(const ANN& n1, const ANN& n2){
	if(n1.tf()!=n2.tf()) return false;
	else if(n1.nlayer()!=n2.nlayer()) return false;
	else if(n1.nIn()!=n2.nIn()) return false;
	else {
		//number of layers
		for(int i=0; i<n1.nlayer(); ++i){
			if(n1.nNodes(i)!=n2.nNodes(i)) return false;
		}
		//pre-/post-conditioning
		for(int i=0; i<n1.nIn(); ++i){
			if(n1.inw()[i]!=n2.inw()[i]) return false;
			if(n1.inb()[i]!=n2.inb()[i]) return false;
		}
		for(int i=0; i<n1.nOut(); ++i){
			if(n1.outw()[i]!=n2.outw()[i]) return false;
			if(n1.outb()[i]!=n2.outb()[i]) return false;
		}
		//bias
		for(int i=0; i<n1.nlayer(); ++i){
			double diff=(n1.bias(i)-n2.bias(i)).norm();
			if(diff>math::constant::ZERO) return false;
		}
		//edge
		for(int i=0; i<n1.nlayer(); ++i){
			double diff=(n1.edge(i)-n2.edge(i)).norm();
			if(diff>math::constant::ZERO) return false;
		}
		//same
		return true;
	}
}

//***********************************************************************
// ANNInit
//***********************************************************************

void ANNInit::defaults(){
	if(NN_PRINT_FUNC>0) std::cout<<"ANNInit::defaults():\n";
	bInit_=0.001;
	wInit_=1;
	sigma_=1.0;
	dist_=rng::dist::Name::NORMAL;
	init_=Init::RAND;
	seed_=-1;
}

std::ostream& operator<<(std::ostream& out, const ANNInit& init){
	char* str=new char[print::len_buf];
	out<<print::buf(str)<<"\n";
	out<<print::title("ANN_INIT",str)<<"\n";
	out<<"b-init = "<<init.bInit_<<"\n";
	out<<"w-init = "<<init.wInit_<<"\n";
	out<<"sigma  = "<<init.sigma_<<"\n";
	out<<"dist   = "<<init.dist_<<"\n";
	out<<"init   = "<<init.init_<<"\n";
	out<<"seed   = "<<init.seed_<<"\n";
	out<<print::title("ANN_INIT",str)<<"\n";
	out<<print::buf(str);
	delete[] str;
	return out;
}

//***********************************************************************
// Cost
//***********************************************************************

/**
* clear all local data
*/
void Cost::clear(){
	if(NN_PRINT_FUNC>0) std::cout<<"Cost::clear():\n";
	dcdz_.clear();
}

/**
* resize data for a given neural network
* @param nn - the neural network for which we will compute the cost function
*/
void Cost::resize(const ANN& nn){
	if(NN_PRINT_FUNC>0) std::cout<<"Cost::resize(const ANN&):\n";
	dcdz_.resize(nn.nlayer());
	for(int n=0; n<nn.nlayer(); ++n){
		dcdz_[n]=VecXd::Zero(nn.node(n+1).size());
	}
	grad_.resize(nn.size());
}

/**
* compute gradient of error given the derivative of the cost function w.r.t. the output (dcdo)
* @param nn - the neural network for which we will compute the gradient
* @param dcdo - the derivative of the cost function w.r.t. the output
* @return grad - the gradient of the cost function w.r.t. each parameter of the network
*/
const VecXd& Cost::grad(const ANN& nn, const VecXd& dcdo){
	if(NN_PRINT_FUNC>0) std::cout<<"Cost::grad(const ANN&,const VecXd&):\n";
	const int nlayer=nn.nlayer();
	//compute delta for the output layer
	const int size=nn.outw().size();
	for(int i=0; i<size; ++i) dcdz_[nlayer-1][i]=nn.outw()[i]*dcdo[i]*nn.dadz(nlayer-1)[i];
	//back-propogate the error
	for(int l=nlayer-1; l>0; --l){
		dcdz_[l-1].noalias()=nn.dadz(l-1).cwiseProduct(nn.edge(l).transpose()*dcdz_[l]);
	}
	int count=0;
	//gradient w.r.t bias
	if(NN_PRINT_STATUS>1) std::cout<<"computing gradient w.r.t. bias\n";
	for(int l=0; l<nlayer; ++l){
		for(int n=0; n<dcdz_[l].size(); ++n){
			grad_[count++]=dcdz_[l][n];//bias(l,n)
		}
	}
	//gradient w.r.t. edges
	if(NN_PRINT_STATUS>1) std::cout<<"computing gradient w.r.t. edges\n";
	for(int l=0; l<nlayer; ++l){
		for(int m=0; m<nn.edge(l).cols(); ++m){
			const double node=nn.node(l)(m);
			for(int n=0; n<nn.edge(l).rows(); ++n){
				grad_[count++]=dcdz_[l][n]*node;//edge(l,n,m)
			}
		}
	}
	//return the gradient
	return grad_;
}

//***********************************************************************
// DOutDVal
//***********************************************************************

/**
* clear all local data
*/
void DOutDVal::clear(){
	if(NN_PRINT_FUNC>0) std::cout<<"DOutDVal::clear():\n";
	dodi_.resize(0,0);
	doda_.clear();
}

/**
* resize data for a given neural network
* @param nn - the neural network for which we will compute the gradient
*/
void DOutDVal::resize(const ANN& nn){
	if(NN_PRINT_FUNC>0) std::cout<<"DOutDVal::resize(const ANN&):\n";
	dodi_=MatXd::Zero(nn.out().size(),nn.in().size());
	doda_.resize(nn.nlayer()+1);
	for(int n=0; n<nn.nlayer()+1; ++n){
		doda_[n]=MatXd::Zero(nn.out().size(),nn.node(n).size());
	}
}

/**
* compute the gradient of output w.r.t. all other node values (e.g. doda_ and dodi_)
* @param nn - the neural network for which we will compute the gradient
*/
void DOutDVal::grad(const ANN& nn){
	//back-propogate the gradient (n.b. do/dz_{o}=outw_ "gradient of out_ w.r.t. the input of out_ is outw_")
	doda_.back()=nn.outw().asDiagonal();
	for(int l=nn.nlayer()-1; l>=0; --l){
		doda_[l].noalias()=doda_[l+1]*nn.dadz(l).asDiagonal()*nn.edge(l);
	}
	//compute gradient of out_ w.r.t. in_ (effect of input scaling)
	dodi_.noalias()=doda_[0]*nn.inw().asDiagonal();
}

//***********************************************************************
// DOutDP
//***********************************************************************

/**
* clear all local data
*/
void DOutDP::clear(){
	if(NN_PRINT_FUNC>0) std::cout<<"DOutDP::clear():\n";
	dodz_.clear();
	dodb_.clear();
	dodw_.clear();
}

/**
* resize data for a given neural network
* @param nn - the neural network for which we will compute the gradient
*/
void DOutDP::resize(const ANN& nn){
	if(NN_PRINT_FUNC>0) std::cout<<"DOutDP::resize(const ANN&):\n";
	dodz_.resize(nn.nlayer());
	for(int n=0; n<nn.nlayer(); ++n){
		dodz_[n]=MatXd::Zero(nn.out().size(),nn.bias(n).size());
	}
	dodb_.resize(nn.nOut());
	for(int n=0; n<nn.nOut(); ++n){
		dodb_[n].resize(nn.nlayer());
		for(int l=0; l<nn.nlayer(); ++l){
			dodb_[n][l]=VecXd::Zero(nn.bias(l).size());
		}
	}
	dodw_.resize(nn.nOut());
	for(int n=0; n<nn.nOut(); ++n){
		dodw_[n].resize(nn.nlayer());
		for(int l=0; l<nn.nlayer(); ++l){
			dodw_[n][l]=MatXd::Zero(nn.edge(l).rows(),nn.edge(l).cols());
		}
	}
}

/**
* compute the gradient of output w.r.t. parameters
* @param nn - the neural network for which we will compute the gradient
*/
void DOutDP::grad(const ANN& nn){
	//back-propogate the gradient (n.b. do/dz_{o}=outw_ "gradient of out_ w.r.t. the input of out_ is outw_")
	dodz_.back()=nn.outw().cwiseProduct(nn.dadz(nn.nlayer()-1)).asDiagonal();
	for(int l=nn.nlayer()-1; l>0; --l){
		dodz_[l-1].noalias()=dodz_[l]*nn.edge(l)*nn.dadz(l-1).asDiagonal();
	}
	//compute the gradient of the output w.r.t. the biases
	for(int n=0; n<nn.nOut(); ++n){
		for(int l=0; l<nn.nlayer(); ++l){
			for(int i=0; i<nn.bias(l).size(); ++i){
				dodb_[n][l](i)=dodz_[l](n,i);
			}
		}
	}
	//compute the gradient of the output w.r.t. the weights
	for(int n=0; n<nn.nOut(); ++n){
		for(int l=0; l<nn.nlayer(); ++l){
			for(int j=0; j<nn.edge(l).cols(); ++j){
				const double node=nn.node(l)[j];
				for(int i=0; i<nn.edge(l).rows(); ++i){
					dodw_[n][l](i,j)=dodz_[l](n,i)*node;
				}
			}
		}
	}
}

//***********************************************************************
// D2OutDPDVal
//***********************************************************************

/**
* clear all local data
*/
void D2OutDPDVal::clear(){
	nnc_.clear();
	dOutDVal_.clear();
	d2odpda_.clear();
}

/**
* resize data for a given neural network
* @param nn - the neural network for which we will compute the gradient
*/
void D2OutDPDVal::resize(const ANN& nn){
	//gradient of the ouput with respect to the input
	dOutDVal_.resize(nn);
	//second derivative
	d2odpda_.resize(nn.size());
}

/**
* compute the gradient of the gradient of output w.r.t. the weights
* @param nn - the neural network for which we will compute the gradient
*/
void D2OutDPDVal::grad(const ANN& nn){
	//local variables
	int count=0;
	//make copy of the network 
	nnc_=nn;
	//loop over all biases
	for(int l=0; l<nnc_.nlayer(); ++l){
		for(int n=0; n<nnc_.bias(l).size(); ++n){
			const double delta=nnc_.bias(l)[n]/100.0;
			//point 1
			nnc_.bias(l)[n]=nn.bias(l)[n]-delta;
			nnc_.execute();
			dOutDVal_.grad(nnc_);
			pt1_=dOutDVal_.dodi();
			//point 2
			nnc_.bias(l)[n]=nn.bias(l)[n]+delta;
			nnc_.execute();
			dOutDVal_.grad(nnc_);
			pt2_=dOutDVal_.dodi();
			//gradient
			d2odpda_[count++].noalias()=0.5*(pt2_-pt1_)/delta;
		}
	}
	//loop over all weights
	for(int l=0; l<nnc_.nlayer(); ++l){
		for(int n=0; n<nn.edge(l).size(); ++n){
			const double delta=nnc_.edge(l)(n)/100.0;
			//point 1
			nnc_.edge(l)(n)=nn.edge(l)(n)-delta;
			nnc_.execute();
			dOutDVal_.grad(nnc_);
			pt1_=dOutDVal_.dodi();
			//point 2
			nnc_.edge(l)(n)=nn.edge(l)(n)+delta;
			nnc_.execute();
			dOutDVal_.grad(nnc_);
			pt2_=dOutDVal_.dodi();
			//gradient
			d2odpda_[count++].noalias()=0.5*(pt2_-pt1_)/delta;
		}
	}
}

}

//***********************************************************************
// serialization
//***********************************************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const NN::ANN& obj){
		if(NN_PRINT_FUNC>0) std::cout<<"nbytes(const NN::ANN&):\n";
		int size=0;
		size+=sizeof(NN::Transfer);//transfer function type
		size+=sizeof(int);//nlayer_
		if(obj.nlayer()>0){
			size+=sizeof(int)*(obj.nlayer()+1);//number of nodes in each layer
			for(int l=0; l<obj.nlayer(); ++l) size+=obj.bias(l).size()*sizeof(double);//bias
			for(int l=0; l<obj.nlayer(); ++l) size+=obj.edge(l).size()*sizeof(double);//edge
			size+=obj.nIn()*sizeof(double);//pre-scale
			size+=obj.nIn()*sizeof(double);//pre-bias
			size+=obj.nOut()*sizeof(double);//post-scale
			size+=obj.nOut()*sizeof(double);//post-bias
		}
		return size;
	}
	
	template <> int nbytes(const NN::ANNInit& obj){
		if(NN_PRINT_FUNC>0) std::cout<<"nbytes(const NN::ANNInit&):\n";
		int size=0;
		size+=sizeof(double);//bInit_
		size+=sizeof(double);//wInit_
		size+=sizeof(double);//sigma_
		size+=sizeof(rng::dist::Name);
		size+=sizeof(NN::Init);
		size+=sizeof(int);//seed_
		return size;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const NN::ANN& obj, char* arr){
		if(NN_PRINT_FUNC>0) std::cout<<"pack(const NN::ANN&,char*):\n";
		int pos=0;
		int tempInt=0;
		//transfer function type
		std::memcpy(arr+pos,&(obj.tf()),sizeof(NN::Transfer)); pos+=sizeof(NN::Transfer);
		//nlayer_
		std::memcpy(arr+pos,&(tempInt=obj.nlayer()),sizeof(int)); pos+=sizeof(int);
		if(obj.nlayer()>0){
			//number of nodes in each layer
			for(int l=0; l<obj.nlayer()+1; ++l){
				std::memcpy(arr+pos,&(tempInt=obj.nNodes(l)),sizeof(int)); pos+=sizeof(int);
			}
			//bias
			for(int l=0; l<obj.nlayer(); ++l){
				std::memcpy(arr+pos,obj.bias(l).data(),obj.bias(l).size()*sizeof(double)); pos+=obj.bias(l).size()*sizeof(double);
			}
			//edge
			for(int l=0; l<obj.nlayer(); ++l){
				std::memcpy(arr+pos,obj.edge(l).data(),obj.edge(l).size()*sizeof(double)); pos+=obj.edge(l).size()*sizeof(double);
			}
			//pre-scale
			std::memcpy(arr+pos,obj.inw().data(),obj.inw().size()*sizeof(double)); pos+=obj.inw().size()*sizeof(double);
			//pre-bias
			std::memcpy(arr+pos,obj.inb().data(),obj.inb().size()*sizeof(double)); pos+=obj.inb().size()*sizeof(double);
			//post-scale
			std::memcpy(arr+pos,obj.outw().data(),obj.outw().size()*sizeof(double)); pos+=obj.outw().size()*sizeof(double);
			//post-bias
			std::memcpy(arr+pos,obj.outb().data(),obj.outb().size()*sizeof(double)); pos+=obj.outb().size()*sizeof(double);
		}
		//return bytes written
		return pos;
	}
	
	template <> int pack(const NN::ANNInit& obj, char* arr){
		if(NN_PRINT_FUNC>0) std::cout<<"pack(const NN::ANNInit&,char*):\n";
		int pos=0;
		std::memcpy(arr+pos,&obj.bInit(),sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&obj.wInit(),sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&obj.sigma(),sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&obj.dist(),sizeof(rng::dist::Name)); pos+=sizeof(rng::dist::Name);
		std::memcpy(arr+pos,&obj.init(),sizeof(NN::Init)); pos+=sizeof(NN::Init);
		std::memcpy(arr+pos,&obj.seed(),sizeof(int)); pos+=sizeof(int);
		return pos;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(NN::ANN& obj, const char* arr){
		if(NN_PRINT_FUNC>0) std::cout<<"unpack(NN::ANN&,const char*):\n";
		//local variables
		int pos=0;
		int nlayer=0,nIn=0;
		std::vector<int> nNodes;
		//transfer function type
		std::memcpy(&(obj.tf()),arr+pos,sizeof(NN::Transfer)); pos+=sizeof(NN::Transfer);
		//nlayer
		std::memcpy(&nlayer,arr+pos,sizeof(int)); pos+=sizeof(int);
		if(nlayer>0){
			nNodes.resize(nlayer+1,0);
			//number of nodes in each layer
			for(int i=0; i<nlayer+1; ++i){
				std::memcpy(&nNodes[i],arr+pos,sizeof(int)); pos+=sizeof(int);
			}
			//resize the network
			NN::ANNInit init;
			obj.resize(init,nNodes);
			//bias
			for(int l=0; l<obj.nlayer(); ++l){
				std::memcpy(obj.bias(l).data(),arr+pos,obj.bias(l).size()*sizeof(double)); pos+=obj.bias(l).size()*sizeof(double);
			}
			//edge
			for(int l=0; l<obj.nlayer(); ++l){
				std::memcpy(obj.edge(l).data(),arr+pos,obj.edge(l).size()*sizeof(double)); pos+=obj.edge(l).size()*sizeof(double);
			}
			//pre-scale
			std::memcpy(obj.inw().data(),arr+pos,obj.inw().size()*sizeof(double)); pos+=obj.inw().size()*sizeof(double);
			//pre-bias
			std::memcpy(obj.inb().data(),arr+pos,obj.inb().size()*sizeof(double)); pos+=obj.inb().size()*sizeof(double);
			//post-scale
			std::memcpy(obj.outw().data(),arr+pos,obj.outw().size()*sizeof(double)); pos+=obj.outw().size()*sizeof(double);
			//post-bias
			std::memcpy(obj.outb().data(),arr+pos,obj.outb().size()*sizeof(double)); pos+=obj.outb().size()*sizeof(double);
		}
		//return bytes read
		return pos;
	}
	
	template <> int unpack(NN::ANNInit& obj, const char* arr){
		if(NN_PRINT_FUNC>0) std::cout<<"unpack(NN::ANNInit&,const char*):\n";
		//local variables
		int pos=0;
		std::memcpy(&obj.bInit(),arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.wInit(),arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.sigma(),arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.dist(),arr+pos,sizeof(rng::dist::Name)); pos+=sizeof(rng::dist::Name);
		std::memcpy(&obj.init(),arr+pos,sizeof(NN::Init)); pos+=sizeof(NN::Init);
		std::memcpy(&obj.seed(),arr+pos,sizeof(int)); pos+=sizeof(int);
		return pos;
	}
	
	
}