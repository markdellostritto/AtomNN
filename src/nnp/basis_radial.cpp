// c libraries
#include <cstring>
#include <cstdio>
// c++ libraries
#include <iostream>
#include <vector>
// math
#include "src/math/special.hpp"
// str
#include "src/str/string.hpp"
#include "src/str/token.hpp"
// basis - radial
#include "src/nnp/basis_radial.hpp"

//==== using statements ====

using math::constant::PI;
using math::constant::RadPI;
using math::constant::Rad2;

//*****************************************
// PhiRN - radial function names
//*****************************************

PhiRN PhiRN::read(const char* str){
	if(std::strcmp(str,"GAUSSIAN")==0) return PhiRN::GAUSSIAN;
	else if(std::strcmp(str,"TANH")==0) return PhiRN::TANH;
	else if(std::strcmp(str,"SOFTPLUS")==0) return PhiRN::SOFTPLUS;
	else if(std::strcmp(str,"LOGCOSH")==0) return PhiRN::LOGCOSH;
	else if(std::strcmp(str,"SWISH")==0) return PhiRN::SWISH;
	else if(std::strcmp(str,"MISH")==0) return PhiRN::MISH;
	else return PhiRN::UNKNOWN;
}

const char* PhiRN::name(const PhiRN& phiRN){
	switch(phiRN){
		case PhiRN::GAUSSIAN: return "GAUSSIAN";
		case PhiRN::TANH: return "TANH";
		case PhiRN::SOFTPLUS: return "SOFTPLUS";
		case PhiRN::LOGCOSH: return "LOGCOSH";
		case PhiRN::SWISH: return "SWISH";
		case PhiRN::MISH: return "MISH";
		default: return "UNKNOWN";
	}
}

std::ostream& operator<<(std::ostream& out, const PhiRN& phiRN){
	switch(phiRN){
		case PhiRN::GAUSSIAN: out<<"GAUSSIAN"; break;
		case PhiRN::TANH: out<<"TANH"; break;
		case PhiRN::SOFTPLUS: out<<"SOFTPLUS"; break;
		case PhiRN::LOGCOSH: out<<"LOGCOSH"; break;
		case PhiRN::SWISH: out<<"SWISH"; break;
		case PhiRN::MISH: out<<"MISH"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

//*****************************************
// BasisR - radial basis
//*****************************************

//==== constructors/destructors ====

/**
* constructor
*/
BasisR::BasisR(double rc, cutoff::Name cutname, cutoff::Norm cutnorm, int size, PhiRN phiRN):Basis(rc,cutname,cutnorm,size){
	if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"BasisR(rc,cutoff::Name,cutoff::Norm,int,PhiRN):\n";
	if(phiRN==PhiRN::UNKNOWN) throw std::invalid_argument("BasisR(rc,cutoff::Name,cutoff::Norm,int,PhiRN): invalid radial function type");
	else phiRN_=phiRN;
	resize(size);
}

/**
* destructor
*/
BasisR::~BasisR(){
	clear();
}


//==== operators ====

/**
* print basis
* @param out - the output stream
* @param basis - the basis to print
* @return the output stream
*/
std::ostream& operator<<(std::ostream& out, const BasisR& basis){
	out<<"Basis "<<basis.cutnorm_<<" "<<basis.cutname_<<" "<<basis.rc_<<" "<<basis.phiRN_<<" "<<basis.size_;
	for(int i=0; i<basis.size(); ++i){
		out<<"\n\t"<<basis.rs_[i]<<" "<<basis.eta_[i]<<" ";
	}
	return out;
}

//==== reading/writing ====

/**
* write basis to file
* @param writer - file pointer
* @param basis - the basis to be written
*/
void BasisR::write(FILE* writer,const BasisR& basis){
	if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"BasisR::write(FILE*,const BasisR&):\n";
	const char* str_tcut=cutoff::Name::name(basis.cutname());
	const char* str_norm=cutoff::Norm::name(basis.cutnorm());
	const char* str_phirn=PhiRN::name(basis.phiRN());
	fprintf(writer,"BasisR %s %s %f %s %i\n",str_norm,str_tcut,basis.rc(),str_phirn,basis.size());
	for(int i=0; i<basis.size(); ++i){
		fprintf(writer,"\t%f %f\n",basis.rs(i),basis.eta(i));
	}
}

/**
* read basis from file
* @param writer - file pointer
* @param basis - the basis to be read
*/
void BasisR::read(FILE* reader, BasisR& basis){
	if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"BasisR::read(FILE*, BasisR&):\n";
	//local variables
	char* input=new char[string::M];
	//read header
	Token token(fgets(input,string::M,reader),string::WS); token.next();
	const cutoff::Norm cutnorm=cutoff::Norm::read(token.next().c_str());
	const cutoff::Name cutname=cutoff::Name::read(token.next().c_str());
	const double rc=std::atof(token.next().c_str());
	PhiRN phiRN=PhiRN::read(token.next().c_str());
	const int size=std::atoi(token.next().c_str());
	//initialize
	basis=BasisR(rc,cutname,cutnorm,size,phiRN);
	//read parameters
	for(int i=0; i<basis.size(); ++i){
		token.read(fgets(input,string::M,reader),string::WS);
		basis.rs(i)=std::atof(token.next().c_str());
		basis.eta(i)=std::atof(token.next().c_str());
	}
	//free local variables
	delete[] input;
}

//==== member functions ====

/**
* clear basis
*/
void BasisR::clear(){
	if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"BasisR::clear():\n";
	Basis::clear();
	phiRN_=PhiRN::UNKNOWN;
	rs_.clear();
	eta_.clear();
}

/**
* resize symmetry function and parameter arrays
* @param size - the total number of symmetry functions/parameters
*/
void BasisR::resize(int size){
	if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"BasisR::resize(int):\n";
	Basis::resize(size);
	if(size_>0){
		rs_.resize(size_);
		eta_.resize(size_);
	}
}

/**
* compute symmetry function - function value
* @param dr - the distance between the central atom and a neighboring atom
*/
double BasisR::symmf(double dr, double eta, double rs)const{
	if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"BasisR::symmf(double,double,double)const:\n";
	const double cut=cut_func(dr)*norm_;
	double rval=0;
	switch(phiRN_){
		case PhiRN::GAUSSIAN:{
			rval=exp(-eta*(dr-rs)*(dr-rs))*cut;
		} break;
		case PhiRN::TANH:{
			rval=0.5*(tanh(-eta*(dr-rs))+1.0)*cut;
		} break;
		case PhiRN::SOFTPLUS:{
			const double arg=-eta*(dr-rs);
			if(arg>1.0){
				rval=(arg+math::special::logp1(exp(-arg)))*cut;
			} else {
				rval=math::special::logp1(exp(arg))*cut;
			}
		} break;
		case PhiRN::LOGCOSH:{
			const double arg=-eta*(dr-rs);
			if(arg>1.0){
				rval=(arg+0.5*math::special::logp1(exp(-2.0*arg)))*cut;
			} else {
				rval=0.5*math::special::logp1(exp(2.0*arg))*cut;
			}
		} break;
		case PhiRN::SWISH:{
			const double arg=-eta*(dr-rs);
			if(arg>0.0){
				const double fexp=exp(-arg);
				rval=arg/(1.0+fexp)*cut;
			} else if(arg<0.0){
				const double fexp=exp(arg);
				rval=arg*fexp/(1.0+fexp)*cut;
			}
		} break;
		case PhiRN::MISH:{
			const double arg=-eta*(dr-rs);
			if(arg>0.0){
				const double expf=exp(-arg);
				rval=arg*(2.0*expf+1.0)/(2.0*expf*(expf+1.0)+1.0)*cut;
			} else if(arg<0.0){
				const double expf=exp(arg);
				rval=arg*expf*(expf+2.0)/(expf*(expf+2.0)+2.0)*cut;
			}
		} break;
		default:
			throw std::invalid_argument("BasisR::symm(double): Invalid symmetry function.");
		break;
	}
	return rval;
}

/**
* compute symmetry function - derivative value
* @param dr - the distance between the central atom and a neighboring atom
*/
double BasisR::symmd(double dr, double eta, double rs)const{
	if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"BasisR::symmd(double,double,double)const:\n";
	double rval=0;
	const double cut=cut_func(dr);
	const double gcut=cut_grad(dr);
	switch(phiRN_){
		case PhiRN::GAUSSIAN:{
			rval=exp(-eta*(dr-rs)*(dr-rs))*(-2.0*eta*(dr-rs)*cut+gcut);
		} break;
		case PhiRN::TANH:{
			const double tanhf=tanh(-eta*(dr-rs));
			rval=0.5*(-eta*(1.0-tanhf*tanhf)*cut+(1.0+tanhf)*gcut);
		} break;
		case PhiRN::SOFTPLUS:{
			const double arg=-eta*(dr-rs);
			if(arg>1.0){
				const double expf=exp(-arg);
				rval=(-eta*1.0/(1.0+expf)*cut+(arg+math::special::logp1(expf))*gcut);
			} else {
				const double expf=exp(arg);
				rval=(-eta*expf/(1.0+expf)*cut+math::special::logp1(expf)*gcut);
			}
		} break;
		case PhiRN::LOGCOSH:{
			const double arg=-eta*(dr-rs);
			if(arg>1.0){
				const double fexp=exp(-2.0*arg);
				rval=-eta/(1.0+fexp)*cut+(arg+0.5*math::special::logp1(fexp))*gcut;
			} else {
				const double fexp=exp(2.0*arg);
				rval=-eta*fexp/(1.0+fexp)*cut+0.5*math::special::logp1(fexp)*gcut;
			}
		} break;
		case PhiRN::SWISH:{
			const double arg=-eta*(dr-rs);
			if(arg>0.0){
				const double fexp=exp(-arg);
				const double den=1.0/(1.0+fexp);
				rval=(-eta*(1.0+fexp*(1.0+arg))*den*cut+arg*gcut)*den;
			} else {
				const double fexp=exp(arg);
				const double den=1.0/(1.0+fexp);
				rval=(-eta*(1.0+arg+fexp)*den*cut+arg*gcut)*fexp*den;
			}
		} break;
		case PhiRN::MISH:{
			const double arg=-eta*(dr-rs);
			if(arg>0.0){
				const double expf=exp(-arg);
				const double den=1.0/(2.0*expf*(expf+1.0)+1.0);
				rval=(-eta*(1.0+expf*(4.0+expf*(4.0*arg+6.0+expf*4.0*(arg+1.0))))*den*cut+arg*(2.0*expf+1.0)*gcut)*den;
			} else {
				const double expf=exp(arg);
				const double den=1.0/(expf*(expf+2.0)+2.0);
				rval=(-eta*(expf*(expf*(expf+4.0)+4.0*arg+6.0)+4.0*(arg+1.0))*den*cut+arg*(expf+2.0)*gcut)*den*expf;
			}
		} break;
		default:
			throw std::invalid_argument("BasisR::symm(double): Invalid symmetry function.");
		break;
	}
	return rval;
}

/**
* compute symmetry functions
* @param dr - the distance between the central atom and a neighboring atom
*/
void BasisR::symm(double dr){
	if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"BasisR::symm(double):\n";
	const double cut=cut_func(dr)*norm_;
	switch(phiRN_){
		case PhiRN::GAUSSIAN:{
			for(int i=0; i<size_; ++i){
				symm_[i]=exp(-eta_[i]*(dr-rs_[i])*(dr-rs_[i]))*cut;
			}
		} break;
		case PhiRN::TANH:{
			for(int i=0; i<size_; ++i){
				symm_[i]=0.5*(tanh(-eta_[i]*(dr-rs_[i]))+1.0)*cut;
			}
		} break;
		case PhiRN::SOFTPLUS:{
			for(int i=0; i<size_; ++i){
				const double arg=-eta_[i]*(dr-rs_[i]);
				if(arg>0.0){
					symm_[i]=(arg+math::special::logp1(exp(-arg)))*cut;
				} else {
					symm_[i]=math::special::logp1(exp(arg))*cut;
				}
			}
		} break;
		case PhiRN::LOGCOSH:{
			for(int i=0; i<size_; ++i){
				const double arg=-eta_[i]*(dr-rs_[i]);
				if(arg>0.0){
					symm_[i]=(arg+0.5*math::special::logp1(exp(-2.0*arg)))*cut;
				} else {
					symm_[i]=0.5*math::special::logp1(exp(2.0*arg))*cut;
				}
			}
		} break;
		case PhiRN::SWISH:{
			for(int i=0; i<size_; ++i){
				const double arg=-eta_[i]*(dr-rs_[i]);
				if(arg>0.0){
					const double fexp=exp(-arg);
					symm_[i]=arg/(1.0+fexp)*cut;
				} else if(arg<0.0){
					const double fexp=exp(arg);
					symm_[i]=arg*fexp/(1.0+fexp)*cut;
				}
			}
		} break;
		case PhiRN::MISH:{
			for(int i=0; i<size_; ++i){
				const double arg=-eta_[i]*(dr-rs_[i]);
				if(arg>0.0){
					const double expf=exp(-arg);
					symm_[i]=arg*(2.0*expf+1.0)/(2.0*expf*(expf+1.0)+1.0)*cut;
				} else if(arg<0.0){
					const double expf=exp(arg);
					symm_[i]=arg*expf*(expf+2.0)/(expf*(expf+2.0)+2.0)*cut;
				}
			}
		} break;
		default:
			throw std::invalid_argument("BasisR::symm(double): Invalid symmetry function.");
		break;
	}
}

/**
* compute force
* @param dr - the distance between the central atom and a neighboring atom
* @param dEdG - gradient of energy w.r.t. the inputs
*/
double BasisR::force(double dr, const double* dEdG)const{
	double amp=0;
	const double cut=cut_func(dr);
	const double gcut=cut_grad(dr);
	switch(phiRN_){
		case PhiRN::GAUSSIAN:{
			for(int i=0; i<size_; ++i){
				amp-=dEdG[i]*exp(-eta_[i]*(dr-rs_[i])*(dr-rs_[i]))*(-2.0*eta_[i]*(dr-rs_[i])*cut+gcut);
			}
		} break;
		case PhiRN::TANH:{
			for(int i=0; i<size_; ++i){
				const double tanhf=tanh(-eta_[i]*(dr-rs_[i]));
				amp-=dEdG[i]*0.5*(-eta_[i]*(1.0-tanhf*tanhf)*cut+(1.0+tanhf)*gcut);
			}
		} break;
		case PhiRN::SOFTPLUS:{
			for(int i=0; i<size_; ++i){
				const double arg=-eta_[i]*(dr-rs_[i]);
				if(arg>0.0){
					const double expf=exp(-arg);
					amp-=dEdG[i]*(-eta_[i]*1.0/(1.0+expf)*cut+(arg+math::special::logp1(expf))*gcut);
				} else {
					const double expf=exp(arg);
					amp-=dEdG[i]*(-eta_[i]*expf/(1.0+expf)*cut+math::special::logp1(expf)*gcut);
				}
			}
		} break;
		case PhiRN::LOGCOSH:{
			for(int i=0; i<size_; ++i){
				const double arg=-eta_[i]*(dr-rs_[i]);
				if(arg>0.0){
					const double fexp=exp(-2.0*arg);
					amp-=dEdG[i]*(-eta_[i]/(1.0+fexp)*cut+(arg+0.5*math::special::logp1(fexp))*gcut);
				} else {
					const double fexp=exp(2.0*arg);
					amp-=dEdG[i]*(-eta_[i]*fexp/(1.0+fexp)*cut+0.5*math::special::logp1(fexp)*gcut);
				}
			}
		} break;
		case PhiRN::SWISH:{
			for(int i=0; i<size_; ++i){
				const double arg=-eta_[i]*(dr-rs_[i]);
				if(arg>0){
					const double fexp=exp(-arg);
					const double den=1.0/(1.0+fexp);
					amp-=dEdG[i]*(-eta_[i]*(1.0+fexp*(1.0+arg))*den*cut+arg*gcut)*den;
				} else {
					const double fexp=exp(arg);
					const double den=1.0/(1.0+fexp);
					amp-=dEdG[i]*(-eta_[i]*(1.0+arg+fexp)*den*cut+arg*gcut)*fexp*den;
				}
			}
		} break;
		case PhiRN::MISH:{
			for(int i=0; i<size_; ++i){
				const double arg=-eta_[i]*(dr-rs_[i]);
				if(arg>0.0){
					const double expf=exp(-arg);
					const double den=1.0/(2.0*expf*(expf+1.0)+1.0);
					amp-=dEdG[i]*(-eta_[i]*(1.0+expf*(4.0+expf*(4.0*arg+6.0+expf*4.0*(arg+1.0))))*den*cut+arg*(2.0*expf+1.0)*gcut)*den;
				} else {
					const double expf=exp(arg);
					const double den=1.0/(expf*(expf+2.0)+2.0);
					amp-=dEdG[i]*(-eta_[i]*(expf*(expf*(expf+4.0)+4.0*arg+6.0)+4.0*(arg+1.0))*den*cut+arg*(expf+2.0)*gcut)*den*expf;
				}
			}
		} break;
		default:
			throw std::invalid_argument("BasisR::symm(double): Invalid symmetry function.");
		break;
	}
	return amp*norm_;
}

void BasisR::compute(double dr, double* symm, double* amp)const{
	const double cut=cut_func(dr);
	const double gcut=cut_grad(dr);
	switch(phiRN_){
		case PhiRN::GAUSSIAN:{
			for(int i=0; i<size_; ++i){
				const double expf=exp(-eta_[i]*(dr-rs_[i])*(dr-rs_[i]));
				symm[i]+=expf*cut;
				amp[i]-=norm_*expf*(-2.0*eta_[i]*(dr-rs_[i])*cut+gcut);
			}
		} break;
		case PhiRN::TANH:{
			for(int i=0; i<size_; ++i){
				const double tanhf=tanh(-eta_[i]*(dr-rs_[i]));
				symm[i]+=0.5*(tanhf+1.0)*cut;
				amp[i]-=norm_*0.5*(-eta_[i]*(1.0-tanhf*tanhf)*cut+(1.0+tanhf)*gcut);
			}
		} break;
		case PhiRN::SOFTPLUS:{
			for(int i=0; i<size_; ++i){
				const double arg=-eta_[i]*(dr-rs_[i]);
				if(arg>1.0){
					const double expf=exp(-arg);
					const double logexpf=math::special::logp1(expf);
					symm[i]+=(arg+logexpf)*cut;
					amp[i]-=norm_*(-eta_[i]*1.0/(1.0+expf)*cut+(arg+logexpf)*gcut);
				} else {
					const double expf=exp(arg);
					const double logexpf=math::special::logp1(expf);
					symm[i]+=logexpf*cut;
					amp[i]-=norm_*(-eta_[i]*expf/(1.0+expf)*cut+logexpf*gcut);
				}
			}
		} break;
		case PhiRN::LOGCOSH:{
			for(int i=0; i<size_; ++i){
				const double arg=-eta_[i]*(dr-rs_[i]);
				if(arg>0.0){
					const double fexp=exp(-2.0*arg);
					const double logexpf=math::special::logp1(fexp);
					symm[i]+=(arg+0.5*logexpf)*cut;
					amp[i]-=norm_*(-eta_[i]*cut/(1.0+fexp)+(arg+0.5*logexpf)*gcut);
				} else {
					const double fexp=exp(2.0*arg);
					const double logexpf=math::special::logp1(fexp);
					symm[i]+=0.5*logexpf*cut;
					amp[i]-=norm_*(-eta_[i]*cut*fexp/(1.0+fexp)+0.5*logexpf*gcut);
				}
			}
		} break;
		default:
			throw std::invalid_argument("BasisR::symm(double): Invalid symmetry function.");
		break;
	}
}

//==== serialization ====

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const BasisR& obj){
		if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"nbytes(const BasisR&):\n";
		int size=0;
		size+=sizeof(obj.cutname());//cutoff name
		size+=sizeof(obj.cutnorm());//cutoff normalization
		size+=sizeof(obj.rc());//cutoff radius
		size+=sizeof(obj.size());//number of symmetry functions
		size+=sizeof(obj.phiRN());//name of symmetry functions
		size+=sizeof(double)*obj.size();//rs
		size+=sizeof(double)*obj.size();//eta
		return size;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const BasisR& obj, char* arr){
		if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"pack(const BasisR&,char*):\n";
		int pos=0;
		std::memcpy(arr+pos,&obj.cutname(),sizeof(obj.cutname())); pos+=sizeof(obj.cutname());//cutoff name
		std::memcpy(arr+pos,&obj.cutnorm(),sizeof(obj.cutnorm())); pos+=sizeof(obj.cutnorm());//cutoff normalization
		std::memcpy(arr+pos,&obj.rc(),sizeof(obj.rc())); pos+=sizeof(obj.rc());//cutoff radius
		std::memcpy(arr+pos,&obj.size(),sizeof(obj.size())); pos+=sizeof(obj.size());//number of symmetry functions
		std::memcpy(arr+pos,&obj.phiRN(),sizeof(obj.phiRN())); pos+=sizeof(obj.phiRN());//name of symmetry functions
		if(obj.size()>0){
			std::memcpy(arr+pos,obj.rs().data(),obj.size()*sizeof(double)); pos+=obj.size()*sizeof(double);
			std::memcpy(arr+pos,obj.eta().data(),obj.size()*sizeof(double)); pos+=obj.size()*sizeof(double);
		}
		return pos;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(BasisR& obj, const char* arr){
		if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"unpack(BasisR&,const char*):\n";
		int pos=0;
		cutoff::Name cutname=cutoff::Name::UNKNOWN;
		cutoff::Norm cutnorm=cutoff::Norm::UNKNOWN;
		PhiRN phiRN=PhiRN::UNKNOWN;
		double rc=0;
		int size=0;
		std::memcpy(&cutname,arr+pos,sizeof(cutname)); pos+=sizeof(cutname);
		std::memcpy(&cutnorm,arr+pos,sizeof(cutnorm)); pos+=sizeof(cutnorm);
		std::memcpy(&rc,arr+pos,sizeof(rc)); pos+=sizeof(rc);
		std::memcpy(&size,arr+pos,sizeof(size)); pos+=sizeof(size);
		std::memcpy(&phiRN,arr+pos,sizeof(PhiRN)); pos+=sizeof(PhiRN);
		obj=BasisR(rc,cutname,cutnorm,size,phiRN);
		if(obj.size()>0){
			std::memcpy(obj.rs().data(),arr+pos,obj.size()*sizeof(double)); pos+=obj.size()*sizeof(double);
			std::memcpy(obj.eta().data(),arr+pos,obj.size()*sizeof(double)); pos+=obj.size()*sizeof(double);
		}
		return pos;
	}
	
}
