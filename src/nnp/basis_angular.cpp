// c libraries
#include <cstring>
#include <cstdio>
// c++ libraries
#include <iostream>
#include <vector>
// math
#include "src/math/special.hpp"
// string
#include "src/str/string.hpp"
#include "src/str/token.hpp"
// basis - angular
#include "src/nnp/basis_angular.hpp"

//==== using statements ====

using math::constant::PI;
using math::constant::RadPI;

//*****************************************
// PhiAN - angular function names
//*****************************************

PhiAN PhiAN::read(const char* str){
	if(std::strcmp(str,"GAUSS")==0) return PhiAN::GAUSS;
	else if(std::strcmp(str,"IPOWP")==0) return PhiAN::IPOWP;
	else if(std::strcmp(str,"IPOWS")==0) return PhiAN::IPOWS;
	else if(std::strcmp(str,"SECHP")==0) return PhiAN::SECHP;
	else if(std::strcmp(str,"SECHS")==0) return PhiAN::SECHS;
	else return PhiAN::UNKNOWN;
}

const char* PhiAN::name(const PhiAN& phiAN){
	switch(phiAN){
		case PhiAN::GAUSS: return "GAUSS";
		case PhiAN::IPOWP: return "IPOWP";
		case PhiAN::IPOWS: return "IPOWS";
		case PhiAN::SECHP: return "SECHP";
		case PhiAN::SECHS: return "SECHS";
		default: return "UNKNOWN";
	}
}

std::ostream& operator<<(std::ostream& out, const PhiAN& phiAN){
	switch(phiAN){
		case PhiAN::GAUSS: out<<"GAUSS"; break;
		case PhiAN::IPOWP: out<<"IPOWP"; break;
		case PhiAN::IPOWS: out<<"IPOWS"; break;
		case PhiAN::SECHP: out<<"SECHP"; break;
		case PhiAN::SECHS: out<<"SECHS"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

//==== constructors/destructors ====

/**
* constructor
*/
BasisA::BasisA(double rc, cutoff::Name cutname, cutoff::Norm cutnorm, int size, PhiAN phiAN):Basis(rc,cutname,cutnorm,size){
	if(phiAN==PhiAN::UNKNOWN) throw std::invalid_argument("BasisA(rc,cutoff::Name,cutoff::Norm,int,PhiAN): invalid angular function type");
	else phiAN_=phiAN;
	resize(size);
}

/**
* destructor
*/
BasisA::~BasisA(){
	clear();
}

//==== operators ====

/**
* print basis
* @param out - the output stream
* @param basis - the basis to print
* @return the output stream
*/
std::ostream& operator<<(std::ostream& out, const BasisA& basis){
	out<<"BasisA "<<basis.cutnorm_<<" "<<basis.cutname_<<" "<<basis.rc_<<" "<<basis.phiAN_<<" "<<basis.size_<<" "<<basis.alpha_;
	for(int i=0; i<basis.size(); ++i){
		out<<"\n\t"<<basis.eta_[i]<<" "<<basis.zeta_[i]<<" "<<" "<<basis.lambda_[i]<<" ";
	}
	return out;
}

//==== reading/writing ====

/**
* write basis to file
* @param writer - file pointer
* @param basis - the basis to be written
*/
void BasisA::write(FILE* writer, const BasisA& basis){
	if(BASIS_ANGULAR_PRINT_FUNC>0) std::cout<<"BasisA::write(FILE*):\n";
	const char* str_tcut=cutoff::Name::name(basis.cutname());
	const char* str_norm=cutoff::Norm::name(basis.cutnorm());
	const char* str_phian=PhiAN::name(basis.phiAN());
	fprintf(writer,"BasisA %s %s %f %s %i %i\n",str_norm,str_tcut,basis.rc(),str_phian,basis.size(),basis.alpha());
	for(int i=0; i<basis.size(); ++i){
		fprintf(writer,"\t%f %f %i\n",basis.eta(i),basis.zeta(i),basis.lambda(i));
	}
}

/**
* read basis from file
* @param writer - file pointer
* @param basis - the basis to be read
*/
void BasisA::read(FILE* reader, BasisA& basis){
	if(BASIS_ANGULAR_PRINT_FUNC>0) std::cout<<"BasisA::read(FILE*, BasisA&):\n";
	//local variables
	char* input=new char[string::M];
	//read header
	Token token(fgets(input,string::M,reader),string::WS); token.next();
	const cutoff::Norm cutnorm=cutoff::Norm::read(token.next().c_str());
	const cutoff::Name cutname=cutoff::Name::read(token.next().c_str());
	const double rc=std::atof(token.next().c_str());
	const PhiAN phiAN=PhiAN::read(token.next().c_str());
	const int size=std::atoi(token.next().c_str());
	const int alpha=std::atoi(token.next().c_str());
	//initialize
	basis=BasisA(rc,cutname,cutnorm,size,phiAN);
	//read parameters
	for(int i=0; i<basis.size(); ++i){
		token.read(fgets(input,string::M,reader),string::WS);
		basis.eta(i)=std::atof(token.next().c_str());
		basis.zeta(i)=std::atof(token.next().c_str());
		basis.lambda(i)=std::atoi(token.next().c_str());
	}
	basis.alpha()=alpha;
	basis.init();
	//free local variables
	delete[] input;
}

//==== member functions ====

/**
* clear basis
*/
void BasisA::clear(){
	if(BASIS_ANGULAR_PRINT_FUNC>0) std::cout<<"BasisA::clear():\n";
	Basis::clear();
	phiAN_=PhiAN::UNKNOWN;
	eta_.clear();
	ietap_.clear();
	zeta_.clear();
	lambda_.clear();
}	

void BasisA::resize(int size){
	if(BASIS_ANGULAR_PRINT_FUNC>0) std::cout<<"BasisA::resize(int):\n";
	Basis::resize(size);
	if(size_>0){
		eta_.resize(size_);
		ietap_.resize(size_);
		zeta_.resize(size_);
		lambda_.resize(size_);
		phif_.resize(size_);
		etaf_.resize(3,std::vector<double>(size_));
	}
}

void BasisA::init(){
	if(BASIS_ANGULAR_PRINT_FUNC>0) std::cout<<"BasisA::init():\n";
	for(int i=0; i<size_; ++i){
		ietap_[i]=math::special::powint(1.0/eta_[i],alpha_);
	}
}

/**
* compute symmetry functions
* @param cos - the cosine of the triple
* @param dr - the triple distances: dr={rij,rik,rjk} with i at the vertex
*/
double BasisA::symmf(double cos, const double dr[3], double eta, double zeta, int lambda, int alpha)const{
	if(BASIS_ANGULAR_PRINT_FUNC>0) std::cout<<"BasisA::symm(double,const double*):\n";
	const double c[2]={
		cut_func(dr[0]),//cut(rij)
		cut_func(dr[1])//cut(rik)
		//cut_func(dr[2]) //cut(rjk)
	};
	double rval=0;
	const double cprod=c[0]*c[1]*norm_;
	const double ietap=1.0/math::special::powint(eta,alpha);
	const double rijp=math::special::powint(dr[0],alpha);
	const double rikp=math::special::powint(dr[1],alpha);
	switch(phiAN_){
		case PhiAN::GAUSS:{
			rval=pow(fabs(0.5*(1.0+lambda*cos)),zeta)*cprod*exp(-ietap*(rijp+rikp));
		} break;
		case PhiAN::IPOWP:{
			rval=pow(fabs(0.5*(1.0+lambda*cos)),zeta)*cprod*1.0/((1.0+rijp*ietap)*(1.0+rikp*ietap));
		} break;
		case PhiAN::IPOWS:{
			rval=pow(fabs(0.5*(1.0+lambda*cos)),zeta)*cprod*1.0/(1.0+ietap*(rijp+rikp));
		} break;
		case PhiAN::SECHP:{
			rval=pow(fabs(0.5*(1.0+lambda*cos)),zeta)*cprod*math::special::sech(ietap*rijp)*math::special::sech(ietap*rikp);
		} break;
		case PhiAN::SECHS:{
			rval=pow(fabs(0.5*(1.0+lambda*cos)),zeta)*cprod*math::special::sech(ietap*(rijp+rikp));
		} break;
		default:
			throw std::invalid_argument("BasisA::symm(double): Invalid symmetry function.");
		break;
	}
	return rval;
}

/**
* compute force
* @param phi - stores angular gradients
* @param eta - stores radial gradients
* @param cos - the cosine of the triple
* @param dr - the triple distances: r={rij,rik,rjk} with i at the vertex
* @param dEdG - gradient of energy w.r.t. the inputs
*/
void BasisA::symmd(double& fphi, double* feta, double cos, const double dr[3], double eta, double zeta, int lambda, int alpha)const{
	//compute cutoffs
	const double c[2]={
		cut_func(dr[0]),//cut(rij)
		cut_func(dr[1])//cut(rik)
		//cut_func(dr[2]) //cut(rjk)
	};
	const double g[2]={
		cut_grad(dr[0]),//cut'(rij)
		cut_grad(dr[1])//cut'(rik)
		//cut_grad(dr[2]) //cut'(rjk)
	};
	//compute phi, eta
	fphi=0;
	feta[0]=0;
	feta[1]=0;
	feta[2]=0;
	const double cprod=c[0]*c[1];
	const double ietap=1.0/math::special::powint(eta,alpha);
	const double rijpa=math::special::powint(dr[0],alpha);
	const double rikpa=math::special::powint(dr[1],alpha);
	const double dij=alpha*rijpa/dr[0]*c[0];
	const double dik=alpha*rikpa/dr[1]*c[1];
	switch(phiAN_){
		case PhiAN::GAUSS:{
			//compute distance values
			const double expf=exp(-ietap*(rijpa+rikpa));
			/*
			//compute angular values
			const double cw=fabs(0.5*(1.0+lambda*cos));
			const double cwp=pow(cw,zeta-1.0);
			const double fangle=cw*cwp;
			const double gangle=0.5*cwp*zeta*lambda;
			//compute phi
			fphi-=expf*cprod*gangle;
			//compute eta
			feta[0]-=fangle*(-dij*ietap+g[0])*c[1]*expf;
			feta[1]-=fangle*(-dik*ietap+g[1])*c[0]*expf;
			//feta[2]-=0;
			*/
			//compute angular values
			const double cw=fabs(0.5*(1.0+lambda*cos));
			const double gangle=pow(cw,zeta-1.0)*expf;
			const double fangle=cw*gangle;
			//compute phi
			fphi-=0.5*zeta*lambda*cprod*gangle;
			//compute eta
			feta[0]-=fangle*(-dij*ietap+g[0])*c[1];
			feta[1]-=fangle*(-dik*ietap+g[1])*c[0];
			//feta[2]-=0;
			
		} break;
		case PhiAN::IPOWP:{
			//compute distance values
			const double denij=1.0/(1.0+rijpa*ietap);
			const double denik=1.0/(1.0+rikpa*ietap);
			const double denp=denij*denik;
			/*
			//compute angular values
			const double cw=fabs(0.5*(1.0+lambda*cos));
			const double cwp=pow(cw,zeta-1.0);
			const double fangle=cw*cwp;
			const double gangle=0.5*cwp*zeta*lambda;
			//compute phi
			fphi-=0.5*zeta*lambda*cprod*gangle;
			//compute eta
			feta[0]-=fangle*denp*(-dij*ietap*denij+g[0])*c[1];
			feta[1]-=fangle*denp*(-dik*ietap*denik+g[1])*c[0];
			//feta[2]-=0;
			*/
			//compute angular values
			const double cw=fabs(0.5*(1.0+lambda*cos));
			const double gangle=pow(cw,zeta-1.0)*denp;
			const double fangle=cw*gangle;
			//compute phi
			fphi-=0.5*zeta*lambda*cprod*gangle;
			//compute eta
			feta[0]-=fangle*(-dij*ietap*denij+g[0])*c[1];
			feta[1]-=fangle*(-dik*ietap*denik+g[1])*c[0];
			//feta[2]-=0;
		} break;
		case PhiAN::IPOWS:{
			//compute distance values
			const double den=1.0/(1.0+ietap*(rijpa+rikpa));
			/*
			//compute angular values
			const double cw=fabs(0.5*(1.0+lambda*cos));
			const double cwp=pow(cw,zeta-1.0);
			const double fangle=cw*cwp*dEdG[i];
			const double gangle=0.5*cwp*zeta*lambda;
			//compute phi
			fphi-=0.5*zeta*lambda*cprod*gangle;
			//compute eta
			feta[0]-=fangle*den*(-dij*ietap*den+g[0])*c[1];
			feta[1]-=fangle*den*(-dik*ietap*den+g[1])*c[0];
			//feta[2]-=0;
			*/
			//compute angular values
			const double cw=fabs(0.5*(1.0+lambda*cos));
			const double gangle=pow(cw,zeta-1.0)*den;
			const double fangle=cw*gangle;
			//compute phi
			fphi-=0.5*zeta*lambda*cprod*gangle;
			//compute eta
			feta[0]-=fangle*(-dij*ietap*den+g[0])*c[1];
			feta[1]-=fangle*(-dik*ietap*den+g[1])*c[0];
			//feta[2]-=0;
		} break;
		case PhiAN::SECHP:{
			//compute distance values
			const double sechij=math::special::sech(rijpa*ietap);
			const double sechik=math::special::sech(rikpa*ietap);
			const double tanhij=std::sqrt(1.0-sechij*sechij);
			const double tanhik=std::sqrt(1.0-sechik*sechik);
			const double sechp=sechij*sechik;
			/*
			//compute angular values
			const double cw=fabs(0.5*(1.0+lambda*cos));
			const double cwp=pow(cw,zeta-1.0);
			const double fangle=cw*cwp;
			const double gangle=0.5*cwp*zeta*lambda;
			//compute phi
			fphi-=0.5*zeta*lambda*cprod*gangle;
			//compute eta
			feta[0]-=fangle*sechp*(-dij*ietap*tanhij+g[0])*c[1];
			feta[1]-=fangle*sechp*(-dik*ietap*tanhik+g[1])*c[0];
			//feta[2]-=0;
			*/
			//compute angular values
			const double cw=fabs(0.5*(1.0+lambda*cos));
			const double gangle=pow(cw,zeta-1.0)*sechp;
			const double fangle=cw*gangle;
			//compute phi
			fphi-=0.5*zeta*lambda*cprod*gangle;
			//compute eta
			feta[0]-=fangle*(-dij*ietap*tanhij+g[0])*c[1];
			feta[1]-=fangle*(-dik*ietap*tanhik+g[1])*c[0];
			//feta[2]-=0;
		} break;
		case PhiAN::SECHS:{
			//compute distance values
			const double fsech=math::special::sech(ietap*(rijpa+rikpa));
			const double ftanh=std::sqrt(1.0-fsech*fsech);
			/*
			//compute angular values
			const double cw=fabs(0.5*(1.0+lambda*cos));
			const double cwp=pow(cw,zeta-1.0);
			const double fangle=cw*cwp;
			const double gangle=0.5*cwp*zeta*lambda;
			//compute phi
			fphi-=0.5*zeta*lambda*cprod*gangle;
			//compute eta
			feta[0]-=fangle*fsech*(-dij*ietap*ftanh+g[0])*c[1];
			feta[1]-=fangle*fsech*(-dik*ietap*ftanh+g[1])*c[0];
			//feta[2]-=0;
			*/
			//compute angular values
			const double cw=fabs(0.5*(1.0+lambda*cos));
			const double gangle=pow(cw,zeta-1.0)*fsech;
			const double fangle=cw*gangle;
			//compute phi
			fphi-=0.5*zeta*lambda*cprod*gangle;
			//compute eta
			feta[0]-=fangle*(-dij*ietap*ftanh+g[0])*c[1];
			feta[1]-=fangle*(-dik*ietap*ftanh+g[1])*c[0];
			//feta[2]-=0;
		} break;
		default:
			throw std::invalid_argument("BasisA::symm(double): Invalid symmetry function.");
		break;
	}
	//normalize
	fphi*=-1*norm_;
	feta[0]*=-1*norm_;
	feta[1]*=-1*norm_;
	feta[2]*=-1*norm_;
}

/**
* compute symmetry functions
* @param cos - the cosine of the triple
* @param dr - the triple distances: dr={rij,rik,rjk} with i at the vertex
*/
void BasisA::symm(double cos, const double dr[3]){
	if(BASIS_ANGULAR_PRINT_FUNC>0) std::cout<<"BasisA::symm(double,const double*):\n";
	const double c[2]={
		cut_func(dr[0]),//cut(rij)
		cut_func(dr[1])//cut(rik)
		//cut_func(dr[2]) //cut(rjk)
	};
	const double cprod=c[0]*c[1]*norm_;
	const double rijp=math::special::powint(dr[0],alpha_);
	const double rikp=math::special::powint(dr[1],alpha_);
	switch(phiAN_){
		case PhiAN::GAUSS:{
			for(int i=0; i<size_; ++i){
				symm_[i]=pow(fabs(0.5*(1.0+lambda_[i]*cos)),zeta_[i])*cprod*exp(-ietap_[i]*(rijp+rikp));
			}
		} break;
		case PhiAN::IPOWP:{
			for(int i=0; i<size_; ++i){
				symm_[i]=pow(fabs(0.5*(1.0+lambda_[i]*cos)),zeta_[i])*cprod*1.0/((1.0+ietap_[i]*rijp)*(1.0+ietap_[i]*rikp));
			}
		} break;
		case PhiAN::IPOWS:{
			for(int i=0; i<size_; ++i){
				symm_[i]=pow(fabs(0.5*(1.0+lambda_[i]*cos)),zeta_[i])*cprod*1.0/(1.0+ietap_[i]*(rijp+rikp));
			}
		} break;
		case PhiAN::SECHP:{
			for(int i=0; i<size_; ++i){
				symm_[i]=pow(fabs(0.5*(1.0+lambda_[i]*cos)),zeta_[i])*cprod*math::special::sech(ietap_[i]*rijp)*math::special::sech(ietap_[i]*rikp);
			}
		} break;
		case PhiAN::SECHS:{
			for(int i=0; i<size_; ++i){
				symm_[i]=pow(fabs(0.5*(1.0+lambda_[i]*cos)),zeta_[i])*cprod*math::special::sech(ietap_[i]*(rijp+rikp));
			}
		} break;
		default:
			throw std::invalid_argument("BasisA::symm(double): Invalid symmetry function.");
		break;
	}
}

/**
* compute force
* @param phi - stores angular gradients
* @param eta - stores radial gradients
* @param cos - the cosine of the triple
* @param dr - the triple distances: r={rij,rik,rjk} with i at the vertex
* @param dEdG - gradient of energy w.r.t. the inputs
*/
void BasisA::force(double& phi, double* eta, double cos, const double dr[3], const double* dEdG)const{
	//compute cutoffs
	const double c[2]={
		cut_func(dr[0]),//cut(rij)
		cut_func(dr[1])//cut(rik)
		//cut_func(dr[2]) //cut(rjk)
	};
	const double g[2]={
		cut_grad(dr[0]),//cut'(rij)
		cut_grad(dr[1])//cut'(rik)
		//cut_grad(dr[2]) //cut'(rjk)
	};
	//compute phi, eta
	phi=0;
	eta[0]=0;
	eta[1]=0;
	eta[2]=0;
	const double cprod=c[0]*c[1];
	const double rijp=math::special::powint(dr[0],alpha_);
	const double rikp=math::special::powint(dr[1],alpha_);
	const double dij=alpha_*rijp/dr[0]*c[0];
	const double dik=alpha_*rikp/dr[1]*c[1];
	switch(phiAN_){
		case PhiAN::GAUSS:{
			for(int i=0; i<size_; ++i){
				//compute distance values
				const double expf=exp(-ietap_[i]*(rijp+rikp));
				/*
				//compute angular values
				const double cw=fabs(0.5*(1.0+lambda_[i]*cos));
				const double cwp=pow(cw,zeta_[i]-1.0);
				const double fangle=cw*cwp*dEdG[i];
				const double gangle=0.5*cwp*zeta_[i]*lambda_[i];
				//compute phi
				phi-=dEdG[i]*expf*cprod*gangle;
				//compute eta
				eta[0]-=fangle*(-dij*ietap_[i]+g[0])*c[1]*expf;
				eta[1]-=fangle*(-dik*ietap_[i]+g[1])*c[0]*expf;
				//eta[2]-=0;
				*/
				//compute angular values
				const double cw=fabs(0.5*(1.0+lambda_[i]*cos));
				const double gangle=pow(cw,zeta_[i]-1.0)*dEdG[i]*expf;
				const double fangle=cw*gangle;
				//compute phi
				phi-=0.5*zeta_[i]*lambda_[i]*cprod*gangle;
				//compute eta
				eta[0]-=fangle*(-dij*ietap_[i]+g[0])*c[1];
				eta[1]-=fangle*(-dik*ietap_[i]+g[1])*c[0];
				//eta[2]-=0;
			}
		} break;
		case PhiAN::IPOWP:{
			for(int i=0; i<size_; ++i){
				//compute distance values
				const double denij=1.0/(1.0+ietap_[i]*rijp);
				const double denik=1.0/(1.0+ietap_[i]*rikp);
				const double denp=denij*denik;
				/*
				//compute angular values
				const double cw=fabs(0.5*(1.0+lambda_[i]*cos));
				const double cwp=pow(cw,zeta_[i]-1.0);
				const double fangle=cw*cwp*dEdG[i];
				const double gangle=0.5*cwp*zeta_[i]*lambda_[i];
				//compute phi
				phi-=dEdG[i]*denp*cprod*gangle;
				//compute eta
				eta[0]-=fangle*(-dij*ietap_[i]*denij+g[0])*c[1]*denp;
				eta[1]-=fangle*(-dik*ietap_[i]*denik+g[1])*c[0]*denp;
				//eta[2]-=0;
				*/
				//compute angular values
				const double cw=fabs(0.5*(1.0+lambda_[i]*cos));
				const double gangle=pow(cw,zeta_[i]-1.0)*dEdG[i]*denp;
				const double fangle=cw*gangle;
				//compute phi
				phi-=0.5*zeta_[i]*lambda_[i]*cprod*gangle;
				//compute eta
				eta[0]-=fangle*(-dij*ietap_[i]*denij+g[0])*c[1];
				eta[1]-=fangle*(-dik*ietap_[i]*denik+g[1])*c[0];
				//eta[2]-=0;
			}
		} break;
		case PhiAN::IPOWS:{
			for(int i=0; i<size_; ++i){
				//compute distance values
				const double den=1.0/(1.0+ietap_[i]*(rijp+rikp));
				/*
				//compute angular values
				const double cw=fabs(0.5*(1.0+lambda_[i]*cos));
				const double cwp=pow(cw,zeta_[i]-1.0);
				const double fangle=cw*cwp*dEdG[i];
				const double gangle=0.5*cwp*zeta_[i]*lambda_[i];
				//compute phi
				phi-=dEdG[i]*den*cprod*gangle;
				//compute eta
				eta[0]-=fangle*(-dij*ietap_[i]*den+g[0])*c[1]*den;
				eta[1]-=fangle*(-dik*ietap_[i]*den+g[1])*c[0]*den;
				//eta[2]-=0;
				*/
				//compute angular values
				const double cw=fabs(0.5*(1.0+lambda_[i]*cos));
				const double gangle=pow(cw,zeta_[i]-1.0)*dEdG[i]*den;
				const double fangle=cw*gangle;
				//compute phi
				phi-=0.5*zeta_[i]*lambda_[i]*cprod*gangle;
				//compute eta
				eta[0]-=fangle*(-dij*ietap_[i]*den+g[0])*c[1];
				eta[1]-=fangle*(-dik*ietap_[i]*den+g[1])*c[0];
				//eta[2]-=0;
			}
		} break;
		case PhiAN::SECHP:{
			for(int i=0; i<size_; ++i){
				//compute distance values
				const double sechij=math::special::sech(ietap_[i]*rijp);
				const double sechik=math::special::sech(ietap_[i]*rikp);
				const double tanhij=std::sqrt(1.0-sechij*sechij);
				const double tanhik=std::sqrt(1.0-sechik*sechik);
				const double sechp=sechij*sechik;
				/*
				//compute angular values
				const double cw=fabs(0.5*(1.0+lambda_[i]*cos));
				const double cwp=pow(cw,zeta_[i]-1.0);
				const double fangle=cw*cwp*dEdG[i];
				const double gangle=0.5*cwp*zeta_[i]*lambda_[i];
				//compute phi
				phi-=dEdG[i]*sechp*cprod*gangle;
				//compute eta
				eta[0]-=fangle*(-dij*ietap_[i]*tanhij+g[0])*c[1]*sechp;
				eta[1]-=fangle*(-dik*ietap_[i]*tanhik+g[1])*c[0]*sechp;
				//eta[2]-=0;
				*/
				//compute angular values
				const double cw=fabs(0.5*(1.0+lambda_[i]*cos));
				const double gangle=pow(cw,zeta_[i]-1.0)*dEdG[i]*sechp;
				const double fangle=cw*gangle;
				//compute phi
				phi-=0.5*zeta_[i]*lambda_[i]*cprod*gangle;
				//compute eta
				eta[0]-=fangle*(-dij*ietap_[i]*tanhij+g[0])*c[1];
				eta[1]-=fangle*(-dik*ietap_[i]*tanhik+g[1])*c[0];
				//eta[2]-=0;
			}
		} break;
		case PhiAN::SECHS:{
			for(int i=0; i<size_; ++i){
				//compute distance values
				const double fsech=math::special::sech(ietap_[i]*(rijp+rikp));
				const double ftanh=std::sqrt(1.0-fsech*fsech);
				/*
				//compute angular values
				const double cw=fabs(0.5*(1.0+lambda_[i]*cos));
				const double cwp=pow(cw,zeta_[i]-1.0);
				const double fangle=cw*cwp*dEdG[i];
				const double gangle=0.5*cwp*zeta_[i]*lambda_[i];
				//compute phi
				phi-=dEdG[i]*fsech*cprod*gangle;
				//compute eta
				eta[0]-=fangle*(-dij*ietap_[i]*ftanh+g[0])*c[1]*fsech;
				eta[1]-=fangle*(-dik*ietap_[i]*ftanh+g[1])*c[0]*fsech;
				//eta[2]-=0;
				*/
				//compute angular values
				const double cw=fabs(0.5*(1.0+lambda_[i]*cos));
				const double gangle=pow(cw,zeta_[i]-1.0)*dEdG[i]*fsech;
				const double fangle=cw*gangle;
				//compute phi
				phi-=0.5*zeta_[i]*lambda_[i]*cprod*gangle;
				//compute eta
				eta[0]-=fangle*(-dij*ietap_[i]*ftanh+g[0])*c[1];
				eta[1]-=fangle*(-dik*ietap_[i]*ftanh+g[1])*c[0];
				//eta[2]-=0;
			}
		} break;
		default:
			throw std::invalid_argument("BasisA::force(double&,double*,double,const double[3],const double*)const: Invalid symmetry function.");
		break;
	}
	//normalize
	phi*=norm_;
	eta[0]*=norm_;
	eta[1]*=norm_;
	eta[2]*=norm_;
}

void BasisA::forcep(double cos, const double dr[3]){
	//compute cutoffs
	const double c[2]={
		cut_func(dr[0]),//cut(rij)
		cut_func(dr[1])//cut(rik)
		//cut_func(dr[2]) //cut(rjk)
	};
	const double g[2]={
		cut_grad(dr[0]),//cut'(rij)
		cut_grad(dr[1])//cut'(rik)
		//cut_grad(dr[2]) //cut'(rjk)
	};	
	//compute phi, eta
	for(int i=0; i<size_; ++i) phif_[i]=0;
	for(int j=0; j<size_; ++j) etaf_[0][j]=0;
	for(int j=0; j<size_; ++j) etaf_[1][j]=0;
	for(int j=0; j<size_; ++j) etaf_[2][j]=0;
	const double cprod=c[0]*c[1];
	const double rijp=math::special::powint(dr[0],alpha_);
	const double rikp=math::special::powint(dr[1],alpha_);
	const double dij=alpha_*rijp/dr[0]*c[0];
	const double dik=alpha_*rikp/dr[1]*c[1];
	switch(phiAN_){
		case PhiAN::GAUSS:{
			for(int i=0; i<size_; ++i){
				//compute distance values
				const double expf=exp(-ietap_[i]*(rijp+rikp));
				//compute angular values
				const double cw=fabs(0.5*(1.0+lambda_[i]*cos));
				const double cwp=pow(cw,zeta_[i]-1.0);
				const double fangle=cw*cwp;
				const double gangle=0.5*cwp*zeta_[i]*lambda_[i];
				//compute phi
				phif_[i]=expf*cprod*gangle*norm_;
				//compute eta
				etaf_[0][i]=fangle*(-dij*ietap_[i]+g[0])*c[1]*expf*norm_;
				etaf_[1][i]=fangle*(-dik*ietap_[i]+g[1])*c[0]*expf*norm_;
				//etaf_[2][i]=0;
			}
		} break;
		case PhiAN::IPOWP:{
			for(int i=0; i<size_; ++i){
				//compute distance values
				const double denij=1.0/(1.0+ietap_[i]*rijp);
				const double denik=1.0/(1.0+ietap_[i]*rikp);
				const double denp=denij*denik;
				//compute angular values
				const double cw=fabs(0.5*(1.0+lambda_[i]*cos));
				const double cwp=pow(cw,zeta_[i]-1.0);
				const double fangle=cw*cwp;
				const double gangle=0.5*cwp*zeta_[i]*lambda_[i];
				//compute phi
				phif_[i]=denp*cprod*gangle*norm_;
				//compute eta
				etaf_[0][i]=fangle*(-dij*ietap_[i]*denij+g[0])*c[1]*denp*norm_;
				etaf_[1][i]=fangle*(-dik*ietap_[i]*denik+g[1])*c[0]*denp*norm_;
				//etaf_[2][i]=0;
			}
		} break;
		case PhiAN::IPOWS:{
			for(int i=0; i<size_; ++i){
				//compute distance values
				const double den=1.0/(1.0+ietap_[i]*(rijp+rikp));
				//compute angular values
				const double cw=fabs(0.5*(1.0+lambda_[i]*cos));
				const double cwp=pow(cw,zeta_[i]-1.0);
				const double fangle=cw*cwp;
				const double gangle=0.5*cwp*zeta_[i]*lambda_[i];
				//compute phi
				phif_[i]=den*cprod*gangle*norm_;
				//compute eta
				etaf_[0][i]=fangle*(-dij*ietap_[i]*den+g[0])*c[1]*den*norm_;
				etaf_[1][i]=fangle*(-dik*ietap_[i]*den+g[1])*c[0]*den*norm_;
				//etaf_[2][i]=0;
			}
		} break;
		case PhiAN::SECHP:{
			for(int i=0; i<size_; ++i){
				//compute distance values
				const double sechij=math::special::sech(ietap_[i]*rijp);
				const double sechik=math::special::sech(ietap_[i]*rikp);
				const double tanhij=std::sqrt(1.0-sechij*sechij);
				const double tanhik=std::sqrt(1.0-sechik*sechik);
				const double sechp=sechij*sechik;
				//compute angular values
				const double cw=fabs(0.5*(1.0+lambda_[i]*cos));
				const double cwp=pow(cw,zeta_[i]-1.0);
				const double fangle=cw*cwp;
				const double gangle=0.5*cwp*zeta_[i]*lambda_[i];
				//compute phi
				phif_[i]=sechp*cprod*gangle*norm_;
				//compute eta
				etaf_[0][i]=fangle*(-dij*ietap_[i]*tanhij+g[0])*c[1]*sechp*norm_;
				etaf_[1][i]=fangle*(-dik*ietap_[i]*tanhik+g[1])*c[0]*sechp*norm_;
				//etaf_[2][i]=0;
			}
		} break;
		case PhiAN::SECHS:{
			for(int i=0; i<size_; ++i){
				//compute distance values
				const double fsech=math::special::sech(ietap_[i]*(rijp+rikp));
				const double ftanh=std::sqrt(1.0-fsech*fsech);
				//compute angular values
				const double cw=fabs(0.5*(1.0+lambda_[i]*cos));
				const double cwp=pow(cw,zeta_[i]-1.0);
				const double fangle=cw*cwp;
				const double gangle=0.5*cwp*zeta_[i]*lambda_[i];
				//compute phi
				phif_[i]=fsech*cprod*gangle*norm_;
				//compute eta
				etaf_[0][i]=fangle*(-dij*ietap_[i]*ftanh+g[0])*c[1]*fsech*norm_;
				etaf_[1][i]=fangle*(-dik*ietap_[i]*ftanh+g[1])*c[0]*fsech*norm_;
				//etaf_[2][i]=0;
			}
		} break;
		default:
			throw std::invalid_argument("BasisA::force(double&,double*,double,const double[3],const double*)const: Invalid symmetry function.");
		break;
	}
}

void BasisA::compute(double cos, const double dr[3], double* symm, double* phi, double* eta0, double* eta1, double* eta2)const{
	//compute cutoffs
	const double c[2]={
		cut_func(dr[0]),//cut(rij)
		cut_func(dr[1])//cut(rik)
		//cut_func(dr[2]) //cut(rjk)
	};
	const double g[2]={
		cut_grad(dr[0]),//cut'(rij)
		cut_grad(dr[1])//cut'(rik)
		//cut_grad(dr[2]) //cut'(rjk)
	};	
	//compute phi, eta
	const double cprod=c[0]*c[1];
	const double rijp=math::special::powint(dr[0],alpha_);
	const double rikp=math::special::powint(dr[1],alpha_);
	const double dij=alpha_*rijp/dr[0]*c[0];
	const double dik=alpha_*rikp/dr[1]*c[1];
	switch(phiAN_){
		case PhiAN::GAUSS:{
			for(int i=0; i<size_; ++i){
				//compute distance values
				const double expf=exp(-ietap_[i]*(rijp+rikp));
				//compute angular values
				const double cw=fabs(0.5*(1.0+lambda_[i]*cos));
				const double cwp=pow(cw,zeta_[i]-1.0);
				const double fangle=cw*cwp;
				const double gangle=0.5*cwp*zeta_[i]*lambda_[i];
				//compute symm
				symm[i]+=norm_*pow(fabs(0.5*(1.0+lambda_[i]*cos)),zeta_[i])*cprod*expf;
				//compute phi
				phi[i]-=norm_*expf*cprod*gangle;
				//compute eta
				eta0[i]-=norm_*fangle*(-dij*ietap_[i]+g[0])*c[1]*expf;
				eta1[i]-=norm_*fangle*(-dik*ietap_[i]+g[1])*c[0]*expf;
				//eta2[i]-=0;
			}
		} break;
		case PhiAN::IPOWP:{
			for(int i=0; i<size_; ++i){
				//compute distance values
				const double denij=1.0/(1.0+ietap_[i]*rijp);
				const double denik=1.0/(1.0+ietap_[i]*rikp);
				const double denp=denij*denik;
				//compute angular values
				const double cw=fabs(0.5*(1.0+lambda_[i]*cos));
				const double cwp=pow(cw,zeta_[i]-1.0);
				const double fangle=cw*cwp;
				const double gangle=0.5*cwp*zeta_[i]*lambda_[i];
				//compute symm
				symm[i]+=norm_*pow(fabs(0.5*(1.0+lambda_[i]*cos)),zeta_[i])*cprod*denp;
				//compute phi
				phi[i]-=norm_*denp*cprod*gangle;
				//compute eta
				eta0[i]-=norm_*fangle*(-dij*ietap_[i]*denij+g[0])*c[1]*denp;
				eta1[i]-=norm_*fangle*(-dik*ietap_[i]*denik+g[1])*c[0]*denp;
				//eta2[i]-=0;
			}
		} break;
		case PhiAN::IPOWS:{
			for(int i=0; i<size_; ++i){
				//compute distance values
				const double den=1.0/(1.0+ietap_[i]*(rijp+rikp));
				//compute angular values
				const double cw=fabs(0.5*(1.0+lambda_[i]*cos));
				const double cwp=pow(cw,zeta_[i]-1.0);
				const double fangle=cw*cwp;
				const double gangle=0.5*cwp*zeta_[i]*lambda_[i];
				//compute symm
				symm[i]+=norm_*pow(fabs(0.5*(1.0+lambda_[i]*cos)),zeta_[i])*cprod*den;
				//compute phi
				phi[i]-=norm_*den*cprod*gangle;
				//compute eta
				eta0[i]-=norm_*fangle*(-dij*ietap_[i]*den+g[0])*c[1]*den;
				eta1[i]-=norm_*fangle*(-dik*ietap_[i]*den+g[1])*c[0]*den;
				//eta2[i]-=0;
			}
		} break;
		case PhiAN::SECHP:{
			for(int i=0; i<size_; ++i){
				//compute distance values
				const double sechij=math::special::sech(ietap_[i]*rijp);
				const double sechik=math::special::sech(ietap_[i]*rikp);
				const double tanhij=std::sqrt(1.0-sechij*sechij);
				const double tanhik=std::sqrt(1.0-sechik*sechik);
				const double sechp=sechij*sechik;
				//compute angular values
				const double cw=fabs(0.5*(1.0+lambda_[i]*cos));
				const double cwp=pow(cw,zeta_[i]-1.0);
				const double fangle=cw*cwp;
				const double gangle=0.5*cwp*zeta_[i]*lambda_[i];
				//compute symm
				symm[i]+=norm_*pow(fabs(0.5*(1.0+lambda_[i]*cos)),zeta_[i])*cprod*sechp;
				//compute phi
				phi[i]-=norm_*sechp*cprod*gangle;
				//compute eta
				eta0[i]-=norm_*fangle*(-dij*ietap_[i]*tanhij+g[0])*c[1]*sechp;
				eta1[i]-=norm_*fangle*(-dik*ietap_[i]*tanhik+g[1])*c[0]*sechp;
				//eta2[i]-=0;
			}
		} break;
		case PhiAN::SECHS:{
			for(int i=0; i<size_; ++i){
				//compute distance values
				const double fsech=math::special::sech(ietap_[i]*(rijp+rikp));
				const double ftanh=std::sqrt(1.0-fsech*fsech);
				//compute angular values
				const double cw=fabs(0.5*(1.0+lambda_[i]*cos));
				const double cwp=pow(cw,zeta_[i]-1.0);
				const double fangle=cw*cwp;
				const double gangle=0.5*cwp*zeta_[i]*lambda_[i];
				//compute symm
				symm[i]+=norm_*pow(fabs(0.5*(1.0+lambda_[i]*cos)),zeta_[i])*cprod*fsech;
				//compute phi
				phi[i]-=norm_*fsech*cprod*gangle;
				//compute eta
				eta0[i]-=norm_*fangle*(-dij*ietap_[i]*ftanh+g[0])*c[1]*fsech;
				eta1[i]-=norm_*fangle*(-dik*ietap_[i]*ftanh+g[1])*c[0]*fsech;
				//eta2[i]-=0;
			}
		} break;
		default:
			throw std::invalid_argument("BasisA::symm(double): Invalid symmetry function.");
		break;
	}
}

//==== serialization ====

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const BasisA& obj){
		if(BASIS_ANGULAR_PRINT_FUNC>0) std::cout<<"nbytes(const BasisA&):\n";
		int size=0;
		size+=sizeof(obj.cutname());//cutoff name
		size+=sizeof(obj.cutnorm());//cutoff normalization
		size+=sizeof(obj.rc());//cutoff radius
		size+=sizeof(obj.size());//number of symmetry functions
		size+=sizeof(obj.phiAN());//name of symmetry functions
		size+=sizeof(obj.alpha());//power
		size+=sizeof(double)*obj.size();//eta
		size+=sizeof(double)*obj.size();//zeta
		size+=sizeof(int)*obj.size();//lambda
		return size;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const BasisA& obj, char* arr){
		if(BASIS_ANGULAR_PRINT_FUNC>0) std::cout<<"pack(const BasisA&,char*):\n";
		int pos=0;
		std::memcpy(arr+pos,&obj.cutname(),sizeof(obj.cutname())); pos+=sizeof(obj.cutname());
		std::memcpy(arr+pos,&obj.cutnorm(),sizeof(obj.cutnorm())); pos+=sizeof(obj.cutnorm());
		std::memcpy(arr+pos,&obj.rc(),sizeof(obj.rc())); pos+=sizeof(obj.rc());
		std::memcpy(arr+pos,&obj.size(),sizeof(obj.size())); pos+=sizeof(obj.size());
		std::memcpy(arr+pos,&obj.phiAN(),sizeof(obj.phiAN())); pos+=sizeof(obj.phiAN());
		std::memcpy(arr+pos,&obj.alpha(),sizeof(obj.alpha())); pos+=sizeof(obj.alpha());
		if(obj.size()>0){
			std::memcpy(arr+pos,obj.eta().data(),obj.size()*sizeof(double)); pos+=obj.size()*sizeof(double);
			std::memcpy(arr+pos,obj.zeta().data(),obj.size()*sizeof(double)); pos+=obj.size()*sizeof(double);
			std::memcpy(arr+pos,obj.lambda().data(),obj.size()*sizeof(int)); pos+=obj.size()*sizeof(int);
		}
		return pos;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(BasisA& obj, const char* arr){
		if(BASIS_ANGULAR_PRINT_FUNC>0) std::cout<<"unpack(BasisA&,const char*):\n";
		int pos=0;
		cutoff::Name cutname=cutoff::Name::UNKNOWN;
		cutoff::Norm cutnorm=cutoff::Norm::UNKNOWN;
		double rc=0;
		int size=0;
		int alpha=0;
		PhiAN phiAN=PhiAN::UNKNOWN;
		std::memcpy(&cutname,arr+pos,sizeof(cutname)); pos+=sizeof(cutname);
		std::memcpy(&cutnorm,arr+pos,sizeof(cutnorm)); pos+=sizeof(cutnorm);
		std::memcpy(&rc,arr+pos,sizeof(rc)); pos+=sizeof(rc);
		std::memcpy(&size,arr+pos,sizeof(size)); pos+=sizeof(size);
		std::memcpy(&phiAN,arr+pos,sizeof(PhiAN)); pos+=sizeof(PhiAN);
		std::memcpy(&alpha,arr+pos,sizeof(alpha)); pos+=sizeof(alpha);
		obj=BasisA(rc,cutname,cutnorm,size,phiAN);
		if(size>0){
			std::memcpy(obj.eta().data(),arr+pos,obj.size()*sizeof(double)); pos+=size*sizeof(double);
			std::memcpy(obj.zeta().data(),arr+pos,obj.size()*sizeof(double)); pos+=size*sizeof(double);
			std::memcpy(obj.lambda().data(),arr+pos,obj.size()*sizeof(int)); pos+=size*sizeof(int);
		}
		obj.alpha()=alpha;
		obj.init();
		return pos;
	}
	
}
