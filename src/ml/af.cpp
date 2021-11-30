#include "src/ml/af.hpp"

namespace AF{

//==== type ====

std::ostream& operator<<(std::ostream& out, const Name::type& tf){
	switch(tf){
		case Name::LINEAR: out<<"LINEAR"; break;
		case Name::SIGMOID: out<<"SIGMOID"; break;
		case Name::TANH: out<<"TANH"; break;
		case Name::ISRU: out<<"ISRU"; break;
		case Name::ARCTAN: out<<"ARCTAN"; break;
		case Name::SOFTSIGN: out<<"SOFTSIGN"; break;
		case Name::RELU: out<<"RELU"; break;
		case Name::ELU: out<<"ELU"; break;
		case Name::GELU: out<<"GELU"; break;
		case Name::SOFTPLUS: out<<"SOFTPLUS"; break;
		case Name::SWISH: out<<"SWISH"; break;
		case Name::MISH: out<<"MISH"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

const char* Name::name(const Name::type& tf){
	switch(tf){
		case Name::LINEAR: return "LINEAR";
		case Name::SIGMOID: return "SIGMOID";
		case Name::TANH: return "TANH";
		case Name::ISRU: return "ISRU";
		case Name::ARCTAN: return "ARCTAN";
		case Name::SOFTSIGN: return "SOFTSIGN";
		case Name::RELU: return "RELU";
		case Name::ELU: return "ELU";
		case Name::GELU: return "GELU";
		case Name::SOFTPLUS: return "SOFTPLUS";
		case Name::SWISH: return "SWISH";
		case Name::MISH: return "MISH";
		default: return "UNKNOWN";
	}
}

Name::type Name::read(const char* str){
	if(std::strcmp(str,"LINEAR")==0) return Name::LINEAR;
	else if(std::strcmp(str,"SIGMOID")==0) return Name::SIGMOID;
	else if(std::strcmp(str,"TANH")==0) return Name::TANH;
	else if(std::strcmp(str,"ISRU")==0) return Name::ISRU;
	else if(std::strcmp(str,"ARCTAN")==0) return Name::ARCTAN;
	else if(std::strcmp(str,"SOFTSIGN")==0) return Name::SOFTSIGN;
	else if(std::strcmp(str,"RELU")==0) return Name::RELU;
	else if(std::strcmp(str,"ELU")==0) return Name::ELU;
	else if(std::strcmp(str,"GELU")==0) return Name::GELU;
	else if(std::strcmp(str,"SOFTPLUS")==0) return Name::SOFTPLUS;
	else if(std::strcmp(str,"SWISH")==0) return Name::SWISH;
	else if(std::strcmp(str,"MISH")==0) return Name::MISH;
	else return Name::UNKNOWN;
}

//==== functions ====

std::ostream& operator<<(std::ostream& out, const Base& base){
	return out<<base.name_;
}

Basis* read(const char* str, Base* base){
	std::vector<std::string> strlist;
	string::trim_right(input,string::COMMENT);//trim comments
	if(string::split(input,string::WS,strlist)==0) throw std::invalid_argument("Invalid AF string");
	string::to_upper(strlist.at(0));//convert tag to upper case
	//read name
	Name::type name=Name::read(strlist.at(0).c_str());
	//make functor
	if(base!=NULL) delete base;
	switch(name){
		case Name::LINEAR:{
			base=new LINEAR(); 
		}break;
		case Name::SIGMOID:{
			base=new SIGMOID();
		}break;
		case Name::TANH:{
			base=new TANH();
		}break;
		case Name::ISRU:{
			base=new ISRU();
		}break;
		case Name::ARCTAN:{
			base=new ARCTAN();
		}break;
		case Name::SOFTSIGN:{
			base=new SOFTSIGN();
		}break;
		case Name::RELU:{
			base=new RELU();
		}break;
		case Name::ELU:{
			base=new ELU();
		}break;
		case Name::GELU:{
			base=new GELU();
		}break;
		case Name::SOFTPLUS:{
			const double s=std::atof(strlist.at(1).c_str());
			const double o=std::atof(strlist.at(2).c_str());
			base=new SOFTPLUS(s,o);
		}break;
		default: base=NULL; break;
	}
	return base;
}

void LINEAR::operator()(VecXd& f, VecXd& d){
	for(int i=0; i<d.size(); ++i) d[i]=1.0;
}

void SIGMOID::operator()(VecXd& f, VecXd& d){
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

void TANH::operator()(VecXd& f, VecXd& d){
	const int size=f.size();
	for(int i=0; i<size; ++i) f[i]=tanh(f[i]);
	for(int i=0; i<size; ++i) d[i]=1.0-f[i]*f[i];
}

void ISRU::operator()(VecXd& f, VecXd& d){
	const int size=f.size();
	for(int i=0; i<size; ++i){
		const double isr=1.0/sqrt(1.0+f[i]*f[i]);
		f[i]=f[i]*isr;
		d[i]=isr*isr*isr;
	}
}

void ARCTAN::operator()(VecXd& f, VecXd& d){
	const int size=f.size();
	for(int i=0; i<size; ++i){
		d[i]=(2.0/math::constant::PI)/(1.0+f[i]*f[i]);
		f[i]=(2.0/math::constant::PI)*atan(f[i]);
	}
}

void SOFTSIGN::operator()(VecXd& f, VecXd& d){
	const int size=f.size();
	for(int i=0; i<size; ++i){
		const double inv=1.0/(1.0+fabs(f[i]));
		f[i]=f[i]*inv;
		d[i]=inv*inv;
	}
}

void RELU::operator()(VecXd& f, VecXd& d){
	const int size=f.size();
	for(int i=0; i<size; ++i){
		if(f[i]>0){
			d[i]=1.0;
		} else {
			f[i]=0.0;
			d[i]=0.0;
		}
	}
}

void ELU::operator()(VecXd& f, VecXd& d){
	const int size=f.size();
	for(int i=0; i<size; ++i){
		if(f[i]>0){
			d[i]=1.0;
		} else {
			const double expf=exp(f[i]);
			f[i]=expf-1.0;
			d[i]=expf;
		}
	}
}

void GELU::operator()(VecXd& f, VecXd& d){
	const int size=f.size();
	const double rad2pii=1.0/(math::constant::Rad2*math::constant::RadPI);
	for(int i=0; i<size; ++i){
		const double erff=0.5*(1.0+erf(f[i]/math::constant::Rad2));
		d[i]=erff+f[i]*exp(-0.5*f[i]*f[i])*rad2pii;
		f[i]*=erff;
	}
}

void SOFTPLUS::operator()(VecXd& f, VecXd& d){
	const int size=f.size();
	for(int i=0; i<size; ++i){
		if(f[i]>=1.0){
			//f(x)=x+ln(1+exp(-x))-ln(2)
			const double expf=exp(-f[i]*s_);
			f[i]=si_*(f[i]+math::special::logp1(expf))+o_;
			d[i]=si_/(1.0+expf);
		} else {
			//f(x)=ln(1+exp(x))-ln(2)
			const double expf=exp(f[i]*s_);
			f[i]=si_*math::special::logp1(expf)+o_;
			d[i]=si_*expf/(expf+1.0);
		}
	}
}

}
