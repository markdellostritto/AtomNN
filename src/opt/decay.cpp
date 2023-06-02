// c
#include <cstring>
// c++
#include <iostream>
// str
#include "src/str/string.hpp"
#include "src/str/token.hpp"
// math
#include "src/math/const.hpp"
// opt
#include "src/opt/decay.hpp"

//***************************************************
// decay method
//***************************************************

namespace opt{
namespace decay{

//***************************************************
// decay name
//***************************************************

std::ostream& operator<<(std::ostream& out, const Name& name){
	switch(name){
		case Name::CONST: out<<"CONST"; break;
		case Name::EXP: out<<"EXP"; break;
		case Name::SQRT: out<<"SQRT"; break;
		case Name::INV: out<<"INV"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

Name Name::read(const char* str){
	if(std::strcmp(str,"CONST")==0) return Name::CONST;
	else if(std::strcmp(str,"EXP")==0) return Name::EXP;
	else if(std::strcmp(str,"SQRT")==0) return Name::SQRT;
	else if(std::strcmp(str,"INV")==0) return Name::INV;
	else return Name::UNKNOWN;
}

//***************************************************
// decay function - const
//***************************************************

//==== operators ====

std::ostream& operator<<(std::ostream& out, const Const& obj){
	return out<<obj.name();
}

//==== member functions ====
	
double Const::step(const Objective& obj){
	return obj.gamma();
}

//***************************************************
// decay function - exp
//***************************************************

//==== operators ====

std::ostream& operator<<(std::ostream& out, const Exp& obj){
	return out<<obj.name()<<" alpha "<<obj.alpha_;
}

//==== member functions ====
	
double Exp::step(const Objective& obj){
	if(OPT_DECAY_PRINT_FUNC>0) std::cout<<"opt::decay::Exp::step(const Objective&):\n";
	return obj.gamma()*(1.0-alpha_);
}

void Exp::read(Token& token){
	alpha_=std::atof(token.next().c_str());
	if(alpha_<=0) throw std::invalid_argument("opt::decay::Exp::read(Token&): invalid alpha.");
}

//***************************************************
// decay function - sqrt
//***************************************************

//==== operators ====

std::ostream& operator<<(std::ostream& out, const Sqrt& obj){
	return out<<obj.name()<<" alpha "<<obj.alpha_;
}

//==== member functions ====
	
double Sqrt::step(const Objective& obj){
	if(OPT_DECAY_PRINT_FUNC>0) std::cout<<"opt::decay::Sqrt::step(const Objective&):\n";
	return obj.gamma()*(1.0-0.5*alpha_/(1.0+alpha_*obj.step()));
}

void Sqrt::read(Token& token){
	alpha_=std::atof(token.next().c_str());
	if(alpha_<=0) throw std::invalid_argument("opt::decay::Sqrt::read(Token&): invalid alpha.");
}

//***************************************************
// decay function - inv
//***************************************************

//==== operators ====

std::ostream& operator<<(std::ostream& out, const Inv& obj){
	return out<<obj.name()<<" alpha "<<obj.alpha_;
}

//==== member functions ====
	
double Inv::step(const Objective& obj){
	if(OPT_DECAY_PRINT_FUNC>0) std::cout<<"opt::decay::Inv::step(const Objective&):\n";
	return obj.gamma()*(1.0-alpha_/(1.0+alpha_*obj.step()));
}

void Inv::read(Token& token){
	alpha_=std::atof(token.next().c_str());
	if(alpha_<=0) throw std::invalid_argument("opt::decay::Inv::read(Token&): invalid alpha.");
}

//***************************************************
// factory
//***************************************************

std::ostream& operator<<(std::ostream& out, const std::shared_ptr<Base>& obj){
	switch(obj->name()){
		case Name::CONST: out<<static_cast<Const&>(*obj); break;
		case Name::EXP: out<<static_cast<Exp&>(*obj); break;
		case Name::SQRT: out<<static_cast<Sqrt&>(*obj); break;
		case Name::INV: out<<static_cast<Inv&>(*obj); break;
		case Name::UNKNOWN: out<<"DECAY UNKNOWN"; break;
	}
	return out;
}

std::shared_ptr<Base>& make(std::shared_ptr<Base>& obj, Name name){
	switch(name){
		case Name::CONST: obj.reset(new Const()); break;
		case Name::EXP: obj.reset(new Exp()); break;
		case Name::SQRT: obj.reset(new Sqrt()); break;
		case Name::INV: obj.reset(new Inv()); break;
		case Name::UNKNOWN: throw std::invalid_argument("opt::decay::make(std::shared_ptr<Base>&,Name): Invalid decay name."); break;
	}
	return obj;
}

std::shared_ptr<Base>& read(std::shared_ptr<Base>& obj, Token& token){
	Name name=Name::read(string::to_upper(token.next()).c_str());
	make(obj,name);
	obj->read(token);
	return obj;
}

}
}

//**********************************************
// serialization
//**********************************************

namespace serialize{

//**********************************************
// byte measures
//**********************************************

template <> int nbytes(const std::shared_ptr<opt::decay::Base>& obj){
	if(OPT_DECAY_PRINT_FUNC>0) std::cout<<"serialize::nbytes(const std::shared_ptr<opt::decay::Base>&):\n";
	int size=0;
	size+=sizeof(bool);
	if(obj!=nullptr){
		size+=sizeof(opt::decay::Name);//name
		switch(obj->name()){
			case opt::decay::Name::CONST: size+=nbytes(static_cast<const opt::decay::Const&>(*obj)); break;
			case opt::decay::Name::EXP: size+=nbytes(static_cast<const opt::decay::Exp&>(*obj)); break;
			case opt::decay::Name::SQRT: size+=nbytes(static_cast<const opt::decay::Sqrt&>(*obj)); break;
			case opt::decay::Name::INV: size+=nbytes(static_cast<const opt::decay::Inv&>(*obj)); break;
			case opt::decay::Name::UNKNOWN: throw std::invalid_argument("serialize::nbytes(const std::shared_ptr<opt::decay::Base>&): Invalid decay name."); break;
		}
	}
	return size;
}
template <> int nbytes(const opt::decay::Const& obj){
	if(OPT_DECAY_PRINT_FUNC>0) std::cout<<"serialize::nbytes(const opt::decay::Const&):\n";
	int size=0;
	size+=sizeof(opt::decay::Name);//name
	return size;
}
template <> int nbytes(const opt::decay::Exp& obj){
	if(OPT_DECAY_PRINT_FUNC>0) std::cout<<"serialize::nbytes(const opt::decay::Exp&):\n";
	int size=0;
	size+=sizeof(opt::decay::Name);//name
	size+=sizeof(double);//alpha
	return size;
}
template <> int nbytes(const opt::decay::Sqrt& obj){
	if(OPT_DECAY_PRINT_FUNC>0) std::cout<<"serialize::nbytes(const opt::decay::Sqrt&):\n";
	int size=0;
	size+=sizeof(opt::decay::Name);//name
	size+=sizeof(double);//alpha
	return size;
}
template <> int nbytes(const opt::decay::Inv& obj){
	if(OPT_DECAY_PRINT_FUNC>0) std::cout<<"serialize::nbytes(const opt::decay::Inv&):\n";
	int size=0;
	size+=sizeof(opt::decay::Name);//name
	size+=sizeof(double);//alpha
	return size;
}

//**********************************************
// packing
//**********************************************

template <> int pack(const std::shared_ptr<opt::decay::Base>& obj, char* arr){
	if(OPT_DECAY_PRINT_FUNC>0) std::cout<<"serialize::pack(const std::shared_ptr<opt::decay::Base>&,char*):\n";
	int pos=0;
	const bool null=(obj==nullptr);
	std::memcpy(arr+pos,&null,sizeof(bool)); pos+=sizeof(bool);
	if(!null){
		std::memcpy(arr+pos,&obj->name(),sizeof(opt::decay::Name)); pos+=sizeof(opt::decay::Name);
		switch(obj->name()){
			case opt::decay::Name::CONST: pos+=pack(static_cast<const opt::decay::Const&>(*obj),arr+pos); break;
			case opt::decay::Name::EXP: pos+=pack(static_cast<const opt::decay::Exp&>(*obj),arr+pos); break;
			case opt::decay::Name::SQRT: pos+=pack(static_cast<const opt::decay::Sqrt&>(*obj),arr+pos); break;
			case opt::decay::Name::INV: pos+=pack(static_cast<const opt::decay::Inv&>(*obj),arr+pos); break;
			case opt::decay::Name::UNKNOWN: throw std::invalid_argument("serialize::pack(const std::shared_ptr<opt::decay::Base>&,char*): Invalid decay name."); break;
		}
	}
	return pos;
}
template <> int pack(const opt::decay::Const& obj, char* arr){
	if(OPT_DECAY_PRINT_FUNC>0) std::cout<<"serialize::pack(const opt::decay::Const&,char*):\n";
	int pos=0;
	std::memcpy(arr+pos,&obj.name(),sizeof(opt::decay::Name)); pos+=sizeof(opt::decay::Name);
	return pos;
}
template <> int pack(const opt::decay::Exp& obj, char* arr){
	if(OPT_DECAY_PRINT_FUNC>0) std::cout<<"serialize::pack(const opt::decay::Exp&,char*):\n";
	int pos=0;
	std::memcpy(arr+pos,&obj.name(),sizeof(opt::decay::Name)); pos+=sizeof(opt::decay::Name);
	std::memcpy(arr+pos,&obj.alpha(),sizeof(double)); pos+=sizeof(double);
	return pos;
}
template <> int pack(const opt::decay::Sqrt& obj, char* arr){
	if(OPT_DECAY_PRINT_FUNC>0) std::cout<<"serialize::pack(const opt::decay::Sqrt&,char*):\n";
	int pos=0;
	std::memcpy(arr+pos,&obj.name(),sizeof(opt::decay::Name)); pos+=sizeof(opt::decay::Name);
	std::memcpy(arr+pos,&obj.alpha(),sizeof(double)); pos+=sizeof(double);
	return pos;
}
template <> int pack(const opt::decay::Inv& obj, char* arr){
	if(OPT_DECAY_PRINT_FUNC>0) std::cout<<"serialize::pack(const opt::decay::Inv&,char*):\n";
	int pos=0;
	std::memcpy(arr+pos,&obj.name(),sizeof(opt::decay::Name)); pos+=sizeof(opt::decay::Name);
	std::memcpy(arr+pos,&obj.alpha(),sizeof(double)); pos+=sizeof(double);
	return pos;
}

//**********************************************
// unpacking
//**********************************************

template <> int unpack(std::shared_ptr<opt::decay::Base>& obj, const char* arr){
	if(OPT_DECAY_PRINT_FUNC>0) std::cout<<"serialize::unpack(std::shared_ptr<opt::decay::Base>&,const char*):\n";
	int pos=0;
	bool null=false;
	std::memcpy(&null,arr+pos,sizeof(bool)); pos+=sizeof(bool);
	if(!null){
		opt::decay::Name name=opt::decay::Name::UNKNOWN;
		std::memcpy(&name,arr+pos,sizeof(opt::decay::Name)); pos+=sizeof(opt::decay::Name);
		opt::decay::make(obj,name);
		switch(name){
			case opt::decay::Name::CONST: pos+=unpack(static_cast<opt::decay::Const&>(*obj),arr+pos); break;
			case opt::decay::Name::EXP: pos+=unpack(static_cast<opt::decay::Exp&>(*obj),arr+pos); break;
			case opt::decay::Name::SQRT: pos+=unpack(static_cast<opt::decay::Sqrt&>(*obj),arr+pos); break;
			case opt::decay::Name::INV: pos+=unpack(static_cast<opt::decay::Inv&>(*obj),arr+pos); break;
			case opt::decay::Name::UNKNOWN: throw std::invalid_argument("serialize::unpack(std::shared_ptr<opt::decay::Base>&,const char*): Invalid decay name."); break;
		}
	} else obj.reset();
	return pos;
}
template <> int unpack(opt::decay::Const& obj, const char* arr){
	if(OPT_DECAY_PRINT_FUNC>0) std::cout<<"serialize::unpack(opt::decay::Const&,const char*):\n";
	int pos=0; opt::decay::Name name;
	std::memcpy(&name,arr+pos,sizeof(opt::decay::Name)); pos+=sizeof(opt::decay::Name);
	if(name!=opt::decay::Name::CONST) throw std::invalid_argument("serialize::unpack(opt::decay::Const&,const char*): Invalid decay name.");
	return pos;
}
template <> int unpack(opt::decay::Exp& obj, const char* arr){
	if(OPT_DECAY_PRINT_FUNC>0) std::cout<<"serialize::unpack(opt::decay::Exp&,const char*):\n";
	int pos=0; opt::decay::Name name;
	std::memcpy(&name,arr+pos,sizeof(opt::decay::Name)); pos+=sizeof(opt::decay::Name);
	if(name!=opt::decay::Name::EXP) throw std::invalid_argument("serialize::unpack(opt::decay::Exp&,const char*): Invalid decay name.");
	std::memcpy(&obj.alpha(),arr+pos,sizeof(double)); pos+=sizeof(double);
	return pos;
}
template <> int unpack(opt::decay::Sqrt& obj, const char* arr){
	if(OPT_DECAY_PRINT_FUNC>0) std::cout<<"serialize::unpack(opt::decay::Sqrt&,const char*):\n";
	int pos=0; opt::decay::Name name;
	std::memcpy(&name,arr+pos,sizeof(opt::decay::Name)); pos+=sizeof(opt::decay::Name);
	if(name!=opt::decay::Name::SQRT) throw std::invalid_argument("serialize::unpack(opt::decay::Sqrt&,const char*): Invalid decay name.");
	std::memcpy(&obj.alpha(),arr+pos,sizeof(double)); pos+=sizeof(double);
	return pos;
}
template <> int unpack(opt::decay::Inv& obj, const char* arr){
	if(OPT_DECAY_PRINT_FUNC>0) std::cout<<"serialize::unpack(opt::decay::Inv&,const char*):\n";
	int pos=0; opt::decay::Name name;
	std::memcpy(&name,arr+pos,sizeof(opt::decay::Name)); pos+=sizeof(opt::decay::Name);
	if(name!=opt::decay::Name::INV) throw std::invalid_argument("serialize::unpack(opt::decay::Inv&,const char*): Invalid decay name.");
	std::memcpy(&obj.alpha(),arr+pos,sizeof(double)); pos+=sizeof(double);
	return pos;
}
	
}