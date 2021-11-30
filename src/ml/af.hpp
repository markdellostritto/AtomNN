#pragma once
#ifndef AF_HPP
#define AF_HPP

namespace AF{
	
struct Name{
	//type
	enum type{
		UNKNOWN=0,
		LINEAR=1,
		SIGMOID=2,
		TANH=3,
		ISRU=4,
		ARCTAN=5,
		SOFTSIGN=6,
		RELU=7,
		ELU=8,
		GELU=9,
		SOFTPLUS=10,
		SWISH=11,
		MISH=12
	};
	static type read(const char* str);
	static const char* name(const Name::type& tf);
};
std::ostream& operator<<(std::ostream& out, const Transfer::type& tf);

class Base{
protected:
	Name::type name_;
public:
	AF(){}
	~AF(){}
	
	//operators
	friend std::ostream& operator<<(std::ostream& out, const Base& base);
	virtual void operator()(VecXd& f, VecXd& d)=0;
	
	//member functions
	virtual Base* clone()=0;
	
	//static functions
	static Basis* read(const char* str, Base* base);
	std::string write(
};

class LINEAR{
public:
	LINEAR():name_(Name::LINEAR){}
	void operator()(VecXd& f, VecXd& d);
	std::unique_ptr<Base> clone(){return std::make_unique<LINEAR>(*this);}
};

class SIGMOID{
public:
	SIGMOID():name_(Name::SIGMOID){}
	void operator()(VecXd& f, VecXd& d);
	std::unique_ptr<Base> clone(){return std::make_unique<SIGMOID>(*this);}
};

class TANH{
public:
	TANH():name_(Name::TANH){}
	void operator()(VecXd& f, VecXd& d);
	std::unique_ptr<Base> clone(){return std::make_unique<TANH>(*this);}
};

class ARCTAN{
public:
	ARCTAN():name_(Name::ARCTAN){}
	void operator()(VecXd& f, VecXd& d);
	std::unique_ptr<Base> clone(){return std::make_unique<ARCTAN>(*this);}
};

class ISRU{
public:
	ISRU():name_(Name::ISRU){}
	void operator()(VecXd& f, VecXd& d);
	std::unique_ptr<Base> clone(){return std::make_unique<ISRU>(*this);}
};

class SOFTSIGN{
public:
	SOFTSIGN():name_(Name::SOFTSIGN){}
	void operator()(VecXd& f, VecXd& d);
	std::unique_ptr<Base> clone(){return std::make_unique<SOFTSIGN>(*this);}
};

class RELU{
public:
	RELU():name_(Name::RELU){}
	void operator()(VecXd& f, VecXd& d);
	std::unique_ptr<Base> clone(){return std::make_unique<RELU>(*this);}
};

class ELU{
public:
	ELU():name_(Name::ELU){}
	void operator()(VecXd& f, VecXd& d);
	std::unique_ptr<Base> clone(){return std::make_unique<ELU>(*this);}
};

class GELU{
public:
	GELU():name_(Name::GELU){}
	void operator()(VecXd& f, VecXd& d);
	std::unique_ptr<Base> clone(){return std::make_unique<GELU>(*this);}
};

class SOFTPLUS{
private:
	SOFTPLUS():name_(Name::SOFTPLUS),s_(1.0),si_(1.0),o_(0.0){}
	SOFTPLUS(double s, double o):name_(Name::SOFTPLUS),s_(s),si_(1.0/s),o_(o){}
	double s_,si_,o_;//scale,offset
public:
	void operator()(VecXd& f, VecXd& d);
	std::unique_ptr<Base> clone(){return std::make_unique<SOFTPLUS>(*this);}
};

class SWISH{
public:
	SWISH():name_(Name::SWISH){}
	void operator()(VecXd& f, VecXd& d);
	std::unique_ptr<Base> clone(){return std::make_unique<SWISH>(*this);}
};

class MISH{
public:
	MISH():name_(Name::MISH){}
	void operator()(VecXd& f, VecXd& d);
	std::unique_ptr<Base> clone(){return std::make_unique<MISH>(*this);}
};

}

#endif
