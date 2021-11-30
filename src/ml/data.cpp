#include "src/ml/data.hpp"

void MLData::clear(){
	size_=0;
	dIn_=0;
	dOut_=0;
	in_.clear();
	out_.clear();
}

void MLData::resize(int dIn, int dOut){
	if(dIn<=0) throw std::invalid_argument("MLData::resize(int): invalid dimension - input.");
	if(dOut<=0) throw std::invalid_argument("MLData::resize(int): invalid dimension - output.");
	clear();
	dIn_=dIn;
	dOut_=dOut;
}

void MLData::resize(int size, int dIn, int dOut){
	if(size<=0) throw std::invalid_argument("MLData::resize(int): invalid size.");
	if(dIn<=0) throw std::invalid_argument("MLData::resize(int): invalid dimension - input.");
	if(dOut<=0) throw std::invalid_argument("MLData::resize(int): invalid dimension - output.");
	clear();
	size_=size;
	dIn_=dIn;
	dOut_=dOut;
	in_.resize(size,Eigen::VectorXd::Zero(dIn_));
	out_.resize(size,Eigen::VectorXd::Zero(dOut_));
}

void MLData::push(const Eigen::VectorXd& in, const Eigen::VectorXd& out){
	if(in.size()!=dIn_) throw std::invalid_argument("MLData::resize(int): invalid dimension - input.");
	if(out.size()!=dOut_) throw std::invalid_argument("MLData::resize(int): invalid dimension - output.");
	in_.push_back(in);
	out_.push_back(out);
	++size_;
}