#pragma once
#ifndef DATA_HPP
#define DATA_HPP

// c++ libaries
#include <vector>
// eigen
#include <Eigen/Dense>
// ann - util
# include "src/util/typedef.hpp"

class MLData{
private:
	int size_;
	int dIn_,dOut_;
	std::vector<VecXd> in_;
	std::vector<VecXd> out_;
public:
	//==== constructors/destructors ====
	MLData():size_(0),dIn_(0),dOut_(0){}
	MLData(int size, int dIn, int dOut){resize(size,dIn,dOut);}
	~MLData(){}
	
	//==== access ====
	const int& size()const{return size_;}
	const int& dIn()const{return dIn_;}
	const int& dOut()const{return dOut_;}
	VecXd& in(int n){return in_[n];}
	const VecXd& in(int n)const{return in_[n];}
	std::vector<VecXd>& in(){return in_;}
	const std::vector<VecXd>& in()const{return in_;}
	VecXd& out(int n){return out_[n];}
	const VecXd& out(int n)const{return out_[n];}
	std::vector<VecXd>& out(){return out_;}
	const std::vector<VecXd>& out()const{return out_;}
	
	//==== member functions ====
	void clear();
	void resize(int dIn, int dOut);
	void resize(int size, int dIn, int dOut);
	void push(const Eigen::VectorXd& in, const Eigen::VectorXd& out);
};

#endif