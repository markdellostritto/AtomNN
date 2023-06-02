#pragma once
#ifndef OBJECTIVE_HPP
#define OBJECTIVE_HPP

// c++
#include <iosfwd>
// eigen
#include <Eigen/Dense>
//serialization
#include "src/mem/serialize.hpp"
// opt
#include "src/opt/stop.hpp"
#include "src/opt/loss.hpp"

#ifndef OPT_OBJ_PRINT_FUNC
#define OPT_OBJ_PRINT_FUNC 0
#endif

namespace opt{

//***************************************************
// Objective
//***************************************************

class Objective{
private:
	//count
		int nPrint_;//print data every n steps
		int nWrite_;//write data every n steps
		int step_;//current step
		int count_;//current count
	//stopping
		int max_;//max steps
		Stop stop_;//the type of value determining the end condition
		Loss loss_;//loss function
		double tol_;//stop tolerance
	//status
		double gamma_;//
		double val_,valOld_;//current, old value
		double dv_,dp_;//change in value, p
	//parameters
		int dim_;//dimension of problem
		Eigen::VectorXd p_,pOld_;//current, old parameters
		Eigen::VectorXd g_,gOld_;//current, old gradients
public:
	//==== constructors/destructors ====
	Objective(){defaults();}
	Objective(int dim){defaults();resize(dim);}
	~Objective(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Objective& data);
	
	//==== access ====
	//count
		int& nPrint(){return nPrint_;}
		const int& nPrint()const{return nPrint_;}
		int& nWrite(){return nWrite_;}
		const int& nWrite()const{return nWrite_;}
		int& step(){return step_;}
		const int& step()const{return step_;}
		int& count(){return count_;}
		const int& count()const{return count_;}
	//stopping
		int& max(){return max_;}
		const int& max()const{return max_;}
		Stop& stop(){return stop_;}
		const Stop& stop()const{return stop_;}
		Loss& loss(){return loss_;}
		const Loss& loss()const{return loss_;}
		double& tol(){return tol_;}
		const double& tol()const{return tol_;}
	//status
		double& gamma(){return gamma_;}
		const double& gamma()const{return gamma_;}
		double& val(){return val_;}
		const double& val()const{return val_;}
		double& valOld(){return valOld_;}
		const double& valOld()const{return valOld_;}
		double& dv(){return dv_;}
		const double& dv()const{return dv_;}
		double& dp(){return dp_;}
		const double& dp()const{return dp_;}
	//parameters
		int& dim(){return dim_;}
		const int& dim()const{return dim_;}
		Eigen::VectorXd& p(){return p_;}
		const Eigen::VectorXd& p()const{return p_;}
		Eigen::VectorXd& pOld(){return pOld_;}
		const Eigen::VectorXd& pOld()const{return pOld_;}
		Eigen::VectorXd& g(){return g_;}
		const Eigen::VectorXd& g()const{return g_;}
		Eigen::VectorXd& gOld(){return gOld_;}
		const Eigen::VectorXd& gOld()const{return gOld_;}
	
	//==== member functions ====
	void defaults();
	void clear(){defaults();}
	void resize(int dim);
};

}

namespace serialize{

	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const opt::Objective& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const opt::Objective& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(opt::Objective& obj, const char* arr);
	
}

#endif