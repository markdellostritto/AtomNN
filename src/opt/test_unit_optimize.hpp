// c libaries
#include <ctime>
// c++ libaries
#include <iostream>
// eigen libraries
#include <Eigen/Dense>
// ann - optimization
#include "src/opt/optimize.hpp"
// ann - string
#include "src/str/print.hpp"

//**********************************************************************
// Rosenberg function
//**********************************************************************

struct Rosen{
	double a,b;
	Rosen():a(1.0),b(100.0){};
	Rosen(double aa, double bb):a(aa),b(bb){};
	double operator()(const Eigen::VectorXd& x, Eigen::VectorXd& g){
		g[0]=-2.0*(a-x[0])-4.0*b*x[0]*(x[1]-x[0]*x[0]);
		g[1]=2.0*b*(x[1]-x[0]*x[0]);
		return (a-x[0])*(a-x[0])+b*(x[1]-x[0]*x[0])*(x[1]-x[0]*x[0]);
	};
};

void opt_rosen(Opt::Model& model){
	//rosenberg function
	Rosen rosen;
	//init data
	Opt::Data data(2);
	data.p()[0]=2.0*(1.0*std::rand()/RAND_MAX-0.5);
	data.p()[1]=2.0*(1.0*std::rand()/RAND_MAX-0.5);
	data.pOld()=data.p();
	data.max()=100000;
	data.nPrint()=100;
	data.algo()=model.algo();
	data.stop()=Opt::Stop::FABS;
	data.tol()=1e-8;
	//optimize
	for(int i=0; i<data.max(); ++i){
		//compute the value and gradient
		data.val()=rosen(data.p(),data.g());
		//compute the new position
		model.step(data);
		//calculate the difference
		data.dv()=std::fabs(data.val()-data.valOld());
		data.dp()=(data.p()-data.pOld()).norm();
		//check the break condition
		switch(data.stop()){
			case Opt::Stop::FREL: if(data.dv()<data.tol()) break;
			case Opt::Stop::XREL: if(data.dp()<data.tol()) break;
			case Opt::Stop::FABS: if(data.val()<data.tol()) break;
		}
		//print the status
		data.pOld().noalias()=data.p();//set "old" p value
		data.gOld().noalias()=data.g();//set "old" g value
		data.valOld()=data.val();//set "old" value
		//update step
		++data.step();
	}
	//print
	std::cout<<"n_steps = "<<data.step()<<"\n";
	std::cout<<"val     = "<<data.val()<<"\n";
	std::cout<<"x       = "<<data.p().transpose()<<"\n";
}