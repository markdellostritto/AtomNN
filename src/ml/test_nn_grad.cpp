// c libaries
#include <ctime>
// c++ libaries
#include <iostream>
// str
#include "src/str/print.hpp"
// math
#include "src/math/reduce.hpp"
// ml
#include "src/ml/nn.hpp"

void test_unit_nn_dOdZ(){
	//local function variables
	const int N=100;
	double erra=0,errp=0;
	double time=0;
	NN::DODZ dODZ;
	NN::ANN nn;
	NN::ANNP init;
	std::vector<int> nh(2);
	const int nInp=4;
	const int nOut=3;
	nh[0]=7; nh[1]=5;
	init.sigma()=1.0;
	init.init()=NN::Init::HE;
	init.neuron()=NN::Neuron::TANH;
	Reduce<2> reduce;
	//init rand
	std::srand(std::time(NULL));
	//loop over all samples
	for(int m=0; m<N; ++m){
		//resize the nn
		Eigen::MatrixXd dOutExact,dOutApprox;
		nn.resize(init,nInp,nh,nOut);
		dODZ.resize(nn);
		//set input/output scaling
		nn.inpw()=Eigen::VectorXd::Random(nInp);
		nn.inpb()=Eigen::VectorXd::Random(nInp);
		nn.outw()=Eigen::VectorXd::Random(nOut);
		nn.outb()=Eigen::VectorXd::Random(nOut);
		//initialize the input nodes
		for(int n=0; n<nn.nInp(); ++n) nn.inp()[n]=(1.0*std::rand())/RAND_MAX-0.5;
		Eigen::VectorXd inp=nn.inp();
		//execute the network, compute analytic gradient
		nn.fpbp();
		clock_t start=std::clock();
		dODZ.grad(nn);
		clock_t stop=std::clock();
		time+=((double)(stop-start))/CLOCKS_PER_SEC*1e9/N;
		dOutApprox=dODZ.dodi();
		//compute brute force gradient
		dOutExact=Eigen::MatrixXd::Zero(nn.nOut(),nn.nInp());
		Eigen::VectorXd delta=Eigen::VectorXd::Random(nn.nInp())/100.0;
		for(int i=0; i<nn.nInp(); ++i){
			//point 1
			for(int n=0; n<nn.nInp(); ++n) nn.inp()[n]=inp[n];//reset input
			nn.inp()[i]+=delta[i];//add small change
			nn.fpbp();//execute
			Eigen::VectorXd outNew1=nn.out();//store output
			//point 2
			for(int n=0; n<nn.nInp(); ++n) nn.inp()[n]=inp[n];//reset input
			nn.inp()[i]-=delta[i];//add small change
			nn.fpbp();//execute
			Eigen::VectorXd outNew2=nn.out();//store output
			//gradient
			for(int j=0; j<nn.out().size(); ++j){
				dOutExact(j,i)=0.5*(outNew1[j]-outNew2[j])/delta[i];
			}
		}
		erra+=(dOutExact-dOutApprox).norm();
		errp+=(dOutExact-dOutApprox).norm()/dOutExact.norm()*100;
		for(int i=0; i<dOutExact.size(); ++i) reduce.push(dOutExact(i),dOutApprox(i));
		//std::cout<<"dOutApprox = "<<dOutApprox.transpose()<<"\n";
		//std::cout<<"dOutExact = "<<dOutExact.transpose()<<"\n";
	}
	//compute the error
	erra/=N;
	errp/=N;
	time/=N;
	//print the results
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<"TEST - ANN - DODZ\n";
	std::cout<<"neuron = "<<nn.neuron()<<"\n";
	std::cout<<"config  = "<<nn.nInp()<<" "; for(int i=0; i<nh.size(); ++i) std::cout<<nh[i]<<" "; std::cout<<nn.nOut()<<"\n";
	std::cout<<"erra    = "<<erra<<"\n";
	std::cout<<"errp    = "<<errp<<"\n";
	std::cout<<"r2      = "<<reduce.r2()<<"\n";
	std::cout<<"m       = "<<reduce.m()<<"\n";
	std::cout<<"time    = "<<time<<"\n";
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

void test_unit_nn_dOutDP(){
	//local function variables
	const int N=100;
	double erra=0,errp=0;
	clock_t start,stop;
	double time=0;
	NN::DODP dOutDP;
	NN::ANN nn,nn_copy;
	NN::ANNP init;
	std::vector<int> nh(3);
	const int nInp=4;
	const int nOut=3;
	nh[0]=7; nh[1]=5; nh[2]=4;
	init.sigma()=1.0;
	init.init()=NN::Init::HE;
	init.neuron()=NN::Neuron::TANH;
	Reduce<2> reduce;
	//init rand
	std::srand(std::time(NULL));
	//loop over all samples
	for(int iter=0; iter<N; ++iter){
		//resize the nn
		nn.resize(init,nInp,nh,nOut);
		dOutDP.resize(nn);
		//set input/output scaling
		nn.inpw()=Eigen::VectorXd::Random(nInp);
		nn.inpb()=Eigen::VectorXd::Random(nInp);
		nn.outw()=Eigen::VectorXd::Random(nOut);
		nn.outb()=Eigen::VectorXd::Random(nOut);
		//initialize the input nodes
		for(int n=0; n<nn.nInp(); ++n) nn.inp()[n]=(1.0*std::rand())/RAND_MAX-0.5;
		//execute the network, compute analytic gradient
		nn.fpbp();
		start=std::clock();
		dOutDP.grad(nn);
		stop=std::clock();
		time+=((double)(stop-start))/CLOCKS_PER_SEC*1e9/N;
		std::vector<std::vector<Eigen::VectorXd> > gradBApprox=dOutDP.dodb();
		std::vector<std::vector<Eigen::MatrixXd> > gradWApprox=dOutDP.dodw();
		//compute brute force gradient - bias
		std::vector<std::vector<Eigen::VectorXd> > gradBExact;
		gradBExact.resize(nn.nOut());
		for(int n=0; n<nn.nOut(); ++n){
			gradBExact[n].resize(nn.nlayer());
			for(int l=0; l<nn.nlayer(); ++l){
				gradBExact[n][l]=Eigen::VectorXd::Zero(nn.b(l).size());
			}
		}
		nn_copy=nn;
		for(int l=0; l<nn.nlayer(); ++l){
			for(int i=0; i<nn.b(l).size(); ++i){
				const double delta=nn.b(l)[i]/100.0;
				//point 1
				nn_copy.b(l)[i]=nn.b(l)[i]-delta;
				nn_copy.fpbp();
				const Eigen::VectorXd pt1=nn_copy.out();
				//point 2
				nn_copy.b(l)[i]=nn.b(l)[i]+delta;
				nn_copy.fpbp();
				const Eigen::VectorXd pt2=nn_copy.out();
				//reset
				nn_copy.b(l)[i]=nn.b(l)[i];
				//gradient
				const Eigen::VectorXd grad=0.5*(pt2-pt1)/delta;
				for(int n=0; n<nn.nOut(); ++n){
					gradBExact[n][l](i)=grad(n);
				}
			}
		}
		std::vector<std::vector<Eigen::MatrixXd> > gradWExact;
		gradWExact.resize(nn.nOut());
		for(int n=0; n<nn.nOut(); ++n){
			gradWExact[n].resize(nn.nlayer());
			for(int l=0; l<nn.nlayer(); ++l){
				gradWExact[n][l]=Eigen::MatrixXd::Zero(nn.w(l).rows(),nn.w(l).cols());
			}
		}
		nn_copy=nn;
		for(int l=0; l<nn.nlayer(); ++l){
			for(int j=0; j<nn.w(l).rows(); ++j){
				for(int k=0; k<nn.w(l).cols(); ++k){
					const double delta=nn.w(l)(j,k)/100.0;
					//point 1
					nn_copy.w(l)(j,k)=nn.w(l)(j,k)-delta;
					nn_copy.fpbp();
					const Eigen::VectorXd pt1=nn_copy.out();
					//point 2
					nn_copy.w(l)(j,k)=nn.w(l)(j,k)+delta;
					nn_copy.fpbp();
					const Eigen::VectorXd pt2=nn_copy.out();
					//reset
					nn_copy.w(l)(j,k)=nn.w(l)(j,k);
					//grad
					const Eigen::VectorXd grad=0.5*(pt2-pt1)/delta;
					for(int n=0; n<nn.nOut(); ++n){
						gradWExact[n][l](j,k)=grad[n];
					}
				}
			}
		}
		//compute the error
		for(int n=0; n<nn.nOut(); ++n){
			for(int l=0; l<nn.nlayer(); ++l){
				//std::cout<<"gradBApprox["<<n<<"]["<<l<<"] = \n"<<gradBApprox[n][l]<<"\n";
				//std::cout<<"gradBExact["<<n<<"]["<<l<<"] = \n"<<gradBExact[n][l]<<"\n";
				erra+=(gradBExact[n][l]-gradBApprox[n][l]).norm();
				errp+=(gradBExact[n][l]-gradBApprox[n][l]).norm()/gradBExact[n][l].norm()*100;
				for(int i=0; i<gradBExact[n][l].size(); ++i){
					reduce.push(gradBExact[n][l](i),gradBApprox[n][l](i));
				}
			}
		}
		for(int n=0; n<nn.nOut(); ++n){
			for(int l=0; l<nn.nlayer(); ++l){
				//std::cout<<"gradWApprox["<<n<<"]["<<l<<"] = \n"<<gradWApprox[n][l]<<"\n";
				//std::cout<<"gradWExact["<<n<<"]["<<l<<"] = \n"<<gradWExact[n][l]<<"\n";
				erra+=(gradWExact[n][l]-gradWApprox[n][l]).norm();
				errp+=(gradWExact[n][l]-gradWApprox[n][l]).norm()/gradWExact[n][l].norm()*100;
				for(int i=0; i<gradWExact[n][l].size(); ++i){
					reduce.push(gradWExact[n][l](i),gradWApprox[n][l](i));
				}
			}
		}
	}
	//compute the error
	erra/=N;
	errp/=N;
	time/=N;
	//print the results
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<"TEST - ANN - DODP\n";
	std::cout<<"neuron = "<<nn.neuron()<<"\n";
	std::cout<<"config = "<<nn.nInp()<<" "; for(int i=0; i<nh.size(); ++i) std::cout<<nh[i]<<" "; std::cout<<nn.nOut()<<"\n";
	std::cout<<"erra   = "<<erra<<"\n";
	std::cout<<"errp   = "<<errp<<"\n";
	std::cout<<"r2     = "<<reduce.r2()<<"\n";
	std::cout<<"m      = "<<reduce.m()<<"\n";
	std::cout<<"time   = "<<time<<"\n";
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

void test_unit_nn_dzdi(){
	//local function variables
	const int N=1000;
	double erra=0,errp=0;
	clock_t start,stop;
	double time=0;
	NN::DZDI dZdI;
	NN::ANN nn,nn_copy;
	NN::ANNP init;
	std::vector<int> nh(3);
	const int nInp=4;
	const int nOut=3;
	nh[0]=7; nh[1]=5; nh[2]=4;
	init.sigma()=1.0;
	init.init()=NN::Init::HE;
	init.neuron()=NN::Neuron::TANH;
	Reduce<2> reduce;
	//init rand
	std::srand(std::time(NULL));
	//loop over all samples
	for(int iter=0; iter<N; ++iter){
		//resize the nn
		nn.resize(init,nInp,nh,nOut);
		dZdI.resize(nn);
		//set input/output scaling
		nn.inpw()=Eigen::VectorXd::Random(nInp);
		nn.inpb()=Eigen::VectorXd::Random(nInp);
		nn.outw()=Eigen::VectorXd::Random(nOut);
		nn.outb()=Eigen::VectorXd::Random(nOut);
		//initialize the input nodes
		for(int n=0; n<nn.nInp(); ++n) nn.inp()[n]=(1.0*std::rand())/RAND_MAX-0.5;
		//execute the network, compute analytic gradient
		nn.fpbp();
		dZdI.grad(nn);
		std::vector<MatXd> dzdi_d=dZdI.dzdi();
		//for(int i=0; i<dzdi_d.size(); ++i) std::cout<<"dzdi_d["<<i<<"] = \n"<<dzdi_d[i]<<"\n";
		//brute force
		std::vector<MatXd> dzdi_b(nn.nlayer());
		for(int n=0; n<nn.nlayer(); ++n){
			dzdi_b[n]=MatXd::Zero(nn.nNodes(n),nn.nInp());
		}
		nn_copy=nn;
		for(int n=0; n<nn.nInp(); ++n){
			const double delta=nn.inp()[n]/1000.0;
			//first point
			nn_copy.inp()[n]=nn.inp()[n]-delta;
			nn_copy.fpbp();
			const std::vector<VecXd> z1=nn_copy.z();
			//second point
			nn_copy.inp()[n]=nn.inp()[n]+delta;
			nn_copy.fpbp();
			const std::vector<VecXd> z2=nn_copy.z();
			//reset
			nn_copy.inp()[n]=nn.inp()[n];
			//derivative
			for(int l=0; l<nn.nlayer(); ++l){
				for(int i=0; i<nn.nNodes(l); ++i){
					dzdi_b[l](i,n)=0.5*(z2[l][i]-z1[l][i])/delta;
				}
			}
		}
		//for(int i=0; i<dzdi_b.size(); ++i) std::cout<<"dzdi_b["<<i<<"] = \n"<<dzdi_b[i]<<"\n";
		for(int i=0; i<nn.nlayer(); ++i){
			erra+=(dzdi_b[i]-dzdi_d[i]).norm();
			errp+=(dzdi_b[i]-dzdi_d[i]).norm()/dzdi_b[i].norm()*100.0;
			for(int j=0; j<dzdi_b[i].size(); ++j){
				reduce.push(dzdi_b[i](j),dzdi_d[i](j));
			}
		}
	}
	//compute the error
	erra/=N;
	errp/=N;
	time/=N;
	//print the results
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<"TEST - ANN - DZDI\n";
	std::cout<<"neuron = "<<nn.neuron()<<"\n";
	std::cout<<"config  = "<<nn.nInp()<<" "; for(int i=0; i<nh.size(); ++i) std::cout<<nh[i]<<" "; std::cout<<nn.nOut()<<"\n";
	std::cout<<"erra    = "<<erra<<"\n";
	std::cout<<"errp    = "<<errp<<"\n";
	std::cout<<"r2      = "<<reduce.r2()<<"\n";
	std::cout<<"m       = "<<reduce.m()<<"\n";
	std::cout<<"time    = "<<time<<"\n";
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

void test_unit_nn_d2OdZdI(){
	//local function variables
	const int N=100;
	double erra=0,errp=0;
	clock_t start,stop;
	double timeD=0;
	double timeB=0;
	NN::D2ODZDIN d2OdZdIN;
	NN::D2ODZDI d2OdZdI;
	NN::ANN nn;
	NN::ANNP init;
	std::vector<int> nh(3);
	const int nInp=4;
	const int nOut=1;
	nh[0]=7; nh[1]=5; nh[2]=5;
	nh[0]=48; nh[1]=24; nh[2]=24;
	init.sigma()=1.0;
	init.init()=NN::Init::HE;
	init.neuron()=NN::Neuron::TANH;
	Reduce<2> reduce;
	//init rand
	std::srand(std::time(NULL));
	//loop over all samples
	for(int iter=0; iter<N; ++iter){
		//resize the nn
		nn.resize(init,nInp,nh,nOut);
		d2OdZdIN.resize(nn);
		d2OdZdI.resize(nn);
		//set input/output scaling
		/*
		nn.inpw()=Eigen::VectorXd::Random(nInp);
		nn.inpb()=Eigen::VectorXd::Random(nInp);
		nn.outw()=Eigen::VectorXd::Random(nOut);
		nn.outb()=Eigen::VectorXd::Random(nOut);
		*/
		nn.inpw()=Eigen::VectorXd::Constant(nInp,1.0);
		nn.inpb()=Eigen::VectorXd::Constant(nInp,0.0);
		nn.outw()=Eigen::VectorXd::Constant(nOut,1.0);
		nn.outb()=Eigen::VectorXd::Constant(nOut,0.0);
		
		//initialize the input nodes
		for(int n=0; n<nn.nInp(); ++n) nn.inp()[n]=(1.0*std::rand())/RAND_MAX-0.5;
		//execute the network, compute analytic gradient
		nn.fpbp2();
		start=std::clock();
		d2OdZdI.grad(nn);
		stop=std::clock();
		timeD+=((double)(stop-start))/CLOCKS_PER_SEC*1e9/N;
		start=std::clock();
		d2OdZdIN.grad(nn);
		stop=std::clock();
		timeB+=((double)(stop-start))/CLOCKS_PER_SEC*1e9/N;
		int count=0;
		for(int l=0; l<nn.nlayer(); ++l){
			for(int n=0; n<nn.b(l).size(); ++n){
				//std::cout<<d2OdZdIN.d2odpdi(count)<<"\n";
				//std::cout<<d2OdZdI.d2odbdi()[l][n]<<"\n";
				erra+=(d2OdZdIN.d2odpdi(count)-d2OdZdI.d2odbdi()[l][n]).norm();
				if(d2OdZdIN.d2odpdi(count).norm()>1e-16){
					errp+=(d2OdZdIN.d2odpdi(count)-d2OdZdI.d2odbdi()[l][n]).norm()/d2OdZdIN.d2odpdi(count).norm()*100.0;
				}
				count++;
			}
		}
		for(int l=0; l<nn.nlayer(); ++l){
			for(int m=0; m<nn.w(l).cols(); ++m){
				for(int n=0; n<nn.w(l).rows(); ++n){
					//std::cout<<d2OdZdIN.d2odpdi(count)<<"\n";
					//std::cout<<d2OdZdI.d2odwdi()[l][n][m]<<"\n";
					erra+=(d2OdZdIN.d2odpdi(count)-d2OdZdI.d2odwdi()[l][n][m]).norm();
					if(d2OdZdIN.d2odpdi(count).norm()>1e-16){
						errp+=(d2OdZdIN.d2odpdi(count)-d2OdZdI.d2odwdi()[l][n][m]).norm()/d2OdZdIN.d2odpdi(count).norm()*100.0;
					}
					count++;
				}
			}
		}
	}
	//compute the error
	erra/=N;
	errp/=N;
	timeD/=N;
	timeB/=N;
	//print the results
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<"TEST - ANN - D2ODZDI\n";
	std::cout<<"timeD = "<<timeD<<"\n";
	std::cout<<"timeB = "<<timeB<<"\n";
	std::cout<<"erra  = "<<erra<<"\n";
	std::cout<<"errp  = "<<errp<<"\n";
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

int main(int argc, char* argv[]){
	
	std::srand(std::time(NULL));
	char* str=new char[print::len_buf];
	
	std::cout<<print::buf(str)<<"\n";
	std::cout<<print::title("ANN",str)<<"\n";
	test_unit_nn_dOdZ();
	test_unit_nn_dOutDP();
	test_unit_nn_dzdi();
	test_unit_nn_d2OdZdI();
	std::cout<<print::title("ANN",str)<<"\n";
	std::cout<<print::buf(str)<<"\n";
	
	delete[] str;
	
	return 0;
}