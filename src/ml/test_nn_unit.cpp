// c libaries
#include <ctime>
// c++ libaries
#include <iostream>
// str
#include "src/str/print.hpp"
// ml
#include "src/ml/nn.hpp"

void test_nn_mem(){
	//local function variables
	NN::ANN nn1,nn2;
	NN::ANNP init;
	//resize the nn
	init.sigma()=1.0;
	init.init()=NN::Init::HE;
	init.neuron()=NN::Neuron::TANH;
	std::vector<int> nh(3);
	const int nInp=2;
	const int nOut=3;
	nh[0]=7; nh[1]=5; nh[2]=nOut;
	nn1.resize(init,nInp,nh);
	nn1.inpw()=Eigen::VectorXd::Random(nInp);
	nn1.inpb()=Eigen::VectorXd::Random(nInp);
	nn1.outw()=Eigen::VectorXd::Random(nOut);
	nn1.outb()=Eigen::VectorXd::Random(nOut);
	//pack
	const int size=serialize::nbytes(nn1);
	char* memarr=new char[size];
	serialize::pack(nn1,memarr);
	serialize::unpack(nn2,memarr);
	//error
	const double err_nlayer=(nn1.nlayer()-nn2.nlayer());
	double err_nnodes=0; for(int i=0; i<nn1.nlayer(); ++i) err_nnodes+=(nn1.nNodes(i)-nn2.nNodes(i));
	double err_b=0; for(int i=0; i<nn1.nlayer(); ++i) err_b+=(nn1.b(i)-nn2.b(i)).norm();
	double err_w=0; for(int i=0; i<nn1.nlayer(); ++i) err_w+=(nn1.w(i)-nn2.w(i)).norm();
	const double err_inpw=(nn1.inpw()-nn2.inpw()).norm();
	const double err_inpb=(nn1.inpb()-nn2.inpb()).norm();
	const double err_outw=(nn1.outw()-nn2.outw()).norm();
	const double err_outb=(nn1.inpb()-nn2.inpb()).norm();
	//print
	std::cout<<"TEST - ANN - MEM\n";
	std::cout<<"err - nlayer = "<<err_nlayer<<"\n";
	std::cout<<"err - nnodes = "<<err_nnodes<<"\n";
	std::cout<<"err - bias   = "<<err_b<<"\n";
	std::cout<<"err - weight = "<<err_w<<"\n";
	std::cout<<"err - inpw   = "<<err_inpw<<"\n";
	std::cout<<"err - inpb   = "<<err_inpb<<"\n";
	std::cout<<"err - outw   = "<<err_outw<<"\n";
	std::cout<<"err - outb   = "<<err_outb<<"\n";
}

void test_nn_execute(){
	//local function variables
	const int N=100;
	double erra=0,errp=0;
	const int nInp=2;
	const int nOut=3;
	const int nlayer=3;
	//init rand
	std::srand(std::time(NULL));
	//resize the nn
	NN::ANN nn;
	NN::ANNP init;
	init.sigma()=1.0;
	init.init()=NN::Init::HE;
	init.neuron()=NN::Neuron::TANH;
	std::vector<int> nh(nlayer);
	nh[0]=7; nh[1]=5; nh[2]=3;
	//loop over all samples
	for(int n=0; n<N; ++n){
		nn.resize(init,nInp,nh);
		//set input/output scaling
		nn.inpw()=Eigen::VectorXd::Random(nInp);
		nn.inpb()=Eigen::VectorXd::Random(nInp);
		nn.outw()=Eigen::VectorXd::Random(nOut);
		nn.outb()=Eigen::VectorXd::Random(nOut);
		//initialize the input nodes
		for(int i=0; i<nn.nInp(); ++i) nn.inp()[i]=std::rand()/RAND_MAX-0.5;
		//execute the network
		nn.fpbp2();
		//compute error
		std::vector<Eigen::VectorXd> z(nlayer);
		std::vector<Eigen::VectorXd> a(nlayer);
		std::vector<Eigen::VectorXd> d(nlayer);
		std::vector<Eigen::VectorXd> d2(nlayer);
		for(int i=0; i<nlayer; ++i){
			z[i]=Eigen::VectorXd::Zero(nh[i]);
			a[i]=Eigen::VectorXd::Zero(nh[i]);
			d[i]=Eigen::VectorXd::Zero(nh[i]);
			d2[i]=Eigen::VectorXd::Zero(nh[i]);
		}
		const Eigen::VectorXd inps=nn.inpw().cwiseProduct(nn.inp()+nn.inpb());
		z[0]=nn.w(0)*inps+nn.b(0);
		nn.affpbp2(0)(z[0],a[0],d[0],d2[0]);
		z[1]=nn.w(1)*a[0]+nn.b(1);
		nn.affpbp2(1)(z[1],a[1],d[1],d2[0]);
		z[2]=nn.w(2)*a[1]+nn.b(2);
		nn.affpbp2(2)(z[2],a[2],d[2],d2[0]);
		const Eigen::VectorXd out=nn.outb()+nn.outw().cwiseProduct(a[2]);
		erra+=(out-nn.out()).norm();
		errp+=(out-nn.out()).norm()/out.norm()*100;
	}
	//compute the error
	erra/=N;
	errp/=N;
	char* str=new char[print::len_buf];
	//print the results
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<"TEST - ANN - EXECUTE\n";
	std::cout<<"transfer = "<<nn.neuron()<<"\n";
	std::cout<<"config   = "<<nn.nInp()<<" "; for(int i=0; i<nh.size(); ++i) std::cout<<nh[i]<<" "; std::cout<<"\n";
	std::cout<<"erra     = "<<erra<<"\n";
	std::cout<<"errp     = "<<errp<<"\n";
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

void test_nn_time(){
	//local function variables
	const int N=1000000;
	const int M=4;
	std::vector<double> time_exec(M,0);
	std::vector<double> time_grad(M,0);
	clock_t start,stop;
	std::vector<std::vector<int> > nh(M);
	nh[1].resize(1,5);
	nh[2].resize(2,5);
	nh[3].resize(3,5);
	std::srand(std::time(NULL));
	for(int m=0; m<M; ++m){
		NN::ANN nn;
		NN::ANNP init;
		init.sigma()=1.0;
		init.init()=NN::Init::HE;
		init.neuron()=NN::Neuron::TANH;
		if(nh[m].size()==0) nn.resize(init,3,3);
		else nn.resize(init,3,nh[m],3);
		NN::Cost cost(nn);
		start=std::clock();
		for(int n=0; n<N; ++n){
			//initialize the input nodes
			for(int i=0; i<nn.nInp(); ++i) nn.inp()[i]=std::rand()/RAND_MAX-0.5;
			//execute the network
			volatile Eigen::VectorXd vec=nn.fpbp2();
		}
		stop=std::clock();
		time_exec[m]=((double)(stop-start))/CLOCKS_PER_SEC*1e9/N;
		start=std::clock();
		for(int n=0; n<N; ++n){
			//initialize the input nodes
			for(int i=0; i<nn.nInp(); ++i) nn.inp()[i]=std::rand()/RAND_MAX-0.5;
			Eigen::VectorXd grad=Eigen::VectorXd::Random(nn.size());
			Eigen::VectorXd dcdo=Eigen::VectorXd::Random(3);
			//compute gradient
			cost.grad(nn,dcdo);
		}
		stop=std::clock();
		time_grad[m]=((double)(stop-start))/CLOCKS_PER_SEC*1e9/N;
	}
	std::vector<std::vector<int> > config(M);
	config[0].resize(2,5); config[0].front()=3; config[0].back()=3;
	config[1].resize(3,5); config[1].front()=3; config[1].back()=3;
	config[2].resize(4,5); config[2].front()=3; config[2].back()=3;
	config[3].resize(5,5); config[3].front()=3; config[3].back()=3;
	double time_avg=0;
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<"TEST - ANN - TIME - EXECUTION\n";
	std::cout<<"transfer = TANH\n";
	std::cout<<"time - "<<time_exec[0]<<" ns - "; for(int i=0; i<config[0].size(); ++i) std::cout<<config[0][i]<<" "; std::cout<<"\n";
	std::cout<<"time - "<<time_exec[1]<<" ns - "; for(int i=0; i<config[1].size(); ++i) std::cout<<config[1][i]<<" "; std::cout<<"\n";
	std::cout<<"time - "<<time_exec[2]<<" ns - "; for(int i=0; i<config[2].size(); ++i) std::cout<<config[2][i]<<" "; std::cout<<"\n";
	std::cout<<"time - "<<time_exec[3]<<" ns - "; for(int i=0; i<config[3].size(); ++i) std::cout<<config[3][i]<<" "; std::cout<<"\n";
	time_avg=0;
	for(int i=0; i<time_exec.size(); ++i) time_avg+=time_exec[i];
	std::cout<<"time - avg = "<<time_avg/time_exec.size()<<"\n";
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<"TEST - ANN - TIME - GRADIENT\n";
	std::cout<<"transfer = TANH\n";
	std::cout<<"time - "<<time_grad[0]<<" ns - "; for(int i=0; i<config[0].size(); ++i) std::cout<<config[0][i]<<" "; std::cout<<"\n";
	std::cout<<"time - "<<time_grad[1]<<" ns - "; for(int i=0; i<config[1].size(); ++i) std::cout<<config[1][i]<<" "; std::cout<<"\n";
	std::cout<<"time - "<<time_grad[2]<<" ns - "; for(int i=0; i<config[2].size(); ++i) std::cout<<config[2][i]<<" "; std::cout<<"\n";
	std::cout<<"time - "<<time_grad[3]<<" ns - "; for(int i=0; i<config[3].size(); ++i) std::cout<<config[3][i]<<" "; std::cout<<"\n";
	time_avg=0;
	for(int i=0; i<time_grad.size(); ++i) time_avg+=time_grad[i];
	std::cout<<"time - avg = "<<time_avg/time_grad.size()<<"\n";
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

int main(int argc, char* argv[]){
	
	std::srand(std::time(NULL));
	char* str=new char[print::len_buf];
	
	std::cout<<print::buf(str)<<"\n";
	std::cout<<print::title("ANN",str)<<"\n";
	test_nn_mem();
	test_nn_execute();
	test_nn_time();
	std::cout<<print::title("ANN",str)<<"\n";
	std::cout<<print::buf(str)<<"\n";
	
	delete[] str;
	
	return 0;
}