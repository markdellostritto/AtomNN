// c libaries
#include <ctime>
// c++ libaries
#include <iostream>
// ann - str
#include "src/str/print.hpp"
// ann - ml
#include "src/ml/nn.hpp"
// ann - optimization
#include "src/ml/test_unit_nn.hpp"

void test_unit_nn(){
	//local function variables
	NN::ANN nn,nn_copy;
	NN::ANNInit init;
	//resize the nn
	init.sigma()=1.0;
	init.init()=NN::Init::HE;
	nn.tf()=NN::Transfer::TANH;
	std::vector<int> nh(2);
	nh[0]=7; nh[1]=5;
	nn.resize(init,2,nh,3);
	//pack
	const int size=serialize::nbytes(nn);
	char* memarr=new char[size];
	serialize::pack(nn,memarr);
	serialize::unpack(nn_copy,memarr);
	std::cout<<"nn = \n"<<nn<<"\n";
	std::cout<<"nn_copy = \n"<<nn_copy<<"\n";
}

void test_unit_nn_tfunc(){
	FILE* writer=NULL;
	const int N=1000;
	const double xmin=-5;
	const double xmax=5;
	const double dx=(xmax-xmin)/(N-1.0);
	Eigen::VectorXd ff,fd;
	ff=Eigen::VectorXd::Zero(N);
	fd=Eigen::VectorXd::Zero(N);
	
	//linear
	for(int i=0; i<N; ++i) ff[i]=xmin+dx*i;
	NN::Transfer::tf_lin(ff,fd);
	writer=fopen("tfunc_linear.dat","w");
	if(writer!=NULL){
		fprintf(writer,"#X F D\n");
		for(int i=0; i<N; ++i){
			fprintf(writer,"%f %f %f\n",xmin+dx*i,ff[i],fd[i]);
		}
	}
	fclose(writer); writer=NULL;
	
	//sigmoid
	for(int i=0; i<N; ++i) ff[i]=xmin+dx*i;
	NN::Transfer::tf_sigmoid(ff,fd);
	writer=fopen("tfunc_sigmoid.dat","w");
	if(writer!=NULL){
		fprintf(writer,"#X F D\n");
		for(int i=0; i<N; ++i){
			fprintf(writer,"%f %f %f\n",xmin+dx*i,ff[i],fd[i]);
		}
	}
	fclose(writer); writer=NULL;
	
	//tanh
	for(int i=0; i<N; ++i) ff[i]=xmin+dx*i;
	NN::Transfer::tf_tanh(ff,fd);
	writer=fopen("tfunc_tanh.dat","w");
	if(writer!=NULL){
		fprintf(writer,"#X F D\n");
		for(int i=0; i<N; ++i){
			fprintf(writer,"%f %f %f\n",xmin+dx*i,ff[i],fd[i]);
		}
	}
	fclose(writer); writer=NULL;
	
	//isru
	for(int i=0; i<N; ++i) ff[i]=xmin+dx*i;
	NN::Transfer::tf_isru(ff,fd);
	writer=fopen("tfunc_isru.dat","w");
	if(writer!=NULL){
		fprintf(writer,"#X F D\n");
		for(int i=0; i<N; ++i){
			fprintf(writer,"%f %f %f\n",xmin+dx*i,ff[i],fd[i]);
		}
	}
	fclose(writer); writer=NULL;
	
	//arctan
	for(int i=0; i<N; ++i) ff[i]=xmin+dx*i;
	NN::Transfer::tf_arctan(ff,fd);
	writer=fopen("tfunc_arctan.dat","w");
	if(writer!=NULL){
		fprintf(writer,"#X F D\n");
		for(int i=0; i<N; ++i){
			fprintf(writer,"%f %f %f\n",xmin+dx*i,ff[i],fd[i]);
		}
	}
	fclose(writer); writer=NULL;
	
	//softsign
	for(int i=0; i<N; ++i) ff[i]=xmin+dx*i;
	NN::Transfer::tf_softsign(ff,fd);
	writer=fopen("tfunc_softsign.dat","w");
	if(writer!=NULL){
		fprintf(writer,"#X F D\n");
		for(int i=0; i<N; ++i){
			fprintf(writer,"%f %f %f\n",xmin+dx*i,ff[i],fd[i]);
		}
	}
	fclose(writer); writer=NULL;
	
	//softsign
	for(int i=0; i<N; ++i) ff[i]=xmin+dx*i;
	NN::Transfer::tf_softsign(ff,fd);
	writer=fopen("tfunc_softsign.dat","w");
	if(writer!=NULL){
		fprintf(writer,"#X F D\n");
		for(int i=0; i<N; ++i){
			fprintf(writer,"%f %f %f\n",xmin+dx*i,ff[i],fd[i]);
		}
	}
	fclose(writer); writer=NULL;
	
	//relu
	for(int i=0; i<N; ++i) ff[i]=xmin+dx*i;
	NN::Transfer::tf_relu(ff,fd);
	writer=fopen("tfunc_relu.dat","w");
	if(writer!=NULL){
		fprintf(writer,"#X F D\n");
		for(int i=0; i<N; ++i){
			fprintf(writer,"%f %f %f\n",xmin+dx*i,ff[i],fd[i]);
		}
	}
	fclose(writer); writer=NULL;
	
	//softplus
	for(int i=0; i<N; ++i) ff[i]=xmin+dx*i;
	NN::Transfer::tf_softplus(ff,fd);
	writer=fopen("tfunc_softplus.dat","w");
	if(writer!=NULL){
		fprintf(writer,"#X F D\n");
		for(int i=0; i<N; ++i){
			fprintf(writer,"%f %f %f\n",xmin+dx*i,ff[i],fd[i]);
		}
	}
	fclose(writer); writer=NULL;
	
	//elu
	for(int i=0; i<N; ++i) ff[i]=xmin+dx*i;
	NN::Transfer::tf_elu(ff,fd);
	writer=fopen("tfunc_elu.dat","w");
	if(writer!=NULL){
		fprintf(writer,"#X F D\n");
		for(int i=0; i<N; ++i){
			fprintf(writer,"%f %f %f\n",xmin+dx*i,ff[i],fd[i]);
		}
	}
	fclose(writer); writer=NULL;
	
	//gelu
	for(int i=0; i<N; ++i) ff[i]=xmin+dx*i;
	NN::Transfer::tf_gelu(ff,fd);
	writer=fopen("tfunc_gelu.dat","w");
	if(writer!=NULL){
		fprintf(writer,"#X F D\n");
		for(int i=0; i<N; ++i){
			fprintf(writer,"%f %f %f\n",xmin+dx*i,ff[i],fd[i]);
		}
	}
	fclose(writer); writer=NULL;
	
	//swish
	for(int i=0; i<N; ++i) ff[i]=xmin+dx*i;
	NN::Transfer::tf_swish(ff,fd);
	writer=fopen("tfunc_swish.dat","w");
	if(writer!=NULL){
		fprintf(writer,"#X F D\n");
		for(int i=0; i<N; ++i){
			fprintf(writer,"%f %f %f\n",xmin+dx*i,ff[i],fd[i]);
		}
	}
	fclose(writer); writer=NULL;
	
	//mish
	for(int i=0; i<N; ++i) ff[i]=xmin+dx*i;
	NN::Transfer::tf_mish(ff,fd);
	writer=fopen("tfunc_mish.dat","w");
	if(writer!=NULL){
		fprintf(writer,"#X F D\n");
		for(int i=0; i<N; ++i){
			fprintf(writer,"%f %f %f\n",xmin+dx*i,ff[i],fd[i]);
		}
	}
	fclose(writer); writer=NULL;
	
	//tanhre
	for(int i=0; i<N; ++i) ff[i]=xmin+dx*i;
	NN::Transfer::tf_tanhre(ff,fd);
	writer=fopen("tfunc_tanhre.dat","w");
	if(writer!=NULL){
		fprintf(writer,"#X F D\n");
		for(int i=0; i<N; ++i){
			fprintf(writer,"%f %f %f\n",xmin+dx*i,ff[i],fd[i]);
		}
	}
	fclose(writer); writer=NULL;
	
}

void test_unit_nn_out(){
	//local function variables
	const int N=100;
	double erra=0,errp=0;
	const int nIn=2;
	const int nOut=3;
	//init rand
	std::srand(std::time(NULL));
	//resize the nn
	NN::ANN nn;
	NN::ANNInit init;
	init.sigma()=1.0;
	init.init()=NN::Init::HE;
	nn.tf()=NN::Transfer::TANH;
	std::vector<int> nh(2);
	nh[0]=7; nh[1]=5;
	//loop over all samples
	for(int n=0; n<N; ++n){
		nn.resize(init,nIn,nh,nOut);
		//set input/output scaling
		nn.inw()=Eigen::VectorXd::Random(nIn);
		nn.inb()=Eigen::VectorXd::Random(nIn);
		nn.outw()=Eigen::VectorXd::Random(nOut);
		nn.outb()=Eigen::VectorXd::Random(nOut);
		//initialize the input nodes
		for(int i=0; i<nn.nIn(); ++i) nn.in()[i]=std::rand()/RAND_MAX-0.5;
		//execute the network
		nn.execute();
		//compute error
		Eigen::VectorXd vec0,vec1,vec2,vec3,vec4;
		Eigen::VectorXd grad1,grad2,grad3;
		Eigen::VectorXd gradd1,gradd2,gradd3;
		vec0=nn.inw().cwiseProduct(nn.in()+nn.inb());
		vec1=nn.edge(0)*vec0+nn.bias(0); grad1=vec1; gradd1=vec1;
		#ifndef NN_COMPUTE_D2
		nn.tfp(0)(vec1,grad1);
		#else
		nn.tfp(0)(vec1,grad1,gradd1);
		#endif
		vec2=nn.edge(1)*vec1+nn.bias(1); grad2=vec2; gradd2=vec2;
		#ifndef NN_COMPUTE_D2
		nn.tfp(1)(vec2,grad2);
		#else
		nn.tfp(1)(vec2,grad2,gradd2);
		#endif
		vec3=nn.edge(2)*vec2+nn.bias(2); grad3=vec3; gradd3=vec3;
		#ifndef NN_COMPUTE_D2
		nn.tfp(2)(vec3,grad3);
		#else
		nn.tfp(2)(vec3,grad3,gradd3);
		#endif
		vec4=nn.outb()+nn.outw().cwiseProduct(vec3);
		erra+=(vec4-nn.out()).norm();
		errp+=(vec4-nn.out()).norm()/vec4.norm()*100;
	}
	//compute the error
	erra/=N;
	errp/=N;
	char* str=new char[print::len_buf];
	//print the results
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<"TEST - ANN - OUT\n";
	std::cout<<"transfer = "<<nn.tf()<<"\n";
	std::cout<<"config   = "<<nn.nIn()<<" "; for(int i=0; i<nh.size(); ++i) std::cout<<nh[i]<<" "; std::cout<<nn.nOut()<<"\n";
	std::cout<<"erra     = "<<erra<<"\n";
	std::cout<<"errp     = "<<errp<<"\n";
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

void test_unit_nn_dOutdVal(){
	//local function variables
	const int N=100;
	double erra=0,errp=0;
	double time=0;
	NN::DOutDVal dOutDVal;
	NN::ANN nn;
	NN::ANNInit init;
	std::vector<int> nh(2);
	const int nIn=4;
	const int nOut=3;
	nh[0]=7; nh[1]=5;
	init.sigma()=1.0;
	init.init()=NN::Init::HE;
	nn.tf()=NN::Transfer::TANH;
	//init rand
	std::srand(std::time(NULL));
	//loop over all samples
	for(int m=0; m<N; ++m){
		//resize the nn
		Eigen::MatrixXd dOutExact,dOutApprox;
		nn.resize(init,nIn,nh,nOut);
		dOutDVal.resize(nn);
		//set input/output scaling
		nn.inw()=Eigen::VectorXd::Random(nIn);
		nn.inb()=Eigen::VectorXd::Random(nIn);
		nn.outw()=Eigen::VectorXd::Random(nOut);
		nn.outb()=Eigen::VectorXd::Random(nOut);
		//initialize the input nodes
		for(int n=0; n<nn.nIn(); ++n) nn.in()[n]=(1.0*std::rand())/RAND_MAX-0.5;
		Eigen::VectorXd in=nn.in();
		//execute the network, compute analytic gradient
		nn.execute();
		clock_t start=std::clock();
		dOutDVal.grad(nn);
		clock_t stop=std::clock();
		time+=((double)(stop-start))/CLOCKS_PER_SEC*1e9/N;
		dOutApprox=dOutDVal.dodi();
		//compute brute force gradient
		dOutExact=Eigen::MatrixXd::Zero(nn.nOut(),nn.nIn());
		Eigen::VectorXd delta=Eigen::VectorXd::Random(nn.nIn())/100.0;
		for(int i=0; i<nn.nIn(); ++i){
			//point 1
			for(int n=0; n<nn.nIn(); ++n) nn.in()[n]=in[n];//reset input
			nn.in()[i]+=delta[i];//add small change
			nn.execute();//execute
			Eigen::VectorXd outNew1=nn.out();//store output
			//point 2
			for(int n=0; n<nn.nIn(); ++n) nn.in()[n]=in[n];//reset input
			nn.in()[i]-=delta[i];//add small change
			nn.execute();//execute
			Eigen::VectorXd outNew2=nn.out();//store output
			//gradient
			for(int j=0; j<nn.out().size(); ++j){
				dOutExact(j,i)=0.5*(outNew1[j]-outNew2[j])/delta[i];
			}
		}
		erra+=(dOutExact-dOutApprox).norm();
		errp+=(dOutExact-dOutApprox).norm()/dOutExact.norm()*100;
	}
	//compute the error
	erra/=N;
	errp/=N;
	time/=N;
	//print the results
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<"TEST - ANN - DOutDVal\n";
	std::cout<<"transfer = "<<nn.tf()<<"\n";
	std::cout<<"config   = "<<nn.nIn()<<" "; for(int i=0; i<nh.size(); ++i) std::cout<<nh[i]<<" "; std::cout<<nn.nOut()<<"\n";
	std::cout<<"erra     = "<<erra<<"\n";
	std::cout<<"errp     = "<<errp<<"\n";
	std::cout<<"time     = "<<time<<"\n";
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

void test_unit_nn_dOutDP(){
	//local function variables
	const int N=100;
	double erra=0,errp=0;
	clock_t start,stop;
	double time=0;
	NN::DOutDP dOutDP;
	NN::ANN nn,nn_copy;
	NN::ANNInit init;
	std::vector<int> nh(3);
	const int nIn=4;
	const int nOut=3;
	nh[0]=7; nh[1]=5; nh[2]=4;
	init.sigma()=1.0;
	init.init()=NN::Init::HE;
	nn.tf()=NN::Transfer::TANH;
	//init rand
	std::srand(std::time(NULL));
	//loop over all samples
	for(int iter=0; iter<N; ++iter){
		//resize the nn
		nn.resize(init,nIn,nh,nOut);
		dOutDP.resize(nn);
		//set input/output scaling
		nn.inw()=Eigen::VectorXd::Random(nIn);
		nn.inb()=Eigen::VectorXd::Random(nIn);
		nn.outw()=Eigen::VectorXd::Random(nOut);
		nn.outb()=Eigen::VectorXd::Random(nOut);
		//initialize the input nodes
		for(int n=0; n<nn.nIn(); ++n) nn.in()[n]=(1.0*std::rand())/RAND_MAX-0.5;
		//execute the network, compute analytic gradient
		nn.execute();
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
				gradBExact[n][l]=Eigen::VectorXd::Zero(nn.bias(l).size());
			}
		}
		for(int l=0; l<nn.nlayer(); ++l){
			for(int i=0; i<nn.bias(l).size(); ++i){
				const double delta=nn.bias(l)[i]/100.0;
				//point 1
				nn_copy=nn;
				nn_copy.bias(l)[i]-=delta;
				nn_copy.execute();
				const Eigen::VectorXd pt1=nn_copy.out();
				//point 2
				nn_copy=nn;
				nn_copy.bias(l)[i]+=delta;
				nn_copy.execute();
				const Eigen::VectorXd pt2=nn_copy.out();
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
				gradWExact[n][l]=Eigen::MatrixXd::Zero(nn.edge(l).rows(),nn.edge(l).cols());
			}
		}
		for(int l=0; l<nn.nlayer(); ++l){
			for(int j=0; j<nn.edge(l).rows(); ++j){
				for(int k=0; k<nn.edge(l).cols(); ++k){
					const double delta=nn.edge(l)(j,k)/100.0;
					//point 1
					nn_copy=nn;
					nn_copy.edge(l)(j,k)-=delta;
					nn_copy.execute();
					const Eigen::VectorXd pt1=nn_copy.out();
					//point 2
					nn_copy=nn;
					nn_copy.edge(l)(j,k)+=delta;
					nn_copy.execute();
					const Eigen::VectorXd pt2=nn_copy.out();
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
			}
		}
		for(int n=0; n<nn.nOut(); ++n){
			for(int l=0; l<nn.nlayer(); ++l){
				//std::cout<<"gradWApprox["<<n<<"]["<<l<<"] = \n"<<gradWApprox[n][l]<<"\n";
				//std::cout<<"gradWExact["<<n<<"]["<<l<<"] = \n"<<gradWExact[n][l]<<"\n";
				erra+=(gradWExact[n][l]-gradWApprox[n][l]).norm();
				errp+=(gradWExact[n][l]-gradWApprox[n][l]).norm()/gradWExact[n][l].norm()*100;
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
	std::cout<<"TEST - ANN - DOutDP\n";
	std::cout<<"transfer = "<<nn.tf()<<"\n";
	std::cout<<"config   = "<<nn.nIn()<<" "; for(int i=0; i<nh.size(); ++i) std::cout<<nh[i]<<" "; std::cout<<nn.nOut()<<"\n";
	std::cout<<"erra     = "<<erra<<"\n";
	std::cout<<"errp     = "<<errp<<"\n";
	std::cout<<"time     = "<<time<<"\n";
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

void test_unit_nn_time(){
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
		NN::ANNInit init;
		init.sigma()=1.0;
		init.init()=NN::Init::HE;
		nn.tf()=NN::Transfer::TANH;
		if(nh[m].size()>=0) nn.resize(init,3,3);
		else nn.resize(init,3,nh[m],3);
		NN::Cost cost(nn);
		start=std::clock();
		for(int n=0; n<N; ++n){
			//initialize the input nodes
			for(int i=0; i<nn.nIn(); ++i) nn.in()[i]=std::rand()/RAND_MAX-0.5;
			//execute the network
			nn.execute();
		}
		stop=std::clock();
		time_exec[m]=((double)(stop-start))/CLOCKS_PER_SEC*1e9/N;
		start=std::clock();
		for(int n=0; n<N; ++n){
			//initialize the input nodes
			for(int i=0; i<nn.nIn(); ++i) nn.in()[i]=std::rand()/RAND_MAX-0.5;
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
	test_unit_nn();
	test_unit_nn_tfunc();
	test_unit_nn_out();
	test_unit_nn_dOutdVal();
	test_unit_nn_dOutDP();
	//test_unit_nn_grad2();
	test_unit_nn_time();
	std::cout<<print::title("ANN",str)<<"\n";
	std::cout<<print::buf(str)<<"\n";
	
	delete[] str;
	
	return 0;
}