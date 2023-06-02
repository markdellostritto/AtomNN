// c libaries
#include <ctime>
// c++ libaries
#include <iostream>
// str
#include "src/str/print.hpp"
// ml
#include "src/ml/nn.hpp"

void test_tfunc_deriv(NN::Neuron type){
	//local function variables
	const int N=1000;
	const double xmin=-5;
	const double xmax=5;
	const double dx=(xmax-xmin)/(N-1.0);
	Eigen::VectorXd z,a,d,d2;
	z=Eigen::VectorXd::Zero(N);
	a=Eigen::VectorXd::Zero(N);
	d=Eigen::VectorXd::Zero(N);
	d2=Eigen::VectorXd::Zero(N);
	double errd1=0,errd2=0;
	
	//set the function pointer
	typedef void (*TFP)(const VecXd&,VecXd&,VecXd&,VecXd&);
	TFP func;
	switch(type){
		case NN::Neuron::LINEAR:  func=NN::AFFPBP2::af_lin; break;
		case NN::Neuron::SIGMOID: func=NN::AFFPBP2::af_sigmoid; break;
		case NN::Neuron::TANH:    func=NN::AFFPBP2::af_tanh; break;
		case NN::Neuron::ISRU:    func=NN::AFFPBP2::af_isru; break;
		case NN::Neuron::ARCTAN:  func=NN::AFFPBP2::af_arctan; break;
		case NN::Neuron::RELU:    func=NN::AFFPBP2::af_relu; break;
		case NN::Neuron::GELU:    func=NN::AFFPBP2::af_gelu; break;
		case NN::Neuron::ELU:     func=NN::AFFPBP2::af_elu; break;
		case NN::Neuron::SOFTPLUS:func=NN::AFFPBP2::af_softplus; break;
		case NN::Neuron::SWISH:   func=NN::AFFPBP2::af_swish; break;
		case NN::Neuron::MISH:    func=NN::AFFPBP2::af_mish; break;
		case NN::Neuron::LOGCOSH: func=NN::AFFPBP2::af_logcosh; break;
		case NN::Neuron::TANHRE:  func=NN::AFFPBP2::af_tanhre; break;
		case NN::Neuron::IQUAD:   func=NN::AFFPBP2::af_iquad; break;
		case NN::Neuron::IFABS:   func=NN::AFFPBP2::af_ifabs; break;
		case NN::Neuron::SINSIG:  func=NN::AFFPBP2::af_sinsig; break;
		default: throw std::invalid_argument("Invalid transfer function.");
	}
	
	//compute the derivative
	for(int i=0; i<N; ++i) z[i]=xmin+dx*i;
	(*func)(z,a,d,d2);
	for(int i=1; i<N-1; ++i){
		const double d11=d[i];
		const double d12=0.5*(a[i+1]-a[i-1])/dx;
		errd1+=std::fabs(d12-d11);
		const double d21=d2[i];
		const double d22=0.5*(d[i+1]-d[i-1])/dx;
		errd2+=std::fabs(d22-d21);
	}
	errd1/=N;
	errd2/=N;
	
	//print the results
	std::cout<<"transfer "<<type<<" errd1 "<<errd1<<" errd2 "<<errd2<<"\n";
}

void test_tfunc_time(NN::Neuron type){
	//local function variables
	const int N=10000000;
	const double xmin=-5;
	const double xmax=5;
	const double dx=(xmax-xmin)/(N-1.0);
	Eigen::VectorXd z,a,d,d2;
	z=Eigen::VectorXd::Zero(N);
	a=Eigen::VectorXd::Zero(N);
	d=Eigen::VectorXd::Zero(N);
	d2=Eigen::VectorXd::Zero(N);
	double err=0;
	
	//set the function pointer
	typedef void (*TFP)(const VecXd&,VecXd&,VecXd&,VecXd&);
	TFP func;
	switch(type){
		case NN::Neuron::LINEAR:  func=NN::AFFPBP2::af_lin; break;
		case NN::Neuron::SIGMOID: func=NN::AFFPBP2::af_sigmoid; break;
		case NN::Neuron::TANH:    func=NN::AFFPBP2::af_tanh; break;
		case NN::Neuron::ISRU:    func=NN::AFFPBP2::af_isru; break;
		case NN::Neuron::ARCTAN:  func=NN::AFFPBP2::af_arctan; break;
		case NN::Neuron::RELU:    func=NN::AFFPBP2::af_relu; break;
		case NN::Neuron::GELU:    func=NN::AFFPBP2::af_gelu; break;
		case NN::Neuron::ELU:     func=NN::AFFPBP2::af_elu; break;
		case NN::Neuron::SOFTPLUS:func=NN::AFFPBP2::af_softplus; break;
		case NN::Neuron::SWISH:   func=NN::AFFPBP2::af_swish; break;
		case NN::Neuron::MISH:    func=NN::AFFPBP2::af_mish; break;
		case NN::Neuron::LOGCOSH: func=NN::AFFPBP2::af_logcosh; break;
		case NN::Neuron::TANHRE:  func=NN::AFFPBP2::af_tanhre; break;
		case NN::Neuron::IQUAD:   func=NN::AFFPBP2::af_iquad; break;
		case NN::Neuron::IFABS:   func=NN::AFFPBP2::af_ifabs; break;
		case NN::Neuron::SINSIG:  func=NN::AFFPBP2::af_sinsig; break;
		default: throw std::invalid_argument("Invalid transfer function.");
	}
	
	//compute the derivative
	for(int i=0; i<N; ++i) z[i]=xmin+dx*i;
	clock_t start=std::clock();
	(*func)(z,a,d,d2);
	clock_t stop=std::clock();
	double time=((double)(stop-start))/CLOCKS_PER_SEC*1e9/N;
	
	//print the results
	std::cout<<"transfer "<<type<<" time "<<time<<"\n";
}

void test_tfunc_write(NN::Neuron type, const char* file){
	//local function variables
	const int N=10000;
	const double xmin=-5;
	const double xmax=5;
	const double dx=(xmax-xmin)/(N-1.0);
	Eigen::VectorXd z,a,d,d2;
	z=Eigen::VectorXd::Zero(N);
	a=Eigen::VectorXd::Zero(N);
	d=Eigen::VectorXd::Zero(N);
	d2=Eigen::VectorXd::Zero(N);
	double err=0;
	
	//set the function pointer
	typedef void (*TFP)(const VecXd&,VecXd&,VecXd&,VecXd&);
	TFP func;
	switch(type){
		case NN::Neuron::LINEAR:  func=NN::AFFPBP2::af_lin; break;
		case NN::Neuron::SIGMOID: func=NN::AFFPBP2::af_sigmoid; break;
		case NN::Neuron::TANH:    func=NN::AFFPBP2::af_tanh; break;
		case NN::Neuron::ISRU:    func=NN::AFFPBP2::af_isru; break;
		case NN::Neuron::ARCTAN:  func=NN::AFFPBP2::af_arctan; break;
		case NN::Neuron::RELU:    func=NN::AFFPBP2::af_relu; break;
		case NN::Neuron::GELU:    func=NN::AFFPBP2::af_gelu; break;
		case NN::Neuron::ELU:     func=NN::AFFPBP2::af_elu; break;
		case NN::Neuron::SOFTPLUS:func=NN::AFFPBP2::af_softplus; break;
		case NN::Neuron::SWISH:   func=NN::AFFPBP2::af_swish; break;
		case NN::Neuron::MISH:    func=NN::AFFPBP2::af_mish; break;
		case NN::Neuron::LOGCOSH: func=NN::AFFPBP2::af_logcosh; break;
		case NN::Neuron::TANHRE:  func=NN::AFFPBP2::af_tanhre; break;
		case NN::Neuron::IQUAD:   func=NN::AFFPBP2::af_iquad; break;
		case NN::Neuron::IFABS:   func=NN::AFFPBP2::af_ifabs; break;
		case NN::Neuron::SINSIG:  func=NN::AFFPBP2::af_sinsig; break;
		default: throw std::invalid_argument("Invalid transfer function.");
	}
	
	//compute the derivative
	for(int i=0; i<N; ++i) z[i]=xmin+dx*i;
	(*func)(z,a,d,d2);
	
	//write the results
	FILE* writer=fopen(file,"w");
	if(writer!=NULL){
		fprintf(writer,"z a d d2\n");
		for(int i=0; i<N; ++i){
			fprintf(writer,"%f %f %f %f\n",z[i],a[i],d[i],d2[i]);
		}
		fclose(writer);
		writer=NULL;
	}
}

int main(int argc, char* argv[]){
	std::srand(std::time(NULL));
	char* str=new char[print::len_buf];
	
	std::cout<<print::buf(str)<<"\n";
	std::cout<<print::title("TEST - NN - TRANSFER - DERIV",str)<<"\n";
	test_tfunc_deriv(NN::Neuron::LINEAR);
	test_tfunc_deriv(NN::Neuron::SIGMOID);
	test_tfunc_deriv(NN::Neuron::TANH);
	test_tfunc_deriv(NN::Neuron::ARCTAN);
	test_tfunc_deriv(NN::Neuron::ISRU);
	test_tfunc_deriv(NN::Neuron::RELU);
	test_tfunc_deriv(NN::Neuron::GELU);
	test_tfunc_deriv(NN::Neuron::ELU);
	test_tfunc_deriv(NN::Neuron::SOFTPLUS);
	test_tfunc_deriv(NN::Neuron::SWISH);
	test_tfunc_deriv(NN::Neuron::MISH);
	test_tfunc_deriv(NN::Neuron::LOGCOSH);
	test_tfunc_deriv(NN::Neuron::TANHRE);
	test_tfunc_deriv(NN::Neuron::IQUAD);
	test_tfunc_deriv(NN::Neuron::IFABS);
	test_tfunc_deriv(NN::Neuron::SINSIG);
	std::cout<<print::title("TEST - NN - TRANSFER - DERIV",str)<<"\n";
	std::cout<<print::buf(str)<<"\n";
	
	std::cout<<print::buf(str)<<"\n";
	std::cout<<print::title("TEST - NN - TRANSFER - TIME",str)<<"\n";
	test_tfunc_time(NN::Neuron::LINEAR);
	test_tfunc_time(NN::Neuron::SIGMOID);
	test_tfunc_time(NN::Neuron::TANH);
	test_tfunc_time(NN::Neuron::ARCTAN);
	test_tfunc_time(NN::Neuron::ISRU);
	test_tfunc_time(NN::Neuron::RELU);
	test_tfunc_time(NN::Neuron::GELU);
	test_tfunc_time(NN::Neuron::ELU);
	test_tfunc_time(NN::Neuron::SOFTPLUS);
	test_tfunc_time(NN::Neuron::SWISH);
	test_tfunc_time(NN::Neuron::MISH);
	test_tfunc_time(NN::Neuron::LOGCOSH);
	test_tfunc_time(NN::Neuron::TANHRE);
	test_tfunc_time(NN::Neuron::IQUAD);
	test_tfunc_time(NN::Neuron::IFABS);
	test_tfunc_time(NN::Neuron::SINSIG);
	std::cout<<print::title("TEST - NN - TRANSFER - TIME",str)<<"\n";
	std::cout<<print::buf(str)<<"\n";
	
	//test_tfunc_write(NN::Neuron::SINSIG,"sinsig.dat");
	
	delete[] str;
	
	return 0;
}