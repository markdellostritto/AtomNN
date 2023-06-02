// c
#include <cmath>
#include <cstdlib>
#include <ctime>
// c++
#include <iostream>
#include <vector>
//math
#include "src/math/special.hpp"
//util
#include "src/util/time.hpp"

void test_fma(){
	std::srand(std::time(NULL));
	Clock clock;
	//constants
	const int N=1000;
	std::vector<double> x(N);
	std::vector<double> y(N);
	std::vector<double> z(N);
	for(int i=0; i<N; ++i){
		x[i]=(1.0*std::rand())/RAND_MAX;
		y[i]=(1.0*std::rand())/RAND_MAX;
		z[i]=(1.0*std::rand())/RAND_MAX;
	}
	//standard
	clock.begin();
	for(int i=0; i<N; ++i){
		for(int j=0; j<N; ++j){
			volatile double f=x[j]*y[j]+z[j];
		}
	}
	clock.end();
	const double tstd=clock.duration();
	//fma
	clock.begin();
	for(int i=0; i<N; ++i){
		for(int j=0; j<N; ++j){
			volatile double f=std::fma(x[j],y[j],z[j]);
		}
	}
	clock.end();
	const double tfma=clock.duration();
	//accuracy
	double err=0;
	for(int i=0; i<N; ++i){
		const double f1=x[i]*y[i]+z[i];
		const double f2=std::fma(x[i],y[i],z[i]);
		err+=fabs(f2-f1);
	}
	err/=N;
	//print
	std::cout<<"test - fma\n";
	std::cout<<"time - std = "<<tstd<<"\n";
	std::cout<<"time - fma = "<<tfma<<"\n";
	std::cout<<"err = "<<err<<"\n";
}

void test_sqrtp1m1(){
	std::srand(std::time(NULL));
	Clock clock;
	//constants
	const int N=100000;
	std::vector<double> x(N);
	for(int i=0; i<N; ++i){
		x[i]=(1.0*std::rand())/RAND_MAX;
	}
	//standard
	clock.begin();
	for(int i=0; i<N; ++i){
		volatile double f=std::sqrt(x[i]*x[i]+1.0)-1.0;
	}
	clock.end();
	const double tstd=clock.duration();
	//new
	clock.begin();
	for(int i=0; i<N; ++i){
		volatile double f=std::exp(0.5*math::special::logp1(x[i]*x[i]))-1.0;
	}
	clock.end();
	const double tnew=clock.duration();
	//accuracy
	double err=0;
	for(int i=0; i<N; ++i){
		const double f1=std::sqrt(x[i]*x[i]+1.0)-1.0;
		const double f2=std::exp(0.5*math::special::logp1(x[i]*x[i]))-1.0;
		err+=fabs(f2-f1);
	}
	err/=N;
	//print
	std::cout<<"test - sqrtp1m1\n";
	std::cout<<"time - std = "<<tstd<<"\n";
	std::cout<<"time - new = "<<tnew<<"\n";
	std::cout<<"err = "<<err<<"\n";
}

void test_powint(){
	std::srand(std::time(NULL));
	Clock clock;
	//constants
	const int N=1000000;
	std::vector<double> x(N);
	std::vector<double> p(N);
	for(int i=0; i<N; ++i){
		x[i]=(1.0*std::rand())/RAND_MAX*10.0-5.0;
		p[i]=std::rand()%32-16;
	}
	//standard
	clock.begin();
	for(int i=0; i<N; ++i){
		volatile double f=pow(x[i],p[i]);
	}
	clock.end();
	const double tstd=clock.duration();
	//new
	clock.begin();
	for(int i=0; i<N; ++i){
		volatile double f=math::special::powint(x[i],p[i]);
	}
	clock.end();
	const double tnew=clock.duration();
	//accuracy
	double err=0,errp=0;
	for(int i=0; i<N; ++i){
		const double f1=pow(x[i],p[i]);
		const double f2=math::special::powint(x[i],p[i]);
		err+=fabs(f2-f1);
		errp+=fabs((f2-f1)/f1*100.0);
	}
	err/=N;
	errp/=N;
	//print
	std::cout<<"test - powint\n";
	std::cout<<"time - std = "<<tstd<<"\n";
	std::cout<<"time - new = "<<tnew<<"\n";
	std::cout<<"err  = "<<err<<"\n";
	std::cout<<"errp = "<<errp<<"\n";
}

void test_fmexp(){
	std::srand(std::time(NULL));
	Clock clock;
	//constants
	const int N=1000000;
	std::vector<double> x(N);
	const double xmin=-10.0;
	const double xmax=-10.0;
	for(int i=0; i<N; ++i){
		x[i]=(1.0*std::rand())/RAND_MAX*(xmax-xmin)+xmin;
	}
	//standard
	clock.begin();
	for(int i=0; i<N; ++i){
		volatile double f=exp(x[i]);
	}
	clock.end();
	const double tstd=clock.duration();
	//new
	clock.begin();
	for(int i=0; i<N; ++i){
		volatile double f=math::special::fmexp(x[i]);
	}
	clock.end();
	const double tnew=clock.duration();
	//accuracy
	double err=0,errp=0;
	for(int i=0; i<N; ++i){
		const double f1=exp(x[i]);
		const double f2=math::special::fmexp(x[i]);
		err+=fabs(f2-f1);
		errp+=fabs((f2-f1)/f1*100.0);
	}
	err/=N;
	errp/=N;
	//print
	std::cout<<"test - fmexp\n";
	std::cout<<"time - std = "<<tstd<<"\n";
	std::cout<<"time - new = "<<tnew<<"\n";
	std::cout<<"err  = "<<err<<"\n";
	std::cout<<"errp = "<<errp<<"\n";
}

void test_logp1(){
	std::srand(std::time(NULL));
	Clock clock;
	//constants
	const int N=1000;
	const double xmin=0.001;
	const double xmax=10.0;
	std::vector<double> x(N);
	for(int i=0; i<N; ++i){
		x[i]=(1.0*std::rand())/RAND_MAX*xmax-xmin;
	}
	//standard
	clock.begin();
	for(int i=0; i<N; ++i){
		volatile double f=std::log1p(x[i]);
	}
	clock.end();
	const double tstd=clock.duration();
	//new
	clock.begin();
	for(int i=0; i<N; ++i){
		volatile double f=math::special::logp1(x[i]);
	}
	clock.end();
	const double tnew=clock.duration();
	//accuracy
	double err=0,errp=0;
	for(int i=0; i<N; ++i){
		const double f1=std::log1p(x[i]);
		const double f2=math::special::logp1(x[i]);
		err+=fabs(f2-f1);
		errp+=fabs((f2-f1)/f1*100.0);
	}
	err/=N;
	errp/=N;
	//write
	FILE* writer=fopen("logp1.dat","w");
	if(writer!=NULL){
		fprintf(writer,"#x exact approx\n");
		for(int i=0; i<N; ++i){
			fprintf(writer,"%f %f %f\n",x[i],
				std::log1p(x[i]),math::special::logp1(x[i])
			);
		}
		fclose(writer);
		writer=NULL;
	}
	//print
	std::cout<<"x = ["<<xmin<<" : "<<xmax<<"]\n";
	std::cout<<"test - logp1\n";
	std::cout<<"time - std = "<<tstd<<"\n";
	std::cout<<"time - new = "<<tnew<<"\n";
	std::cout<<"err  = "<<err<<"\n";
	std::cout<<"errp = "<<errp<<"\n";
}

void test_tanh(){
	std::srand(std::time(NULL));
	Clock clock;
	//constants
	const int N=1000000;
	const double xmin=-10.0;
	const double xmax=10.0;
	std::vector<double> x(N);
	for(int i=0; i<N; ++i){
		x[i]=(1.0*std::rand())/RAND_MAX*xmax-xmin;
	}
	//standard
	clock.begin();
	for(int i=0; i<N; ++i){
		volatile double f=std::tanh(x[i]);
	}
	clock.end();
	const double tstd=clock.duration();
	//new
	clock.begin();
	for(int i=0; i<N; ++i){
		volatile double f=math::special::tanh(x[i]);
	}
	clock.end();
	const double tnew=clock.duration();
	//accuracy
	double err=0,errp=0;
	for(int i=0; i<N; ++i){
		const double f1=std::tanh(x[i]);
		const double f2=math::special::tanh(x[i]);
		err+=fabs(f2-f1);
		errp+=fabs((f2-f1)/f1*100.0);
	}
	err/=N;
	errp/=N;
	//print
	std::cout<<"x = ["<<xmin<<" : "<<xmax<<"]\n";
	std::cout<<"test - tanh\n";
	std::cout<<"time - std = "<<tstd<<"\n";
	std::cout<<"time - new = "<<tnew<<"\n";
	std::cout<<"err  = "<<err<<"\n";
	std::cout<<"errp = "<<errp<<"\n";
}

void test_sech(){
	std::srand(std::time(NULL));
	Clock clock;
	//constants
	const int N=1000000;
	const double xmin=-10.0;
	const double xmax=10.0;
	std::vector<double> x(N);
	for(int i=0; i<N; ++i){
		x[i]=(1.0*std::rand())/RAND_MAX*xmax-xmin;
	}
	//standard
	clock.begin();
	for(int i=0; i<N; ++i){
		volatile double f=1.0/std::cosh(x[i]);
	}
	clock.end();
	const double tstd=clock.duration();
	//new
	clock.begin();
	for(int i=0; i<N; ++i){
		volatile double f=math::special::sech(x[i]);
	}
	clock.end();
	const double tnew=clock.duration();
	//accuracy
	double err=0,errp=0;
	for(int i=0; i<N; ++i){
		const double f1=1.0/std::cosh(x[i]);
		const double f2=math::special::sech(x[i]);
		err+=fabs(f2-f1);
		errp+=fabs((f2-f1)/f1*100.0);
	}
	err/=N;
	errp/=N;
	//print
	std::cout<<"x = ["<<xmin<<" : "<<xmax<<"]\n";
	std::cout<<"test - sech\n";
	std::cout<<"time - std = "<<tstd<<"\n";
	std::cout<<"time - new = "<<tnew<<"\n";
	std::cout<<"err  = "<<err<<"\n";
	std::cout<<"errp = "<<errp<<"\n";
}

void test_tanhsech(){
	std::srand(std::time(NULL));
	Clock clock;
	//constants
	const int N=1000000;
	const double xmin=-10.0;
	const double xmax=10.0;
	std::vector<double> x(N);
	for(int i=0; i<N; ++i){
		x[i]=(1.0*std::rand())/RAND_MAX*xmax-xmin;
	}
	//standard
	clock.begin();
	for(int i=0; i<N; ++i){
		volatile double fsech=1.0/std::cosh(x[i]);
		volatile double ftanh=std::tanh(x[i]);
	}
	clock.end();
	const double tstd=clock.duration();
	//new
	clock.begin();
	for(int i=0; i<N; ++i){
		double fsech,ftanh;
		math::special::tanhsech(x[i],ftanh,fsech);
	}
	clock.end();
	const double tnew=clock.duration();
	//accuracy
	double err=0,errp=0;
	for(int i=0; i<N; ++i){
		const double ftanh1=std::tanh(x[i]);
		const double fsech1=1.0/std::cosh(x[i]);
		double fsech2,ftanh2;
		math::special::tanhsech(x[i],ftanh2,fsech2);
		err+=fabs(ftanh2-ftanh1);
		err+=fabs(fsech2-fsech1);
		errp+=fabs((ftanh2-ftanh1)/ftanh1)*100.0;
		errp+=fabs((fsech2-fsech1)/fsech1)*100.0;
	}
	err/=N;
	errp/=N;
	//print
	std::cout<<"x = ["<<xmin<<" : "<<xmax<<"]\n";
	std::cout<<"test - tanhsech\n";
	std::cout<<"time - std = "<<tstd<<"\n";
	std::cout<<"time - new = "<<tnew<<"\n";
	std::cout<<"err  = "<<err<<"\n";
	std::cout<<"errp = "<<errp<<"\n";
}

int main(int argc, char* argv[]){
	
	test_fma();
	test_sqrtp1m1();
	test_powint();
	test_fmexp();
	test_logp1();
	test_tanh();
	test_sech();
	test_tanhsech();
	
	return 1;
}