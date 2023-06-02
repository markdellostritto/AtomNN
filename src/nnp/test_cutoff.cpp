// c libaries
#include <ctime>
// c++ libaries
#include <iostream>
// str
#include "src/str/print.hpp"
#include "src/str/string.hpp"
// cutoff
#include "cutoff.hpp"
// basis - radial
#include "basis.hpp"

//**********************************************
// cutoff
//**********************************************

void test_cutoff(cutoff::Name name){
	//constants
	const double rc=6.0;
	const int N=500;
	const double dr=rc/(N-1.0);
	Basis basis(rc,name,cutoff::Norm::UNIT,1);
	//compute error
	double integral=0.5*(basis.cut_func(0.0)+basis.cut_func(rc));
	for(int i=N-2; i>=1; --i){
		integral+=basis.cut_func(i/(N-1.0)*rc);
	}
	integral*=dr;
	double errg=0;
	for(int i=N-2; i>=1; --i){
		const double g=0.5*(basis.cut_func((i+1.0)/(N-1.0)*rc)-basis.cut_func((i-1.0)/(N-1.0)*rc))/dr;
		errg+=std::fabs(g-basis.cut_grad(i/(N-1.0)*rc));
	}
	errg/=(N-1.0);
	//compute time
	clock_t start,stop;
	const int Nt=1e5;
	std::srand(std::time(NULL));
	start=std::clock();
	for(int i=Nt-1; i>=0; i--) volatile double val=basis.cut_func(std::rand()/RAND_MAX*rc);
	stop=std::clock();
	const double timef=((double)(stop-start))/CLOCKS_PER_SEC*1e9/Nt;
	start=std::clock();
	for(int i=Nt-1; i>=0; i--) volatile double val=basis.cut_grad(std::rand()/RAND_MAX*rc);
	stop=std::clock();
	const double timed=((double)(stop-start))/CLOCKS_PER_SEC*1e9/Nt;
	//print
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<"TEST - UNIT - CUTOFF\n";
	std::cout<<"cutoff          = "<<name<<"\n";
	std::cout<<"err  - integral = "<<std::fabs(0.5*rc-integral)<<"\n";
	std::cout<<"err  - gradient = "<<errg<<"\n";
	std::cout<<"time - function = "<<timef<<" ns\n";
	std::cout<<"time - gradient = "<<timed<<" ns\n";
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

//**********************************************
// main
//**********************************************

int main(int argc, char* argv[]){
	
	std::srand(std::time(NULL));
	char* str=new char[print::len_buf];
	
	std::cout<<print::buf(str)<<"\n";
	std::cout<<print::title("TEST - BASIS - CUTOFF",str)<<"\n";
	test_cutoff(cutoff::Name::STEP);
	test_cutoff(cutoff::Name::COS);
	test_cutoff(cutoff::Name::TANH);
	test_cutoff(cutoff::Name::BUMP2);
	test_cutoff(cutoff::Name::BUMP4);
	std::cout<<print::title("TEST - BASIS - CUTOFF",str)<<"\n";
	std::cout<<print::buf(str)<<"\n";
	
	delete[] str;
	
	return 0;
}
