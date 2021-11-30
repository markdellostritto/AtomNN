// c libaries
#include <ctime>
// c++ libaries
#include <iostream>
// ann - str
#include "src/str/print.hpp"
#include "src/str/string.hpp"
// ann - ml
#include "src/ml/nn.hpp"
// ann - nnp
#include "src/nnp/nnp.hpp"
#include "src/nnp/basis_radial.hpp"
#include "src/nnp/basis_angular.hpp"
#include "src/nnp/test_unit_nnp.hpp"
// ann - cutoff
#include "cutoff.hpp"
// ann - symmetry functions
#include "symm_radial_t1.hpp"
#include "symm_radial_g1.hpp"
#include "symm_radial_g2.hpp"
#include "symm_angular_g3.hpp"
#include "symm_angular_g4.hpp"

//**********************************************
// cutoff
//**********************************************

void test_cutoff_cos(){
	const double rc=6.0;
	const int N=100;
	const double dr=rc/(N-1.0);
	cutoff::Cos cutf=cutoff::Cos(rc);
	//compute error
	double integral=0.5*(cutf.val(0.0)+cutf.val(rc));
	for(int i=N-2; i>=1; --i){
		integral+=cutf.val(i/(N-1.0)*rc);
	}
	integral*=dr;
	double errg=0;
	for(int i=N-2; i>=1; --i){
		const double g=0.5*(cutf.val((i+1.0)/(N-1.0)*rc)-cutf.val((i-1.0)/(N-1.0)*rc))/dr;
		errg+=std::fabs(g-cutf.grad(i/(N-1.0)*rc));
	}
	errg/=(N-1.0);
	//compute time
	double timef,timed;
	clock_t start,stop;
	const int Nt=1e5;
	std::srand(std::time(NULL));
	start=std::clock();
	for(int i=Nt-1; i>=0; i--) volatile double val=cutf.val(std::rand()/RAND_MAX*rc);
	stop=std::clock();
	timef=((double)(stop-start))/CLOCKS_PER_SEC*1e9/Nt;
	start=std::clock();
	for(int i=Nt-1; i>=0; i--) volatile double val=cutf.grad(std::rand()/RAND_MAX*rc);
	stop=std::clock();
	timed=((double)(stop-start))/CLOCKS_PER_SEC*1e9/Nt;
	//print
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<"TEST - UNIT - CUTOFF - COS\n";
	std::cout<<"err  - integral = "<<std::fabs(0.5*rc-integral)<<"\n";
	std::cout<<"err  - gradient = "<<errg<<"\n";
	std::cout<<"time - function = "<<timef<<" ns\n";
	std::cout<<"time - gradient = "<<timed<<" ns\n";
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

//**********************************************
// symm
//**********************************************

void test_symm_t1(){
	const double rc=6.0;
	const int N=100;
	const double dr=rc/(N-1.0);
	cutoff::Cos cutf=cutoff::Cos(rc);
	//compute error
	double errg=0,r;
	PhiR_T1 t1(1.4268,0.56278905);
	for(int i=N-2; i>=1; --i){
		r=(i+1.0)/(N-1.0)*rc;
		const double f2=t1.val(r,cutf.val(r));
		r=(i-1.0)/(N-1.0)*rc;
		const double f1=t1.val(r,cutf.val(r));
		const double g=0.5*(f2-f1)/dr;
		r=i/(N-1.0)*rc;
		const double cut=cutf.val(r);
		const double gcut=cutf.grad(r);
		errg+=std::fabs(g-t1.grad(r,cut,gcut));
	}
	errg/=(N-1.0);
	//compute time
	double timef,timed;
	clock_t start,stop;
	const int Nt=1e5;
	std::srand(std::time(NULL));
	start=std::clock();
	for(int i=Nt-1; i>=0; i--){
		r=std::rand()/RAND_MAX*rc;
		volatile double val=t1.val(r,cutf.val(r));
	}
	stop=std::clock();
	timef=((double)(stop-start))/CLOCKS_PER_SEC*1e9/Nt;
	start=std::clock();
	for(int i=Nt-1; i>=0; i--){
		r=std::rand()/RAND_MAX*rc;
		const double cut=cutf.val(r);
		const double gcut=cutf.grad(r);
		volatile double val=t1.grad(r,cut,gcut);
	}
	stop=std::clock();
	timed=((double)(stop-start))/CLOCKS_PER_SEC*1e9/Nt;
	//serialization
	PhiR_T1 t1c(1.5427856487,0.816578674);
	const int size=serialize::nbytes(t1);
	char* arr=new char[size];
	const int size_pack=serialize::pack(t1,arr);
	const int size_unpack=serialize::unpack(t1c,arr);
	const double errs=(
		std::abs(t1.rs-t1c.rs)
		+std::abs(t1.eta-t1c.eta)
		+std::abs(size_pack-size)
		+std::abs(size_unpack-size)
	);
	delete[] arr;
	//print
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<"TEST - UNIT - SYMM - T1\n";
	std::cout<<"N  = "<<N<<"\n";
	std::cout<<"rc = "<<rc<<"\n";
	std::cout<<"err - grad = "<<errg<<"\n";
	std::cout<<"time - function = "<<timef<<" ns\n";
	std::cout<<"time - gradient = "<<timed<<" ns\n";
	std::cout<<"err - s = "<<errs<<"\n";
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

void test_symm_g1(){
	const double rc=6.0;
	const int N=100;
	const double dr=rc/(N-1.0);
	cutoff::Cos cutf=cutoff::Cos(rc);
	//compute error
	double errg=0,r;
	PhiR_G1 g1;
	for(int i=N-2; i>=1; --i){
		r=(i+1.0)/(N-1.0)*rc;
		const double f2=g1.val(r,cutf.val(r));
		r=(i-1.0)/(N-1.0)*rc;
		const double f1=g1.val(r,cutf.val(r));
		const double g=0.5*(f2-f1)/dr;
		r=i/(N-1.0)*rc;
		const double cut=cutf.val(r);
		const double gcut=cutf.grad(r);
		errg+=std::fabs(g-g1.grad(r,cut,gcut));
	}
	errg/=(N-1.0);
	//compute time
	double timef,timed;
	clock_t start,stop;
	const int Nt=1e5;
	std::srand(std::time(NULL));
	start=std::clock();
	for(int i=Nt-1; i>=0; i--){
		r=std::rand()/RAND_MAX*rc;
		volatile double val=g1.val(r,cutf.val(r));
	}
	stop=std::clock();
	timef=((double)(stop-start))/CLOCKS_PER_SEC*1e9/Nt;
	start=std::clock();
	for(int i=Nt-1; i>=0; i--){
		r=std::rand()/RAND_MAX*rc;
		const double cut=cutf.val(r);
		const double gcut=cutf.grad(r);
		volatile double val=g1.grad(r,cut,gcut);
	}
	stop=std::clock();
	timed=((double)(stop-start))/CLOCKS_PER_SEC*1e9/Nt;
	//serialization
	PhiR_G1 g1c;
	const int size=serialize::nbytes(g1);
	char* arr=new char[size];
	const int size_pack=serialize::pack(g1,arr);
	const int size_unpack=serialize::unpack(g1c,arr);
	const double errs=(
		std::abs(size_pack-size)
		+std::abs(size_unpack-size)
	);
	delete[] arr;
	//print
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<"TEST - UNIT - SYMM - G1\n";
	std::cout<<"N  = "<<N<<"\n";
	std::cout<<"rc = "<<rc<<"\n";
	std::cout<<"err - grad = "<<errg<<"\n";
	std::cout<<"time - function = "<<timef<<" ns\n";
	std::cout<<"time - gradient = "<<timed<<" ns\n";
	std::cout<<"err - s = "<<errs<<"\n";
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

void test_symm_g2(){
	const double rc=6.0;
	const int N=100;
	cutoff::Cos cutf=cutoff::Cos(rc);
	//compute error
	const double dr=rc/(N-1.0);
	double errg=0,r;
	PhiR_G2 g2(1.4268,0.56278905);
	for(int i=N-2; i>=1; --i){
		r=(i+1.0)/(N-1.0)*rc;
		const double f2=g2.val(r,cutf.val(r));
		r=(i-1.0)/(N-1.0)*rc;
		const double f1=g2.val(r,cutf.val(r));
		const double g=0.5*(f2-f1)/dr;
		r=i/(N-1.0)*rc;
		const double cut=cutf.val(r);
		const double gcut=cutf.grad(r);
		errg+=std::fabs(g-g2.grad(r,cut,gcut));
	}
	errg/=(N-1.0);
	//compute time
	double timef,timed;
	clock_t start,stop;
	const int Nt=1e5;
	std::srand(std::time(NULL));
	start=std::clock();
	for(int i=Nt-1; i>=0; i--){
		r=std::rand()/RAND_MAX*rc;
		volatile double val=g2.val(r,cutf.val(r));
	}
	stop=std::clock();
	timef=((double)(stop-start))/CLOCKS_PER_SEC*1e9/Nt;
	start=std::clock();
	for(int i=Nt-1; i>=0; i--){
		r=std::rand()/RAND_MAX*rc;
		const double cut=cutf.val(r);
		const double gcut=cutf.grad(r);
		volatile double val=g2.grad(r,cut,gcut);
	}
	stop=std::clock();
	timed=((double)(stop-start))/CLOCKS_PER_SEC*1e9/Nt;
	//serialization
	PhiR_G2 g2c;
	const int size=serialize::nbytes(g2);
	char* arr=new char[size];
	const int size_pack=serialize::pack(g2,arr);
	const int size_unpack=serialize::unpack(g2c,arr);
	const double errs=(
		std::abs(g2.rs-g2c.rs)
		+std::abs(g2.eta-g2c.eta)
		+std::abs(size_pack-size)
		+std::abs(size_unpack-size)
	);
	delete[] arr;
	//print
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<"TEST - UNIT - SYMM - G2\n";
	std::cout<<"N  = "<<N<<"\n";
	std::cout<<"rc = "<<rc<<"\n";
	std::cout<<"err - grad = "<<errg<<"\n";
	std::cout<<"time - function = "<<timef<<" ns\n";
	std::cout<<"time - gradient = "<<timed<<" ns\n";
	std::cout<<"err - s = "<<errs<<"\n";
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

void test_symm_g3(){
	//local variables
	const double rc=6.0;
	const int N=100;
	const double dr=rc/(N-1.0);
	double errgd[3]={0,0,0};
	double errga=0;
	double r[3],c[3];
	PhiA_G3 g3(1.4268,2.5,1);
	const double cos=1.0/std::sqrt(2.0);
	cutoff::Cos cutf=cutoff::Cos(rc);
	//grad - dist - 0
	for(int i=N-2; i>=1; --i){
		//assign arrays
		r[0]=1.472896; r[1]=1.5728; r[2]=1.9587892;
		c[0]=cutf.val(r[0]); c[1]=cutf.val(r[1]); c[2]=cutf.val(r[2]);
		//first point
		r[0]=(i+1.0)/(N-1.0)*rc; c[0]=cutf.val(r[0]);
		const double f2=g3.val(cos,r,c);
		//second point
		r[0]=(i-1.0)/(N-1.0)*rc; c[0]=cutf.val(r[0]);
		const double f1=g3.val(cos,r,c);
		//gradient - approx
		const double g=0.5*(f2-f1)/dr;
		//gradient - exact
		r[0]=i/(N-1.0)*rc; c[0]=cutf.val(r[0]);
		const double gcut=cutf.grad(r[0]);
		//error
		errgd[0]+=std::fabs(g-g3.grad_dist_0(r,c,gcut)*g3.angle(cos));
	}
	errgd[0]/=(N-1.0);
	//grad - dist - 1
	for(int i=N-2; i>=1; --i){
		//assign arrays
		r[0]=1.472896; r[1]=1.5728; r[2]=1.9587892;
		c[0]=cutf.val(r[0]); c[1]=cutf.val(r[1]); c[2]=cutf.val(r[2]);
		//first point
		r[1]=(i+1.0)/(N-1.0)*rc; c[1]=cutf.val(r[1]);
		const double f2=g3.val(cos,r,c);
		//second point
		r[1]=(i-1.0)/(N-1.0)*rc; c[1]=cutf.val(r[1]);
		const double f1=g3.val(cos,r,c);
		//gradient - approx
		const double g=0.5*(f2-f1)/dr;
		//gradient - exact
		r[1]=i/(N-1.0)*rc; c[1]=cutf.val(r[1]);
		const double gcut=cutf.grad(r[1]);
		//error
		errgd[1]+=std::fabs(g-g3.grad_dist_1(r,c,gcut)*g3.angle(cos));
	}
	errgd[1]/=(N-1.0);
	//grad - dist - 2
	for(int i=N-2; i>=1; --i){
		//assign arrays
		r[0]=1.472896; r[1]=1.5728; r[2]=1.9587892;
		c[0]=cutf.val(r[0]); c[1]=cutf.val(r[1]); c[2]=cutf.val(r[2]);
		//first point
		r[2]=(i+1.0)/(N-1.0)*rc; c[2]=cutf.val(r[2]);
		const double f2=g3.val(cos,r,c);
		//second point
		r[2]=(i-1.0)/(N-1.0)*rc; c[2]=cutf.val(r[2]);
		const double f1=g3.val(cos,r,c);
		//gradient - approx
		const double g=0.5*(f2-f1)/dr;
		//gradient - exact
		r[2]=i/(N-1.0)*rc; c[2]=cutf.val(r[2]);
		const double gcut=cutf.grad(r[2]);
		//error
		errgd[2]+=std::fabs(g-g3.grad_dist_2(r,c,gcut)*g3.angle(cos));
	}
	errgd[2]/=(N-1.0);
	//grad - angle
	for(int i=N-2; i>=1; --i){
		double cosv;
		//assign arrays
		r[0]=1.472896; r[1]=1.5728; r[2]=1.9587892;
		c[0]=cutf.val(r[0]); c[1]=cutf.val(r[1]); c[2]=cutf.val(r[2]);
		//first point
		cosv=(i+1.0)/(N-1.0)*math::constant::PI;
		const double f2=g3.val(cosv,r,c);
		//second point
		cosv=(i-1.0)/(N-1.0)*math::constant::PI;
		const double f1=g3.val(cosv,r,c);
		//gradient - approx
		const double g=0.5*(f2-f1)/dr;
		//gradient - exact
		cosv=i/(N-1.0)*math::constant::PI;
		const double gcut=cutf.grad(r[2]);
		//error
		errga+=std::fabs(g-g3.grad_angle(cosv)*g3.dist(r,c));
	}
	errga/=(N-1.0);
	//time - value
	double timef,timed;
	clock_t start,stop;
	const int Nt=1e5;
	std::srand(std::time(NULL));
	start=std::clock();
	for(int i=Nt-1; i>=0; i--){
		r[0]=std::rand()/RAND_MAX*rc;
		r[1]=std::rand()/RAND_MAX*rc;
		r[2]=std::rand()/RAND_MAX*rc;
		c[0]=cutf.val(r[0]);
		c[1]=cutf.val(r[1]);
		c[2]=cutf.val(r[2]);
		volatile double val=g3.val(cos,r,c);
	}
	stop=std::clock();
	timef=((double)(stop-start))/CLOCKS_PER_SEC*1e9/Nt;
	start=std::clock();
	for(int i=Nt-1; i>=0; i--){
		r[0]=std::rand()/RAND_MAX*rc;
		r[1]=std::rand()/RAND_MAX*rc;
		r[2]=std::rand()/RAND_MAX*rc;
		c[0]=cutf.val(r[0]);
		c[1]=cutf.val(r[1]);
		c[2]=cutf.val(r[2]);
		const double gcut=cutf.grad(r[0]);
		volatile double val=g3.grad_dist_2(r,c,gcut)*g3.angle(cos);
	}
	stop=std::clock();
	timed=((double)(stop-start))/CLOCKS_PER_SEC*1e9/Nt;
	//serialization
	PhiA_G3 g3c;
	const int size=serialize::nbytes(g3);
	char* arr=new char[size];
	const int size_pack=serialize::pack(g3,arr);
	const int size_unpack=serialize::unpack(g3c,arr);
	const double errs=(
		std::abs(g3.zeta-g3c.zeta)
		+std::abs(g3.eta-g3c.eta)
		+std::abs(g3.lambda-g3c.lambda)
		+std::abs(size_pack-size)
		+std::abs(size_unpack-size)
	);
	delete[] arr;
	//print
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<"TEST - UNIT - SYMM - G3\n";
	std::cout<<"N  = "<<N<<"\n";
	std::cout<<"rc = "<<rc<<"\n";
	std::cout<<"err - grad - dist[0] = "<<errgd[0]<<"\n";
	std::cout<<"err - grad - dist[1] = "<<errgd[1]<<"\n";
	std::cout<<"err - grad - dist[2] = "<<errgd[2]<<"\n";
	std::cout<<"err - grad - angle   = "<<errga<<"\n";
	std::cout<<"err - s = "<<errs<<"\n";
	std::cout<<"time - function = "<<timef<<" ns\n";
	std::cout<<"time - gradient = "<<timed<<" ns\n";
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

void test_symm_g4(){
	//local variables
	const double rc=6.0;
	const int N=100;
	const double dr=rc/(N-1.0);
	double errgd[3]={0,0,0};
	double errga=0;
	double r[3],c[3];
	PhiA_G4 g4(1.4268,2.5,1);
	const double cos=1.0/std::sqrt(2.0);
	cutoff::Cos cutf=cutoff::Cos(rc);
	//grad - dist - 0
	for(int i=N-2; i>=1; --i){
		//assign arrays
		r[0]=1.472896; r[1]=1.5728; r[2]=1.9587892;
		c[0]=cutf.val(r[0]); c[1]=cutf.val(r[1]); c[2]=cutf.val(r[2]);
		//first point
		r[0]=(i+1.0)/(N-1.0)*rc; c[0]=cutf.val(r[0]);
		const double f2=g4.val(cos,r,c);
		//second point
		r[0]=(i-1.0)/(N-1.0)*rc; c[0]=cutf.val(r[0]);
		const double f1=g4.val(cos,r,c);
		//gradient - approx
		const double g=0.5*(f2-f1)/dr;
		//gradient - exact
		r[0]=i/(N-1.0)*rc; c[0]=cutf.val(r[0]);
		const double gcut=cutf.grad(r[0]);
		//error
		errgd[0]+=std::fabs(g-g4.grad_dist_0(r,c,gcut)*g4.angle(cos));
	}
	errgd[0]/=(N-1.0);
	//grad - dist - 1
	for(int i=N-2; i>=1; --i){
		//assign arrays
		r[0]=1.472896; r[1]=1.5728; r[2]=1.9587892;
		c[0]=cutf.val(r[0]); c[1]=cutf.val(r[1]); c[2]=cutf.val(r[2]);
		//first point
		r[1]=(i+1.0)/(N-1.0)*rc; c[1]=cutf.val(r[1]);
		const double f2=g4.val(cos,r,c);
		//second point
		r[1]=(i-1.0)/(N-1.0)*rc; c[1]=cutf.val(r[1]);
		const double f1=g4.val(cos,r,c);
		//gradient - approx
		const double g=0.5*(f2-f1)/dr;
		//gradient - exact
		r[1]=i/(N-1.0)*rc; c[1]=cutf.val(r[1]);
		const double gcut=cutf.grad(r[1]);
		//error
		errgd[1]+=std::fabs(g-g4.grad_dist_1(r,c,gcut)*g4.angle(cos));
	}
	errgd[1]/=(N-1.0);
	//grad - dist - 2
	for(int i=N-2; i>=1; --i){
		//assign arrays
		r[0]=1.472896; r[1]=1.5728; r[2]=1.9587892;
		c[0]=cutf.val(r[0]); c[1]=cutf.val(r[1]); c[2]=cutf.val(r[2]);
		//first point
		r[2]=(i+1.0)/(N-1.0)*rc; c[2]=cutf.val(r[2]);
		const double f2=g4.val(cos,r,c);
		//second point
		r[2]=(i-1.0)/(N-1.0)*rc; c[2]=cutf.val(r[2]);
		const double f1=g4.val(cos,r,c);
		//gradient - approx
		const double g=0.5*(f2-f1)/dr;
		//gradient - exact
		r[2]=i/(N-1.0)*rc; c[2]=cutf.val(r[2]);
		const double gcut=cutf.grad(r[2]);
		errgd[2]+=std::fabs(g-g4.grad_dist_2(r,c,gcut)*g4.angle(cos));
	}
	errgd[2]/=(N-1.0);
	//grad - angle
	for(int i=N-2; i>=1; --i){
		double cosv;
		//assign arrays
		r[0]=1.472896; r[1]=1.5728; r[2]=1.9587892;
		c[0]=cutf.val(r[0]); c[1]=cutf.val(r[1]); c[2]=cutf.val(r[2]);
		//first point
		cosv=(i+1.0)/(N-1.0)*math::constant::PI;
		const double f2=g4.val(cosv,r,c);
		//second point
		cosv=(i-1.0)/(N-1.0)*math::constant::PI;
		const double f1=g4.val(cosv,r,c);
		//gradient - approx
		const double g=0.5*(f2-f1)/dr;
		//gradient - exact
		cosv=i/(N-1.0)*math::constant::PI;
		errga+=std::fabs(g-g4.grad_angle(cosv)*g4.dist(r,c));
	}
	errga/=(N-1.0);
	//time - value
	double timef,timed;
	clock_t start,stop;
	const int Nt=1e5;
	std::srand(std::time(NULL));
	start=std::clock();
	for(int i=Nt-1; i>=0; i--){
		r[0]=std::rand()/RAND_MAX*rc;
		r[1]=std::rand()/RAND_MAX*rc;
		r[2]=std::rand()/RAND_MAX*rc;
		c[0]=cutf.val(r[0]);
		c[1]=cutf.val(r[1]);
		c[2]=cutf.val(r[2]);
		volatile double val=g4.val(cos,r,c);
	}
	stop=std::clock();
	timef=((double)(stop-start))/CLOCKS_PER_SEC*1e9/Nt;
	start=std::clock();
	for(int i=Nt-1; i>=0; i--){
		r[0]=std::rand()/RAND_MAX*rc;
		r[1]=std::rand()/RAND_MAX*rc;
		r[2]=std::rand()/RAND_MAX*rc;
		c[0]=cutf.val(r[0]);
		c[1]=cutf.val(r[1]);
		c[2]=cutf.val(r[2]);
		const double gcut=cutf.grad(r[0]);
		volatile double val=g4.grad_dist_2(r,c,gcut)*g4.angle(cos);
	}
	stop=std::clock();
	timed=((double)(stop-start))/CLOCKS_PER_SEC*1e9/Nt;
	//serialization
	PhiA_G4 g4c;
	const int size=serialize::nbytes(g4);
	char* arr=new char[size];
	const int size_pack=serialize::pack(g4,arr);
	const int size_unpack=serialize::unpack(g4c,arr);
	const double errs=(
		std::abs(g4.zeta-g4c.zeta)
		+std::abs(g4.eta-g4c.eta)
		+std::abs(g4.lambda-g4c.lambda)
		+std::abs(size_pack-size)
		+std::abs(size_unpack-size)
	);
	delete[] arr;
	//print
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<"TEST - UNIT - SYMM - G4\n";
	std::cout<<"N  = "<<N<<"\n";
	std::cout<<"rc = "<<rc<<"\n";
	std::cout<<"err - gradient - dist[0] = "<<errgd[0]<<"\n";
	std::cout<<"err - gradient - dist[1] = "<<errgd[1]<<"\n";
	std::cout<<"err - gradient - dist[2] = "<<errgd[2]<<"\n";
	std::cout<<"err - gradient - angle   = "<<errga<<"\n";
	std::cout<<"err - s = "<<errs<<"\n";
	std::cout<<"time - function = "<<timef<<" ns\n";
	std::cout<<"time - gradient = "<<timed<<" ns\n";
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

//**********************************************
// nn_pot
//**********************************************

void test_unit_nnh(){
	NNH nnh,nnh_copy;
	NN::ANNInit init;
	//initialize neural network
	std::cout<<"initializing neural network\n";
	std::vector<int> nh(4);
	nh[0]=12; nh[1]=8; nh[2]=4; nh[3]=2;
	init.sigma()=1.0;
	init.init()=NN::Init::HE;
	init.seed()=-1;
	nnh.nn().tf()=NN::Transfer::ARCTAN;
	nnh.nn().resize(init,nh);
	//initialize basis
	std::cout<<"initializing basis\n";
	BasisR basisR; basisR.init_G2(6,cutoff::Norm::UNIT,cutoff::Name::COS,10.0);
	BasisA basisA; basisA.init_G4(4,cutoff::Norm::UNIT,cutoff::Name::COS,10.0);
	std::vector<Atom> species(2);
	species[0].name()="Ar";
	species[0].id()=string::hash("Ar");
	species[0].mass()=5.0;
	species[0].energy()=-7.0;
	species[1].name()="Ne";
	species[1].id()=string::hash("Ne");
	species[1].mass()=12.0;
	species[1].energy()=-9.0;
	nnh.atom()=species[0];
	nnh.resize(species.size());
	nnh.basisR(0)=basisR;
	nnh.basisR(1)=basisR;
	nnh.basisA(0,0)=basisA;
	nnh.basisA(0,1)=basisA;
	nnh.basisA(1,1)=basisA;
	nnh.init_input();
	//print
	std::cout<<nnh<<"\n";
	//pack
	std::cout<<"serialization\n";
	const int memsize=serialize::nbytes(nnh);
	char* memarr=new char[memsize];
	serialize::pack(nnh,memarr);
	serialize::unpack(nnh_copy,memarr);
	delete[] memarr;
	//print
	std::cout<<nnh_copy<<"\n";
}

void test_unit_nnp(){
	NNH nnh;
	NNP nnp,nnp_copy;
	NN::ANNInit init;
	//initialize neural network
	std::cout<<"initializing neural network\n";
	std::vector<int> nh(4);
	nh[0]=12; nh[1]=8; nh[2]=4; nh[3]=2;
	init.sigma()=1.0;
	init.init()=NN::Init::HE;
	init.seed()=-1;
	nnh.nn().tf()=NN::Transfer::ARCTAN;
	nnh.nn().resize(init,nh);
	//initialize basis
	std::cout<<"initializing basis\n";
	BasisR basisR; basisR.init_G2(6,cutoff::Norm::UNIT,cutoff::Name::COS,10.0);
	BasisA basisA; basisA.init_G4(4,cutoff::Norm::UNIT,cutoff::Name::COS,10.0);
	std::vector<Atom> species(2);
	species[0].name()="Ar";
	species[0].id()=string::hash("Ar");
	species[0].mass()=5.0;
	species[0].energy()=-7.0;
	species[1].name()="Ne";
	species[1].id()=string::hash("Ne");
	species[1].mass()=12.0;
	species[1].energy()=-9.0;
	nnh.atom()=species[0];
	nnh.resize(species.size());
	nnh.basisR(0)=basisR;
	nnh.basisR(1)=basisR;
	nnh.basisA(0,0)=basisA;
	nnh.basisA(0,1)=basisA;
	nnh.basisA(1,1)=basisA;
	nnh.init_input();
	//resize potential
	std::cout<<"resizing potential\n";
	nnp.resize(species);
	nnp.nnh(0)=nnh;
	nnp.nnh(1)=nnh;
	//print
	std::cout<<nnp<<"\n";
	//pack
	std::cout<<"serialization\n";
	const int memsize=serialize::nbytes(nnp);
	char* memarr=new char[memsize];
	serialize::pack(nnp,memarr);
	serialize::unpack(nnp_copy,memarr);
	delete[] memarr;
	//print
	std::cout<<nnp_copy<<"\n";
}

void test_unit_nnp_csymm(){
	/*
	//local function variables
	std::vector<Atom> atoms;
	NNP nnpot;
	Structure struc_small;
	Structure struc_large;
	AtomType atomT;
	atomT.name=true; atomT.an=false; atomT.type=true; atomT.index=false;
	atomT.posn=true; atomT.force=false; atomT.symm=true; atomT.charge=false;
	atomT.neigh=true;
	//set the atoms
	std::cout<<"setting atoms\n";
	atoms.resize(1);
	atoms[0].name()="Ar";
	atoms[0].mass()=22.90;
	atoms[0].energy()=0.0;
	atoms[0].charge()=0.0;
	//resize the nnpot
	std::cout<<"setting nnpot\n";
	nnpot.rc()=9.0;
	nnpot.resize(atoms);
	NNP::read_basis("basis_Ar",nnpot,"Ar");
	//print the nnpot
	std::cout<<nnpot<<"\n";
	//read the structures
	std::cout<<"reading structures\n";
	VASP::POSCAR::read("Ar.vasp",atomT,struc_small);
	VASP::POSCAR::read("Ar_l4.vasp",atomT,struc_large);
	//set the type
	for(int i=0; i<struc_small.nAtoms(); ++i) struc_small.type(i)=nnpot.index(struc_small.name(i));
	for(int i=0; i<struc_large.nAtoms(); ++i) struc_large.type(i)=nnpot.index(struc_large.name(i));
	//compute neighbor lists
	Structure::neigh_list(struc_small,nnpot.rc());
	Structure::neigh_list(struc_large,nnpot.rc());
	//init the symmetry functions
	std::cout<<"initializing symmetry functions\n";
	nnpot.init(struc_small);
	nnpot.init(struc_large);
	//compute the symmetry functions
	std::cout<<"computing symmetry functions\n";
	nnpot.symm(struc_small);
	nnpot.symm(struc_large);
	//find average symmetry function
	Eigen::VectorXd symm_small=Eigen::VectorXd::Zero(struc_small.symm(0).size());
	Eigen::VectorXd symm_large=Eigen::VectorXd::Zero(struc_large.symm(0).size());
	std::cout<<"symm_small = "<<struc_small.symm(0).transpose()<<"\n";
	std::cout<<"symm_large = "<<struc_large.symm(0).transpose()<<"\n";
	*/
	/*for(int i=0; i<struc_small.nAtoms(); ++i){
		std::cout<<"symm - small - "<<i<<" = "<<struc_small.symm(i).transpose()<<"\n";
	}
	for(int i=0; i<struc_large.nAtoms(); ++i){
		std::cout<<"symm - large - "<<i<<" = "<<struc_large.symm(i).transpose()<<"\n";
	}*/
}

//**********************************************
// main
//**********************************************

int main(int argc, char* argv[]){
	
	std::srand(std::time(NULL));
	char* str=new char[print::len_buf];
	
	std::cout<<print::buf(str)<<"\n";
	std::cout<<print::title("SYMM",str)<<"\n";
	test_cutoff_cos();
	test_symm_t1();
	test_symm_g1();
	test_symm_g2();
	test_symm_g3();
	test_symm_g4();
	std::cout<<print::title("SYMM",str)<<"\n";
	std::cout<<print::buf(str)<<"\n";
	
	std::cout<<print::buf(str)<<"\n";
	std::cout<<print::title("NN_POT",str)<<"\n";
	test_unit_nnh();
	test_unit_nnp();
	test_unit_nnp_csymm();
	std::cout<<print::title("NN_POT",str)<<"\n";
	std::cout<<print::buf(str)<<"\n";
	
	delete[] str;
	
	return 0;
}
