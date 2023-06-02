// c libaries
#include <ctime>
// c++ libaries
#include <iostream>
// str
#include "src/str/print.hpp"
#include "src/str/string.hpp"
// basis - angular
#include "basis_angular.hpp"
// math
#include "src/math/reduce.hpp"

//**********************************************
// symm
//**********************************************

void test_symm_grad_phi(PhiAN phian){
	//rand
	std::srand(std::time(NULL));
	//error
	double erra=0;
	double errp=0;
	double timef=0;
	double timed=0;
	Reduce<2> reduce;
	const int N=500;
	const int M=100;
	for(int iter=0; iter<M; ++iter){
		//constants
		const double rc=6.0;
		const double da=math::constant::PI/(N-1.0);
		const double dc=std::cos(da);
		BasisA basisA(rc,cutoff::Name::COS,cutoff::Norm::UNIT,1,phian);
		const double zeta=(1.0*std::rand())/RAND_MAX*(8.0-1.0)+1.0;
		int alpha=0.0;
		if(phian==PhiAN::GAUSS) alpha=2;
		else if(phian==PhiAN::IPOWP || phian==PhiAN::IPOWS) alpha=5;
		else if(phian==PhiAN::SECHP || phian==PhiAN::SECHS) alpha=2;
		const int lambda=((1.0*std::rand())/RAND_MAX<0.5)?1:-1;
		const double eta=(1.0*std::rand())/RAND_MAX*0.5*rc*5;
		const double rs=(1.0*std::rand())/RAND_MAX*0.5*rc*5;
		const double dr[3]={
			(1.0*std::rand())/RAND_MAX*(0.2*rc-1.0)+0.0,
			(1.0*std::rand())/RAND_MAX*(0.2*rc-1.0)+0.0,
			(1.0*std::rand())/RAND_MAX*(0.2*rc-1.0)+0.0
		};
		double cos=0;
		//compute error
		//std::cout<<"eta = "<<eta<<"\n";
		//std::cout<<"zeta = "<<zeta<<"\n";
		//std::cout<<"alpha = "<<alpha<<"\n";
		//std::cout<<"lambda = "<<lambda<<"\n";
		//std::cout<<"dr = "<<dr[0]<<" "<<dr[1]<<" "<<dr[2]<<"\n";
		for(int i=1; i<N-1; ++i){
			double fphi;
			double feta[3];
			//finite difference
			const double cos2=std::cos((i+1.0)/(N-1.0)*da);
			const double f2=basisA.symmf(cos2,dr,eta,zeta,lambda,alpha);
			const double cos1=std::cos((i-1.0)/(N-1.0)*da);
			const double f1=basisA.symmf(cos1,dr,eta,zeta,lambda,alpha);
			const double g=(f2-f1)/(cos2-cos1);
			//exact
			cos=std::cos((i*1.0)/(N-1.0)*da);
			basisA.symmd(fphi,feta,cos,dr,eta,zeta,lambda,alpha);
			erra+=std::fabs(g-fphi);
			errp+=std::fabs(g-fphi)/std::fabs(g)*100.0;
			reduce.push(g,fphi);
			//std::cout<<"g "<<g<<" "<<fphi<<"\n";
		}
	}
	const int Mt=1;
	for(int iter=0; iter<Mt; ++iter){
		//constants
		const double rc=6.0;
		const double da=math::constant::PI/(N-1.0);
		const double dc=std::cos(da);
		BasisA basisA(rc,cutoff::Name::COS,cutoff::Norm::UNIT,1,phian);
		const double zeta=(1.0*std::rand())/RAND_MAX*(8.0-1.0)+1.0;
		int alpha=0.0;
		if(phian==PhiAN::GAUSS) alpha=2;
		else if(phian==PhiAN::IPOWP || phian==PhiAN::IPOWS) alpha=5;
		else if(phian==PhiAN::SECHP || phian==PhiAN::SECHS) alpha=2;
		const int lambda=((1.0*std::rand())/RAND_MAX<0.5)?1:-1;
		const double eta=(1.0*std::rand())/RAND_MAX*0.5*rc*5;
		const double rs=(1.0*std::rand())/RAND_MAX*0.5*rc*5;
		const double dr[3]={
			(1.0*std::rand())/RAND_MAX*(0.2*rc-1.0)+0.0,
			(1.0*std::rand())/RAND_MAX*(0.2*rc-1.0)+0.0,
			(1.0*std::rand())/RAND_MAX*(0.2*rc-1.0)+0.0
		};
		double cos=0;
		//compute time
		clock_t start,stop;
		const int Nt=1e5;
		start=std::clock();
		for(int i=0; i<Nt; ++i){
			cos=std::cos((i+1.0)/(N-1.0)*math::constant::PI);
			volatile double val=basisA.symmf(cos,dr,eta,zeta,lambda,alpha);
		}
		stop=std::clock();
		timef+=((double)(stop-start))/CLOCKS_PER_SEC*1e9/Nt;
		start=std::clock();
		for(int i=0; i<Nt; ++i){
			cos=std::cos((i+1.0)/(N-1.0)*math::constant::PI);
			double fphi;
			double feta[3];
			basisA.symmd(fphi,feta,cos,dr,eta,zeta,lambda,alpha);
		}
		stop=std::clock();
		timed+=((double)(stop-start))/CLOCKS_PER_SEC*1e9/Nt;
	}
	timef/=Mt;
	timed/=Mt;
	erra/=(N-1.0)*M;
	errp/=(N-1.0)*M;
	//print
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<"TEST - SYMM - GRAD - PHI\n";
	std::cout<<"NAME = "<<phian<<"\n";
	std::cout<<"err - grad - abs = "<<erra<<"\n";
	std::cout<<"err - grad - per = "<<errp<<"\n";
	std::cout<<"correlation - m  = "<<reduce.m()<<"\n";
	std::cout<<"correlation - r2 = "<<reduce.r2()<<"\n";
	std::cout<<"time - function = "<<timef<<" ns\n";
	std::cout<<"time - gradient = "<<timed<<" ns\n";
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

void test_symm_grad_dr(PhiAN phian, int index){
	//rand
	std::srand(std::time(NULL));
	//error
	double erra=0;
	double errp=0;
	double timef=0;
	double timed=0;
	const int M=100;
	const int N=500;
	Reduce<2> reduce;
	for(int iter=0; iter<M; ++iter){
		//constants
		const double rc=6.0;
		const double da=math::constant::PI/(N-1.0);
		const double dc=std::cos(da);
		BasisA basisA(rc,cutoff::Name::COS,cutoff::Norm::UNIT,1,phian);
		const double dr=rc/(N-1.0);
		const double cos=std::cos((1.0*std::rand())/RAND_MAX*math::constant::PI);
		const double zeta=((1.0*std::rand())/RAND_MAX*(8.0-1.0)+1.0)/5.0;
		int alpha=0;
		if(phian==PhiAN::GAUSS) alpha=2;
		else if(phian==PhiAN::IPOWP || phian==PhiAN::IPOWS) alpha=5;
		else if(phian==PhiAN::SECHP || phian==PhiAN::SECHS) alpha=2;
		const int lambda=((1.0*std::rand())/RAND_MAX<0.5)?1:-1;
		const double eta=(1.0*std::rand())/RAND_MAX*0.5*rc*5;
		const double rs=(1.0*std::rand())/RAND_MAX*0.5*rc*5;
		double rr[3]={
			(1.0*std::rand())/RAND_MAX*(0.2*rc-1.0)+0.0,
			(1.0*std::rand())/RAND_MAX*(0.2*rc-1.0)+0.0,
			(1.0*std::rand())/RAND_MAX*(0.2*rc-1.0)+0.0
		};
		//compute error
		//std::cout<<"eta = "<<eta<<"\n";
		//std::cout<<"zeta = "<<zeta<<"\n";
		//std::cout<<"alpha = "<<alpha<<"\n";
		//std::cout<<"lambda = "<<lambda<<"\n";
		//std::cout<<"dr = "<<rr[0]<<" "<<rr[1]<<" "<<rr[2]<<"\n";
		for(int i=1; i<N-1; ++i){
			double fphi;
			double feta[3];
			//finite difference
			const double r2=(i+1.0)/(N-1.0)*rc;
			rr[index]=r2;
			const double f2=basisA.symmf(cos,rr,eta,zeta,lambda,alpha);
			const double r1=(i-1.0)/(N-1.0)*rc;
			rr[index]=r1;
			const double f1=basisA.symmf(cos,rr,eta,zeta,lambda,alpha);
			const double g=(f2-f1)/(r2-r1);
			//exact
			const double r0=(i*1.0)/(N-1.0)*rc;
			rr[index]=r0;
			basisA.symmd(fphi,feta,cos,rr,eta,zeta,lambda,alpha);
			erra+=std::fabs(g-feta[index]);
			if(g!=0) errp+=std::fabs(g-feta[index])/std::fabs(g)*100.0;
			reduce.push(g,feta[index]);
			//std::cout<<"g "<<g<<" "<<feta[index]<<"\n";
		}
	}
	const int Mt=50;
	for(int iter=0; iter<Mt; ++iter){
		//constants
		const double rc=6.0;
		const double da=math::constant::PI/(N-1.0);
		const double dc=std::cos(da);
		BasisA basisA(rc,cutoff::Name::COS,cutoff::Norm::UNIT,1,phian);
		const double dr=rc/(N-1.0);
		const double cos=std::cos((1.0*std::rand())/RAND_MAX*math::constant::PI);
		const double zeta=((1.0*std::rand())/RAND_MAX*(8.0-1.0)+1.0)/5.0;
		int alpha=0;
		if(phian==PhiAN::GAUSS) alpha=2;
		else if(phian==PhiAN::IPOWP || phian==PhiAN::IPOWS) alpha=5;
		else if(phian==PhiAN::SECHP || phian==PhiAN::SECHS) alpha=2;
		const int lambda=((1.0*std::rand())/RAND_MAX<0.5)?1:-1;
		const double eta=(1.0*std::rand())/RAND_MAX*0.5*rc*5;
		const double rs=(1.0*std::rand())/RAND_MAX*0.5*rc*5;
		double rr[3]={
			(1.0*std::rand())/RAND_MAX*(0.2*rc-1.0)+0.0,
			(1.0*std::rand())/RAND_MAX*(0.2*rc-1.0)+0.0,
			(1.0*std::rand())/RAND_MAX*(0.2*rc-1.0)+0.0
		};
		//compute time
		clock_t start,stop;
		const int Nt=1e5;
		start=std::clock();
		for(int i=0; i<Nt; ++i){
			volatile double val=basisA.symmf(cos,rr,eta,zeta,lambda,alpha);
		}
		stop=std::clock();
		timef+=((double)(stop-start))/CLOCKS_PER_SEC*1e9/Nt;
		start=std::clock();
		for(int i=0; i<Nt; ++i){
			double fphi;
			double feta[3];
			basisA.symmd(fphi,feta,cos,rr,eta,zeta,lambda,alpha);
		}
		stop=std::clock();
		timed+=((double)(stop-start))/CLOCKS_PER_SEC*1e9/Nt;
	}
	timef/=Mt;
	timed/=Mt;
	erra/=(N-1.0)*(M);
	errp/=(N-1.0)*(M);
	//print
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<"TEST - SYMM - GRAD - DR["<<index<<"]\n";
	std::cout<<"NAME = "<<phian<<"\n";
	std::cout<<"err - grad - abs = "<<erra<<"\n";
	std::cout<<"err - grad - per = "<<errp<<"\n";
	std::cout<<"correlation - m  = "<<reduce.m()<<"\n";
	std::cout<<"correlation - r2 = "<<reduce.r2()<<"\n";
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
	std::cout<<print::title("TEST - BASIS - ANGULAR - GRADIENT",str)<<"\n";
	test_symm_grad_phi(PhiAN::GAUSS);
	test_symm_grad_phi(PhiAN::IPOWP);
	test_symm_grad_phi(PhiAN::IPOWS);
	test_symm_grad_phi(PhiAN::SECHP);
	test_symm_grad_phi(PhiAN::SECHS);
	std::cout<<print::buf(str)<<"\n";
	
	test_symm_grad_dr(PhiAN::GAUSS,0);
	test_symm_grad_dr(PhiAN::GAUSS,1);
	test_symm_grad_dr(PhiAN::IPOWP,0);
	test_symm_grad_dr(PhiAN::IPOWP,1);
	test_symm_grad_dr(PhiAN::IPOWS,0);
	test_symm_grad_dr(PhiAN::IPOWS,1);
	test_symm_grad_dr(PhiAN::SECHP,0);
	test_symm_grad_dr(PhiAN::SECHP,1);
	test_symm_grad_dr(PhiAN::SECHS,0);
	test_symm_grad_dr(PhiAN::SECHS,1);
	
	delete[] str;
	
	return 0;
}
