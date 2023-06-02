// c libaries
#include <ctime>
// c++ libaries
#include <iostream>
// str
#include "src/str/print.hpp"
#include "src/str/string.hpp"
// basis - radial
#include "basis_radial.hpp"

//**********************************************
// symm
//**********************************************

void test_symm_grad(PhiRN phirn){
	//rand
	std::srand(std::time(NULL));
	//constants
	const double rc=6.0;
	const int N=100;
	const double dr=rc/(N-1.0);
	BasisR basisR(rc,cutoff::Name::COS,cutoff::Norm::UNIT,1,phirn);
	const double eta=(1.0*std::rand())/RAND_MAX*0.5*rc;
	const double rs=(1.0*std::rand())/RAND_MAX*0.5*rc;
	double r=0;
	//compute error
	double errg=0;
	for(int i=1; i<N-1; ++i){
		//finite difference
		const double r2=(i+1.0)/(N-1.0)*rc;
		const double f2=basisR.symmf(r2,eta,rs);
		const double r1=(i-1.0)/(N-1.0)*rc;
		const double f1=basisR.symmf(r1,eta,rs);
		const double g=(f2-f1)/(r2-r1);
		//exact
		r=i/(N-1.0)*rc;
		errg+=std::fabs(g-basisR.symmd(r,eta,rs));
	}
	errg/=(N-1.0);
	//compute time
	clock_t start,stop;
	const int Nt=1e5;
	start=std::clock();
	for(int i=0; i<Nt; ++i){
		r=std::rand()/RAND_MAX*rc;
		volatile double val=basisR.symmf(r,eta,rs);
	}
	stop=std::clock();
	const double timef=((double)(stop-start))/CLOCKS_PER_SEC*1e9/Nt;
	start=std::clock();
	for(int i=0; i<Nt; ++i){
		r=std::rand()/RAND_MAX*rc;
		volatile double val=basisR.symmd(r,eta,rs);
	}
	stop=std::clock();
	const double timed=((double)(stop-start))/CLOCKS_PER_SEC*1e9/Nt;
	//print
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<"TEST - UNIT - SYMM\n";
	std::cout<<"NAME = "<<phirn<<"\n";
	std::cout<<"err - grad = "<<errg<<"\n";
	std::cout<<"time - function = "<<timef<<" ns\n";
	std::cout<<"time - gradient = "<<timed<<" ns\n";
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

void write_symm(PhiRN phirn, const char* fname){
	//constants
	const double rc=6.0;
	const int N=100;
	const double dr=rc/(N-1.0);
	const double nf=6;
	BasisR basisR(rc,cutoff::Name::COS,cutoff::Norm::UNIT,nf,phirn);
	std::vector<double> rs(nf,1.0);
	std::vector<double> eta(nf,0.0);
	eta[0]=0.10;
	eta[1]=0.25;
	eta[2]=0.50;
	eta[3]=1.00;
	eta[4]=2.00;
	eta[5]=5.00;
	//open file
	FILE* writer=fopen(fname,"w");
	if(writer!=NULL){
		for(int i=0; i<N; ++i){
			const double r=i/(N-1.0)*rc;
			fprintf(writer,"%f ",r);
			for(int j=0; j<nf; ++j){
				const double f=basisR.symmf(r,eta[j],rs[j]);
				const double d=basisR.symmd(r,eta[j],rs[j]);
				fprintf(writer,"%f %f ",f,d);
			}
			fprintf(writer,"\n",r);
		}
		fclose(writer);
		writer=NULL;
	} else std::cout<<"ERROR: Could not open output file.\n";
}

//**********************************************
// main
//**********************************************

int main(int argc, char* argv[]){
	
	std::srand(std::time(NULL));
	char* str=new char[print::len_buf];
	
	std::cout<<print::buf(str)<<"\n";
	std::cout<<print::title("TEST - BASIS - RADIAL - GRADIENT",str)<<"\n";
	test_symm_grad(PhiRN::GAUSSIAN);
	test_symm_grad(PhiRN::TANH);
	test_symm_grad(PhiRN::SOFTPLUS);
	test_symm_grad(PhiRN::LOGCOSH);
	test_symm_grad(PhiRN::SWISH);
	test_symm_grad(PhiRN::MISH);
	std::cout<<print::title("TEST - BASIS - RADIAL - GRADIENT",str)<<"\n";
	std::cout<<print::buf(str)<<"\n";
	
	std::cout<<print::buf(str)<<"\n";
	std::cout<<print::title("TEST - BASIS - RADIAL - WRITE",str)<<"\n";
	write_symm(PhiRN::GAUSSIAN,"symm_gauss.dat");
	write_symm(PhiRN::TANH,"symm_tanh.dat");
	write_symm(PhiRN::SOFTPLUS,"symm_softplus.dat");
	write_symm(PhiRN::LOGCOSH,"symm_logcosh.dat");
	write_symm(PhiRN::SWISH,"symm_swish.dat");
	write_symm(PhiRN::MISH,"symm_mish.dat");
	std::cout<<print::title("TEST - BASIS - RADIAL - WRITE",str)<<"\n";
	std::cout<<print::buf(str)<<"\n";
	
	delete[] str;
	
	return 0;
}
