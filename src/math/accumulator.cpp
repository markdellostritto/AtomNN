//c++ libraries
#include <iostream>
#include <stdexcept>
#include <complex>
// ann - acuumulator
#include "src/math/accumulator.hpp"

//***********************************************
//Velocity
//***********************************************

/*void Vel::push(double pos){
	if(t_==0) vel_=pos;
	if(t_==1) vel_=(pos-vel_)/ts_;
}*/

//***********************************************
//Distribution
//***********************************************

//constructors/destructors

Dist::Dist(const Dist& dist){
	if(DEBUG_ACC_DIST>0) std::cout<<"Dist::Dost(const Dist&):\n";
	//init
	nbins_=0; buf_=0;
	bc_=0; gc_=0;
	len_=0; min_=0; max_=0;
	hist_=NULL;
	bufx_=NULL;
	bufy_=NULL;
	//copy
	min_=dist.min();
	max_=dist.max();
	nbins_=dist.nbins();
	buf_=dist.buf();
	bc_=dist.bc();
	gc_=dist.gc();
	if(nbins_>0) len_=(max_-min_)/nbins_;
	if(nbins_>0) hist_=new double[nbins_];
	if(buf_>0) bufx_=new double[buf_];
	if(buf_>0) bufy_=new double[buf_];
	for(int i=0; i<nbins_; ++i) hist_[i]=dist.hist(i);
	for(int i=0; i<buf_; ++i) bufx_[i]=dist.bufx(i);
	for(int i=0; i<buf_; ++i) bufy_[i]=dist.bufy(i);
}

Dist::~Dist(){
	if(DEBUG_ACC_DIST>0) std::cout<<"Dist::~Dist():\n";
	if(hist_!=NULL) delete[] hist_;
	if(bufx_!=NULL) delete[] bufx_;
	if(bufy_!=NULL) delete[] bufy_;
}

//operators

bool operator==(const Dist& dist1, const Dist& dist2){
	return (
		dist1.nbins()==dist2.nbins() &&
		dist1.len()==dist2.len() &&
		dist1.min()==dist2.min() && 
		dist1.max()==dist2.max()
	);
}

Dist& Dist::operator+=(const Dist& dist){
	//check that the dist is compatible
	if(*this!=dist) throw std::invalid_argument("Incompatible Dist.");
	//average the two histograms
	for(int i=0; i<nbins_; ++i) hist_[i]=0.5*(hist_[i]+dist.hist(i));
	//update the global count
	gc_+=dist.gc();
	//push the other buffer into own buffer
	for(int n=0; n<dist.bc(); ++n) push(dist.bufx(n));
	for(int n=0; n<dist.bc(); ++n) push(dist.bufy(n));
	//return dist
	return *this;
}

Dist& Dist::operator=(const Dist& dist){
	if(DEBUG_ACC_DIST>0) std::cout<<"Dist::operator=(const Dist&):\n";
	clear();
	min_=dist.min();
	max_=dist.max();
	nbins_=dist.nbins();
	buf_=dist.buf();
	bc_=dist.bc();
	gc_=dist.gc();
	if(nbins_>0) len_=(max_-min_)/nbins_;
	if(nbins_>0) hist_=new double[nbins_];
	if(buf_>0) bufx_=new double[buf_];
	if(buf_>0) bufy_=new double[buf_];
	for(int i=0; i<nbins_; ++i) hist_[i]=dist.hist(i);
	for(int i=0; i<buf_; ++i) bufx_[i]=dist.bufx(i);
	for(int i=0; i<buf_; ++i) bufy_[i]=dist.bufy(i);
	return *this;
}

//member functions

void Dist::clear(){
	if(DEBUG_ACC_DIST>0) std::cout<<"Dist::clear():\n";
	nbins_=0; buf_=0;
	bc_=0; gc_=0;
	len_=0; min_=0; max_=0;
	norm_=true;
	if(hist_!=NULL) delete[] hist_; hist_=NULL;
	if(bufx_!=NULL) delete[] bufx_; bufx_=NULL;
	if(bufy_!=NULL) delete[] bufy_; bufy_=NULL;
}

void Dist::init(double min, double max, int nbins, int buf){
	if(DEBUG_ACC_DIST>0) std::cout<<"Dist::init(double min, double max, int nbins, int buf=1):\n";
	if(min>=max) throw std::runtime_error("Invalid Dist limits.");
	if(nbins==0) throw std::runtime_error("Invalid Dist nbins.");
	if(buf==0) throw std::runtime_error("Invalid Dist buf.");
	if(hist_!=NULL) delete[] hist_; hist_=NULL;
	if(bufx_!=NULL) delete[] bufx_; bufx_=NULL;
	if(bufy_!=NULL) delete[] bufy_; bufy_=NULL;
	bc_=0; gc_=0;
	min_=min;
	max_=max;
	nbins_=nbins;
	buf_=buf;
	if(nbins_>0) len_=(max-min)/nbins_;
	if(nbins_>0) hist_=new double[nbins_];
	if(buf_>0) bufx_=new double[buf_];
	if(buf_>0) bufy_=new double[buf_];
}

int Dist::bin(double x){
	if(DEBUG_ACC_DIST>0) std::cout<<"Dist::bin(double):\n";
	int uLim=nbins_;
	int lLim=0;
	int mid;
	while(uLim-lLim>1){
		mid=lLim+(uLim-lLim)/2;
		if(min_+len_*lLim<=x && x<=min_+len_*mid) uLim=mid;
		else lLim=mid;
	}
	return lLim;
}

void Dist::push(double x){
	if(DEBUG_ACC_DIST>0) std::cout<<"Dist::push(double):\n";
	bufx_[bc_]=x;
	bufy_[bc_]=1.0;
	++bc_;
	if(bc_==buf_){
		if(norm_){
			for(int i=0; i<nbins_; ++i) hist_[i]*=gc_/(gc_+1.0);
			++gc_;
			for(int i=0; i<buf_; ++i) hist_[bin(bufx_[i])]+=bufy_[i]/(buf_*gc_);
			bc_=0;
		} else {
			++gc_;
			for(int i=0; i<buf_; ++i) hist_[bin(bufx_[i])]+=bufy_[i];
			bc_=0;
		}
	}
}

void Dist::push(double x, double y){
	if(DEBUG_ACC_DIST>0) std::cout<<"Dist::push(double,double):\n";
	bufx_[bc_]=x;
	bufy_[bc_]=y;
	++bc_;
	if(bc_==buf_){
		if(norm_){
			for(int i=0; i<nbins_; ++i) hist_[i]*=gc_/(gc_+1.0);
			++gc_;
			for(int i=0; i<buf_; ++i) hist_[bin(bufx_[i])]+=bufy_[i]/(buf_*gc_);
			bc_=0;
		} else {
			++gc_;
			for(int i=0; i<buf_; ++i) hist_[bin(bufx_[i])]+=bufy_[i];
			bc_=0;
		}
	}
}

//***********************************************
//Fourier
//***********************************************

#ifdef ACC_FFT

//constants

const std::complex<double> Fourier::I=std::complex<double>(0.0,1.0);

//operators

bool operator==(const Fourier& f1, const Fourier& f2){
	return (f1.N()==f2.N());
}

Fourier& Fourier::operator+=(const Fourier& fourier){
	//check compatibility
	if(*this!=fourier) throw std::invalid_argument("Incompatible Fourier");
	//average the fourier transforms
	for(int n=0; n<N_; ++n){
		f_[n]=0.5*(f_[n]+fourier.f(n));
	}
	//push other buffer onto own buffer
	for(int i=0; i<fourier.n(); ++i){
		push(fourier.fft().in(i));
	}
	//return fourier
	return *this;
}

//member functions

void Fourier::clear(){
	fft_.clear();
	if(f_!=NULL) delete[] f_;
	f_=NULL;
	n_=0;
	N_=0;
}

void Fourier::init(int N){
	n_=0;
	N_=N;
	fft_=fourier::FFT_R2C(N_);
	f_=new fftw_complex[N_];
	for(int i=N_-1; i>=0; --i){f_[i][0]=0;f_[i][1]=0;};
}

void Fourier::push(double x){
	if(n_<N_-1) fft_.in(n_++)=x;
	else {
		fft_.in(n_++)=x;
		fft_.transformf();
		for(int k=N_-1; k>=0; --k){
			f_[k][0]+=fft_.out(k)[0];
			f_[k][1]+=fft_.out(k)[1];
		}
		n_=0;
	}
}

#endif