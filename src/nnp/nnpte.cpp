// mpi
#include <mpi.h>
// c libraries
#include <cstdio>
#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
#include <cmath>
#elif (defined __ICC || defined __INTEL_COMPILER)
#include <mathimf.h> //intel math library
#else
#include <cmath>
#endif
// c++ libraries
#include <iostream>
#include <exception>
#include <algorithm>
#include <random>
#include <chrono>
// structure
#include "src/struc/structure.hpp"
#include "src/struc/neighbor.hpp"
// format
#include "src/format/file_struc.hpp"
#include "src/format/format.hpp"
// math
#include "src/math/reduce.hpp"
#include "src/math/corr.hpp"
#include "src/math/special.hpp"
// string
#include "src/str/string.hpp"
#include "src/str/token.hpp"
#include "src/str/print.hpp"
// chem
#include "src/chem/units.hpp"
#include "src/chem/alias.hpp"
// thread
#include "src/thread/comm.hpp"
#include "src/thread/dist.hpp"
#include "src/thread/mpif.hpp"
// util
#include "src/util/compiler.hpp"
#include "src/util/time.hpp"
// torch
#include "src/torch/pot.hpp"
#include "src/torch/pot_gauss_long.hpp"
#include "src/torch/pot_ldamp_long.hpp"
#include "src/torch/pot_ldamp_cut.hpp"
// nnpte
#include "src/nnp/nnpte.hpp"

static bool compare_pair(const std::pair<int,double>& p1, const std::pair<int,double>& p2){
	return p1.first<p2.first;
}

using math::constant::LOG2;

//************************************************************
// MPI Communicators
//************************************************************

thread::Comm WORLD;// all processors
thread::Comm BATCH;// subgroup for each element of the batch

//************************************************************
// serialization
//************************************************************

namespace serialize{

//**********************************************
// byte measures
//**********************************************

template <> int nbytes(const NNPTE& obj){
   if(NNPTE_PRINT_FUNC>0) std::cout<<"nbytes(const NNPTE&)\n";
	int size=0;
	//nnp
		size+=nbytes(obj.nnp_);
	//input/output
		size+=nbytes(obj.file_params_);
		size+=nbytes(obj.file_error_);
		size+=nbytes(obj.file_ann_);
		size+=nbytes(obj.file_restart_);
	//flags
		size+=sizeof(bool);//restart
		size+=sizeof(bool);//pre-conditioning
		size+=sizeof(bool);//wparams
	//optimization
		size+=nbytes(obj.batch_);
		size+=nbytes(obj.obj_);
		size+=nbytes(obj.algo_);
		size+=nbytes(obj.decay_);
		size+=sizeof(double);//delta_
	//return the size
		return size;
}

//**********************************************
// packing
//**********************************************

template <> int pack(const NNPTE& obj, char* arr){
	if(NNPTE_PRINT_FUNC>0) std::cout<<"pack(const NNPTE&,char*)\n";
	int pos=0;
	//nnp
		pos+=pack(obj.nnp_,arr+pos);
	//input/output
		pos+=pack(obj.file_params_,arr+pos);
		pos+=pack(obj.file_error_,arr+pos);
		pos+=pack(obj.file_ann_,arr+pos);
		pos+=pack(obj.file_restart_,arr+pos);
	//flags
		std::memcpy(arr+pos,&obj.restart_,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(arr+pos,&obj.preCond_,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(arr+pos,&obj.wparams_,sizeof(bool)); pos+=sizeof(bool);
	//optimization
		pos+=pack(obj.batch_,arr+pos);
		pos+=pack(obj.obj_,arr+pos);
		pos+=pack(obj.algo_,arr+pos);
		pos+=pack(obj.decay_,arr+pos);
		std::memcpy(arr+pos,&obj.delta_,sizeof(double)); pos+=sizeof(double);
	//return bytes written
		return pos;
}

//**********************************************
// unpacking
//**********************************************

template <> int unpack(NNPTE& obj, const char* arr){
	if(NNPTE_PRINT_FUNC>0) std::cout<<"unpack(const NNPTE&,char*)\n";
	int pos=0;
	//nnp
		pos+=unpack(obj.nnp_,arr+pos);
	//input/output
		pos+=unpack(obj.file_params_,arr+pos);
		pos+=unpack(obj.file_error_,arr+pos);
		pos+=unpack(obj.file_ann_,arr+pos);
		pos+=unpack(obj.file_restart_,arr+pos);
	//flags
		std::memcpy(&obj.restart_,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(&obj.preCond_,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(&obj.wparams_,arr+pos,sizeof(bool)); pos+=sizeof(bool);
	//optimization
		pos+=unpack(obj.batch_,arr+pos);
		pos+=unpack(obj.obj_,arr+pos);
		pos+=unpack(obj.algo_,arr+pos);
		pos+=unpack(obj.decay_,arr+pos);
		std::memcpy(&obj.delta_,arr+pos,sizeof(double)); pos+=sizeof(double);
		obj.deltai()=1.0/obj.delta();
		obj.delta2()=obj.delta()*obj.delta();
	//return bytes read
		return pos;
}
	
}

//************************************************************
// Mode
//************************************************************

std::ostream& operator<<(std::ostream& out, const Mode& mode){
	switch(mode){
		case Mode::TRAIN: out<<"TRAIN"; break;
		case Mode::TEST: out<<"TEST"; break;
		case Mode::SYMM: out<<"SYMM"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

const char* Mode::name(const Mode& mode){
	switch(mode){
		case Mode::TRAIN: return "TRAIN";
		case Mode::TEST: return "TEST";
		case Mode::SYMM: return "SYMM";
		default: return "UNKNOWN";
	}
}

Mode Mode::read(const char* str){
	if(std::strcmp(str,"TRAIN")==0) return Mode::TRAIN;
	else if(std::strcmp(str,"TEST")==0) return Mode::TEST;
	else if(std::strcmp(str,"SYMM")==0) return Mode::SYMM;
	else return Mode::UNKNOWN;
}

//************************************************************
// NNPTE - Neural Network Potential - Optimization
//************************************************************

std::ostream& operator<<(std::ostream& out, const NNPTE& nnpte){
	char* str=new char[print::len_buf];
	out<<print::buf(str)<<"\n";
	out<<print::title("NNPTE",str)<<"\n";
	//files
	out<<"file_params  = "<<nnpte.file_params_<<"\n";
	out<<"file_error   = "<<nnpte.file_error_<<"\n";
	out<<"file_ann     = "<<nnpte.file_ann_<<"\n";
	out<<"file_restart = "<<nnpte.file_restart_<<"\n";
	//flags
	out<<"restart      = "<<nnpte.restart_<<"\n";
	out<<"pre-cond     = "<<nnpte.preCond_<<"\n";
	out<<"wparams      = "<<nnpte.wparams_<<"\n";
	//optimization
	out<<"batch        = "<<nnpte.batch_<<"\n";
	out<<"algo         = "<<nnpte.algo_<<"\n";
	out<<"decay        = "<<nnpte.decay_<<"\n";
	out<<"n_print      = "<<nnpte.obj().nPrint()<<"\n";
	out<<"n_write      = "<<nnpte.obj().nWrite()<<"\n";
	out<<"max          = "<<nnpte.obj().max()<<"\n";
	out<<"stop         = "<<nnpte.obj().stop()<<"\n";
	out<<"loss         = "<<nnpte.obj().loss()<<"\n";
	out<<"tol          = "<<nnpte.obj().tol()<<"\n";
	out<<"gamma        = "<<nnpte.obj().gamma()<<"\n";
	out<<"delta        = "<<nnpte.delta()<<"\n";
	out<<print::buf(str);
	delete[] str;
	return out;
}

NNPTE::NNPTE(){
	if(NNPTE_PRINT_FUNC>0) std::cout<<"NNP::NNPTE():\n";
	defaults();
};

void NNPTE::defaults(){
	if(NNPTE_PRINT_FUNC>0) std::cout<<"NNP::defaults():\n";
	//nnp
		nTypes_=0;
	//input/output
		file_params_="nnp_params.dat";
		file_error_="nnp_error.dat";
		file_restart_="nnpte.restart";
		file_ann_="ann";
	//flags
		restart_=false;
		preCond_=false;
		wparams_=false;
	//optimization
		delta_=1.0;
		deltai_=1.0;
	//error
		error_[0]=0;//loss - train
		error_[1]=0;//loss - val
		error_[2]=0;//rmse - train
		error_[3]=0;//rmse - val
}

void NNPTE::clear(){
	if(NNPTE_PRINT_FUNC>0) std::cout<<"NNP::clear():\n";
	//elements
		nTypes_=0;
		gElement_.clear();
		pElement_.clear();
	//nnp
		nnp_.clear();
	//optimization
		batch_.clear();
		obj_.clear();
	//error
		error_[0]=0;//loss - train
		error_[1]=0;//loss - val
		error_[2]=0;//rmse - train
		error_[3]=0;//rmse - val
}

void NNPTE::write_restart(const char* file){
	if(NNPTE_PRINT_FUNC>1) std::cout<<"NNPTE::write_restart(const char*):\n";
	//local variables
	char* arr=NULL;
	FILE* writer=NULL;
	bool error=false;
	try{
		//open file
		writer=fopen(file,"wb");
		if(writer==NULL) throw std::runtime_error(std::string("NNPTE::write_restart(const char*): Could not open file: ")+file);
		//allocate buffer
		const int nBytes=serialize::nbytes(*this);
		arr=new char[nBytes];
		if(arr==NULL) throw std::runtime_error("NNPTE::write_restart(const char*): Could not allocate memory.");
		//write to buffer
		serialize::pack(*this,arr);
		//write to file
		const int nWrite=fwrite(arr,sizeof(char),nBytes,writer);
		if(nWrite!=nBytes) throw std::runtime_error("NNPTE::write_restart(const char*): Write error.");
		//close the file, free memory
		delete[] arr; arr=NULL;
		fclose(writer); writer=NULL;
	}catch(std::exception& e){
		std::cout<<e.what()<<"\n";
		error=true;
	}
	//free local variables
	if(arr!=NULL) delete[] arr;
	if(writer!=NULL) fclose(writer);
	if(error) throw std::runtime_error("NNPTE::write_restart(const char*): Failed to write");
}

void NNPTE::read_restart(const char* file){
	if(NNPTE_PRINT_FUNC>0) std::cout<<"NNPTE::read_restart(const char*):\n";
	//local variables
	char* arr=NULL;
	FILE* reader=NULL;
	bool error=false;
	try{
		//open file
		reader=fopen(file,"rb");
		if(reader==NULL) throw std::runtime_error(std::string("NNPTE::read_restart(const char*): Could not open file: ")+std::string(file));
		//find size
		std::fseek(reader,0,SEEK_END);
		const int nBytes=std::ftell(reader);
		std::fseek(reader,0,SEEK_SET);
		//allocate buffer
		arr=new char[nBytes];
		if(arr==NULL) throw std::runtime_error("NNPTE::read_restart(const char*): Could not allocate memory.");
		//read from file
		const int nRead=fread(arr,sizeof(char),nBytes,reader);
		if(nRead!=nBytes) throw std::runtime_error("NNPTE::read_restart(const char*): Read error.");
		//read from buffer
		serialize::unpack(*this,arr);
		//close the file, free memory
		delete[] arr; arr=NULL;
		fclose(reader); reader=NULL;
	}catch(std::exception& e){
		std::cout<<e.what()<<"\n";
		error=true;
	}
	//free local variables
	if(arr!=NULL) delete[] arr;
	if(reader!=NULL) fclose(reader);
	if(error) throw std::runtime_error("NNPTE::read_restart(const char*): Failed to read");
}

void NNPTE::train(int batchSize, const std::vector<Structure>& struc_train, const std::vector<Structure>& struc_val){
	if(NNPTE_PRINT_FUNC>0) std::cout<<"NNPTE::train(NNP&,std::vector<Structure>&,int):\n";
	//====== local function variables ======
	//statistics
		std::vector<int> N;//total number of inputs for each element
		std::vector<Eigen::VectorXd> avg_in;//average of the inputs for each element (nnp_.nSpecies_ x nInput_)
		std::vector<Eigen::VectorXd> max_in;//max of the inputs for each element (nnp_.nSpecies_ x nInput_)
		std::vector<Eigen::VectorXd> min_in;//min of the inputs for each element (nnp_.nSpecies_ x nInput_)
		std::vector<Eigen::VectorXd> dev_in;//average of the stddev for each element (nnp_.nSpecies_ x nInput_)
	//timing
		Clock clock;
	//random
		std::default_random_engine generator;

	if(NNPTE_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"training NN potential\n";
	
	//====== check the parameters ======
	if(batchSize<=0) throw std::invalid_argument("NNPTE::train(int): Invalid batch size.");
	if(struc_train.size()==0) throw std::invalid_argument("NNPTE::train(int): No training data provided.");
	if(struc_val.size()==0) throw std::invalid_argument("NNPTE::train(int): No validation data provided.");
	
	//====== get the number of structures ======
	double nBatchF=(1.0*batchSize)/BATCH.size();
	double nTrainF=(1.0*struc_train.size())/BATCH.size();
	double nValF=(1.0*struc_val.size())/BATCH.size();
	MPI_Allreduce(MPI_IN_PLACE,&nBatchF,1,MPI_DOUBLE,MPI_SUM,WORLD.mpic());
	MPI_Allreduce(MPI_IN_PLACE,&nTrainF,1,MPI_DOUBLE,MPI_SUM,WORLD.mpic());
	MPI_Allreduce(MPI_IN_PLACE,&nValF,1,MPI_DOUBLE,MPI_SUM,WORLD.mpic());
	const int nBatch=std::round(nBatchF);
	const int nTrain=std::round(nTrainF);
	const int nVal=std::round(nValF);
	
	//====== set the distributions over the atoms ======
	dist_atomt.resize(struc_train.size());
	dist_atomv.resize(struc_val.size());
	for(int i=0; i<struc_train.size(); ++i) dist_atomt[i].init(BATCH.size(),BATCH.rank(),struc_train[i].nAtoms());
	for(int i=0; i<struc_val.size(); ++i) dist_atomv[i].init(BATCH.size(),BATCH.rank(),struc_val[i].nAtoms());
	
	//====== initialize the random number generator ======
	rngen_=std::mt19937(std::chrono::system_clock::now().time_since_epoch().count());
	
	//====== resize the optimization data ======
	if(NNPTE_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"resizing the optimization data\n";
	//set the number of types
	nTypes_=nnp_.ntypes();
	//resize per-element arrays
	pElement_.resize(nTypes_);
	gElement_.resize(nTypes_);
	grad_.resize(nTypes_);
	for(int n=0; n<nTypes_; ++n){
		const int nn_size=nnp_.nnh(n).nn().size();
		pElement_[n]=Eigen::VectorXd::Zero(nn_size);
		gElement_[n]=Eigen::VectorXd::Zero(nn_size);
		grad_[n]=Eigen::VectorXd::Zero(nn_size);
	}
	//resize gradient objects
	if(NNPTE_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"resizing gradient data\n";
	cost_.resize(nTypes_);
	for(int n=0; n<nTypes_; ++n){
		cost_[n].resize(nnp_.nnh(n).nn());
	}
	
	//====== compute the number of atoms of each element ======
	if(NNPTE_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"computing the number of atoms of each element\n";
	std::vector<double> nAtoms_(nTypes_,0);
	for(int i=0; i<struc_train.size(); ++i){
		for(int j=0; j<struc_train[i].nAtoms(); ++j){
			++nAtoms_[struc_train[i].type(j)];
		}
	}
	MPI_Allreduce(MPI_IN_PLACE,nAtoms_.data(),nTypes_,MPI_DOUBLE,MPI_SUM,WORLD.mpic());
	for(int i=0; i<nTypes_; ++i) nAtoms_[i]/=BATCH.size();
	if(NNPTE_PRINT_DATA>-1 && WORLD.rank()==0){
		char* strbuf=new char[print::len_buf];
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("ATOM - DATA",strbuf)<<"\n";
		for(int i=0; i<nTypes_; ++i){
			const std::string& name=nnp_.nnh(i).type().name();
			const int index=nnp_.index(nnp_.nnh(i).type().name());
			std::cout<<name<<"("<<index<<") - "<<(int)nAtoms_[i]<<"\n";
		}
		std::cout<<print::buf(strbuf)<<"\n";
		delete[] strbuf;
	}
	
	//====== set the indices and batch size ======
	if(NNPTE_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"setting indices and batch\n";
	batch_.resize(batchSize,struc_train.size());
	
	//====== collect input statistics ======
	//resize arrays
	N.resize(nTypes_);
	max_in.resize(nTypes_);
	min_in.resize(nTypes_);
	avg_in.resize(nTypes_);
	dev_in.resize(nTypes_);
	for(int n=0; n<nTypes_; ++n){
		const int nInput=nnp_.nnh(n).nInput();
		max_in[n]=Eigen::VectorXd::Constant(nInput,-1.0*std::numeric_limits<double>::max());
		min_in[n]=Eigen::VectorXd::Constant(nInput,1.0*std::numeric_limits<double>::max());
		avg_in[n]=Eigen::VectorXd::Zero(nInput);
		dev_in[n]=Eigen::VectorXd::Zero(nInput);
	}
	//compute the total number
	if(NNPTE_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"compute the total number\n";
	for(int n=0; n<struc_train.size(); ++n){
		for(int i=0; i<struc_train[n].nAtoms(); ++i){
			++N[struc_train[n].type(i)];
		}
	}
	//accumulate the number
	for(int i=0; i<nTypes_; ++i){
		double Nloc=(1.0*N[i])/BATCH.size();//normalize by the size of the BATCH group
		MPI_Allreduce(MPI_IN_PLACE,&Nloc,1,MPI_DOUBLE,MPI_SUM,WORLD.mpic());
		N[i]=static_cast<int>(std::round(Nloc));
	}
	//compute the max/min
	if(NNPTE_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"compute the max/min\n";
	for(int n=0; n<struc_train.size(); ++n){
		const Structure& strucl=struc_train[n];
		for(int i=0; i<strucl.nAtoms(); ++i){
			const int index=strucl.type(i);
			for(int k=0; k<nnp_.nnh(index).nInput(); ++k){
				if(strucl.symm(i)[k]>max_in[index][k]) max_in[index][k]=strucl.symm(i)[k];
				if(strucl.symm(i)[k]<min_in[index][k]) min_in[index][k]=strucl.symm(i)[k];
			}
		}
	}
	//accumulate the min/max
	for(int i=0; i<nTypes_; ++i){
		MPI_Allreduce(MPI_IN_PLACE,min_in[i].data(),min_in[i].size(),MPI_DOUBLE,MPI_MIN,WORLD.mpic());
		MPI_Allreduce(MPI_IN_PLACE,max_in[i].data(),max_in[i].size(),MPI_DOUBLE,MPI_MAX,WORLD.mpic());
	}
	//compute the average
	if(NNPTE_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"compute the average\n";
	for(int n=0; n<struc_train.size(); ++n){
		for(int i=0; i<struc_train[n].nAtoms(); ++i){
			avg_in[struc_train[n].type(i)].noalias()+=struc_train[n].symm(i);
		}
	}
	//accumulate the average
	for(int i=0; i<nTypes_; ++i){
		avg_in[i]/=BATCH.size();//normalize by the size of the BATCH group
		MPI_Allreduce(MPI_IN_PLACE,avg_in[i].data(),avg_in[i].size(),MPI_DOUBLE,MPI_SUM,WORLD.mpic());
		avg_in[i]/=N[i];
	}
	//compute the stddev
	if(NNPTE_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"compute the stddev\n";
	for(int n=0; n<struc_train.size(); ++n){
		const Structure& strucl=struc_train[n];
		for(int i=0; i<strucl.nAtoms(); ++i){
			const int index=strucl.type(i);
			dev_in[index].noalias()+=(avg_in[index]-strucl.symm(i)).cwiseProduct(avg_in[index]-strucl.symm(i));
		}
	}
	//accumulate the stddev
	for(int i=0; i<dev_in.size(); ++i){
		for(int j=0; j<dev_in[i].size(); ++j){
			dev_in[i][j]/=BATCH.size();//normalize by the size of the BATCH group
			MPI_Allreduce(MPI_IN_PLACE,&dev_in[i][j],1,MPI_DOUBLE,MPI_SUM,WORLD.mpic());
			dev_in[i][j]=sqrt(dev_in[i][j]/(N[i]-1.0));
		}
	}
	
	//====== precondition the input ======
	std::vector<Eigen::VectorXd> inpb_(nTypes_);//input bias
	std::vector<Eigen::VectorXd> inpw_(nTypes_);//input weight
	for(int n=0; n<nTypes_; ++n){
		inpb_[n]=Eigen::VectorXd::Constant(nnp_.nnh(n).nInput(),0.0);
		inpw_[n]=Eigen::VectorXd::Constant(nnp_.nnh(n).nInput(),1.0);
	}
	if(preCond_){
		if(NNPTE_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"pre-conditioning input\n";
		//set the preconditioning vectors - bias
		for(int i=0; i<inpb_.size(); ++i){
			inpb_[i]=-1*avg_in[i];
		}
		//set the preconditioning vectors - weight
		for(int i=0; i<inpw_.size(); ++i){
			for(int j=0; j<inpw_[i].size(); ++j){
				if(dev_in[i][j]==0) inpw_[i][j]=1;
				else inpw_[i][j]=1.0/(1.0*dev_in[i][j]);
			}
		}
	}
	
	//====== set the bias for each of the species ======
	if(NNPTE_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"setting the bias for each species\n";
	for(int n=0; n<nTypes_; ++n){
		NN::ANN& nn_=nnp_.nnh(n).nn();
		for(int i=0; i<nn_.nInp(); ++i) nn_.inpb()[i]=inpb_[n][i];
		for(int i=0; i<nn_.nInp(); ++i) nn_.inpw()[i]=inpw_[n][i];
		nn_.outb()[0]=0.0;
		nn_.outw()[0]=1.0;
	}
	
	//====== initialize the optimization data ======
	const int nParams=nnp_.size();
	if(restart_){
		//restart
		if(WORLD.rank()==0) std::cout<<"restarting optimization\n";
		if(nParams!=obj_.dim()) throw std::runtime_error(
			std::string("NNPTE::train(int): Network has ")
			+std::to_string(nParams)+std::string(" while opt has ")
			+std::to_string(obj_.dim())+std::string(" parameters.")
		);
	} else {
		//from scratch
		if(WORLD.rank()==0) std::cout<<"starting from scratch\n";
		//resize the optimization objects
		obj_.resize(nParams);
		algo_->resize(nParams);
		//load random initial values in the per-element arrays
		for(int n=0; n<nTypes_; ++n){
			nnp_.nnh(n).nn()>>pElement_[n];
			gElement_[n]=Eigen::VectorXd::Random(nnp_.nnh(n).nn().size())*1e-6;
		}
		//load initial values from per-element arrays into global arrays
		int count=0;
		for(int n=0; n<nTypes_; ++n){
			for(int m=0; m<pElement_[n].size(); ++m){
				obj_.p()[count]=pElement_[n][m];
				obj_.g()[count]=gElement_[n][m];
				++count;
			}
		}
	}
	
	//====== print input statistics and bias ======
	if(NNPTE_PRINT_DATA>-1 && WORLD.rank()==0){
		char* strbuf=new char[print::len_buf];
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("OPT - DATA",strbuf)<<"\n";
		std::cout<<"N-PARAMS    = \n\t"<<nParams<<"\n";
		std::cout<<"AVG - INPUT = \n"; for(int i=0; i<avg_in.size(); ++i) std::cout<<"\t"<<avg_in[i].transpose()<<"\n";
		std::cout<<"MAX - INPUT = \n"; for(int i=0; i<max_in.size(); ++i) std::cout<<"\t"<<max_in[i].transpose()<<"\n";
		std::cout<<"MIN - INPUT = \n"; for(int i=0; i<min_in.size(); ++i) std::cout<<"\t"<<min_in[i].transpose()<<"\n";
		std::cout<<"DEV - INPUT = \n"; for(int i=0; i<dev_in.size(); ++i) std::cout<<"\t"<<dev_in[i].transpose()<<"\n";
		std::cout<<"PRE-BIAS    = \n"; for(int i=0; i<inpb_.size(); ++i) std::cout<<"\t"<<inpb_[i].transpose()<<"\n";
		std::cout<<"PRE-SCALE   = \n"; for(int i=0; i<inpw_.size(); ++i) std::cout<<"\t"<<inpw_[i].transpose()<<"\n";
		std::cout<<print::buf(strbuf)<<"\n";
		delete[] strbuf;
	}
	
	//====== execute the optimization ======
	if(NNPTE_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"executing the optimization\n";
	//optimization variables
	bool fbreak=false;
	const double nBatchi_=1.0/nBatch;
	const double nVali_=1.0/nVal;
	//bcast parameters
	MPI_Bcast(obj_.p().data(),obj_.p().size(),MPI_DOUBLE,0,WORLD.mpic());
	//allocate status vectors
	std::vector<int> step;
	std::vector<double> gamma,rmse_g,loss_t,loss_v,rmse_t,rmse_v;
	std::vector<Eigen::VectorXd> params;
	if(WORLD.rank()==0){
		int size=obj_.max()/obj_.nPrint();
		if(size==0) ++size;
		step.resize(size);
		gamma.resize(size);
		rmse_g.resize(size);
		loss_t.resize(size);
		loss_v.resize(size);
		rmse_t.resize(size);
		rmse_v.resize(size);
		params.resize(size);
	}
	//print status header to standard output
	if(WORLD.rank()==0) printf("opt gamma rmse_g loss_t loss_v rmse_t rmse_v\n");
	//start the clock
	clock.begin();
	
	//begin optimization
	Eigen::VectorXd gtot_=Eigen::VectorXd::Zero(obj_.dim());
	for(int iter=0; iter<obj_.max(); ++iter){
		double error_sum_[4]={0.0,0.0,0.0,0.0};
		//compute the error and gradient
		//error(obj_.p(),struc_train,struc_val);
		error2(obj_.p(),struc_train,struc_val);
		//pack the gradient
		int count=0;
		for(int n=0; n<nTypes_; ++n){
			std::memcpy(gtot_.data()+count,gElement_[n].data(),gElement_[n].size()*sizeof(double));
			count+=gElement_[n].size();
		}
		//accumulate gradient and error
		obj_.g().setZero();
		MPI_Reduce(gtot_.data(),obj_.g().data(),gtot_.size(),MPI_DOUBLE,MPI_SUM,0,WORLD.mpic());
		MPI_Reduce(error_,error_sum_,4,MPI_DOUBLE,MPI_SUM,0,WORLD.mpic());
		if(WORLD.rank()==0){
			//compute error averaged over the batch
			error_[0]=error_sum_[0]*nBatchi_;//loss - train
			error_[1]=error_sum_[1]*nVali_;//loss - val
			error_[2]=sqrt(error_sum_[2]*nBatchi_);//rmse - train
			error_[3]=sqrt(error_sum_[3]*nVali_);//rmse - val
			//compute gradient averaged over the batch
			obj_.g()*=nBatchi_;
			//print/write error
			if(obj_.step()%obj_.nPrint()==0){
				const int t=iter/obj_.nPrint();
				step[t]=obj_.count();
				gamma[t]=obj_.gamma();
				rmse_g[t]=sqrt(obj_.g().squaredNorm()/nParams);
				loss_t[t]=error_[0];
				loss_v[t]=error_[1];
				rmse_t[t]=error_[2];
				rmse_v[t]=error_[3];
				params[t]=obj_.p();
				printf("%8i %12.10e %12.10e %12.10e %12.10e %12.10e %12.10e\n",step[t],gamma[t],rmse_g[t],loss_t[t],loss_v[t],rmse_t[t],rmse_v[t]);
			}
			//write the basis and potentials
			if(obj_.step()%obj_.nWrite()==0){
				if(NNPTE_PRINT_STATUS>1) std::cout<<"writing the restart file and potentials\n";
				//write restart file
				const std::string file_restart=file_restart_+"."+std::to_string(obj_.count());
				this->write_restart(file_restart.c_str());
				//write potential file
				const std::string file_ann=file_ann_+"."+std::to_string(obj_.count());
				NNP::write(file_ann.c_str(),nnp_);
			}
			//compute the new position
			obj_.val()=error_[0];//loss - train
			obj_.gamma()=decay_->step(obj_);
			algo_->step(obj_);
			//compute the difference
			obj_.dv()=std::fabs(obj_.val()-obj_.valOld());
			obj_.dp()=(obj_.p()-obj_.pOld()).norm();
			//set the new "old" values
			obj_.valOld()=obj_.val();//set "old" value
			obj_.pOld()=obj_.p();//set "old" p value
			obj_.gOld()=obj_.g();//set "old" g value
			//check the break condition
			switch(obj_.stop()){
				case opt::Stop::FABS: fbreak=(obj_.val()<obj_.tol()); break;
				case opt::Stop::FREL: fbreak=(obj_.dv()<obj_.tol()); break;
				case opt::Stop::XREL: fbreak=(obj_.dp()<obj_.tol()); break;
			}
		}
		//bcast parameters
		MPI_Bcast(obj_.p().data(),obj_.p().size(),MPI_DOUBLE,0,WORLD.mpic());
		//bcast break condition
		MPI_Bcast(&fbreak,1,MPI_C_BOOL,0,WORLD.mpic());
		if(fbreak) break;
		//increment step
		++obj_.step();
		++obj_.count();
	}
	/*
	//begin optimization
	std::vector<Eigen::VectorXd> gElementT=gElement_;
	for(int iter=0; iter<obj_.max(); ++iter){
		double error_sum_[4]={0.0,0.0,0.0,0.0};
		//compute the error and gradient
		//error(obj_.p(),struc_train,struc_val);
		error2(obj_.p(),struc_train,struc_val);
		//accumulate error
		MPI_Reduce(error_,error_sum_,4,MPI_DOUBLE,MPI_SUM,0,WORLD.mpic());
		//accumulate gradient
		for(int n=0; n<nTypes_; ++n){
			gElementT[n].setZero();
			MPI_Reduce(gElement_[n].data(),gElementT[n].data(),gElementT[n].size(),MPI_DOUBLE,MPI_SUM,0,WORLD.mpic());
		}
		if(WORLD.rank()==0){
			//compute error averaged over the batch
			error_[0]=error_sum_[0]*nBatchi_;//loss - train
			error_[1]=error_sum_[1]*nVali_;//loss - val
			error_[2]=sqrt(error_sum_[2]*nBatchi_);//rmse - train
			error_[3]=sqrt(error_sum_[3]*nVali_);//rmse - val
			//compute gradient averaged over the batch
			for(int n=0; n<nTypes_; ++n) gElementT[n]*=nBatchi_;
			//pack the gradient
			int count=0;
			for(int n=0; n<nTypes_; ++n){
				std::memcpy(obj_.g().data()+count,gElementT[n].data(),gElementT[n].size()*sizeof(double));
				count+=gElementT[n].size();
			}
			//print/write error
			if(obj_.step()%obj_.nPrint()==0){
				const int t=iter/obj_.nPrint();
				step[t]=obj_.count();
				gamma[t]=obj_.gamma();
				rmse_g[t]=sqrt(obj_.g().squaredNorm()/nParams);
				loss_t[t]=error_[0];
				loss_v[t]=error_[1];
				rmse_t[t]=error_[2];
				rmse_v[t]=error_[3];
				params[t]=obj_.p();
				printf("%8i %12.10e %12.10e %12.10e %12.10e %12.10e %12.10e\n",step[t],gamma[t],rmse_g[t],loss_t[t],loss_v[t],rmse_t[t],rmse_v[t]);
			}
			//write the basis and potentials
			if(obj_.step()%obj_.nWrite()==0){
				if(NNPTE_PRINT_STATUS>1) std::cout<<"writing the restart file and potentials\n";
				//write restart file
				const std::string file_restart=file_restart_+"."+std::to_string(obj_.count());
				this->write_restart(file_restart.c_str());
				//write potential file
				const std::string file_ann=file_ann_+"."+std::to_string(obj_.count());
				NNP::write(file_ann.c_str(),nnp_);
			}
			//compute the new position
			obj_.val()=error_[0];//loss - train
			obj_.gamma()=decay_->step(obj_);
			algo_->step(obj_);
			//compute the difference
			obj_.dv()=std::fabs(obj_.val()-obj_.valOld());
			obj_.dp()=(obj_.p()-obj_.pOld()).norm();
			//set the new "old" values
			obj_.valOld()=obj_.val();//set "old" value
			obj_.pOld()=obj_.p();//set "old" p value
			obj_.gOld()=obj_.g();//set "old" g value
			//check the break condition
			switch(obj_.stop()){
				case opt::Stop::FABS: fbreak=(obj_.val()<obj_.tol()); break;
				case opt::Stop::FREL: fbreak=(obj_.dv()<obj_.tol()); break;
				case opt::Stop::XREL: fbreak=(obj_.dp()<obj_.tol()); break;
			}
		}
		//bcast parameters
		MPI_Bcast(obj_.p().data(),obj_.p().size(),MPI_DOUBLE,0,WORLD.mpic());
		//bcast break condition
		MPI_Bcast(&fbreak,1,MPI_C_BOOL,0,WORLD.mpic());
		if(fbreak) break;
		//increment step
		++obj_.step();
		++obj_.count();
	}
	*/
	//compute the training time
	clock.end();
	double time_train=clock.duration();
	MPI_Allreduce(MPI_IN_PLACE,&time_train,1,MPI_DOUBLE,MPI_SUM,WORLD.mpic());
	time_train/=WORLD.size();
	MPI_Barrier(WORLD.mpic());
	
	//====== write the error ======
	if(WORLD.rank()==0){
		FILE* writer_error_=NULL;
		if(!restart_){
			writer_error_=fopen(file_error_.c_str(),"w");
			fprintf(writer_error_,"#STEP GAMMA RMSE_GRAD LOSS_TRAIN LOSS_VAL RMSE_TRAIN RMSE_VAL\n");
		} else {
			writer_error_=fopen(file_error_.c_str(),"a");
		}
		if(writer_error_==NULL) throw std::runtime_error("NNPTE::train(int): Could not open error record file.");
		for(int t=0; t<step.size(); ++t){
			fprintf(writer_error_,
				"%6i %12.10e %12.10e %12.10e %12.10e %12.10e %12.10e\n",
				step[t],gamma[t],rmse_g[t],loss_t[t],loss_v[t],rmse_t[t],rmse_v[t]
			);
		}
		fclose(writer_error_);
		writer_error_=NULL;
	}
	
	//====== write the parameters ======
	if(WORLD.rank()==0 && wparams_){
		FILE* writer_p_=NULL;
		if(!restart_) writer_p_=fopen(file_params_.c_str(),"w");
		else writer_p_=fopen(file_params_.c_str(),"a");
		if(writer_p_==NULL) throw std::runtime_error("NNPTE::train(int): Could not open error record file.");
		for(int t=0; t<step.size(); ++t){
			for(int i=0; i<params[t].size(); ++i){
				fprintf(writer_p_,"%.12f ",params[t][i]);
			}
			fprintf(writer_p_,"\n");
		}
		fclose(writer_p_);
		writer_p_=NULL;
	}
	
	//====== unpack final parameters ======
	if(NNPTE_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"packing final parameters into neural network\n";
	//unpack from global to per-element arrays
	int count=0;
	for(int n=0; n<nTypes_; ++n){
		for(int m=0; m<pElement_[n].size(); ++m){
			pElement_[n][m]=obj_.p()[count];
			gElement_[n][m]=obj_.g()[count];
			++count;
		}
	}
	//pack from per-element arrays into neural networks
	for(int n=0; n<nTypes_; ++n) nnp_.nnh(n).nn()<<pElement_[n];
	
	if(NNPTE_PRINT_DATA>-1 && WORLD.rank()==0){
		char* strbuf=new char[print::len_buf];
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("TRAIN - SUMMARY",strbuf)<<"\n";
		std::cout<<"N-STEP = "<<obj_.step()<<"\n";
		std::cout<<"TIME   = "<<time_train<<"\n";
		if(NNPTE_PRINT_DATA>1){
			std::cout<<"p = "; for(int i=0; i<obj_.p().size(); ++i) std::cout<<obj_.p()[i]<<" "; std::cout<<"\n";
		}
		std::cout<<print::buf(strbuf)<<"\n";
		delete[] strbuf;
	}
}

void NNPTE::error(const Eigen::VectorXd& x, const std::vector<Structure>& struc_train, const std::vector<Structure>& struc_val){
	if(NNPTE_PRINT_FUNC>0) std::cout<<"NNPTE::error(const Eigen::VectorXd&):\n";
	
	//====== reset the error ======
	error_[0]=0; //loss - training
	error_[1]=0; //loss - validation
	error_[2]=0; //rmse - training
	error_[3]=0; //rmse - validation
	
	//====== unpack total parameters into element arrays ======
	if(NNPTE_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"unpacking total parameters into element arrays\n";
	int count=0;
	for(int n=0; n<nTypes_; ++n){
		std::memcpy(pElement_[n].data(),x.data()+count,pElement_[n].size()*sizeof(double));
		count+=pElement_[n].size();
	}
	
	//====== unpack arrays into element nn's ======
	if(NNPTE_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"unpacking arrays into element nn's\n";
	for(int n=0; n<nTypes_; ++n) nnp_.nnh(n).nn()<<pElement_[n];
	
	//====== reset the gradients ======
	if(NNPTE_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"resetting gradients\n";
	for(int n=0; n<nTypes_; ++n) gElement_[n].setZero();
	
	//====== randomize the batch ======
	if(NNPTE_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"randomizing the batch\n";
	for(int i=0; i<batch_.size(); ++i) batch_[i]=batch_.data((batch_.count()++)%batch_.capacity());
	std::sort(batch_.elements(),batch_.elements()+batch_.size());
	if(batch_.count()>=batch_.capacity()){
		std::shuffle(batch_.data(),batch_.data()+batch_.capacity(),rngen_);
		MPI_Bcast(batch_.data(),batch_.capacity(),MPI_INT,0,BATCH.mpic());
		batch_.count()=0;
	}
	if(NNPTE_PRINT_DATA>1 && WORLD.rank()==0){std::cout<<"batch = "; for(int i=0; i<batch_.size(); ++i) std::cout<<batch_[i]<<" "; std::cout<<"\n";}
	
	//====== compute training error and gradient ======
	if(NNPTE_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"computing training error and gradient\n";
	for(int i=0; i<batch_.size(); ++i){
		const int ii=batch_[i];
		//**** compute the energy ****
		double energy=0;
		for(int n=0; n<dist_atomt[ii].size(); ++n){
			//get the index of the atom within the local processor subset
			const int m=dist_atomt[ii].index(n);
			//find the element index in the nn potential
			const int type=struc_train[ii].type(m);
			//execute the network
			nnp_.nnh(type).nn().fp(struc_train[ii].symm(m));
			//add the atom energy to the total
			energy+=nnp_.nnh(type).nn().out()[0];
		}
		//**** accumulate energy across BATCH communicator ****
		MPI_Allreduce(MPI_IN_PLACE,&energy,1,MPI_DOUBLE,MPI_SUM,BATCH.mpic());
		//**** compute the energy difference normalized by number of atoms ****
		const double norm=1.0/struc_train[ii].nAtoms();
		const double dE=(energy-struc_train[ii].energy())*norm;
		Eigen::VectorXd dcdo=Eigen::VectorXd::Constant(1,norm);
		switch(obj_.loss()){
			case opt::Loss::MSE:{
				error_[0]+=0.5*dE*dE;//loss - train
				dcdo*=dE;
			} break;
			case opt::Loss::MAE:{
				error_[0]+=std::fabs(dE);//loss - train
				dcdo*=math::special::sgn(dE);
			} break;
			case opt::Loss::HUBER:{
				const double arg=dE*deltai_;
				const double sqrtf=sqrt(1.0+arg*arg);
				error_[0]+=delta2_*(sqrtf-1.0);//loss - train
				dcdo*=dE/sqrtf;
			} break;
			case opt::Loss::ASINH:{
				const double arg=dE*deltai_;
				const double sqrtf=sqrt(1.0+arg*arg);
				const double logf=log(arg+sqrtf);
				error_[0]+=delta2_*(1.0-sqrtf+arg*logf);//loss - train
				dcdo*=logf*delta_;
			} break;
			default: break;
		}
		error_[2]+=dE*dE;//rmse - train
		//**** compute the gradient ****
		for(int n=0; n<dist_atomt[ii].size(); ++n){
			//get the index of the atom within the local processor subset
			const int m=dist_atomt[ii].index(n);
			//find the element index in the nn potential
			const int type=struc_train[ii].type(m);
			//execute the network
			nnp_.nnh(type).nn().fpbp(struc_train[ii].symm(m));
			//compute the gradient 
			gElement_[type].noalias()+=cost_[type].grad(nnp_.nnh(type).nn(),dcdo);
		}
	}
	
	//====== compute validation error ======
	if(obj_.step()%obj_.nPrint()==0 || obj_.step()%obj_.nWrite()==0){
		if(NNPTE_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"computing validation error\n";
		for(int i=0; i<struc_val.size(); ++i){
			//**** compute the energy ****
			double energy=0;
			for(int n=0; n<dist_atomv[i].size(); ++n){
				//get the index of the atom within the local processor subset
				const int m=dist_atomv[i].index(n);
				//find the element index in the nn potential
				const int type=struc_val[i].type(m);
				//execute the network
				nnp_.nnh(type).nn().fp(struc_val[i].symm(m));
				//add the energy to the total
				energy+=nnp_.nnh(type).nn().out()[0];
			}
			//**** accumulate energy ****
			MPI_Allreduce(MPI_IN_PLACE,&energy,1,MPI_DOUBLE,MPI_SUM,BATCH.mpic());
			//**** compute error ****
			const double dE=(energy-struc_val[i].energy())/struc_val[i].nAtoms();
			switch(obj_.loss()){
				case opt::Loss::MSE:{
					error_[1]+=0.5*dE*dE;//loss - val
				} break;
				case opt::Loss::MAE:{
					error_[1]+=std::fabs(dE);//loss - val
				} break;
				case opt::Loss::HUBER:{
					const double arg=dE*deltai_;
					error_[1]+=delta2_*(sqrt(1.0+arg*arg)-1.0);//loss - val
				} break;
				case opt::Loss::ASINH:{
					const double arg=dE*deltai_;
					const double sqrtf=sqrt(1.0+arg*arg);
					error_[1]+=delta2_*(1.0-sqrtf+arg*log(arg+sqrtf));//loss - val
				} break;
				default: break;
			}
			error_[3]+=dE*dE;//rmse - val
		}
	}
	
	//====== normalize w.r.t. batch size ======
	//note: we sum these quantities over WORLD, meaning that we are summing over duplicates in each BATCH
	//this normalization step corrects for this double counting
	const double batchsi=1.0/(1.0*BATCH.size());
	error_[0]*=batchsi;//loss - train
	error_[1]*=batchsi;//loss - validation
	error_[2]*=batchsi;//rmse - train
	error_[3]*=batchsi;//rmse - validation
}

void NNPTE::error2(const Eigen::VectorXd& x, const std::vector<Structure>& struc_train, const std::vector<Structure>& struc_val){
	if(NNPTE_PRINT_FUNC>0) std::cout<<"NNPTE::error(const Eigen::VectorXd&):\n";
	
	//====== reset the error ======
	error_[0]=0; //loss - training
	error_[1]=0; //loss - validation
	error_[2]=0; //rmse - training
	error_[3]=0; //rmse - validation
	
	//====== unpack total parameters into element arrays ======
	if(NNPTE_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"unpacking total parameters into element arrays\n";
	int count=0;
	for(int n=0; n<nTypes_; ++n){
		std::memcpy(pElement_[n].data(),x.data()+count,pElement_[n].size()*sizeof(double));
		count+=pElement_[n].size();
	}
	
	//====== unpack arrays into element nn's ======
	if(NNPTE_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"unpacking arrays into element nn's\n";
	for(int n=0; n<nTypes_; ++n) nnp_.nnh(n).nn()<<pElement_[n];
	
	//====== reset the gradients ======
	if(NNPTE_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"resetting gradients\n";
	for(int n=0; n<nTypes_; ++n) gElement_[n].setZero();
	
	//====== randomize the batch ======
	if(NNPTE_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"randomizing the batch\n";
	for(int i=0; i<batch_.size(); ++i) batch_[i]=batch_.data((batch_.count()++)%batch_.capacity());
	std::sort(batch_.elements(),batch_.elements()+batch_.size());
	if(batch_.count()>=batch_.capacity()){
		std::shuffle(batch_.data(),batch_.data()+batch_.capacity(),rngen_);
		MPI_Bcast(batch_.data(),batch_.capacity(),MPI_INT,0,BATCH.mpic());
		batch_.count()=0;
	}
	if(NNPTE_PRINT_DATA>1 && WORLD.rank()==0){std::cout<<"batch = "; for(int i=0; i<batch_.size(); ++i) std::cout<<batch_[i]<<" "; std::cout<<"\n";}
	
	//====== compute training error and gradient ======
	if(NNPTE_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"computing training error and gradient\n";
	for(int i=0; i<batch_.size(); ++i){
		//set batch
		const int ii=batch_[i];
		//reset gradients
		for(int n=0; n<nTypes_; ++n) grad_[n].setZero();
		//**** compute the energy ****
		double energy=0;
		const Eigen::VectorXd dcdo=Eigen::VectorXd::Constant(1,1);
		for(int n=0; n<dist_atomt[ii].size(); ++n){
			//get the index of the atom within the local processor subset
			const int m=dist_atomt[ii].index(n);
			//find the element index in the nn potential
			const int type=struc_train[ii].type(m);
			//execute the network
			nnp_.nnh(type).nn().fpbp(struc_train[ii].symm(m));
			//add the atom energy to the total
			energy+=nnp_.nnh(type).nn().out()[0];
			//compute gradient
			grad_[type].noalias()+=cost_[type].grad(nnp_.nnh(type).nn(),dcdo);
		}
		//**** accumulate energy across BATCH communicator ****
		MPI_Allreduce(MPI_IN_PLACE,&energy,1,MPI_DOUBLE,MPI_SUM,BATCH.mpic());
		//**** compute the energy difference normalized by number of atoms ****
		const double norm=1.0/struc_train[ii].nAtoms();
		const double dE=(energy-struc_train[ii].energy())*norm;
		//**** compute the error and parameter gradients ****
		double gfac=norm;
		switch(obj_.loss()){
			case opt::Loss::MSE:{
				error_[0]+=0.5*dE*dE;//loss - train
				gfac*=dE;
			} break;
			case opt::Loss::MAE:{
				error_[0]+=std::fabs(dE);//loss - train
				gfac*=math::special::sgn(dE);
			} break;
			case opt::Loss::HUBER:{
				const double arg=dE*deltai_;
				const double sqrtf=sqrt(1.0+arg*arg);
				error_[0]+=delta2_*(sqrtf-1.0);//loss - train
				gfac*=dE/sqrtf;
			} break;
			case opt::Loss::ASINH:{
				const double arg=dE*deltai_;
				const double sqrtf=sqrt(1.0+arg*arg);
				const double logf=log(arg+sqrtf);
				error_[0]+=delta2_*(1.0-sqrtf+arg*logf);//loss - train
				gfac*=logf*delta_;
			} break;
			default: break;
		}
		error_[2]+=dE*dE;//rmse - train
		//**** compute the gradient ****
		for(int j=0; j<nTypes_; ++j){
			gElement_[j].noalias()+=grad_[j]*gfac;
		}
	}
	
	//====== compute validation error ======
	if(obj_.step()%obj_.nPrint()==0 || obj_.step()%obj_.nWrite()==0){
		if(NNPTE_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"computing validation error\n";
		for(int i=0; i<struc_val.size(); ++i){
			//**** compute the energy ****
			double energy=0;
			for(int n=0; n<dist_atomv[i].size(); ++n){
				//get the index of the atom within the local processor subset
				const int m=dist_atomv[i].index(n);
				//find the element index in the nn potential
				const int type=struc_val[i].type(m);
				//execute the network
				nnp_.nnh(type).nn().fp(struc_val[i].symm(m));
				//add the energy to the total
				energy+=nnp_.nnh(type).nn().out()[0];
			}
			//**** accumulate energy ****
			MPI_Allreduce(MPI_IN_PLACE,&energy,1,MPI_DOUBLE,MPI_SUM,BATCH.mpic());
			//**** compute error ****
			const double dE=(energy-struc_val[i].energy())/struc_val[i].nAtoms();
			switch(obj_.loss()){
				case opt::Loss::MSE:{
					error_[1]+=0.5*dE*dE;//loss - val
				} break;
				case opt::Loss::MAE:{
					error_[1]+=std::fabs(dE);//loss - val
				} break;
				case opt::Loss::HUBER:{
					const double arg=dE*deltai_;
					error_[1]+=delta2_*(sqrt(1.0+arg*arg)-1.0);//loss - val
				} break;
				case opt::Loss::ASINH:{
					const double arg=dE*deltai_;
					const double sqrtf=sqrt(1.0+arg*arg);
					error_[1]+=delta2_*(1.0-sqrtf+arg*log(arg+sqrtf));//loss - val
				} break;
				default: break;
			}
			error_[3]+=dE*dE;//rmse - val
		}
	}
	
	//====== normalize w.r.t. batch size ======
	//note: we sum these quantities over WORLD, meaning that we are summing over duplicates in each BATCH
	//this normalization step corrects for this double counting
	const double batchsi=1.0/(1.0*BATCH.size());
	error_[0]*=batchsi;//loss - train
	error_[1]*=batchsi;//loss - validation
	error_[2]*=batchsi;//rmse - train
	error_[3]*=batchsi;//rmse - validation
}

void NNPTE::read(const char* file, NNPTE& nnpte){
	if(NN_PRINT_FUNC>0) std::cout<<"NNPTE::read(const char*,NNPTE&):\n";
	//local variables
	FILE* reader=NULL;
	//open the file
	reader=std::fopen(file,"r");
	if(reader!=NULL){
		NNPTE::read(reader,nnpte);
		std::fclose(reader);
		reader=NULL;
	} else throw std::runtime_error(std::string("ERROR: Could not open \"")+std::string(file)+std::string("\" for reading.\n"));
}

void NNPTE::read(FILE* reader, NNPTE& nnpte){
	//==== local variables ====
	char* input=new char[string::M];
	Token token;
	//==== rewind reader ====
	std::rewind(reader);
	//==== read parameters ====
	while(fgets(input,string::M,reader)!=NULL){
		token.read(string::trim_right(input,string::COMMENT),string::WS);
		if(token.end()) continue;//skip empty lines
		const std::string tag=string::to_upper(token.next());
		//nnp
		if(tag=="R_CUT"){
			nnpte.nnp().rc()=std::atof(token.next().c_str());
		}
		//files
		if(tag=="FILE_ERROR"){
			nnpte.file_error()=token.next();
		} else if(tag=="FILE_PARAMS"){
			nnpte.file_params()=token.next();
		} else if(tag=="FILE_ANN"){
			nnpte.file_ann()=token.next();
		} else if(tag=="FILE_RESTART"){
			nnpte.file_restart()=token.next();
		}
		//flags
		if(tag=="RESTART"){//read restart file
			nnpte.restart()=string::boolean(token.next().c_str());//restarting
		} else if(tag=="PRE_COND"){//whether to precondition the inputs
			nnpte.preCond()=string::boolean(token.next().c_str());
		} else if(tag=="WRITE_PARAMS"){
			nnpte.wparams()=string::boolean(token.next().c_str());
		} 
		//optimization
		if(tag=="LOSS"){
			nnpte.obj().loss()=opt::Loss::read(string::to_upper(token.next()).c_str());
		} else if(tag=="STOP"){
			nnpte.obj().stop()=opt::Stop::read(string::to_upper(token.next()).c_str());
		} else if(tag=="MAX_ITER"){
			nnpte.obj().max()=std::atoi(token.next().c_str());
		} else if(tag=="N_PRINT"){
			nnpte.obj().nPrint()=std::atoi(token.next().c_str());
		} else if(tag=="N_WRITE"){
			nnpte.obj().nWrite()=std::atoi(token.next().c_str());
		} else if(tag=="TOL"){
			nnpte.obj().tol()=std::atof(token.next().c_str());
		} else if(tag=="GAMMA"){
			nnpte.obj().gamma()=std::atof(token.next().c_str());
		} else if(tag=="ALGO"){
			opt::algo::read(nnpte.algo(),token);
		} else if(tag=="DECAY"){
			opt::decay::read(nnpte.decay(),token);
		} else if(tag=="DELTA"){
			nnpte.delta()=std::atof(token.next().c_str());
			nnpte.deltai()=1.0/nnpte.delta();
			nnpte.delta2()=nnpte.delta()*nnpte.delta();
		} 
	}
	//==== free local variables ====
	delete[] input;
}

//************************************************************
// MAIN
//************************************************************

int main(int argc, char* argv[]){
	//======== global variables ========
	//units
		units::System unitsys=units::System::UNKNOWN;
	//mode
		Mode mode=Mode::TRAIN;
	//atom format
		AtomType atomT;
		atomT.name=true; atomT.an=true; atomT.type=true; atomT.index=true;
		atomT.posn=true; atomT.force=false; atomT.symm=true; atomT.charge=false;
	//flags - compute
		struct Compute{
			bool coul=false;  //compute - external potential - coul
			bool vdw=false;   //compute - external potential - vdw
			bool force=false; //compute - forces
			bool norm=false;  //compute - energy normalization
			bool zero=false;  //compute - zero point energy
		} compute;
	//flags - writing
		struct Write{
			bool energy=false; //writing - energies
			bool force=false;  //writing - forces
			bool ewald=false;  //writing - ewald energies
			bool input=false;  //writing - inputs
		} write;
	//external potentials
		ptnl::PotGaussLong pot_coul;
		ptnl::PotLDampLong pot_vdw;
	//nn potential - opt
		int nBatch=-1;
		std::vector<Type> types;//unique atomic species
		NNPTE nnpte;//nn potential optimization data
		std::vector<std::vector<int> > nh;//hidden layer configuration
		NN::ANNP annp;//neural network initialization parameters
	//data names
		static const char* const dnames[] = {"TRAINING","VALIDATION","TESTING"};
	//structures - format
		FILE_FORMAT::type format;//format of training data
	//structures - data
		std::vector<int> nstrucs(3,0);
		std::vector<std::vector<std::string> > data(3); //data files
		std::vector<std::vector<std::string> > files(3); //structure files
		std::vector<std::vector<Structure> > strucs(3); //structures
		std::vector<std::vector<int> > indices(3);
		std::vector<Alias> aliases;
	//mpi data distribution
		std::vector<thread::Dist> dist(4);
	//timing
		Clock clock,clock_wall; //time objects
		double time_wall=0;     //total wall time
		std::vector<double> time_energy(3,0.0);
		std::vector<double> time_force(3,0.0);
		std::vector<double> time_symm(3,0.0);
	//file i/o
		FILE* reader=NULL;
		char* paramfile=new char[string::M];
		char* input=new char[string::M];
		char* strbuf=new char[print::len_buf];
		bool read_pot=false;
		std::string file_pot;
		std::vector<std::string> files_basis;//file - stores basis
	//string
		Token token;
	
	try{
		//************************************************************************************
		// LOADING/INITIALIZATION
		//************************************************************************************
		
		//======== initialize mpi ========
		MPI_Init(&argc,&argv);
		WORLD.mpic()=MPI_COMM_WORLD;
		MPI_Comm_size(WORLD.mpic(),&WORLD.size());
		MPI_Comm_rank(WORLD.mpic(),&WORLD.rank());
		
		//======== start wall clock ========
		if(WORLD.rank()==0) clock_wall.begin();
		
		//======== print title ========
		if(WORLD.rank()==0){
			std::cout<<print::buf(strbuf,'*')<<"\n";
			std::cout<<print::buf(strbuf,'*')<<"\n";
			std::cout<<print::title("ATOMIC NEURAL NETWORK",strbuf,' ')<<"\n";
			std::cout<<print::buf(strbuf,'*')<<"\n";
			std::cout<<print::buf(strbuf,'*')<<"\n";
		}
		
		//======== print compiler information ========
		if(WORLD.rank()==0){
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("COMPILER",strbuf)<<"\n";
			std::cout<<"date     = "<<compiler::date()<<"\n";
			std::cout<<"time     = "<<compiler::time()<<"\n";
			std::cout<<"compiler = "<<compiler::name()<<"\n";
			std::cout<<"version  = "<<compiler::version()<<"\n";
			std::cout<<"standard = "<<compiler::standard()<<"\n";
			std::cout<<"arch     = "<<compiler::arch()<<"\n";
			std::cout<<"instr    = "<<compiler::instr()<<"\n";
			std::cout<<"os       = "<<compiler::os()<<"\n";
			std::cout<<"omp      = "<<compiler::omp()<<"\n";
			std::cout<<print::buf(strbuf)<<"\n";
		}
		
		//======== print mathematical constants ========
		if(WORLD.rank()==0){
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("MATHEMATICAL CONSTANTS",strbuf)<<"\n";
			std::printf("PI    = %.15f\n",math::constant::PI);
			std::printf("RadPI = %.15f\n",math::constant::RadPI);
			std::printf("Rad2  = %.15f\n",math::constant::Rad2);
			std::printf("Log2  = %.15f\n",math::constant::LOG2);
			std::printf("Eps<D> = %.15e\n",std::numeric_limits<double>::epsilon());
			std::printf("Min<D> = %.15e\n",std::numeric_limits<double>::min());
			std::printf("Max<D> = %.15e\n",std::numeric_limits<double>::max());
			std::printf("Min<I> = %i\n",std::numeric_limits<int>::min());
			std::printf("Max<I> = %i\n",std::numeric_limits<int>::max());
			std::cout<<print::buf(strbuf)<<"\n";
		}
		
		//======== print physical constants ========
		if(WORLD.rank()==0){
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("PHYSICAL CONSTANTS",strbuf)<<"\n";
			std::printf("bohr-r  (A)  = %.12f\n",units::BOHR);
			std::printf("hartree (eV) = %.12f\n",units::HARTREE);
			std::cout<<print::buf(strbuf)<<"\n";
		}
		
		//======== set mpi data ========
		{
			int* ranks=new int[WORLD.size()];
			MPI_Gather(&WORLD.rank(),1,MPI_INT,ranks,1,MPI_INT,0,WORLD.mpic());
			if(WORLD.rank()==0){
				std::cout<<print::buf(strbuf)<<"\n";
				std::cout<<print::title("MPI",strbuf)<<"\n";
				std::cout<<"world - size = "<<WORLD.size()<<"\n"<<std::flush;
				std::cout<<print::buf(strbuf)<<"\n";
				std::cout<<std::flush;
			}
			delete[] ranks;
		}
		
		//======== rank 0 reads from file ========
		if(WORLD.rank()==0){
			
			//======== check the arguments ========
			if(argc!=2) throw std::invalid_argument("Invalid number of arguments.");
			
			//======== load the parameter file ========
			if(NNPTE_PRINT_STATUS>0) std::cout<<"reading parameter file\n";
			std::strcpy(paramfile,argv[1]);
			
			//======== open the parameter file ========
			if(NNPTE_PRINT_STATUS>0) std::cout<<"opening parameter file\n";
			reader=fopen(paramfile,"r");
			if(reader==NULL) throw std::runtime_error(std::string("I/O Error: Could not open parameter file: ")+paramfile);
			
			//======== read in the parameters ========
			if(NNPTE_PRINT_STATUS>-1) std::cout<<"reading parameters\n";
			while(fgets(input,string::M,reader)!=NULL){
				token.read(string::trim_right(input,string::COMMENT),string::WS);
				if(token.end()) continue;//skip empty line
				const std::string tag=string::to_upper(token.next());
				//general
				if(tag=="UNITS"){//units
					unitsys=units::System::read(string::to_upper(token.next()).c_str());
				} else if(tag=="FORMAT"){//simulation format
					format=FILE_FORMAT::read(string::to_upper(token.next()).c_str());
				} 
				//data and execution mode
				if(tag=="MODE"){//mode of calculation
					mode=Mode::read(string::to_upper(token.next()).c_str());
				} else if(tag=="DATA_TRAIN"){//data - training
					data[0].push_back(token.next());
				} else if(tag=="DATA_VAL"){//data - validation
					data[1].push_back(token.next());
				} else if(tag=="DATA_TEST"){//data - testing
					data[2].push_back(token.next());
				}
				//atom
				if(tag=="ATOM"){//atom - name/mass/energy
					//process the string
					const std::string name=token.next();
					const std::string atomtag=string::to_upper(token.next());
					const int id=string::hash(name);
					//look for the atom name in the existing list of atom names
					int index=-1;
					for(int i=0; i<types.size(); ++i){
						if(name==types[i].name()){index=i;break;}
					}
					//if atom is not found, add it
					if(index<0){
						index=types.size();
						types.push_back(Type());
						types.back().name()=name;
						types.back().id()=id;
						files_basis.resize(files_basis.size()+1);
						nh.resize(nh.size()+1);
					}
					//set tag value
					if(atomtag=="MASS"){
						types[index].mass().flag()=true;
						types[index].mass().val()=std::atof(token.next().c_str());
					} else if(atomtag=="CHARGE"){
						types[index].charge().flag()=true;
						types[index].charge().val()=std::atof(token.next().c_str());
						atomT.charge=true;
					} else if(atomtag=="CHI"){
						types[index].chi().flag()=true;
						types[index].chi().val()=std::atof(token.next().c_str());
						atomT.chi=true;
					} else if(atomtag=="ETA"){
						types[index].eta().flag()=true;
						types[index].eta().val()=std::atof(token.next().c_str());
						atomT.eta=true;
					} else if(atomtag=="ENERGY"){
						types[index].energy().flag()=true;
						types[index].energy().val()=std::atof(token.next().c_str());
					} else if(atomtag=="RVDW"){
						types[index].rvdw().flag()=true;
						types[index].rvdw().val()=std::atof(token.next().c_str());
					} else if(atomtag=="RCOV"){
						types[index].rcov().flag()=true;
						types[index].rcov().val()=std::atof(token.next().c_str());
					} else if(atomtag=="C6"){
						types[index].c6().flag()=true;
						types[index].c6().val()=std::atof(token.next().c_str());
					} else if(atomtag=="BASIS"){
						files_basis[index]=token.next();
					} else if(atomtag=="NH"){
						nh[index].clear();
						while(!token.end()) nh[index].push_back(std::atoi(token.next().c_str()));
					}
				} 
				//neural network potential
				if(tag=="READ_POT"){
					file_pot=token.next();
					read_pot=true;
				}
				//batch
				if(tag=="N_BATCH"){//size of the batch
					nBatch=std::atoi(token.next().c_str());
				} 
				//flags - writing
				if(tag=="WRITE"){
					const std::string wtype=string::to_upper(token.next());
					if(wtype=="ENERGY") write.energy=string::boolean(token.next().c_str());
					else if(wtype=="FORCE") write.force=string::boolean(token.next().c_str());
					else if(wtype=="EWALD") write.ewald=string::boolean(token.next().c_str());
					else if(wtype=="INPUT") write.input=string::boolean(token.next().c_str());
				}
				//flags - compute
				if(tag=="COMPUTE"){
					const std::string ctype=string::to_upper(token.next());
					if(ctype=="COUL") compute.coul=string::boolean(token.next().c_str());
					else if(ctype=="VDW") compute.vdw=string::boolean(token.next().c_str());
					else if(ctype=="FORCE") compute.force=string::boolean(token.next().c_str());
					else if(ctype=="NORM") compute.norm=string::boolean(token.next().c_str());
					else if(ctype=="ZERO") compute.zero=string::boolean(token.next().c_str());
				}
				//potential 
				if(tag=="POT_COUL"){
					token.next();
					pot_coul.read(token);
				} else if(tag=="POT_VDW"){
					token.next();
					pot_vdw.read(token);
				}
				//alias
				if(tag=="ALIAS"){
					aliases.push_back(Alias());
					Alias::read(token,aliases.back());
				}
			}
			
			//======== set atom flags =========
			if(NNPTE_PRINT_STATUS>0) std::cout<<"setting atom flags\n";
			atomT.force=compute.force;
			atomT.charge=compute.coul;
			
			//======== read - nnpte =========
			if(NNPTE_PRINT_STATUS>0) std::cout<<"reading neural network training parameters\n";
			NNPTE::read(reader,nnpte);
			
			//======== read - annp =========
			if(NNPTE_PRINT_STATUS>0) std::cout<<"reading neural network parameters\n";
			NN::ANNP::read(reader,annp);
			
			//======== close parameter file ========
			if(NNPTE_PRINT_STATUS>0) std::cout<<"closing parameter file\n";
			fclose(reader);
			reader=NULL;
			
			//======== (restart == false) ========
			if(!nnpte.restart_){
				//======== (read potential == false) ========
				if(!read_pot){
					//resize the potential
					if(NNPTE_PRINT_STATUS>-1) std::cout<<"resizing potential\n";
					nnpte.nnp().resize(types);
					//read basis files
					if(NNPTE_PRINT_STATUS>-1) std::cout<<"reading basis files\n";
					if(files_basis.size()!=nnpte.nnp().ntypes()) throw std::runtime_error("main(int,char**): invalid number of basis files.");
					for(int i=0; i<nnpte.nnp().ntypes(); ++i){
						const char* file=files_basis[i].c_str();
						const char* atomName=types[i].name().c_str();
						NNP::read_basis(file,nnpte.nnp(),atomName);
					}
					//initialize the neural network hamiltonians
					if(NNPTE_PRINT_STATUS>-1) std::cout<<"initializing neural network hamiltonians\n";
					for(int i=0; i<nnpte.nnp().ntypes(); ++i){
						NNH& nnhl=nnpte.nnp().nnh(i);
						nnhl.type()=types[i];
						nnhl.nn().resize(annp,nnhl.nInput(),nh[i],1);
						nnhl.dOdZ().resize(nnhl.nn());
					}
				}
				//======== (read potential == true) ========
				if(read_pot){
					if(NNPTE_PRINT_STATUS>-1) std::cout<<"reading potential\n";
					NNP::read(file_pot.c_str(),nnpte.nnp());
				}
			}
			//======== (restart == true) ========
			if(nnpte.restart_){
				if(NNPTE_PRINT_STATUS>-1) std::cout<<"reading restart file\n";
				const std::string file=nnpte.file_restart_;
				nnpte.read_restart(file.c_str());
				nnpte.restart()=true;
			}
			
			//======== print parameters ========
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("GENERAL PARAMETERS",strbuf)<<"\n";
			std::cout<<"read_pot  = "<<read_pot<<"\n";
			std::cout<<"atom_type = "<<atomT<<"\n";
			std::cout<<"format    = "<<format<<"\n";
			std::cout<<"units     = "<<unitsys<<"\n";
			std::cout<<"mode      = "<<mode<<"\n";
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("DATA FILES",strbuf)<<"\n";
			std::cout<<"data_train = \n"; for(int i=0; i<data[0].size(); ++i) std::cout<<"\t\t"<<data[0][i]<<"\n";
			std::cout<<"data_val   = \n"; for(int i=0; i<data[1].size(); ++i) std::cout<<"\t\t"<<data[1][i]<<"\n";
			std::cout<<"data_test  = \n"; for(int i=0; i<data[2].size(); ++i) std::cout<<"\t\t"<<data[2][i]<<"\n";
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("WRITE FLAGS",strbuf)<<"\n";
			std::cout<<"energy = "<<write.energy<<"\n";
			std::cout<<"ewald  = "<<write.ewald<<"\n";
			std::cout<<"inputs = "<<write.input<<"\n";
			std::cout<<"force  = "<<write.force<<"\n";
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("COMPUTE FLAGS",strbuf)<<"\n";
			std::cout<<"coul  = "<<compute.coul<<"\n";
			std::cout<<"vdw   = "<<compute.vdw<<"\n";
			std::cout<<"force = "<<compute.force<<"\n";
			std::cout<<"norm  = "<<compute.norm<<"\n";
			std::cout<<"zero  = "<<compute.zero<<"\n";
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("EXTERNAL POTENTIAL",strbuf)<<"\n";
			if(compute.coul) std::cout<<"COUL = "<<pot_coul<<"\n";
			if(compute.vdw)  std::cout<<"VDW  = "<<pot_vdw<<"\n";
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("TYPES",strbuf)<<"\n";
			for(int i=0; i<types.size(); ++i){
				std::cout<<types[i]<<"\n";
			}
			std::cout<<print::title("ALIAS",strbuf)<<"\n";
			for(int i=0; i<aliases.size(); ++i){
				std::cout<<aliases[i]<<"\n";
			}
			std::cout<<annp<<"\n";
			std::cout<<nnpte<<"\n";
			std::cout<<nnpte.nnp()<<"\n";
			
			//========= check the data =========
			if(mode==Mode::TRAIN && data[0].size()==0) throw std::invalid_argument("No data provided - training.");
			if(mode==Mode::TRAIN && data[1].size()==0) throw std::invalid_argument("No data provided - validation.");
			if(mode==Mode::TEST  && data[2].size()==0) throw std::invalid_argument("No data provided - testing.");
			if(mode==Mode::UNKNOWN) throw std::invalid_argument("Invalid calculation mode");
			if(format==FILE_FORMAT::UNKNOWN) throw std::invalid_argument("Invalid file format.");
			if(unitsys==units::System::UNKNOWN) throw std::invalid_argument("Invalid unit system.");
			if(types.size()==0) throw std::invalid_argument("Invalid number of types.");
		}
		
		//======== bcast the parameters ========
		if(NNPTE_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"broadcasting parameters\n";
		//general parameters
		MPI_Bcast(&unitsys,1,MPI_INT,0,WORLD.mpic());
		//mode
		MPI_Bcast(&mode,1,MPI_INT,0,WORLD.mpic());
		//atom type
		thread::bcast(WORLD.mpic(),0,atomT);
		thread::bcast(WORLD.mpic(),0,annp);
		//flags - compute
		MPI_Bcast(&compute.coul,1,MPI_C_BOOL,0,WORLD.mpic());
		MPI_Bcast(&compute.vdw,1,MPI_C_BOOL,0,WORLD.mpic());
		MPI_Bcast(&compute.force,1,MPI_C_BOOL,0,WORLD.mpic());
		MPI_Bcast(&compute.norm,1,MPI_C_BOOL,0,WORLD.mpic());
		MPI_Bcast(&compute.zero,1,MPI_C_BOOL,0,WORLD.mpic());
		//flags - writing
		MPI_Bcast(&write.energy,1,MPI_C_BOOL,0,WORLD.mpic());
		MPI_Bcast(&write.force,1,MPI_C_BOOL,0,WORLD.mpic());
		MPI_Bcast(&write.ewald,1,MPI_C_BOOL,0,WORLD.mpic());
		MPI_Bcast(&write.input,1,MPI_C_BOOL,0,WORLD.mpic());
		//external potential
		if(compute.coul) thread::bcast(WORLD.mpic(),0,pot_coul);
		if(compute.vdw) thread::bcast(WORLD.mpic(),0,pot_vdw);
		//batch
		MPI_Bcast(&nBatch,1,MPI_INT,0,WORLD.mpic());
		//structures - format
		MPI_Bcast(&format,1,MPI_INT,0,WORLD.mpic());
		//nnpte
		thread::bcast(WORLD.mpic(),0,nnpte);
		//alias
		int naliases=aliases.size();
		MPI_Bcast(&naliases,1,MPI_INT,0,WORLD.mpic());
		if(WORLD.rank()!=0) aliases.resize(naliases);
		for(int i=0; i<aliases.size(); ++i){
			thread::bcast(WORLD.mpic(),0,aliases[i]);
		}
		
		//======== set the unit system ========
		if(NNPTE_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"setting the unit system\n";
		units::consts::init(unitsys);
		
		//************************************************************************************
		// READ DATA
		//************************************************************************************
		
		//======== rank 0 reads the data files (lists of structure files) ========
		if(WORLD.rank()==0){
			if(NNPTE_PRINT_STATUS>-1) std::cout<<"reading data\n";
			//==== read data ====
			for(int n=0; n<3; ++n){
				for(int i=0; i<data[n].size(); ++i){
					//open the data file
					if(NNPTE_PRINT_DATA>0) std::cout<<"data["<<n<<"]["<<i<<"]: "<<data[n][i]<<"\n";
					reader=fopen(data[n][i].c_str(),"r");
					if(reader==NULL) throw std::runtime_error(std::string("I/O Error: Could not open data file: ")+data[n][i]);
					//read in the data
					while(fgets(input,string::M,reader)!=NULL){
						if(!string::empty(input)) files[n].push_back(std::string(string::trim(input)));
					}
					//close the file
					fclose(reader); reader=NULL;
				}
			}
			//==== print the files ====
			if(NNPTE_PRINT_DATA>1){
				for(int n=0; n<3; ++n){
					if(files[n].size()>0){
						std::cout<<print::buf(strbuf)<<"\n";
						std::cout<<print::title("FILES",strbuf)<<"\n";
						for(int i=0; i<files[n].size(); ++i) std::cout<<"\t"<<files[n][i]<<"\n";
						std::cout<<print::buf(strbuf)<<"\n";
					}
				}
			}
		}
		
		//======== bcast the file names =======
		if(NNPTE_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"bcasting file names\n";
		//bcast names
		for(int n=0; n<3; ++n) thread::bcast(WORLD.mpic(),0,files[n]);
		//set number of structures
		for(int n=0; n<3; ++n) nstrucs[n]=files[n].size();
		//print number of structures
		if(WORLD.rank()==0){
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("DATA - SIZE",strbuf)<<"\n";
			std::cout<<"ntrain = "<<nstrucs[0]<<"\n";
			std::cout<<"nval   = "<<nstrucs[1]<<"\n";
			std::cout<<"ntest  = "<<nstrucs[2]<<"\n";
			std::cout<<"nbatch = "<<nBatch<<"\n";
			std::cout<<print::buf(strbuf)<<"\n";
		}
		//check the batch size
		if(nBatch<=0) throw std::invalid_argument("Invalid batch size.");
		if(nBatch>nstrucs[0]) throw std::invalid_argument("Invalid batch size.");
		
		//======== initializing batch communicator ========
		if(NNPTE_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"initializing batch communicator\n";
		//split WORLD into BATCH
		BATCH=WORLD.split(WORLD.color(WORLD.ncomm(nBatch)));
		//print batch communicators
		{
			if(WORLD.rank()==0){
				std::cout<<print::buf(strbuf)<<"\n";
				std::cout<<print::title("BATCH COMMUNICATORS",strbuf)<<"\n";
				std::cout<<std::flush;
			}
			MPI_Barrier(WORLD.mpic());
			const int sizeb=serialize::nbytes(BATCH);
			const int sizet=WORLD.size()*serialize::nbytes(BATCH);
			char* arrb=new char[sizeb];
			char* arrt=new char[sizet];
			serialize::pack(BATCH,arrb);
			MPI_Gather(arrb,sizeb,MPI_CHAR,arrt,sizeb,MPI_CHAR,0,WORLD.mpic());
			if(WORLD.rank()==0){
				for(int i=0; i<WORLD.size(); ++i){
					thread::Comm tmp;
					serialize::unpack(tmp,arrt+i*sizeb);
					std::cout<<"BATCH["<<i<<"] = "<<tmp<<"\n";
				}
			}
			delete[] arrb;
			delete[] arrt;
			if(WORLD.rank()==0){
				std::cout<<print::buf(strbuf)<<"\n";
				std::cout<<std::flush;
			}
			MPI_Barrier(WORLD.mpic());
		}
		
		//======== generate thread distributions ========
		if(NNPTE_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"generating thread distributions\n";
		//thread dist - divide structures equally among the batch groups
		dist[0].init(BATCH.ncomm(),BATCH.color(),nstrucs[0]);//train
		dist[1].init(BATCH.ncomm(),BATCH.color(),nstrucs[1]);//validation
		dist[2].init(BATCH.ncomm(),BATCH.color(),nstrucs[2]);//test
		dist[3].init(BATCH.ncomm(),BATCH.color(),nBatch);//batch
		//print
		if(WORLD.rank()==0){
			std::string str;
			std::cout<<"thread_dist_train   = "<<thread::Dist::size(str,BATCH.ncomm(),nstrucs[0])<<"\n";
			std::cout<<"thread_dist_val     = "<<thread::Dist::size(str,BATCH.ncomm(),nstrucs[1])<<"\n";
			std::cout<<"thread_dist_test    = "<<thread::Dist::size(str,BATCH.ncomm(),nstrucs[2])<<"\n";
			std::cout<<"thread_dist_batch   = "<<thread::Dist::size(str,BATCH.ncomm(),nBatch)<<"\n";
			std::cout<<"thread_offset_train = "<<thread::Dist::offset(str,BATCH.ncomm(),nstrucs[0])<<"\n";
			std::cout<<"thread_offset_val   = "<<thread::Dist::offset(str,BATCH.ncomm(),nstrucs[1])<<"\n";
			std::cout<<"thread_offset_test  = "<<thread::Dist::offset(str,BATCH.ncomm(),nstrucs[2])<<"\n";
			std::cout<<"thread_offset_batch = "<<thread::Dist::offset(str,BATCH.ncomm(),nBatch)<<"\n";
		}
		
		//======== gen indices (random shuffle) ========
		for(int n=0; n<3; ++n){
			indices[n].resize(nstrucs[n],-1);
			for(int i=0; i<indices[n].size(); ++i) indices[n][i]=i;
			std::random_shuffle(indices[n].begin(),indices[n].end());
			thread::bcast(WORLD.mpic(),0,indices[n]);
		}
		
		//======== read the structures ========
		if(NNPTE_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"reading structures\n";
		for(int n=0; n<3; ++n){
			if(files[n].size()>0){
				//resize structure array
				strucs[n].resize(dist[n].size());
				//rank 0 of batch group reads structures
				if(BATCH.rank()==0){
					for(int i=0; i<dist[n].size(); ++i){
						const std::string& file=files[n][indices[n][dist[n].index(i)]];
						read_struc(file.c_str(),format,atomT,strucs[n][i]);
						if(NNPTE_PRINT_DATA>1) std::cout<<"\t"<<file<<" "<<strucs[n][i].energy()<<"\n";
					}
				}
				//broadcast structures to all other procs in the BATCH group
				for(int i=0; i<dist[n].size(); ++i){
					thread::bcast(BATCH.mpic(),0,strucs[n][i]);
				}
			}
		}
		
		//======== apply alias ========
		if(NNPTE_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"applying aliases\n";
		for(int n=0; n<3; ++n){
			for(int i=0; i<dist[n].size(); ++i){
				Structure& strucl=strucs[n][i];
				for(int j=0; j<strucl.nAtoms(); ++j){
					for(int k=0; k<aliases.size(); ++k){
						for(int l=0; l<aliases[k].labels().size(); ++l){
							if(strucl.name(j)==aliases[k].labels()[l]){
								strucl.name(j)=aliases[k].alias();
							}
						}
					}
				}
			}
		}
		
		//======== check the structures ========
		if(NNPTE_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"checking the structures\n";
		if(BATCH.rank()==0){
			for(int n=0; n<3; ++n){
				for(int i=0; i<dist[n].size(); ++i){
					const std::string filename=files[n][indices[n][dist[n].index(i)]];
					const Structure& strucl=strucs[n][i];
					if(strucl.nAtoms()==0) throw std::runtime_error(std::string("ERROR: Structure \"")+filename+std::string(" has zero atoms."));
					if(std::isinf(strucl.energy())) throw std::runtime_error(std::string("ERROR: Structure \"")+filename+std::string(" has inf energy."));
					if(strucl.energy()!=strucl.energy()) throw std::runtime_error(std::string("ERROR: Structure \"")+filename+std::string(" has nan energy."));
					if(std::fabs(strucl.energy())<math::constant::ZERO) std::cout<<"*********** WARNING: Structure \""<<filename<<"\" has ZERO energy. ***********";
					if(compute.force){
						for(int j=0; j<strucl.nAtoms(); ++j){
							const double force=strucl.force(j).squaredNorm();
							if(std::isinf(force)) std::cout<<"WARNING: Atom \""<<strucl.name(j)<<strucl.index(j)<<"\" in \""<<filename<<" has inf force.\n";
							if(force!=force) std::cout<<"WARNING: Atom \""<<strucl.name(j)<<strucl.index(j)<<"\" in \""<<filename<<" has nan force.\n";
						}
					}
					if(NNPTE_PRINT_DATA>1) std::cout<<"\t"<<filename<<" "<<strucl.energy()<<" "<<WORLD.rank()<<"\n";
				}
			}
		}
		MPI_Barrier(WORLD.mpic());
		
		//************************************************************************************
		// ATOM PROPERTIES
		//************************************************************************************
		
		//======== set atom properties ========
		if(WORLD.rank()==0) std::cout<<"setting atomic properties\n";
		
		//======== set the indices ========
		if(WORLD.rank()==0) std::cout<<"setting the indices\n";
		for(int n=0; n<3; ++n){
			for(int i=0; i<dist[n].size(); ++i){
				for(int j=0; j<strucs[n][i].nAtoms(); ++j){
					strucs[n][i].index(j)=j;
				}
			}
		}
		
		//======== set the types ========
		if(WORLD.rank()==0) std::cout<<"setting the types\n";
		for(int n=0; n<3; ++n){
			for(int i=0; i<dist[n].size(); ++i){
				for(int j=0; j<strucs[n][i].nAtoms(); ++j){
					strucs[n][i].type(j)=nnpte.nnp_.index(strucs[n][i].name(j));
				}
			}
		}
		
		//======== set the charges ========
		if(atomT.charge){
			if(WORLD.rank()==0) std::cout<<"setting charges\n";
			for(int n=0; n<3; ++n){
				for(int i=0; i<dist[n].size(); ++i){
					for(int j=0; j<strucs[n][i].nAtoms(); ++j){
						strucs[n][i].charge(j)=nnpte.nnp_.nnh(strucs[n][i].type(j)).type().charge().val();
					}
				}
			}
		}
		
		//======== set the electronegativities ========
		if(atomT.chi){
			if(WORLD.rank()==0) std::cout<<"setting electronegativities\n";
			for(int n=0; n<3; ++n){
				for(int i=0; i<dist[n].size(); ++i){
					for(int j=0; j<strucs[n][i].nAtoms(); ++j){
						strucs[n][i].chi(j)=nnpte.nnp_.nnh(strucs[n][i].type(j)).type().chi().val();
					}
				}
			}
		}
		
		//======== set the idempotentials ========
		if(atomT.eta){
			if(WORLD.rank()==0) std::cout<<"setting idempotentials\n";
			for(int n=0; n<3; ++n){
				for(int i=0; i<dist[n].size(); ++i){
					for(int j=0; j<strucs[n][i].nAtoms(); ++j){
						strucs[n][i].eta(j)=nnpte.nnp_.nnh(strucs[n][i].type(j)).type().eta().val();
					}
				}
			}
		}
		
		//************************************************************************************
		// EXTERNAL POTENTIALS
		//************************************************************************************
		
		if(compute.coul && compute.vdw) throw std::invalid_argument("Can't have charge and vdw interactions.");
		
		//======== compute ewald energies ========
		if(compute.coul){
			if(WORLD.rank()==0) std::cout<<"computing ewald energies\n";
			for(int n=0; n<3; ++n){
				std::vector<double> ewald(dist[n].size(),std::numeric_limits<double>::max());
				for(int i=BATCH.rank(); i<dist[n].size(); i+=BATCH.size()){
					NeighborList nlist(strucs[n][i],pot_coul.rc());
					ewald[i]=pot_coul.energy(strucs[n][i],nlist);
				}
				MPI_Allreduce(MPI_IN_PLACE,ewald.data(),ewald.size(),MPI_DOUBLE,MPI_MIN,BATCH.mpic());
				for(int i=0; i<dist[n].size(); ++i){
					strucs[n][i].ewald()=ewald[i];
					strucs[n][i].energy()-=ewald[i];
				}
			}
		}
		
		//======== compute vdw energies ========
		if(compute.vdw){
			if(WORLD.rank()==0) std::cout<<"computing vdw energies\n";
			Reduce<1> ralpha,rerrer,rerrek;
			Reduce<1> rNK;
			std::vector<Reduce<1> > rnk(3);
			//compute parameters
			pot_vdw.resize(nnpte.nnp().ntypes());
			pot_vdw.ksl().rc()=pot_vdw.rc();
			pot_vdw.ksl().prec()=pot_vdw.prec();
			for(int i=0; i<nnpte.nnp().ntypes(); ++i){
				const double ri=nnpte.nnp().nnh(i).type().rvdw().val();
				const double ci=nnpte.nnp().nnh(i).type().c6().val();
				pot_vdw.rvdw()(i,i)=ri;
				pot_vdw.c6()(i,i)=ci;
			}
			pot_vdw.init();
			if(WORLD.rank()==0){
				for(int i=0; i<nnpte.nnp().ntypes(); ++i){
					const std::string ni=nnpte.nnp().nnh(i).type().name();
					for(int j=0; j<nnpte.nnp().ntypes(); ++j){
						const std::string nj=nnpte.nnp().nnh(j).type().name();
						std::cout<<"c6("<<ni<<","<<nj<<") = "<<pot_vdw.c6()(i,j)<<"\n";
						std::cout<<"rvdw("<<ni<<","<<nj<<") = "<<pot_vdw.rvdw()(i,j)<<"\n";
					}
				}
			}
			//compute energy
			for(int n=0; n<3; ++n){
				std::vector<double> evdw(dist[n].size(),std::numeric_limits<double>::max());
				for(int i=BATCH.rank(); i<dist[n].size(); i+=BATCH.size()){
					NeighborList nlist(strucs[n][i],pot_vdw.rc());
					evdw[i]=pot_vdw.energy(strucs[n][i],nlist);
					ralpha.push(pot_vdw.ksl().alpha());
					rerrer.push(pot_vdw.ksl().errEr());
					rerrek.push(pot_vdw.ksl().errEk());
					rnk[0].push(pot_vdw.ksl().nk()[0]*1.0);
					rnk[1].push(pot_vdw.ksl().nk()[1]*1.0);
					rnk[2].push(pot_vdw.ksl().nk()[2]*1.0);
					rNK.push(pot_vdw.ksl().nk().prod());
				}
				MPI_Allreduce(MPI_IN_PLACE,evdw.data(),evdw.size(),MPI_DOUBLE,MPI_MIN,BATCH.mpic());
				for(int i=0; i<dist[n].size(); ++i){
					strucs[n][i].ewald()=evdw[i];
					strucs[n][i].energy()-=evdw[i];
				}
			}
			if(WORLD.rank()==0){
				std::cout<<"alpha = "<<ralpha.avg()<<" "<<ralpha.min()<<" "<<ralpha.max()<<" "<<ralpha.dev()<<"\n";
				std::cout<<"errEr = "<<rerrer.avg()<<" "<<rerrer.min()<<" "<<rerrer.max()<<" "<<rerrer.dev()<<"\n";
				std::cout<<"errEk = "<<rerrek.avg()<<" "<<rerrek.min()<<" "<<rerrek.max()<<" "<<rerrek.dev()<<"\n";
				std::cout<<"nk[0] = "<<rnk[0].avg()<<" "<<rnk[0].min()<<" "<<rnk[0].max()<<" "<<rnk[0].dev()<<"\n";
				std::cout<<"nk[1] = "<<rnk[1].avg()<<" "<<rnk[1].min()<<" "<<rnk[1].max()<<" "<<rnk[1].dev()<<"\n";
				std::cout<<"nk[2] = "<<rnk[2].avg()<<" "<<rnk[2].min()<<" "<<rnk[2].max()<<" "<<rnk[2].dev()<<"\n";
				std::cout<<"nk    = "<<rNK.avg()<<" "<<rNK.min()<<" "<<rNK.max()<<" "<<rNK.dev()<<"\n";
			}
		}
		
		//************************************************************************************
		// SET INPUTS
		//************************************************************************************
		
		//======== initialize the symmetry functions ========
		if(NNPTE_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"initializing symmetry functions\n";
		for(int n=0; n<3; ++n){
			for(int i=0; i<dist[n].size(); ++i){
				NNP::init(nnpte.nnp_,strucs[n][i]);
			}
		}
		
		//======== compute the symmetry functions ========
		if(NNPTE_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"setting the inputs (symmetry functions)\n";
		for(int n=0; n<3; ++n){
			clock.begin();
			if(dist[n].size()>0){
				//compute symmetry functions
				for(int i=BATCH.rank(); i<dist[n].size(); i+=BATCH.size()){
					if(NNPTE_PRINT_STATUS>0) std::cout<<"structure-train["<<i<<"]\n";
					NeighborList nlist(strucs[n][i],nnpte.nnp_.rc());
					NNP::symm(nnpte.nnp_,strucs[n][i],nlist);
				}
				MPI_Barrier(BATCH.mpic());
				//bcast symmetry functions
				for(int i=0; i<BATCH.size(); ++i){
					const int root=i;
					for(int j=root; j<dist[n].size(); j+=BATCH.size()){
						thread::bcast(BATCH.mpic(),root,strucs[n][j]);
					}
				}
				MPI_Barrier(BATCH.mpic());
			}
			clock.end();
			time_symm[n]=clock.duration();
		}
		MPI_Barrier(WORLD.mpic());
		
		//======== print the memory ========
		{
			//compute memory
			int meml[3]={0,0,0};
			for(int n=0; n<3; ++n) for(int i=0; i<dist[n].size(); ++i) meml[n]+=serialize::nbytes(strucs[n][i]);
			//allocate arrays
			std::vector<std::vector<int> > mem(3,std::vector<int>(WORLD.size(),0));
			//gather memory
			for(int n=0; n<3; ++n) MPI_Gather(&meml[n],1,MPI_INT,mem[n].data(),1,MPI_INT,0,WORLD.mpic());
			//compute total
			std::vector<double> memt(3,0.0);
			for(int n=0; n<3; ++n) for(int i=0; i<WORLD.size(); ++i) memt[n]+=mem[n][i];
			//print
			if(WORLD.rank()==0){
				std::cout<<print::buf(strbuf)<<"\n";
				std::cout<<print::title("MEMORY",strbuf)<<"\n";
				std::cout<<"memory unit - MB\n";
				std::cout<<"mem - train - loc = "; for(int i=0; i<WORLD.size(); ++i) std::cout<<(1.0*mem[0][i])/1e6<<" "; std::cout<<"\n";
				std::cout<<"mem - val   - loc = "; for(int i=0; i<WORLD.size(); ++i) std::cout<<(1.0*mem[1][i])/1e6<<" "; std::cout<<"\n";
				std::cout<<"mem - test  - loc = "; for(int i=0; i<WORLD.size(); ++i) std::cout<<(1.0*mem[2][i])/1e6<<" "; std::cout<<"\n";
				std::cout<<"mem - train - tot = "<<(1.0*memt[0])/1e6<<"\n";
				std::cout<<"mem - val   - tot = "<<(1.0*memt[1])/1e6<<"\n";
				std::cout<<"mem - test  - tot = "<<(1.0*memt[2])/1e6<<"\n";
				std::cout<<print::buf(strbuf)<<"\n";
			}
		}
		
		//************************************************************************************
		// TRAINING
		//************************************************************************************
		
		//======== subtract ground-state energies ========
		if(NNPTE_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"subtracting ground-state energies\n";
		for(int n=0; n<3; ++n){
			for(int i=0; i<dist[n].size(); ++i){
				for(int j=0; j<strucs[n][i].nAtoms(); ++j){
					strucs[n][i].energy()-=nnpte.nnp().nnh(strucs[n][i].type(j)).type().energy().val();
				}
			}
		}
		
		//======== train the nn potential ========
		if(mode==Mode::TRAIN){
			if(NNPTE_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"training the nn potential\n";
			nnpte.train(dist[3].size(),strucs[0],strucs[1]);
		}
		MPI_Barrier(WORLD.mpic());
		
		//======== add ground-state energies ========
		if(NNPTE_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"adding ground-state energies\n";
		for(int n=0; n<3; ++n){
			for(int i=0; i<dist[n].size(); ++i){
				for(int j=0; j<strucs[n][i].nAtoms(); ++j){
					strucs[n][i].energy()+=nnpte.nnp().nnh(strucs[n][i].type(j)).type().energy().val();
				}
			}
		}
		
		//************************************************************************************
		// EVALUATION
		//************************************************************************************
		
		//======== statistical data - energies/forces/errors ========
		std::vector<double> kendall(3,0);
		std::vector<Reduce<1> > r1_energy(3);
		std::vector<Reduce<2> > r2_energy(3);
		std::vector<Reduce<1> > r1_force(3);
		std::vector<std::vector<Reduce<2> > > r2_force(3,std::vector<Reduce<2> >(3));
		
		//======== compute the final energies ========
		if(NNPTE_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"computing final energies\n";
		for(int n=0; n<3; ++n){
			if(dist[n].size()>0){
				std::vector<double> energy_n(nstrucs[n],std::numeric_limits<double>::max());
				std::vector<double> energy_r(nstrucs[n],std::numeric_limits<double>::max());
				std::vector<double> energy_n_t(nstrucs[n],std::numeric_limits<double>::max());
				std::vector<double> energy_r_t(nstrucs[n],std::numeric_limits<double>::max());
				std::vector<int> natoms(nstrucs[n],0); std::vector<int> natoms_t(nstrucs[n],0);
				//compute energies
				clock.begin();
				for(int i=0; i<dist[n].size(); ++i){
					if(NNPTE_PRINT_STATUS>0) std::cout<<"structure["<<WORLD.rank()<<"]["<<i<<"]\i";
					energy_r[dist[n].index(i)]=strucs[n][i].energy();
					energy_n[dist[n].index(i)]=NNP::energy(nnpte.nnp_,strucs[n][i]);
					natoms[dist[n].index(i)]=strucs[n][i].nAtoms();
				}
				clock.end();
				time_energy[n]=clock.duration();
				if(compute.zero){
					for(int i=0; i<dist[n].size(); ++i){
						for(int j=0; j<strucs[n][i].nAtoms(); ++j){
							energy_r[dist[n].index(i)]-=nnpte.nnp().nnh(strucs[n][i].type(j)).type().energy().val();
							energy_n[dist[n].index(i)]-=nnpte.nnp().nnh(strucs[n][i].type(j)).type().energy().val();
						}
					}
				}
				MPI_Reduce(energy_r.data(),energy_r_t.data(),nstrucs[n],MPI_DOUBLE,MPI_MIN,0,WORLD.mpic());
				MPI_Reduce(energy_n.data(),energy_n_t.data(),nstrucs[n],MPI_DOUBLE,MPI_MIN,0,WORLD.mpic());
				MPI_Reduce(natoms.data(),natoms_t.data(),nstrucs[n],MPI_INT,MPI_MAX,0,WORLD.mpic());
				//accumulate statistics
				for(int i=0; i<nstrucs[n]; ++i){
					r1_energy[n].push(std::fabs(energy_r_t[i]-energy_n_t[i])/natoms_t[i]);
					r2_energy[n].push(energy_r_t[i]/natoms_t[i],energy_n_t[i]/natoms_t[i]);
				}
				kendall[n]=math::corr::kendall(energy_r_t,energy_n_t);
				//normalize
				if(compute.norm){
					for(int i=0; i<nstrucs[n]; ++i) energy_r_t[i]/=natoms_t[i];
					for(int i=0; i<nstrucs[n]; ++i) energy_n_t[i]/=natoms_t[i];
				}
				//write energies
				if(write.energy && WORLD.rank()==0){
					std::string file;
					switch(n){
						case 0: file="nnp_energy_train.dat"; break;
						case 1: file="nnp_energy_val.dat"; break;
						case 2: file="nnp_energy_test.dat"; break;
						default: file="ERROR.dat"; break;
					}
					FILE* writer=fopen(file.c_str(),"w");
					if(writer==NULL) std::cout<<"WARNING: Could not open file: \""<<file<<"\"\n";
					else{
						std::vector<std::pair<int,double> > energy_r_pair(nstrucs[n]);
						std::vector<std::pair<int,double> > energy_n_pair(nstrucs[n]);
						for(int i=0; i<nstrucs[n]; ++i){
							energy_r_pair[i].first=indices[n][i];
							energy_r_pair[i].second=energy_r_t[i];
							energy_n_pair[i].first=indices[n][i];
							energy_n_pair[i].second=energy_n_t[i];
						}
						std::sort(energy_r_pair.begin(),energy_r_pair.end(),compare_pair);
						std::sort(energy_n_pair.begin(),energy_n_pair.end(),compare_pair);
						fprintf(writer,"#STRUCTURE ENERGY_REF ENERGY_NN\n");
						for(int i=0; i<nstrucs[n]; ++i){
							fprintf(writer,"%s %f %f\n",files[n][i].c_str(),energy_r_pair[i].second,energy_n_pair[i].second);
						}
						fclose(writer); writer=NULL;
					}
				}
			}
		}
		
		//======== compute the final ewald energies ========
		if(NNPTE_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"computing final ewald energies\n";
		for(int n=0; n<3; ++n){
			if(write.ewald && dist[n].size()>0){
				std::vector<double> energy_r(nstrucs[n],std::numeric_limits<double>::max());
				std::vector<double> energy_r_t(nstrucs[n],std::numeric_limits<double>::max());
				std::vector<int> natoms(nstrucs[n],0); std::vector<int> natoms_t(nstrucs[n],0);
				//compute energies
				clock.begin();
				for(int i=0; i<dist[n].size(); ++i){
					if(NNPTE_PRINT_STATUS>0) std::cout<<"structure-train["<<WORLD.rank()<<"]["<<i<<"]\n";
					energy_r[dist[n].index(i)]=strucs[n][i].ewald();
					natoms[dist[n].index(i)]=strucs[n][i].nAtoms();
				}
				clock.end();
				MPI_Reduce(energy_r.data(),energy_r_t.data(),nstrucs[n],MPI_DOUBLE,MPI_MIN,0,WORLD.mpic());
				MPI_Reduce(natoms.data(),natoms_t.data(),nstrucs[n],MPI_INT,MPI_MAX,0,WORLD.mpic());
				//normalize
				if(compute.norm){
					for(int i=0; i<nstrucs[n]; ++i) energy_r_t[i]/=natoms_t[i];
				}
				//write energies
				if(write.energy && WORLD.rank()==0){
					std::string file;
					switch(n){
						case 0: file="nnp_ewald_train.dat"; break;
						case 1: file="nnp_ewald_val.dat"; break;
						case 2: file="nnp_ewald_test.dat"; break;
						default: file="ERROR.dat"; break;
					}
					FILE* writer=fopen(file.c_str(),"w");
					if(writer==NULL) std::cout<<"WARNING: Could not open file: \""<<file<<"\"\n";
					else{
						std::vector<std::pair<int,double> > energy_r_pair(nstrucs[n]);
						for(int i=0; i<nstrucs[n]; ++i){
							energy_r_pair[i].first=indices[n][i];
							energy_r_pair[i].second=energy_r_t[i];
						}
						std::sort(energy_r_pair.begin(),energy_r_pair.end(),compare_pair);
						fprintf(writer,"#STRUCTURE ENERGY_EWALD\n");
						for(int i=0; i<nstrucs[n]; ++i){
							fprintf(writer,"%s %f\n",files[n][i].c_str(),energy_r_pair[i].second);
						}
						fclose(writer); writer=NULL;
					}
				}
			}
		}
		
		//======== compute the final forces ========
		if(NNPTE_PRINT_STATUS>-1 && WORLD.rank()==0 && compute.force) std::cout<<"computing final forces\n";
		if(compute.force && write.force){
			for(int n=0; n<3; ++n){
				if(dist[n].size()>0){
					//compute forces
					clock.begin();
					for(int i=0; i<dist[n].size(); ++i){
						if(NNPTE_PRINT_STATUS>0) std::cout<<"structure["<<n<<"]["<<i<<"]\n";
						Structure& struc=strucs[n][i];
						//compute exact forces
						std::vector<Eigen::Vector3d> f_r(struc.nAtoms());
						for(int j=0; j<struc.nAtoms(); ++j) f_r[j]=struc.force(j);
						//compute nn forces
						NeighborList nlist(struc,nnpte.nnp_.rc());
						NNP::force(nnpte.nnp_,struc,nlist);
						std::vector<Eigen::Vector3d> f_n(struc.nAtoms());
						for(int j=0; j<struc.nAtoms(); ++j) f_n[j]=struc.force(j);
						//compute statistics
						if(BATCH.rank()==0){
							for(int j=0; j<struc.nAtoms(); ++j){
								r1_force[n].push((f_r[j]-f_n[j]).norm());
								r2_force[n][0].push(f_r[j][0],f_n[j][0]);
								r2_force[n][1].push(f_r[j][1],f_n[j][1]);
								r2_force[n][2].push(f_r[j][2],f_n[j][2]);
							}
						}
					}
					clock.end();
					time_force[n]=clock.duration();
					//accumulate statistics
					std::vector<Reduce<1> > r1fv(WORLD.size());
					thread::gather(r1_force[n],r1fv,WORLD.mpic());
					if(WORLD.rank()==0) for(int i=1; i<WORLD.size(); ++i) r1_force[n]+=r1fv[i];
					for(int i=0; i<3; ++i){
						std::vector<Reduce<2> > r2fv(WORLD.size());
						thread::gather(r2_force[n][i],r2fv,WORLD.mpic());
						if(WORLD.rank()==0) for(int j=1; j<WORLD.size(); ++j) r2_force[n][i]+=r2fv[j];
					}
				}
			}
		}
		
		//======== write the inputs ========
		if(write.input){
			for(int nn=0; nn<3; ++nn){
				if(dist[nn].size()>0){
					std::string file;
					switch(nn){
						case 0: file="nnp_inputs_train.dat"; break;
						case 1: file="nnp_inputs_val.dat"; break;
						case 2: file="nnp_inputs_test.dat"; break;
						default: file="ERROR.dat"; break;
					}
					for(int ii=0; ii<WORLD.size(); ++ii){
						if(WORLD.rank()==ii){
							FILE* writer=NULL;
							if(ii==0) writer=fopen(file.c_str(),"w");
							else writer=fopen(file.c_str(),"a");
							if(writer!=NULL){
								for(int n=0; n<dist[nn].size(); ++n){
									for(int i=0; i<strucs[nn][n].nAtoms(); ++i){
										fprintf(writer,"%s%i ",strucs[nn][n].name(i).c_str(),i);
										for(int j=0; j<strucs[nn][n].symm(i).size(); ++j){
											fprintf(writer,"%f ",strucs[nn][n].symm(i)[j]);
										}
										fprintf(writer,"\n");
									}
								}
								fclose(writer); writer=NULL;
							} else std::cout<<"WARNING: Could not open inputs file for training structures\n";
						}
						MPI_Barrier(WORLD.mpic());
					}
				}
			}
		}
		
		//======== stop the wall clock ========
		if(WORLD.rank()==0) clock_wall.end();
		if(WORLD.rank()==0) time_wall=clock_wall.duration();
		
		//************************************************************************************
		// OUTPUT
		//************************************************************************************
		
		//======== print the timing info ========
		for(int n=0; n<3; ++n){
			MPI_Allreduce(MPI_IN_PLACE,&time_symm[n],1,MPI_DOUBLE,MPI_SUM,WORLD.mpic()); 
			MPI_Allreduce(MPI_IN_PLACE,&time_energy[n],1,MPI_DOUBLE,MPI_SUM,WORLD.mpic()); 
			MPI_Allreduce(MPI_IN_PLACE,&time_force[n],1,MPI_DOUBLE,MPI_SUM,WORLD.mpic());
			time_symm[n]/=WORLD.size();
			time_energy[n]/=WORLD.size();
			time_force[n]/=WORLD.size();
		}
		if(WORLD.rank()==0){
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("TIMING (S)",strbuf)<<"\n";
			if(strucs[0].size()>0){
				std::cout<<"time - symm   - train = "<<time_symm[0]<<"\n";
				std::cout<<"time - energy - train = "<<time_energy[0]<<"\n";
				std::cout<<"time - force  - train = "<<time_force[0]<<"\n";
			}
			if(strucs[1].size()>0){
				std::cout<<"time - symm   - val   = "<<time_symm[1]<<"\n";
				std::cout<<"time - energy - val   = "<<time_energy[1]<<"\n";
				std::cout<<"time - force  - val   = "<<time_force[1]<<"\n";
			}
			if(strucs[2].size()>0){
				std::cout<<"time - symm   - test  = "<<time_symm[2]<<"\n";
				std::cout<<"time - energy - test  = "<<time_energy[2]<<"\n";
				std::cout<<"time - force  - test  = "<<time_force[2]<<"\n";
			}
			std::cout<<"time - wall           = "<<time_wall<<"\n";
			std::cout<<print::buf(strbuf)<<"\n";
		}
		
		//======== print the error statistics ========
		if(WORLD.rank()==0){
			for(int n=0; n<3; ++n){
				if(nstrucs[n]>0){
					std::cout<<print::buf(strbuf)<<"\n";
					if(n==0) std::cout<<print::title("ERROR - STATISTICS - TRAINING",strbuf)<<"\n";
					else if(n==1) std::cout<<print::title("ERROR - STATISTICS - VALIDATION",strbuf)<<"\n";
					else if(n==2) std::cout<<print::title("ERROR - STATISTICS - TESTING",strbuf)<<"\n";
					std::cout<<"\tERROR - AVG - "<<dnames[n]<<" - ENERGY/ATOM = "<<r1_energy[n].avg()<<"\n";
					std::cout<<"\tERROR - DEV - "<<dnames[n]<<" - ENERGY/ATOM = "<<r1_energy[n].dev()<<"\n";
					std::cout<<"\tERROR - MAX - "<<dnames[n]<<" - ENERGY/ATOM = "<<r1_energy[n].max()<<"\n";
					std::cout<<"\tM/R2 - "<<dnames[n]<<" - ENERGY/ATOM = "<<r2_energy[n].m()<<" "<<r2_energy[n].r2()<<"\n";
					std::cout<<"\tKENDALL - "<<dnames[n]<<" = "<<kendall[n]<<"\n";
					if(compute.force){
					std::cout<<"FORCE:\n";
					std::cout<<"\tERROR - AVG - FORCE - "<<dnames[n]<<" = "<<r1_force[n].avg()<<"\n";
					std::cout<<"\tERROR - DEV - FORCE - "<<dnames[n]<<" = "<<r1_force[n].dev()<<"\n";
					std::cout<<"\tERROR - MAX - FORCE - "<<dnames[n]<<" = "<<r1_force[n].max()<<"\n";
					std::cout<<"\tM  (FX,FY,FZ) = "<<r2_force[n][0].m() <<" "<<r2_force[n][1].m() <<" "<<r2_force[n][2].m() <<"\n";
					std::cout<<"\tR2 (FX,FY,FZ) = "<<r2_force[n][0].r2()<<" "<<r2_force[n][1].r2()<<" "<<r2_force[n][2].r2()<<"\n";
					}
					std::cout<<print::buf(strbuf)<<"\n";
				}
			}
		}
		
		//======== write the nn's ========
		if(NNPTE_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"writing the nn's\n";
		if(WORLD.rank()==0){
			NNP::write(nnpte.file_ann_.c_str(),nnpte.nnp_);
		}
		//======== write restart file ========
		if(NNPTE_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"writing restart file\n";
		if(WORLD.rank()==0){
			nnpte.write_restart(nnpte.file_restart_.c_str());
		}
		
		//======== finalize mpi ========
		if(NNPTE_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"finalizing mpi\n";
		std::cout<<std::flush;
		MPI_Comm_free(&BATCH.mpic());
		MPI_Barrier(WORLD.mpic());
		MPI_Finalize();
	}catch(std::exception& e){
		std::cout<<"ERROR in nnpte::main(int,char**):\n";
		std::cout<<e.what()<<"\n";
	}
	
	//======== free local variables ========
	delete[] paramfile;
	delete[] input;
	delete[] strbuf;
	
	return 0;
}
