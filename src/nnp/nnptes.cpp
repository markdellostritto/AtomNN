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
#include "src/format/file.hpp"
#include "src/format/format.hpp"
// math
#include "src/math/reduce.hpp"
#include "src/math/func.hpp"
// string
#include "src/str/string.hpp"
#include "src/str/print.hpp"
#include "src/str/token.hpp"
// chem
#include "src/chem/units.hpp"
#include "src/chem/ptable.hpp"
// thread
#include "src/thread/comm.hpp"
#include "src/thread/dist.hpp"
#include "src/thread/mpif.hpp"
// util
#include "src/util/compiler.hpp"
#include "src/util/time.hpp"
// nnptes
#include "src/nnp/nnptes.hpp"

static bool compare_pair(const std::pair<int,double>& p1, const std::pair<int,double>& p2){
	return p1.first<p2.first;
}

//************************************************************
// MPI Communicators
//************************************************************

thread::Comm WORLD;//all processors
thread::Comm BATCH;//group of nproc/nBatch processors handling each element of the batch

//************************************************************
// serialization
//************************************************************

namespace serialize{

//**********************************************
// byte measures
//**********************************************

template <> int nbytes(const NNPTES& obj){
   if(NNPTES_PRINT_FUNC>0) std::cout<<"nbytes(const NNPTES&)\n";
	int size=0;
	//input/output
		size+=nbytes(obj.file_params_);
		size+=nbytes(obj.file_error_);
		size+=nbytes(obj.file_ann_);
		size+=nbytes(obj.file_restart_);
	//flags
		size+=sizeof(bool);//restart
		size+=sizeof(bool);//pre-conditioning
		size+=sizeof(bool);//force
		size+=sizeof(bool);//symm
		size+=sizeof(bool);//norm
		size+=sizeof(bool);//wparams
	//nnp
		size+=nbytes(obj.nnp_);
	//optimization
		size+=sizeof(Opt::Loss);
		size+=nbytes(obj.batch_);
		size+=nbytes(obj.data_);
		size+=sizeof(bool);//nullmodel
		if(obj.model_!=nullptr){
			switch(obj.data_.algo()){
				case Opt::Algo::SGD:      size+=nbytes(static_cast<const Opt::SGD&>(*obj.model_)); break;
				case Opt::Algo::SDM:      size+=nbytes(static_cast<const Opt::SDM&>(*obj.model_)); break;
				case Opt::Algo::NAG:      size+=nbytes(static_cast<const Opt::NAG&>(*obj.model_)); break;
				case Opt::Algo::ADAGRAD:  size+=nbytes(static_cast<const Opt::ADAGRAD&>(*obj.model_)); break;
				case Opt::Algo::ADADELTA: size+=nbytes(static_cast<const Opt::ADADELTA&>(*obj.model_)); break;
				case Opt::Algo::RMSPROP:  size+=nbytes(static_cast<const Opt::RMSPROP&>(*obj.model_)); break;
				case Opt::Algo::ADAM:     size+=nbytes(static_cast<const Opt::ADAM&>(*obj.model_)); break;
				case Opt::Algo::NADAM:    size+=nbytes(static_cast<const Opt::NADAM&>(*obj.model_)); break;
				case Opt::Algo::AMSGRAD:  size+=nbytes(static_cast<const Opt::AMSGRAD&>(*obj.model_)); break;
				case Opt::Algo::BFGS:     size+=nbytes(static_cast<const Opt::BFGS&>(*obj.model_)); break;
				case Opt::Algo::RPROP:    size+=nbytes(static_cast<const Opt::RPROP&>(*obj.model_)); break;
				case Opt::Algo::CG:       size+=nbytes(static_cast<const Opt::CG&>(*obj.model_)); break;
				default: throw std::runtime_error("nbytes(const NNPotOpt&): Invalid optimization method."); break;
			}
		}
	//error
		size+=sizeof(double);//error_scale_
		size+=sizeof(double);//error_train_
		size+=sizeof(double);//error_val_
	//return the size
		return size;
}

//**********************************************
// packing
//**********************************************

template <> int pack(const NNPTES& obj, char* arr){
	if(NNPTES_PRINT_FUNC>0) std::cout<<"pack(const NNPTES&,char*)\n";
	int pos=0;
	//input/output
		pos+=pack(obj.file_params_,arr+pos);
		pos+=pack(obj.file_error_,arr+pos);
		pos+=pack(obj.file_ann_,arr+pos);
		pos+=pack(obj.file_restart_,arr+pos);
	//flags
		std::memcpy(arr+pos,&obj.restart_,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(arr+pos,&obj.preCond_,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(arr+pos,&obj.force_,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(arr+pos,&obj.symm_,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(arr+pos,&obj.norm_,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(arr+pos,&obj.wparams_,sizeof(bool)); pos+=sizeof(bool);
	//nnp
		pos+=pack(obj.nnp_,arr+pos);
	//optimization
		std::memcpy(arr+pos,&obj.loss_,sizeof(Opt::Loss)); pos+=sizeof(Opt::Loss);
		pos+=pack(obj.batch_,arr+pos);
		pos+=pack(obj.data_,arr+pos);
		bool nullmodel=(obj.model_==nullptr)?true:false;
		std::memcpy(arr+pos,&nullmodel,sizeof(bool)); pos+=sizeof(bool);
		if(obj.model_!=nullptr){
			switch(obj.data_.algo()){
				case Opt::Algo::SGD:      pos+=pack(static_cast<const Opt::SGD&>(*obj.model_),arr+pos); break;
				case Opt::Algo::SDM:      pos+=pack(static_cast<const Opt::SDM&>(*obj.model_),arr+pos); break;
				case Opt::Algo::NAG:      pos+=pack(static_cast<const Opt::NAG&>(*obj.model_),arr+pos); break;
				case Opt::Algo::ADAGRAD:  pos+=pack(static_cast<const Opt::ADAGRAD&>(*obj.model_),arr+pos); break;
				case Opt::Algo::ADADELTA: pos+=pack(static_cast<const Opt::ADADELTA&>(*obj.model_),arr+pos); break;
				case Opt::Algo::RMSPROP:  pos+=pack(static_cast<const Opt::RMSPROP&>(*obj.model_),arr+pos); break;
				case Opt::Algo::ADAM:     pos+=pack(static_cast<const Opt::ADAM&>(*obj.model_),arr+pos); break;
				case Opt::Algo::NADAM:    pos+=pack(static_cast<const Opt::NADAM&>(*obj.model_),arr+pos); break;
				case Opt::Algo::AMSGRAD:  pos+=pack(static_cast<const Opt::AMSGRAD&>(*obj.model_),arr+pos); break;
				case Opt::Algo::BFGS:     pos+=pack(static_cast<const Opt::BFGS&>(*obj.model_),arr+pos); break;
				case Opt::Algo::RPROP:    pos+=pack(static_cast<const Opt::RPROP&>(*obj.model_),arr+pos); break;
				case Opt::Algo::CG:       pos+=pack(static_cast<const Opt::CG&>(*obj.model_),arr+pos); break;
				default: throw std::runtime_error("pack(const NNPTES&,char*): Invalid optimization method."); break;
			}
		}
	//error
		std::memcpy(arr+pos,&obj.error_scale_,sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&obj.error_train_,sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&obj.error_val_,sizeof(double)); pos+=sizeof(double);
	//return bytes written
		return pos;
}

//**********************************************
// unpacking
//**********************************************

template <> int unpack(NNPTES& obj, const char* arr){
	if(NNPTES_PRINT_FUNC>0) std::cout<<"unpack(const NNPTES&,char*)\n";
	int pos=0;
	//input/output
		pos+=unpack(obj.file_params_,arr+pos);
		pos+=unpack(obj.file_error_,arr+pos);
		pos+=unpack(obj.file_ann_,arr+pos);
		pos+=unpack(obj.file_restart_,arr+pos);
	//flags
		std::memcpy(&obj.restart_,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(&obj.preCond_,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(&obj.force_,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(&obj.symm_,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(&obj.norm_,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(&obj.wparams_,arr+pos,sizeof(bool)); pos+=sizeof(bool);
	//nnp
		pos+=unpack(obj.nnp_,arr+pos);
	//optimization
		std::memcpy(&obj.loss_,arr+pos,sizeof(Opt::Loss)); pos+=sizeof(Opt::Loss);
		pos+=unpack(obj.batch_,arr+pos);
		pos+=unpack(obj.data_,arr+pos);
		bool nullmodel=true;
		std::memcpy(&nullmodel,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		if(!nullmodel){
			switch(obj.data_.algo()){
				case Opt::Algo::SGD:
					obj.model_.reset(new Opt::SGD());
					pos+=unpack(static_cast<Opt::SGD&>(*obj.model_),arr+pos);
				break;
				case Opt::Algo::SDM:
					obj.model_.reset(new Opt::SDM());
					pos+=unpack(static_cast<Opt::SDM&>(*obj.model_),arr+pos);
				break;
				case Opt::Algo::NAG:
					obj.model_.reset(new Opt::NAG());
					pos+=unpack(static_cast<Opt::NAG&>(*obj.model_),arr+pos);
				break;
				case Opt::Algo::ADAGRAD:
					obj.model_.reset(new Opt::ADAGRAD());
					pos+=unpack(static_cast<Opt::ADAGRAD&>(*obj.model_),arr+pos);
				break;
				case Opt::Algo::ADADELTA:
					obj.model_.reset(new Opt::ADADELTA());
					pos+=unpack(static_cast<Opt::ADADELTA&>(*obj.model_),arr+pos);
				break;
				case Opt::Algo::RMSPROP:
					obj.model_.reset(new Opt::RMSPROP());
					pos+=unpack(static_cast<Opt::RMSPROP&>(*obj.model_),arr+pos);
				break;
				case Opt::Algo::ADAM:
					obj.model_.reset(new Opt::ADAM());
					pos+=unpack(static_cast<Opt::ADAM&>(*obj.model_),arr+pos);
				break;
				case Opt::Algo::NADAM:
					obj.model_.reset(new Opt::NADAM());
					pos+=unpack(static_cast<Opt::NADAM&>(*obj.model_),arr+pos);
				break;
				case Opt::Algo::AMSGRAD:
					obj.model_.reset(new Opt::AMSGRAD());
					pos+=unpack(static_cast<Opt::AMSGRAD&>(*obj.model_),arr+pos);
				break;
				case Opt::Algo::BFGS:
					obj.model_.reset(new Opt::BFGS());
					pos+=unpack(static_cast<Opt::BFGS&>(*obj.model_),arr+pos);
				break;
				case Opt::Algo::RPROP:
					obj.model_.reset(new Opt::RPROP());
					pos+=unpack(static_cast<Opt::RPROP&>(*obj.model_),arr+pos);
				break;
				case Opt::Algo::CG:
					obj.model_.reset(new Opt::CG());
					pos+=unpack(static_cast<Opt::CG&>(*obj.model_),arr+pos);
				break;
				default:
					throw std::runtime_error("unpack(NNPTES&,const char*): Invalid optimization method.");
				break;
			}
		}
	//error
		std::memcpy(&obj.error_scale_,arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.error_train_,arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.error_val_,arr+pos,sizeof(double)); pos+=sizeof(double);
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
// NNPTES - Neural Network Potential - Optimization
//************************************************************

std::ostream& operator<<(std::ostream& out, const NNPTES& nnptes){
	char* str=new char[print::len_buf];
	out<<print::buf(str)<<"\n";
	out<<print::title("NNP - TRAIN",str)<<"\n";
	out<<"PRE-COND     = "<<nnptes.preCond_<<"\n";
	out<<"RESTART      = "<<nnptes.restart_<<"\n";
	out<<"FORCE        = "<<nnptes.force_<<"\n";
	out<<"SYMM         = "<<nnptes.symm_<<"\n";
	out<<"NORM         = "<<nnptes.norm_<<"\n";
	out<<"BATCH        = "<<nnptes.batch_<<"\n";
	out<<"LOSS         = "<<nnptes.loss_<<"\n";
	out<<"POT          = "<<nnptes.pot_<<"\n";
	out<<"ERROR_S      = "<<nnptes.error_scale_<<"\n";
	out<<"FILE_ANN     = "<<nnptes.file_ann_<<"\n";
	out<<"FILE_PARAMS  = "<<nnptes.file_params_<<"\n";
	out<<"FILE_ERROR   = "<<nnptes.file_error_<<"\n";
	out<<"FILE_RESTART = "<<nnptes.file_restart_<<"\n";
	out<<print::title("NNP - TRAIN",str)<<"\n";
	out<<print::buf(str);
	delete[] str;
	return out;
}

NNPTES::NNPTES(){
	if(NNPTES_PRINT_FUNC>0) std::cout<<"NNP::NNPTES():\n";
	defaults();
};

void NNPTES::defaults(){
	if(NNPTES_PRINT_FUNC>0) std::cout<<"NNP::defaults():\n";
	//input/output
		file_params_="nnp_params.dat";
		file_error_="nnp_error.dat";
		file_ann_="ann";
		file_restart_="nnptes.restart";
	//flags
		restart_=false;
		preCond_=false;
		force_=true;
		symm_=true;
		norm_=false;
		wparams_=false;
	//nnp	
		nElements_=0;
		gElement_.clear();
		pElement_.clear();
		nnp_.clear();
	//optimization
		loss_=Opt::Loss::MSE;
		identity_=Eigen::VectorXd::Identity(1,1);
	//error
		error_scale_=1.0;
		error_train_=0;
		error_val_=0;
}

void NNPTES::clear(){
	if(NNPTES_PRINT_FUNC>0) std::cout<<"NNP::clear():\n";
	//elements
		nElements_=0;
		gElement_.clear();
		pElement_.clear();
	//nn
		nnp_.clear();
	//optimization
		batch_.clear();
		data_.clear();
		identity_=Eigen::VectorXd::Identity(1,1);
	//error
		error_train_=0;
		error_val_=0;
}

void NNPTES::write_restart(const char* file){
	if(NNPTES_PRINT_FUNC>1) std::cout<<"NNPTES::write_restart(const char*):\n";
	//local variables
	char* arr=NULL;
	FILE* writer=NULL;
	bool error=false;
	try{
		//open file
		writer=fopen(file,"wb");
		if(writer==NULL) throw std::runtime_error(std::string("NNPTES::write_restart(const char*): Could not open file: ")+file);
		//allocate buffer
		const int nBytes=serialize::nbytes(*this);
		arr=new char[nBytes];
		if(arr==NULL) throw std::runtime_error("NNPTES::write_restart(const char*): Could not allocate memory.");
		//write to buffer
		serialize::pack(*this,arr);
		//write to file
		const int nWrite=fwrite(arr,sizeof(char),nBytes,writer);
		if(nWrite!=nBytes) throw std::runtime_error("NNPTES::write_restart(const char*): Write error.");
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
	if(error) throw std::runtime_error("NNPTES::write_restart(const char*): Failed to write");
}

void NNPTES::read_restart(const char* file){
	if(NNPTES_PRINT_FUNC>0) std::cout<<"NNPTES::read_restart(const char*):\n";
	//local variables
	char* arr=NULL;
	FILE* reader=NULL;
	bool error=false;
	try{
		//open file
		reader=fopen(file,"rb");
		if(reader==NULL) throw std::runtime_error(std::string("NNPTES::read_restart(const char*): Could not open file: ")+std::string(file));
		//find size
		std::fseek(reader,0,SEEK_END);
		const int nBytes=std::ftell(reader);
		std::fseek(reader,0,SEEK_SET);
		//allocate buffer
		arr=new char[nBytes];
		if(arr==NULL) throw std::runtime_error("NNPTES::read_restart(const char*): Could not allocate memory.");
		//read from file
		const int nRead=fread(arr,sizeof(char),nBytes,reader);
		if(nRead!=nBytes) throw std::runtime_error("NNPTES::read_restart(const char*): Read error.");
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
	if(error) throw std::runtime_error("NNPTES::read_restart(const char*): Failed to read");
}

void NNPTES::train(int batchSize, std::vector<Structure>& struc_train, std::vector<Structure>& struc_val){
	if(NNPTES_PRINT_FUNC>0) std::cout<<"NNPTES::train(NNP&,std::vector<Structure>&,int):\n";
	//====== local function variables ======
		char* strbuf=new char[print::len_buf];
	//statistics
		std::vector<int> N;//total number of inputs for each element
		std::vector<Eigen::VectorXd> avg_in;//average of the inputs for each element (nnp_.nSpecies_ x nInput_)
		std::vector<Eigen::VectorXd> max_in;//max of the inputs for each element (nnp_.nSpecies_ x nInput_)
		std::vector<Eigen::VectorXd> min_in;//min of the inputs for each element (nnp_.nSpecies_ x nInput_)
		std::vector<Eigen::VectorXd> dev_in;//average of the stddev for each element (nnp_.nSpecies_ x nInput_)
	//timing
		Clock clock;
		
	if(NNPTES_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"training NN potential\n";
	
	//====== compute the spin interactions ====
	if(NNPTES_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"computing spin interactions\n";
	Jt_.resize(struc_train.size());
	for(int n=0; n<struc_train.size(); ++n){
		pot_.rc()=8.0;
		NeighborList nlist(struc_train[n],pot_.rc());
		pot_.J(struc_train[n],nlist,Jt_[n]);
	}
	Jv_.resize(struc_val.size());
	for(int n=0; n<struc_val.size(); ++n){
		pot_.rc()=8.0;
		NeighborList nlist(struc_val[n],pot_.rc());
		pot_.J(struc_val[n],nlist,Jv_[n]);
	}
	
	//====== check the parameters ======
	if(batchSize<=0) throw std::invalid_argument("NNPTES::train(int): Invalid batch size.");
	if(struc_train.size()==0) throw std::invalid_argument("NNPTES::train(int): No training data provided.");
	if(struc_val.size()==0) throw std::invalid_argument("NNPTES::train(int): No validation data provided.");
	
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
	
	//====== compute the number of atoms of each element ======
	if(NNPTES_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"computing the number of atoms of each element\n";
	if(nElements_<=0) nElements_=nnp_.ntypes();
	else if(nElements_!=nnp_.ntypes()) throw std::invalid_argument("NNPTES::train(int): Invalid number of elements in the potential.");
	//compute the number of atoms of each species, set the type
	std::vector<double> nAtoms_(nElements_,0);
	for(int i=0; i<struc_train.size(); ++i){
		for(int j=0; j<struc_train[i].nAtoms(); ++j){
			++nAtoms_[struc_train[i].type(j)];
		}
	}
	MPI_Allreduce(MPI_IN_PLACE,nAtoms_.data(),nElements_,MPI_DOUBLE,MPI_SUM,WORLD.mpic());
	for(int i=0; i<nElements_; ++i) nAtoms_[i]/=BATCH.size();
	if(NNPTES_PRINT_DATA>-1 && WORLD.rank()==0){
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("ATOM - DATA",strbuf)<<"\n";
		for(int i=0; i<nElements_; ++i){
			const std::string& name=nnp_.nnh(i).type().name();
			const int index=nnp_.index(nnp_.nnh(i).type().name());
			std::cout<<name<<"("<<index<<") - "<<(int)nAtoms_[i]<<"\n";
		}
		std::cout<<print::title("ATOM - DATA",strbuf)<<"\n";
		std::cout<<print::buf(strbuf)<<"\n";
	}
	
	//====== set the indices and batch size ======
	if(NNPTES_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"setting indices and batch\n";
	batch_.resize(batchSize,struc_train.size());
	
	//====== collect input statistics ======
	//resize arrays
	N.resize(nElements_);
	avg_in.resize(nElements_);
	max_in.resize(nElements_);
	min_in.resize(nElements_);
	dev_in.resize(nElements_);
	for(int n=0; n<nElements_; ++n){
		const int nInput=nnp_.nnh(n).nInput();
		avg_in[n]=Eigen::VectorXd::Zero(nInput);
		max_in[n]=Eigen::VectorXd::Zero(nInput);
		min_in[n]=Eigen::VectorXd::Zero(nInput);
		dev_in[n]=Eigen::VectorXd::Zero(nInput);
	}
	//compute the total number
	if(NNPTES_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"compute the total number\n";
	for(int n=0; n<struc_train.size(); ++n){
		for(int i=0; i<struc_train[n].nAtoms(); ++i){
			++N[struc_train[n].type(i)];
		}
	}
	//accumulate the number
	for(int i=0; i<nElements_; ++i){
		double Nloc=(1.0*N[i])/BATCH.size();//normalize by the size of the BATCH group
		double tmp=0;
		MPI_Allreduce(&Nloc,&tmp,1,MPI_DOUBLE,MPI_SUM,WORLD.mpic());
		N[i]=static_cast<int>(std::round(tmp));
	}
	//compute the max/min
	if(NNPTES_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"compute the max/min\n";
	for(int n=0; n<struc_train.size(); ++n){
		const Structure& strucl=struc_train[n];
		for(int i=0; i<strucl.nAtoms(); ++i){
			const int index=strucl.type(i);
			max_in[index].noalias()-=strucl.symm(i);
			min_in[index].noalias()+=strucl.symm(i);
		}
	}
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
	for(int i=0; i<nElements_; ++i){
		MPI_Allreduce(MPI_IN_PLACE,min_in[i].data(),min_in[i].size(),MPI_DOUBLE,MPI_MIN,WORLD.mpic());
		MPI_Allreduce(MPI_IN_PLACE,max_in[i].data(),max_in[i].size(),MPI_DOUBLE,MPI_MIN,WORLD.mpic());
	}
	//compute the average
	if(NNPTES_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"compute the average\n";
	for(int n=0; n<struc_train.size(); ++n){
		for(int i=0; i<struc_train[n].nAtoms(); ++i){
			avg_in[struc_train[n].type(i)].noalias()+=struc_train[n].symm(i);
		}
	}
	//accumulate the average
	for(int i=0; i<nElements_; ++i){
		MPI_Allreduce(MPI_IN_PLACE,avg_in[i].data(),avg_in[i].size(),MPI_DOUBLE,MPI_SUM,WORLD.mpic());
		avg_in[i]/=N[i];
	}
	//compute the stddev
	if(NNPTES_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"compute the stddev\n";
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
			double tmp=0;
			dev_in[i][j]/=BATCH.size();//normalize by the size of the BATCH group
			MPI_Allreduce(&dev_in[i][j],&tmp,1,MPI_DOUBLE,MPI_SUM,WORLD.mpic());
			dev_in[i][j]=sqrt(tmp/(N[i]-1.0));
		}
	}
	
	//====== precondition the input ======
	std::vector<Eigen::VectorXd> inb_(nElements_);//input bias
	std::vector<Eigen::VectorXd> inw_(nElements_);//input weight
	for(int n=0; n<nElements_; ++n){
		inb_[n]=Eigen::VectorXd::Constant(nnp_.nnh(n).nInput(),0.0);
		inw_[n]=Eigen::VectorXd::Constant(nnp_.nnh(n).nInput(),1.0);
	}
	if(preCond_){
		if(NNPTES_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"pre-conditioning input\n";
		//set the preconditioning vectors - bias
		for(int i=0; i<inb_.size(); ++i){
			inb_[i]=-1*avg_in[i];
		}
		//set the preconditioning vectors - weight
		for(int i=0; i<inw_.size(); ++i){
			for(int j=0; j<inw_[i].size(); ++j){
				if(dev_in[i][j]==0) inw_[i][j]=1;
				else inw_[i][j]=1.0/(1.0*dev_in[i][j]);
			}
		}
	}
	
	//====== set the bias for each of the species ======
	if(NNPTES_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"setting the bias for each species\n";
	for(int n=0; n<nElements_; ++n){
		NN::ANN& nn_=nnp_.nnh(n).nn();
		for(int i=0; i<nn_.nIn(); ++i) nn_.inb()[i]=inb_[n][i];
		for(int i=0; i<nn_.nIn(); ++i) nn_.inw()[i]=inw_[n][i];
		nn_.outb()[0]=0.0;
		nn_.outw()[0]=1.0;
	}
	
	//====== resize the optimization data ======
	if(NNPTES_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"initializing the optimization data\n";
	//resize per-element arrays
	pElement_.resize(nElements_);
	gElement_.resize(nElements_);
	for(int n=0; n<nElements_; ++n){
		const int nn_size=nnp_.nnh(n).nn().size();
		pElement_[n]=Eigen::VectorXd::Zero(nn_size);
		gElement_[n]=Eigen::VectorXd::Zero(nn_size);
	}
	//resize gradient objects
	if(NNPTES_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"resizing gradient data\n";
	cost_.resize(nElements_);
	for(int i=0; i<nElements_; ++i){
		cost_[i].resize(nnp_.nnh(i).nn());
	}
	
	//====== initialize the optimization data ======
	const int nParams=nnp_.size();
	if(restart_){
		//restart
		if(WORLD.rank()==0) std::cout<<"restarting optimization\n";
		if(nParams!=data_.dim()) throw std::runtime_error(
			std::string("NNPTES::train(int): Network has ")
			+std::to_string(nParams)+std::string(" while opt has ")
			+std::to_string(data_.dim())+std::string(" parameters.")
		);
	} else {
		//from scratch
		if(WORLD.rank()==0) std::cout<<"starting from scratch\n";
		//resize the optimization objects
		data_.init(nParams);
		model_->init(nParams);
		//load random initial values in the per-element arrays
		for(int n=0; n<nElements_; ++n){
			nnp_.nnh(n).nn()>>pElement_[n];
			gElement_[n]=Eigen::VectorXd::Random(nnp_.nnh(n).nn().size())*1e-6;
		}
		//load initial values from per-element arrays into global arrays
		int count=0;
		for(int n=0; n<nElements_; ++n){
			for(int m=0; m<pElement_[n].size(); ++m){
				data_.p()[count]=pElement_[n][m];
				data_.g()[count]=gElement_[n][m];
				++count;
			}
		}
	}
	
	//====== print input statistics and bias ======
	if(NNPTES_PRINT_DATA>-1 && WORLD.rank()==0){
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("OPT - DATA",strbuf)<<"\n";
		std::cout<<"N-PARAMS    = \n\t"<<nParams<<"\n";
		std::cout<<"AVG - INPUT = \n"; for(int i=0; i<avg_in.size(); ++i) std::cout<<"\t"<<avg_in[i].transpose()<<"\n";
		std::cout<<"MAX - INPUT = \n"; for(int i=0; i<max_in.size(); ++i) std::cout<<"\t"<<max_in[i].transpose()<<"\n";
		std::cout<<"MIN - INPUT = \n"; for(int i=0; i<min_in.size(); ++i) std::cout<<"\t"<<min_in[i].transpose()<<"\n";
		std::cout<<"DEV - INPUT = \n"; for(int i=0; i<dev_in.size(); ++i) std::cout<<"\t"<<dev_in[i].transpose()<<"\n";
		std::cout<<"PRE-BIAS    = \n"; for(int i=0; i<inb_.size(); ++i) std::cout<<"\t"<<inb_[i].transpose()<<"\n";
		std::cout<<"PRE-SCALE   = \n"; for(int i=0; i<inw_.size(); ++i) std::cout<<"\t"<<inw_[i].transpose()<<"\n";
		std::cout<<print::title("OPT - DATA",strbuf)<<"\n";
		std::cout<<print::buf(strbuf)<<"\n";
	}
	
	//====== execute the optimization ======
	if(NNPTES_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"executing the optimization\n";
	//optimization variables
	bool fbreak=false;
	identity_=Eigen::VectorXd::Constant(1,1);
	const double nBatchi_=1.0/nBatch;
	const double nVali_=1.0/nVal;
	//bcast parameters
	MPI_Bcast(data_.p().data(),data_.p().size(),MPI_DOUBLE,0,WORLD.mpic());
	//allocate status vectors
	std::vector<int> step;
	std::vector<double> gamma,err_t,err_v;
	std::vector<Eigen::VectorXd> params;
	if(WORLD.rank()==0){
		int size=data_.max()/data_.nPrint();
		if(size==0) ++size;
		step.resize(size);
		gamma.resize(size);
		err_t.resize(size);
		err_v.resize(size);
		params.resize(size);
	}
	//weight mask (for regularization)
	Eigen::VectorXd maskWeight;
	if(WORLD.rank()==0 && model_->lambda()>0){
		int count=0;
		maskWeight.resize(nParams);
		for(int n=0; n<nElements_; ++n){
			for(int i=0; i<nnp_.nnh(n).nn().nBias(); ++i) maskWeight[count++]=0.0;
			for(int i=0; i<nnp_.nnh(n).nn().nWeight(); ++i) maskWeight[count++]=1.0;
		}
	}
	MPI_Barrier(WORLD.mpic());
	std::vector<Eigen::VectorXd> gElementT=gElement_;
	//print status header to standard output
	if(WORLD.rank()==0) printf("opt gamma err_t err_v\n");
	//start the clock
	clock.begin();
	//begin optimization
	for(int iter=0; iter<data_.max(); ++iter){
		double error_train_sum_=0,error_val_sum_=0;
		//compute the error and gradient
		error(data_.p(),struc_train,struc_val);
		//accumulate error
		MPI_Reduce(&error_train_,&error_train_sum_,1,MPI_DOUBLE,MPI_SUM,0,WORLD.mpic());
		MPI_Reduce(&error_val_,&error_val_sum_,1,MPI_DOUBLE,MPI_SUM,0,WORLD.mpic());
		//accumulate gradient
		for(int n=0; n<nElements_; ++n){
			gElementT[n].setZero();
			MPI_Reduce(gElement_[n].data(),gElementT[n].data(),gElementT[n].size(),MPI_DOUBLE,MPI_SUM,0,WORLD.mpic());
		}
		if(WORLD.rank()==0){
			//compute error averaged over the batch
			error_train_=error_train_sum_*nBatchi_;
			error_val_=error_val_sum_*nVali_;
			//compute gradient averaged over the batch
			for(int n=0; n<nElements_; ++n) gElementT[n]*=nBatchi_;
			//pack the gradient
			int count=0;
			for(int n=0; n<nElements_; ++n){
				std::memcpy(data_.g().data()+count,gElementT[n].data(),gElementT[n].size()*sizeof(double));
				count+=gElementT[n].size();
			}
			//print/write error
			if(data_.step()%data_.nPrint()==0){
				const int t=iter/data_.nPrint();
				step[t]=data_.count();
				gamma[t]=model_->gamma();
				err_t[t]=std::sqrt(2.0*error_train_);
				err_v[t]=std::sqrt(2.0*error_val_);
				params[t]=data_.p();
				printf("%8i %12.10f %12.10f %12.10f\n",step[t],gamma[t],err_t[t],err_v[t]);
			}
			//write the basis and potentials
			if(data_.step()%data_.nWrite()==0){
				if(NNPTES_PRINT_STATUS>1) std::cout<<"writing the restart file and potentials\n";
				//write restart file
				const std::string file_restart=file_restart_+"."+std::to_string(data_.count());
				this->write_restart(file_restart.c_str());
				//write potential file
				const std::string file_ann=file_ann_+"."+std::to_string(data_.count());
				NNP::write(file_ann.c_str(),nnp_);
			}
			//compute the new position
			data_.val()=error_train_;
			model_->step(data_);
			//update weights - regularization
			if(model_->lambda()>0.0){
				const double fac=-1.0*model_->lambda()*model_->gamma();
				data_.p().noalias()+=fac*maskWeight.cwiseProduct(data_.pOld());
			}
			//update weights - mixing
			if(model_->mix()>0.0){
				data_.p()*=(1.0-model_->mix());
				data_.p().noalias()+=data_.pOld()*model_->mix();
			}
			//compute the difference
			data_.dv()=std::fabs(data_.val()-data_.valOld());
			data_.dp()=(data_.p()-data_.pOld()).norm();
			//set the new "old" values
			data_.valOld()=data_.val();//set "old" value
			data_.pOld()=data_.p();//set "old" p value
			data_.gOld()=data_.g();//set "old" g value
			//check the break condition
			switch(data_.stop()){
				case Opt::Stop::FABS: fbreak=(data_.val()<data_.tol()); break;
				case Opt::Stop::FREL: fbreak=(data_.dv()<data_.tol()); break;
				case Opt::Stop::XREL: fbreak=(data_.dp()<data_.tol()); break;
			}
		}
		//bcast parameters
		MPI_Bcast(data_.p().data(),data_.p().size(),MPI_DOUBLE,0,WORLD.mpic());
		//bcast break condition
		MPI_Bcast(&fbreak,1,MPI_C_BOOL,0,WORLD.mpic());
		if(fbreak) break;
		//increment step
		++data_.step();
		++data_.count();
	}
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
			fprintf(writer_error_,"#STEP GAMMA ERROR_RMS_TRAIN ERROR_RMS_VAL\n");
		} else {
			writer_error_=fopen(file_error_.c_str(),"a");
		}
		if(writer_error_==NULL) throw std::runtime_error("NNPTES::train(int): Could not open error record file.");
		for(int t=0; t<step.size(); ++t){
			fprintf(writer_error_,"%6i %12.10f %12.10f %12.10f\n",step[t],gamma[t],err_t[t],err_v[t]);
		}
		fclose(writer_error_);
		writer_error_=NULL;
	}
	
	//====== write the parameters ======
	if(WORLD.rank()==0){
		FILE* writer_p_=NULL;
		if(!restart_) writer_p_=fopen(file_params_.c_str(),"w");
		else writer_p_=fopen(file_params_.c_str(),"a");
		if(writer_p_==NULL) throw std::runtime_error("NNPTES::train(int): Could not open error record file.");
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
	if(NNPTES_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"packing final parameters into neural network\n";
	//unpack from global to per-element arrays
	int count=0;
	for(int n=0; n<nElements_; ++n){
		for(int m=0; m<pElement_[n].size(); ++m){
			pElement_[n][m]=data_.p()[count];
			gElement_[n][m]=data_.g()[count];
			++count;
		}
	}
	//pack from per-element arrays into neural networks
	for(int n=0; n<nElements_; ++n) nnp_.nnh(n).nn()<<pElement_[n];
	
	if(NNPTES_PRINT_DATA>-1 && WORLD.rank()==0){
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("TRAIN - SUMMARY",strbuf)<<"\n";
		std::cout<<"N-STEP = "<<data_.step()<<"\n";
		std::cout<<"TIME   = "<<time_train<<"\n";
		if(NNPTES_PRINT_DATA>1){
			std::cout<<"p = "; for(int i=0; i<data_.p().size(); ++i) std::cout<<data_.p()[i]<<" "; std::cout<<"\n";
		}
		std::cout<<print::title("TRAIN - SUMMARY",strbuf)<<"\n";
		std::cout<<print::buf(strbuf)<<"\n";
	}
	
	delete[] strbuf;
}

double NNPTES::error(const Eigen::VectorXd& x, std::vector<Structure>& struc_train, std::vector<Structure>& struc_val){
	if(NNPTES_PRINT_FUNC>0) std::cout<<"NNPTES::error(const Eigen::VectorXd&):\n";
	//====== local variables ======
	std::vector<Eigen::VectorXd> grad=gElement_;//resize as gElement_
	
	//====== reset the error ======
	error_train_=0; //error - training
	error_val_=0;   //error - validation
	
	//====== unpack total parameters into element arrays ======
	if(NNPTES_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"unpacking total parameters into element arrays\n";
	int count=0;
	for(int n=0; n<nElements_; ++n){
		std::memcpy(pElement_[n].data(),x.data()+count,pElement_[n].size()*sizeof(double));
		count+=pElement_[n].size();
	}
	
	//====== unpack arrays into element nn's ======
	if(NNPTES_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"unpacking arrays into element nn's\n";
	for(int n=0; n<nElements_; ++n) nnp_.nnh(n).nn()<<pElement_[n];
	
	//====== reset the gradients ======
	if(NNPTES_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"resetting gradients\n";
	for(int n=0; n<nElements_; ++n) gElement_[n].setZero();
	
	//====== randomize the batch ======
	if(NNPTES_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"randomizing the batch\n";
	for(int i=0; i<batch_.size(); ++i) batch_[i]=batch_.data((batch_.count()++)%batch_.capacity());
	std::sort(batch_.elements(),batch_.elements()+batch_.size());
	if(batch_.count()>=batch_.capacity()){
		std::shuffle(batch_.data(),batch_.data()+batch_.capacity(),rngen_);
		MPI_Bcast(batch_.data(),batch_.capacity(),MPI_INT,0,BATCH.mpic());
		batch_.count()=0;
	}
	if(NNPTES_PRINT_DATA>1 && WORLD.rank()==0){std::cout<<"batch = "; for(int i=0; i<batch_.size(); ++i) std::cout<<batch_[i]<<" "; std::cout<<"\n";}
	
	//====== compute training error and gradient ======
	if(NNPTES_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"computing training error and gradient\n";
	for(int i=0; i<batch_.size(); ++i){
		const int ii=batch_[i];
		const int nAtoms=struc_train[ii].nAtoms();
		//**** compute the energy ****
		double energyV=0;
		for(int n=0; n<nAtoms; ++n) struc_train[ii].js(n)=0.0;
		for(int n=0; n<dist_atomt[ii].size(); ++n){
			//get the index of the atom within the local processor subset
			const int m=dist_atomt[ii].index(n);
			//find the element index in the nn potential
			const int type=struc_train[ii].type(m);
			//execute the network
			nnp_.nnh(type).nn().execute(struc_train[ii].symm(m));
			//add the atom energy to the total
			energyV+=nnp_.nnh(type).nn().out()[0]+nnp_.nnh(type).type().energy().val()*0.0;
			struc_train[ii].js(m)=nnp_.nnh(type).nn().out()[1]+nnp_.nnh(type).type().js().val();
		}
		//**** accumulate energy across BATCH communicator ****
		MPI_Allreduce(MPI_IN_PLACE,&energyV,1,MPI_DOUBLE,MPI_SUM,BATCH.mpic());
		//**** accumulate js across BATCH communicator ****
		MPI_Allreduce(MPI_IN_PLACE,struc_train[ii].js().data(),nAtoms,MPI_DOUBLE,MPI_SUM,BATCH.mpic());
		//**** compute the spin energy ****
		double energyS=0;
		for(int si=0; si<nAtoms; ++si){
			for(int sj=0; sj<nAtoms; ++sj){
				energyS+=struc_train[ii].js(sj)*Jt_[ii](sj,si)*struc_train[ii].js(si);
			}
		}
		energyS*=0.5;
		//**** compute the total energy and error ****
		const double scale=error_scale_/nAtoms;
		const double energyT=energyS+energyV;
		const double dE=scale*(energyT-struc_train[ii].energy());
		double gpre=0;
		switch(loss_){
			case Opt::Loss::MSE:{
				error_train_+=0.5*dE*dE;
				gpre=1.0;
			}break;
			case Opt::Loss::MAE:{
				const double mag=fabs(dE);
				error_train_+=mag;
				gpre=1.0/mag;
			}break;
			case Opt::Loss::HUBER:{
				const double rad=sqrt(dE*dE+1.0);
				error_train_+=rad-1.0;
				gpre=1.0/rad;
			}break;
			default:{
				gpre=0;
			}break;
		}
		gpre*=scale*dE;
		//**** scale and sum atomic gradients ****
		for(int j=0; j<nElements_; ++j) grad[j].setZero();
		for(int n=0; n<dist_atomt[ii].size(); ++n){
			//get the index of the atom within the local processor subset
			const int m=dist_atomt[ii].index(n);
			//find the element index in the nn potential
			const int type=struc_train[ii].type(m);
			//execute the network
			nnp_.nnh(type).nn().execute(struc_train[ii].symm(m));
			//compute spin prefactor
			double gpres=0;
			for(int j=0; j<nAtoms; ++j){
				gpres+=struc_train[ii].js(j)*Jt_[ii](j,m);
			}
			//compute dcda
			Eigen::VectorXd dcda(2);
			dcda<<gpre,gpre*gpres;
			//compute gradient
			grad[type].noalias()+=cost_[type].grad(nnp_.nnh(type).nn(),dcda);
		}
		//**** accumulate gradient across the BATCH communicator ****
		for(int j=0; j<nElements_; ++j){
			MPI_Allreduce(MPI_IN_PLACE,grad[j].data(),grad[j].size(),MPI_DOUBLE,MPI_SUM,BATCH.mpic());
		}
		//**** add gradient to total ****
		for(int j=0; j<nElements_; ++j){
			gElement_[j].noalias()+=grad[j];
		}
	}
	
	//====== compute validation error and gradient ======
	if(data_.step()%data_.nPrint()==0 || data_.step()%data_.nWrite()==0){
		if(NNPTES_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"computing validation error and gradient\n";
		for(int i=0; i<struc_val.size(); ++i){
			const int nAtoms=struc_val[i].nAtoms();
			//**** compute the energy ****
			double energyV=0;
			for(int n=0; n<struc_val[i].nAtoms(); ++n) struc_val[i].js(n)=0.0;
			for(int n=0; n<dist_atomv[i].size(); ++n){
				//get the index of the atom within the local processor subset
				const int m=dist_atomv[i].index(n);
				//find the element index in the nn potential
				const int type=struc_val[i].type(m);
				//execute the network
				nnp_.nnh(type).nn().execute(struc_val[i].symm(m));
				//add the energy to the total
				energyV+=nnp_.nnh(type).nn().out()[0]+nnp_.nnh(type).type().energy().val()*0.0;
				struc_val[i].js(m)=nnp_.nnh(type).nn().out()[1]+nnp_.nnh(type).type().js().val();
			}
			//**** accumulate energy ****
			MPI_Allreduce(MPI_IN_PLACE,&energyV,1,MPI_DOUBLE,MPI_SUM,BATCH.mpic());
			//**** accumulate js across BATCH communicator ****
			MPI_Allreduce(MPI_IN_PLACE,struc_val[i].js().data(),nAtoms,MPI_DOUBLE,MPI_SUM,BATCH.mpic());
			//**** compute the spin energy ****
			double energyS=0;
			for(int si=0; si<nAtoms; ++si){
				for(int sj=0; sj<nAtoms; ++sj){
					energyS+=struc_val[i].js(sj)*Jv_[i](sj,si)*struc_val[i].js(si);
				}
			}
			energyS*=0.5;
			//**** compute error ****
			const double energyT=energyS+energyV;
			std::cout<<"energy["<<i<<"] = "<<energyS<<" "<<energyV<<" "<<energyT<<" "<<struc_val[i].energy()<<"\n";
			const double dE=error_scale_*(energyT-struc_val[i].energy())/nAtoms;
			double error=0;
			switch(loss_){
				case Opt::Loss::MSE:{
					error_val_+=0.5*dE*dE;
					error=0.5*dE*dE;
				} break;
				case Opt::Loss::MAE:{
					error_val_+=std::fabs(dE);
					error=std::fabs(dE);
				} break;
				case Opt::Loss::HUBER:{
					error_val_+=(sqrt(1.0+(dE*dE))-1.0);
					error=(sqrt(1.0+(dE*dE))-1.0);
				} break;
				default: break;
			}
		}
	}
	
	//====== normalize w.r.t. batch size ======
	//note: we sum these quantities over WORLD, meaning that we are summing over duplicates in each BATCH
	//this normalization step corrects for this double counting
	const double batchsi=1.0/(1.0*BATCH.size());
	error_train_*=batchsi;
	error_val_*=batchsi;
	for(int j=0; j<nElements_; ++j) gElement_[j]*=batchsi;
	
	//====== return the error ======
	if(NNPTES_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"returning the error\n";
	return error_train_;
}

void NNPTES::read(const char* file, NNPTES& nnptes){
	if(NN_PRINT_FUNC>0) std::cout<<"NNPTES::read(const char*,NNPTES&):\n";
	//local variables
	FILE* reader=NULL;
	//open the file
	reader=std::fopen(file,"r");
	if(reader!=NULL){
		NNPTES::read(reader,nnptes);
		std::fclose(reader);
		reader=NULL;
	} else throw std::runtime_error(std::string("ERROR: Could not open \"")+std::string(file)+std::string("\" for reading.\n"));
}

void NNPTES::read(FILE* reader, NNPTES& nnptes){
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
		//files
		if(tag=="FILE_ERROR"){
			nnptes.file_error()=token.next();
		} else if(tag=="FILE_PARAMS"){
			nnptes.file_params()=token.next();
		} else if(tag=="FILE_ANN"){
			nnptes.file_ann()=token.next();
		} else if(tag=="FILE_RESTART"){
			nnptes.file_restart()=token.next();
		}
		//flags
		if(tag=="RESTART"){//read restart file
			nnptes.restart()=string::boolean(token.next().c_str());//restarting
		} else if(tag=="PRE_COND"){//whether to precondition the inputs
			nnptes.preCond()=string::boolean(token.next().c_str());
		} else if(tag=="CALC_FORCE"){//compute force at end
			nnptes.force()=string::boolean(token.next().c_str());
		} else if(tag=="WRITE_PARAMS"){
			nnptes.wparams()=string::boolean(token.next().c_str());
		} else if(tag=="NORM"){//normalize energy
			nnptes.norm()=string::boolean(token.next().c_str());
		}
		//optimization
		if(tag=="ERROR_SCALE"){
			nnptes.error_scale()=std::atof(token.next().c_str());
		} else if(tag=="LOSS"){
			nnptes.loss()=Opt::Loss::read(string::to_upper(token.next()).c_str());
		}
		//spin potential
		if(tag=="POT"){
			nnptes.pot().read(token);
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
	//structures - format
		AtomType atomT;
		atomT.name=true; atomT.an=true; atomT.type=true; atomT.index=true;
		atomT.posn=true; atomT.force=false; atomT.symm=true; atomT.charge=false;
		atomT.chi=false; atomT.eta=false; atomT.spin=true; atomT.js=true;
		FILE_FORMAT::type format;//format of training data
	//nn potential - opt
		int nBatch=-1;
		std::vector<Type> types;//unique atomic species
		NNPTES nnptes;//nn potential optimization data
		Opt::Model* model_param_=NULL;//optimization model
		std::vector<std::vector<int> > nh;//hidden layer configuration
		NN::ANNP annp;//neural network initialization parameters
	//structures - data
		std::vector<std::string> data_train;  //data files - training
		std::vector<std::string> data_val;    //data files - validation
		std::vector<std::string> data_test;   //data files - testing
		std::vector<std::string> files_train; //structure files - training
		std::vector<std::string> files_val;   //structure files - validation
		std::vector<std::string> files_test;  //structure files - testing
		std::vector<Structure> struc_train;   //structures - training
		std::vector<Structure> struc_val;     //structures - validation
		std::vector<Structure> struc_test;    //structures - testing
	//mpi data distribution
		thread::Dist dist_batch; //data distribution - batch
		thread::Dist dist_train; //data distribution - training
		thread::Dist dist_val;   //data distribution - validation
		thread::Dist dist_test;  //data distribution - testing
	//timing
		Clock clock,clock_wall;     //time objects
		double time_wall=0;         //total wall time
		double time_energy_train=0; //compute time - energies - training
		double time_energy_val=0;   //compute time - energies - validation
		double time_energy_test=0;  //compute time - energies - testing
		double time_force_train=0;  //compute time - forces - training
		double time_force_val=0;    //compute time - forces - validation
		double time_force_test=0;   //compute time - forces - testing
		double time_symm_train=0;   //compute time - symmetry functions - training
		double time_symm_val=0;     //compute time - symmetry functions - validation
		double time_symm_test=0;    //compute time - symmetry functions - test
	//file i/o
		FILE* reader=NULL;
		char* paramfile=new char[string::M];
		char* input=new char[string::M];
		char* strbuf=new char[print::len_buf];
		bool read_pot=false;
		std::string file_pot;
		std::vector<std::string> files_basis;//file - stores basis
		Token token;
	//writing
		bool write_energy=false; //writing - energies
		bool write_input=false;  //writing - inputs
		bool write_symm=false;   //writing - symmetry functions
		bool write_force=false;  //writing - forces
		
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
			std::cout<<"date     = "<<compiler::date()<<"\n";
			std::cout<<"time     = "<<compiler::time()<<"\n";
			std::cout<<"compiler = "<<compiler::name()<<"\n";
			std::cout<<"version  = "<<compiler::version()<<"\n";
			std::cout<<"standard = "<<compiler::standard()<<"\n";
			std::cout<<"arch     = "<<compiler::arch()<<"\n";
			std::cout<<"instr    = "<<compiler::instr()<<"\n";
			std::cout<<"os       = "<<compiler::os()<<"\n";
			std::cout<<"omp      = "<<compiler::omp()<<"\n";
		}
		
		//======== print mathematical constants ========
		if(WORLD.rank()==0){
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("MATHEMATICAL CONSTANTS",strbuf)<<"\n";
			std::printf("PI    = %.15f\n",math::constant::PI);
			std::printf("RadPI = %.15f\n",math::constant::RadPI);
			std::printf("Rad2  = %.15f\n",math::constant::Rad2);
			std::cout<<print::title("MATHEMATICAL CONSTANTS",strbuf)<<"\n";
			std::cout<<print::buf(strbuf)<<"\n";
		}
		
		//======== print physical constants ========
		if(WORLD.rank()==0){
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("PHYSICAL CONSTANTS",strbuf)<<"\n";
			std::printf("bohr-r  (A)  = %.12f\n",units::BOHR);
			std::printf("hartree (eV) = %.12f\n",units::HARTREE);
			std::cout<<print::title("PHYSICAL CONSTANTS",strbuf)<<"\n";
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
				std::cout<<print::title("MPI",strbuf)<<"\n";
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
			if(NNPTES_PRINT_STATUS>0) std::cout<<"reading parameter file\n";
			std::strcpy(paramfile,argv[1]);
			
			//======== open the parameter file ========
			if(NNPTES_PRINT_STATUS>0) std::cout<<"opening parameter file\n";
			reader=fopen(paramfile,"r");
			if(reader==NULL) throw std::runtime_error(std::string("I/O Error: Could not open parameter file: ")+paramfile);
			
			//======== read in the parameters ========
			if(NNPTES_PRINT_STATUS>-1) std::cout<<"reading parameters\n";
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
					data_train.push_back(token.next());
				} else if(tag=="DATA_VAL"){//data - validation
					data_val.push_back(token.next());
				} else if(tag=="DATA_TEST"){//data - testing
					data_test.push_back(token.next());
				} else if(tag=="ATOM"){//atom - name/mass/energy
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
					} else if(atomtag=="JS"){
						types[index].js().flag()=true;
						types[index].js().val()=std::atof(token.next().c_str());
					} else if(atomtag=="BASIS"){
						files_basis[index]=token.next();
					} else if(atomtag=="NH"){
						nh[index].clear();
						while(!token.end()) nh[index].push_back(std::atoi(token.next().c_str()));
					}
				} 
				//neural network potential
				if(tag=="R_CUT"){//distance cutoff
					nnptes.nnp().rc()=std::atof(token.next().c_str());
				} else if(tag=="READ_POT"){
					file_pot=token.next();
					read_pot=true;
				}
				//nnp train - flags
				if(tag=="N_BATCH"){//size of the batch
					nBatch=std::atoi(token.next().c_str());
				} 
				//writing
				if(tag=="WRITE_ENERGY"){//whether to write the final energies
					write_energy=string::boolean(token.next().c_str());
				} else if(tag=="WRITE_INPUT"){//whether to write the final energies
					write_input=string::boolean(token.next().c_str());
				} else if(tag=="WRITE_FORCE"){//whether to write the final forces
					write_force=string::boolean(token.next().c_str());
				} else if(tag=="WRITE_SYMM"){//print symmetry functions
					write_symm=string::boolean(token.next().c_str());
				} 
			}
			
			//======== set atom flags ========
			atomT.force=nnptes.force();
			
			//======== read - nnptes =========
			if(NNPTES_PRINT_STATUS>0) std::cout<<"reading neural network training parameters\n";
			NNPTES::read(reader,nnptes);
			
			//======== read - annp =========
			if(NNPTES_PRINT_STATUS>0) std::cout<<"reading neural network parameters\n";
			NN::ANNP::read(reader,annp);
			
			//======== read optimization data =========
			if(NNPTES_PRINT_STATUS>0) std::cout<<"reading optimization data\n";
			Opt::read(nnptes.data_,reader);
			
			//======== read optimization model ========
			if(NNPTES_PRINT_STATUS>0) std::cout<<"reading optimization model\n";
			switch(nnptes.data().algo()){
				case Opt::Algo::SGD:
					nnptes.model().reset(new Opt::SGD());
					Opt::read(static_cast<Opt::SGD&>(*nnptes.model()),reader);
					model_param_=new Opt::SGD(static_cast<const Opt::SGD&>(*nnptes.model()));
				break;
				case Opt::Algo::SDM:
					nnptes.model().reset(new Opt::SDM());
					Opt::read(static_cast<Opt::SDM&>(*nnptes.model()),reader);
					model_param_=new Opt::SDM(static_cast<const Opt::SDM&>(*nnptes.model()));
				break;
				case Opt::Algo::NAG:
					nnptes.model().reset(new Opt::NAG());
					Opt::read(static_cast<Opt::NAG&>(*nnptes.model()),reader);
					model_param_=new Opt::NAG(static_cast<const Opt::NAG&>(*nnptes.model()));
				break;
				case Opt::Algo::ADAGRAD:
					nnptes.model().reset(new Opt::ADAGRAD());
					Opt::read(static_cast<Opt::ADAGRAD&>(*nnptes.model()),reader);
					model_param_=new Opt::ADAGRAD(static_cast<const Opt::ADAGRAD&>(*nnptes.model()));
				break;
				case Opt::Algo::ADADELTA:
					nnptes.model().reset(new Opt::ADADELTA());
					Opt::read(static_cast<Opt::ADADELTA&>(*nnptes.model()),reader);
					model_param_=new Opt::ADADELTA(static_cast<const Opt::ADADELTA&>(*nnptes.model()));
				break;
				case Opt::Algo::RMSPROP:
					nnptes.model().reset(new Opt::RMSPROP());
					Opt::read(static_cast<Opt::RMSPROP&>(*nnptes.model()),reader);
					model_param_=new Opt::RMSPROP(static_cast<const Opt::RMSPROP&>(*nnptes.model()));
				break;
				case Opt::Algo::ADAM:
					nnptes.model().reset(new Opt::ADAM());
					Opt::read(static_cast<Opt::ADAM&>(*nnptes.model()),reader);
					model_param_=new Opt::ADAM(static_cast<const Opt::ADAM&>(*nnptes.model()));
				break;
				case Opt::Algo::NADAM:
					nnptes.model().reset(new Opt::NADAM());
					Opt::read(static_cast<Opt::NADAM&>(*nnptes.model()),reader);
					model_param_=new Opt::NADAM(static_cast<const Opt::NADAM&>(*nnptes.model()));
				break;
				case Opt::Algo::AMSGRAD:
					nnptes.model().reset(new Opt::AMSGRAD());
					Opt::read(static_cast<Opt::AMSGRAD&>(*nnptes.model()),reader);
					model_param_=new Opt::AMSGRAD(static_cast<const Opt::AMSGRAD&>(*nnptes.model()));
				break;
				case Opt::Algo::BFGS:
					nnptes.model().reset(new Opt::BFGS());
					Opt::read(static_cast<Opt::BFGS&>(*nnptes.model()),reader);
					model_param_=new Opt::BFGS(static_cast<const Opt::BFGS&>(*nnptes.model()));
				break;
				case Opt::Algo::RPROP:
					nnptes.model().reset(new Opt::RPROP());
					Opt::read(static_cast<Opt::RPROP&>(*nnptes.model()),reader);
					model_param_=new Opt::RPROP(static_cast<const Opt::RPROP&>(*nnptes.model()));
				break;
				case Opt::Algo::CG:
					nnptes.model().reset(new Opt::CG());
					Opt::read(static_cast<Opt::CG&>(*nnptes.model()),reader);
					model_param_=new Opt::CG(static_cast<const Opt::CG&>(*nnptes.model()));
				break;
				default:
					throw std::invalid_argument("Invalid optimization algorithm.");
				break;
			}
			
			//======== close parameter file ========
			if(NNPTES_PRINT_STATUS>0) std::cout<<"closing parameter file\n";
			fclose(reader);
			reader=NULL;
			
			//======== check if we compute symmetry functions ========
			if(mode==Mode::SYMM) nnptes.symm()=true;
			else if(mode==Mode::TRAIN || mode==Mode::TEST){
				if(format==FILE_FORMAT::BINARY) nnptes.symm()=false;
				else nnptes.symm()=true;
			}
			
			//========= check the data =========
			if(mode==Mode::TRAIN && data_train.size()==0) throw std::invalid_argument("No training data provided.");
			if(mode==Mode::TRAIN && data_val.size()==0) throw std::invalid_argument("No validation data provided.");
			if(mode==Mode::TEST && data_test.size()==0) throw std::invalid_argument("No test data provided.");
		}
		
		//======== bcast the parameters ========
		if(NNPTES_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"broadcasting parameters\n";
		//general parameters
		MPI_Bcast(&unitsys,1,MPI_INT,0,WORLD.mpic());
		//nnp_opt
		MPI_Bcast(&nBatch,1,MPI_INT,0,WORLD.mpic());
		thread::bcast(WORLD.mpic(),0,nnptes);
		//file i/o
		MPI_Bcast(&format,1,MPI_INT,0,WORLD.mpic());
		//writing
		MPI_Bcast(&write_energy,1,MPI_C_BOOL,0,WORLD.mpic());
		MPI_Bcast(&write_force,1,MPI_C_BOOL,0,WORLD.mpic());
		MPI_Bcast(&write_input,1,MPI_C_BOOL,0,WORLD.mpic());
		MPI_Bcast(&write_symm,1,MPI_C_BOOL,0,WORLD.mpic());
		//mode
		MPI_Bcast(&mode,1,MPI_INT,0,WORLD.mpic());
		//atom type
		thread::bcast(WORLD.mpic(),0,atomT);
		thread::bcast(WORLD.mpic(),0,annp);
		
		//======== print parameters ========
		if(WORLD.rank()==0){
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("GENERAL PARAMETERS",strbuf)<<"\n";
			std::cout<<"READ_POT   = "<<read_pot<<"\n";
			std::cout<<"ATOM_T     = "<<atomT<<"\n";
			std::cout<<"FORMAT     = "<<format<<"\n";
			std::cout<<"UNITS      = "<<unitsys<<"\n";
			std::cout<<"MODE       = "<<mode<<"\n";
			std::cout<<"DATA_TRAIN = \n"; for(int i=0; i<data_train.size(); ++i) std::cout<<"\t\t"<<data_train[i]<<"\n";
			std::cout<<"DATA_VAL   = \n"; for(int i=0; i<data_val.size(); ++i) std::cout<<"\t\t"<<data_val[i]<<"\n";
			std::cout<<"DATA_TEST  = \n"; for(int i=0; i<data_test.size(); ++i) std::cout<<"\t\t"<<data_test[i]<<"\n";
			std::cout<<print::title("GENERAL PARAMETERS",strbuf)<<"\n";
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("WRITING",strbuf)<<"\n";
			std::cout<<"WRITE_ENERGY = "<<write_energy<<"\n";
			std::cout<<"WRITE_INPUTS = "<<write_input<<"\n";
			std::cout<<"WRITE_SYMM   = "<<write_symm<<"\n";
			std::cout<<"WRITE_FORCE  = "<<write_force<<"\n";
			std::cout<<print::title("WRITING",strbuf)<<"\n";
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("ATOMS",strbuf)<<"\n";
			for(int i=0; i<types.size(); ++i){
				std::cout<<types[i]<<"\n";
			}
			std::cout<<print::title("ATOMS",strbuf)<<"\n";
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<annp<<"\n";
			std::cout<<nnptes<<"\n";
		}
		MPI_Barrier(WORLD.mpic());
		
		//======== check the parameters ========
		if(mode==Mode::UNKNOWN) throw std::invalid_argument("Invalid calculation mode");
		if(format==FILE_FORMAT::UNKNOWN) throw std::invalid_argument("Invalid file format.");
		if(unitsys==units::System::UNKNOWN) throw std::invalid_argument("Invalid unit system.");
		if(nnptes.loss_==Opt::Loss::UNKNOWN) throw std::invalid_argument("Invalid loss function.");
		if(nnptes.error_scale_<=0) throw std::invalid_argument("Invalid error scaling.");
		
		//======== set the unit system ========
		if(NNPTES_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"setting the unit system\n";
		units::consts::init(unitsys);
		
		//************************************************************************************
		// READ/INITIALIZE NN-POT
		//************************************************************************************
		
		//======== initialize the potential (rank 0) ========
		if(NNPTES_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"initializing the potential\n";
		if(WORLD.rank()==0){
			//======== read the basis (if not restarting) ========
			if(!nnptes.restart_){
				if(!read_pot){
					//resize the potential
					if(NNPTES_PRINT_STATUS>-1) std::cout<<"resizing potential\n";
					nnptes.nnp_.resize(types);
					//read basis files
					if(NNPTES_PRINT_STATUS>-1) std::cout<<"reading basis files\n";
					if(files_basis.size()!=nnptes.nnp_.ntypes()) throw std::runtime_error("main(int,char**): invalid number of basis files.");
					for(int i=0; i<nnptes.nnp_.ntypes(); ++i){
						const char* file=files_basis[i].c_str();
						const char* atomName=types[i].name().c_str();
						NNP::read_basis(file,nnptes.nnp_,atomName);
					}
					//initialize the neural network hamiltonians
					if(NNPTES_PRINT_STATUS>-1) std::cout<<"initializing neural network hamiltonians\n";
					for(int i=0; i<nnptes.nnp_.ntypes(); ++i){
						NNH& nnhl=nnptes.nnp_.nnh(i);
						nnhl.type()=types[i];
						nnhl.nn().resize(annp,nnhl.nInput(),nh[i],2);
						nnhl.dOutDVal().resize(nnhl.nn());
					}
				} else {
					if(NNPTES_PRINT_STATUS>-1) std::cout<<"reading potential\n";
					NNP::read(file_pot.c_str(),nnptes.nnp_);
				}
			}
			//======== read restart file (if restarting) ========
			if(nnptes.restart_){
				if(NNPTES_PRINT_STATUS>-1) std::cout<<"reading restart file\n";
				const std::string file=nnptes.file_restart_;
				nnptes.read_restart(file.c_str());
				nnptes.restart()=true;
			}
			//======== print the potential ========
			if(WORLD.rank()==0) std::cout<<"printing the potential\n";
			std::cout<<nnptes.nnp_<<"\n";
		}
		
		//======== bcast the potential ========
		if(WORLD.rank()==0) std::cout<<"bcasting the potential\n";
		thread::bcast(WORLD.mpic(),0,nnptes.nnp_);
		
		//************************************************************************************
		// READ DATA
		//************************************************************************************
		
		//======== rank 0 reads the data files (lists of structure files) ========
		if(WORLD.rank()==0){
			//==== read the training data ====
			if(NNPTES_PRINT_STATUS>-1) std::cout<<"reading data - training\n";
			for(int i=0; i<data_train.size(); ++i){
				//open the data file
				if(NNPTES_PRINT_DATA>0) std::cout<<"data file "<<i<<": "<<data_train[i]<<"\n";
				reader=fopen(data_train[i].c_str(),"r");
				if(reader==NULL) throw std::runtime_error(std::string("I/O Error: Could not open data file: ")+data_train[i]);
				//read in the data
				while(fgets(input,string::M,reader)!=NULL){
					if(!string::empty(input)) files_train.push_back(std::string(string::trim(input)));
				}
				//close the file
				fclose(reader);
				reader=NULL;
			}
			//==== read the validation data ====
			if(NNPTES_PRINT_STATUS>-1) std::cout<<"reading data - validation\n";
			for(int i=0; i<data_val.size(); ++i){
				//open the data file
				if(NNPTES_PRINT_DATA>0) std::cout<<"data file "<<i<<": "<<data_val[i]<<"\n";
				reader=fopen(data_val[i].c_str(),"r");
				if(reader==NULL) throw std::runtime_error(std::string("I/O Error: Could not open data file: ")+data_val[i]);
				//read in the data
				while(fgets(input,string::M,reader)!=NULL){
					if(!string::empty(input)) files_val.push_back(std::string(string::trim(input)));
				}
				//close the file
				fclose(reader);
				reader=NULL;
			}
			//==== read the test data ====
			if(NNPTES_PRINT_STATUS>-1) std::cout<<"reading data - testing\n";
			for(int i=0; i<data_test.size(); ++i){
				//open the data file
				if(NNPTES_PRINT_DATA>0) std::cout<<"data file "<<i<<": "<<data_test[i]<<"\n";
				reader=fopen(data_test[i].c_str(),"r");
				if(reader==NULL) throw std::runtime_error(std::string("I/O Error: Could not open data file: ")+data_test[i]);
				//read in the data
				while(fgets(input,string::M,reader)!=NULL){
					if(!string::empty(input)) files_test.push_back(std::string(string::trim(input)));
				}
				//close the file
				fclose(reader);
				reader=NULL;
			}
		
			//==== print the files ====
			if(NNPTES_PRINT_DATA>1){
				if(files_train.size()>0){
					std::cout<<print::buf(strbuf)<<"\n";
					std::cout<<print::title("FILES - TRAIN",strbuf)<<"\n";
					for(int i=0; i<files_train.size(); ++i) std::cout<<"\t"<<files_train[i]<<"\n";
					std::cout<<print::title("FILES - TRAIN",strbuf)<<"\n";
					std::cout<<print::buf(strbuf)<<"\n";
				}
				if(files_val.size()>0){
					std::cout<<print::buf(strbuf)<<"\n";
					std::cout<<print::title("FILES - Stop",strbuf)<<"\n";
					for(int i=0; i<files_val.size(); ++i) std::cout<<"\t"<<files_val[i]<<"\n";
					std::cout<<print::title("FILES - Stop",strbuf)<<"\n";
					std::cout<<print::buf(strbuf)<<"\n";
				}
				if(files_test.size()>0){
					std::cout<<print::buf(strbuf)<<"\n";
					std::cout<<print::title("FILES - TEST",strbuf)<<"\n";
					for(int i=0; i<files_test.size(); ++i) std::cout<<"\t"<<files_test[i]<<"\n";
					std::cout<<print::title("FILES - TEST",strbuf)<<"\n";
					std::cout<<print::buf(strbuf)<<"\n";
				}
			}
		}
		
		//======== bcast the file names =======
		if(NNPTES_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"bcasting file names\n";
		//bcast names
		thread::bcast(WORLD.mpic(),0,files_train);
		thread::bcast(WORLD.mpic(),0,files_val);
		thread::bcast(WORLD.mpic(),0,files_test);
		//set number of structures
		const int nTrain=files_train.size();
		const int nVal=files_val.size();
		const int nTest=files_test.size();
		//print number of structures
		if(WORLD.rank()==0){
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("DATA",strbuf)<<"\n";
			std::cout<<"ntrain = "<<nTrain<<"\n";
			std::cout<<"nval   = "<<nVal<<"\n";
			std::cout<<"ntest  = "<<nTest<<"\n";
			std::cout<<"nbatch = "<<nBatch<<"\n";
			std::cout<<print::title("DATA",strbuf)<<"\n";
			std::cout<<print::buf(strbuf)<<"\n";
		}
		//check the batch size
		if(nBatch<=0) throw std::invalid_argument("Invalid batch size.");
		if(nBatch>nTrain) throw std::invalid_argument("Invalid batch size.");
		
		//======== gen thread dist + offset, batch communicators ========
		//split WORLD into BATCH
		BATCH=WORLD.split(WORLD.color(WORLD.ncomm(nBatch)));
		//print batch communicators
		{
			if(WORLD.rank()==0){
				std::cout<<print::buf(strbuf)<<"\n";
				std::cout<<print::title("BATCH Communicators",strbuf)<<"\n";
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
				std::cout<<print::title("BATCH Communicators",strbuf)<<"\n";
				std::cout<<print::buf(strbuf)<<"\n";
				std::cout<<std::flush;
			}
			MPI_Barrier(WORLD.mpic());
		}
		//thread dist - divide structures equally among the batch groups
		dist_batch.init(BATCH.ncomm(),BATCH.color(),nBatch);
		dist_train.init(BATCH.ncomm(),BATCH.color(),nTrain);
		dist_val.init(BATCH.ncomm(),BATCH.color(),nVal);
		dist_test.init(BATCH.ncomm(),BATCH.color(),nTest);
		//print
		if(WORLD.rank()==0){
			std::string str;
			std::cout<<"thread_dist_batch   = "<<thread::Dist::size(str,BATCH.ncomm(),nBatch)<<"\n";
			std::cout<<"thread_dist_train   = "<<thread::Dist::size(str,BATCH.ncomm(),nTrain)<<"\n";
			std::cout<<"thread_dist_val     = "<<thread::Dist::size(str,BATCH.ncomm(),nVal)<<"\n";
			std::cout<<"thread_dist_test    = "<<thread::Dist::size(str,BATCH.ncomm(),nTest)<<"\n";
			std::cout<<"thread_offset_batch = "<<thread::Dist::offset(str,BATCH.ncomm(),nBatch)<<"\n";
			std::cout<<"thread_offset_train = "<<thread::Dist::offset(str,BATCH.ncomm(),nTrain)<<"\n";
			std::cout<<"thread_offset_val   = "<<thread::Dist::offset(str,BATCH.ncomm(),nVal)<<"\n";
			std::cout<<"thread_offset_test  = "<<thread::Dist::offset(str,BATCH.ncomm(),nTest)<<"\n";
		}
		
		//======== gen indices (random shuffle) ========
		std::vector<int> indices_train(nTrain,0);
		std::vector<int> indices_val(nVal,0);
		std::vector<int> indices_test(nTest,0);
		if(WORLD.rank()==0){
			for(int i=0; i<indices_train.size(); ++i) indices_train[i]=i;
			for(int i=0; i<indices_val.size(); ++i) indices_val[i]=i;
			for(int i=0; i<indices_test.size(); ++i) indices_test[i]=i;
			//std::random_shuffle(indices_train.begin(),indices_train.end());
			//std::random_shuffle(indices_val.begin(),indices_val.end());
			//std::random_shuffle(indices_test.begin(),indices_test.end());
		}
		//======== bcast randomized indices ========
		thread::bcast(WORLD.mpic(),0,indices_train);
		thread::bcast(WORLD.mpic(),0,indices_val);
		thread::bcast(WORLD.mpic(),0,indices_test);
		
		//======== read the structures ========
		//==== training structures ====
		if(NNPTES_PRINT_STATUS>-1) std::cout<<"reading structures - training - color "<<BATCH.color()<<" rank "<<BATCH.rank()<<"\n"<<std::flush;
		if(files_train.size()>0){
			struc_train.resize(dist_train.size());
			//rank 0 of batch group reads structures
			if(BATCH.rank()==0){
				for(int i=0; i<dist_train.size(); ++i){
					const std::string& file=files_train[indices_train[dist_train.index(i)]];
					read_struc(file.c_str(),format,atomT,struc_train[i]);
					if(NNPTES_PRINT_DATA>1) std::cout<<"\t"<<file<<" "<<struc_train[i].energy()<<"\n";
				}
			}
			//broadcast structures to all other procs in the BATCH group
			for(int i=0; i<dist_train.size(); ++i){
				thread::bcast(BATCH.mpic(),0,struc_train[i]);
			}
		}
		//==== validation structures ====
		if(NNPTES_PRINT_STATUS>-1) std::cout<<"reading structures - validation - color "<<BATCH.color()<<" rank "<<BATCH.rank()<<"\n"<<std::flush;
		if(files_val.size()>0){
			struc_val.resize(dist_val.size());
			//rank 0 of batch group reads structures
			if(BATCH.rank()==0){
				for(int i=0; i<dist_val.size(); ++i){
					const std::string& file=files_val[indices_val[dist_val.index(i)]];
					read_struc(file.c_str(),format,atomT,struc_val[i]);
					if(NNPTES_PRINT_DATA>1) std::cout<<"\t"<<file<<" "<<struc_val[i].energy()<<"\n";
				}
			}
			//broadcast structures to all other procs in BATCH group
			for(int i=0; i<dist_val.size(); ++i){
				thread::bcast(BATCH.mpic(),0,struc_val[i]);
			}
		}
		//==== testing structures ====
		if(NNPTES_PRINT_STATUS>-1) std::cout<<"reading structures - testing - color "<<BATCH.color()<<" rank "<<BATCH.rank()<<"\n"<<std::flush;
		if(files_test.size()>0){
			struc_test.resize(dist_test.size());
			//rank 0 of batch group reads structures
			if(BATCH.rank()==0){
				for(int i=0; i<dist_test.size(); ++i){
					const std::string& file=files_test[indices_test[dist_test.index(i)]];
					read_struc(file.c_str(),format,atomT,struc_test[i]);
					if(NNPTES_PRINT_DATA>1) std::cout<<"\t"<<file<<" "<<struc_test[i].energy()<<"\n";
				}
			}
			//broadcast structures to all other procs in group
			for(int i=0; i<dist_test.size(); ++i){
				thread::bcast(BATCH.mpic(),0,struc_test[i]);
			}
		}
		MPI_Barrier(WORLD.mpic());
		
		//======== check the structures ========
		if(NNPTES_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"checking the structures\n";
		//==== training structures ====
		if(BATCH.rank()==0){
			for(int i=0; i<dist_train.size(); ++i){
				const std::string filename=files_train[indices_train[dist_train.index(i)]];
				const Structure& strucl=struc_train[i];
				if(strucl.nAtoms()==0) throw std::runtime_error(std::string("ERROR: Structure \"")+filename+std::string(" has zero atoms."));
				if(std::isinf(strucl.energy())) throw std::runtime_error(std::string("ERROR: Structure \"")+filename+std::string(" has inf energy."));
				if(strucl.energy()!=strucl.energy()) throw std::runtime_error(std::string("ERROR: Structure \"")+filename+std::string(" has nan energy."));
				if(std::fabs(strucl.energy())<math::constant::ZERO) std::cout<<"*********** WARNING: Structure \""<<filename<<"\" has ZERO energy. ***********";
				if(nnptes.force_){
					for(int n=0; n<strucl.nAtoms(); ++n){
						const double force=strucl.force(n).squaredNorm();
						if(std::isinf(force)) std::cout<<"WARNING: Atom \""<<strucl.name(n)<<strucl.index(n)<<"\" in \""<<filename<<" has inf force.\n";
						if(force!=force) std::cout<<"WARNING: Atom \""<<strucl.name(n)<<strucl.index(n)<<"\" in \""<<filename<<" has nan force.\n";
					}
				}
				if(NNPTES_PRINT_DATA>1) std::cout<<"\t"<<filename<<" "<<strucl.energy()<<" "<<WORLD.rank()<<"\n";
			}
		}
		//==== validation structures ====
		if(BATCH.rank()==0){
			for(int i=0; i<dist_val.size(); ++i){
				const std::string filename=files_val[indices_val[dist_val.index(i)]];
				const Structure& strucl=struc_val[i];
				if(strucl.nAtoms()==0) throw std::runtime_error(std::string("ERROR: Structure \"")+filename+std::string(" has zero atoms."));
				if(std::isinf(strucl.energy())) throw std::runtime_error(std::string("ERROR: Structure \"")+filename+std::string(" has inf energy."));
				if(strucl.energy()!=strucl.energy()) throw std::runtime_error(std::string("ERROR: Structure \"")+filename+std::string(" has nan energy."));
				if(std::fabs(strucl.energy())<math::constant::ZERO) std::cout<<"*********** WARNING: Structure \""<<filename<<"\" has ZERO energy. ***********";
				if(nnptes.force_){
					for(int n=0; n<strucl.nAtoms(); ++n){
						const double force=strucl.force(n).squaredNorm();
						if(std::isinf(force)) std::cout<<"WARNING: Atom \""<<strucl.name(n)<<strucl.index(n)<<"\" in \""<<filename<<" has inf force.\n";
						if(force!=force) std::cout<<"WARNING: Atom \""<<strucl.name(n)<<strucl.index(n)<<"\" in \""<<filename<<" has nan force.\n";
					}
				}
			}
			if(NNPTES_PRINT_DATA>1) for(int i=0; i<dist_val.size(); ++i) std::cout<<"\t"<<files_val[dist_val.index(i)]<<" "<<struc_val[i].energy()<<"\n";
		}
		//==== testing structures ====
		if(BATCH.rank()==0){
			for(int i=0; i<dist_test.size(); ++i){
				const std::string filename=files_test[indices_test[dist_test.index(i)]];
				const Structure& strucl=struc_test[i];
				if(strucl.nAtoms()==0) throw std::runtime_error(std::string("ERROR: Structure \"")+filename+std::string(" has zero atoms."));
				if(std::isinf(strucl.energy())) throw std::runtime_error(std::string("ERROR: Structure \"")+filename+std::string(" has inf energy."));
				if(strucl.energy()!=strucl.energy()) throw std::runtime_error(std::string("ERROR: Structure \"")+filename+std::string(" has nan energy."));
				if(std::fabs(strucl.energy())<math::constant::ZERO) std::cout<<"*********** WARNING: Structure \""<<filename<<"\" has ZERO energy. ***********";
				if(nnptes.force_){
					for(int n=0; n<strucl.nAtoms(); ++n){
						const double force=strucl.force(n).squaredNorm();
						if(std::isinf(force)) std::cout<<"WARNING: Atom \""<<strucl.name(n)<<strucl.index(n)<<"\" in \""<<filename<<" has inf force.\n";
						if(force!=force) std::cout<<"WARNING: Atom \""<<strucl.name(n)<<strucl.index(n)<<"\" in \""<<filename<<" has nan force.\n";
					}
				}
			}
			if(NNPTES_PRINT_DATA>1) for(int i=0; i<dist_test.size(); ++i) std::cout<<"\t"<<files_test[indices_test[dist_test.index(i)]]<<" "<<struc_test[i].energy()<<"\n";
		}
		MPI_Barrier(WORLD.mpic());
		
		//************************************************************************************
		// ATOM PROPERTIES
		//************************************************************************************
		
		//======== set atom properties ========
		if(WORLD.rank()==0) std::cout<<"setting atomic properties\n";
		
		//======== set the indices ========
		if(WORLD.rank()==0) std::cout<<"setting indices\n";
		for(int i=0; i<dist_train.size(); ++i){
			for(int n=0; n<struc_train[i].nAtoms(); ++n){
				struc_train[i].index(n)=n;
			}
		}
		for(int i=0; i<dist_val.size(); ++i){
			for(int n=0; n<struc_val[i].nAtoms(); ++n){
				struc_val[i].index(n)=n;
			}
		}
		for(int i=0; i<dist_test.size(); ++i){
			for(int n=0; n<struc_test[i].nAtoms(); ++n){
				struc_test[i].index(n)=n;
			}
		}
		
		//======== set the atomic numbers ========
		if(WORLD.rank()==0) std::cout<<"setting atomic numbers\n";
		for(int i=0; i<dist_train.size(); ++i){
			for(int n=0; n<struc_train[i].nAtoms(); ++n){
				struc_train[i].an(n)=ptable::an(struc_train[i].name(n).c_str());
			}
		}
		for(int i=0; i<dist_val.size(); ++i){
			for(int n=0; n<struc_val[i].nAtoms(); ++n){
				struc_val[i].an(n)=ptable::an(struc_val[i].name(n).c_str());
			}
		}
		for(int i=0; i<dist_test.size(); ++i){
			for(int n=0; n<struc_test[i].nAtoms(); ++n){
				struc_test[i].an(n)=ptable::an(struc_test[i].name(n).c_str());
			}
		}
		
		//======== set the types ========
		if(WORLD.rank()==0) std::cout<<"setting types\n";
		for(int i=0; i<dist_train.size(); ++i){
			for(int n=0; n<struc_train[i].nAtoms(); ++n){
				struc_train[i].type(n)=nnptes.nnp_.index(struc_train[i].name(n));
			}
		}
		for(int i=0; i<dist_val.size(); ++i){
			for(int n=0; n<struc_val[i].nAtoms(); ++n){
				struc_val[i].type(n)=nnptes.nnp_.index(struc_val[i].name(n));
			}
		}
		for(int i=0; i<dist_test.size(); ++i){
			for(int n=0; n<struc_test[i].nAtoms(); ++n){
				struc_test[i].type(n)=nnptes.nnp_.index(struc_test[i].name(n));
			}
		}
		
		//======== set the charges ========
		if(atomT.charge){
			if(WORLD.rank()==0) std::cout<<"setting charges\n";
			//==== set charges - training ====
			for(int i=0; i<dist_train.size(); ++i){
				for(int n=0; n<struc_train[i].nAtoms(); ++n){
					for(int j=0; j<nnptes.nnp_.ntypes(); ++j){
						if(nnptes.nnp_.nnh(j).type().name()==struc_train[i].name(n)){
							struc_train[i].charge(n)=nnptes.nnp_.nnh(j).type().charge().val();
							break;
						}
					}
				}
			}
			//==== set charges - validation ====
			for(int i=0; i<dist_val.size(); ++i){
				for(int n=0; n<struc_val[i].nAtoms(); ++n){
					for(int j=0; j<nnptes.nnp_.ntypes(); ++j){
						if(nnptes.nnp_.nnh(j).type().name()==struc_val[i].name(n)){
							struc_val[i].charge(n)=nnptes.nnp_.nnh(j).type().charge().val();
							break;
						}
					}
				}
			}
			//==== set charges - testing ====
			for(int i=0; i<dist_test.size(); ++i){
				for(int n=0; n<struc_test[i].nAtoms(); ++n){
					for(int j=0; j<nnptes.nnp_.ntypes(); ++j){
						if(nnptes.nnp_.nnh(j).type().name()==struc_test[i].name(n)){
							struc_test[i].charge(n)=nnptes.nnp_.nnh(j).type().charge().val();
							break;
						}
					}
				}
			}
		}
		
		//======== set the electronegativities ========
		if(atomT.chi){
			if(WORLD.rank()==0) std::cout<<"setting electronegativities\n";
			//==== set electronegativities - training ====
			for(int i=0; i<dist_train.size(); ++i){
				for(int n=0; n<struc_train[i].nAtoms(); ++n){
					for(int j=0; j<nnptes.nnp_.ntypes(); ++j){
						if(nnptes.nnp_.nnh(j).type().name()==struc_train[i].name(n)){
							struc_train[i].chi(n)=nnptes.nnp_.nnh(j).type().chi().val();
							break;
						}
					}
				}
			}
			//==== set electronegativities - validation ====
			for(int i=0; i<dist_val.size(); ++i){
				for(int n=0; n<struc_val[i].nAtoms(); ++n){
					for(int j=0; j<nnptes.nnp_.ntypes(); ++j){
						if(nnptes.nnp_.nnh(j).type().name()==struc_val[i].name(n)){
							struc_val[i].chi(n)=nnptes.nnp_.nnh(j).type().chi().val();
							break;
						}
					}
				}
			}
			//==== set electronegativities - testing ====
			for(int i=0; i<dist_test.size(); ++i){
				for(int n=0; n<struc_test[i].nAtoms(); ++n){
					for(int j=0; j<nnptes.nnp_.ntypes(); ++j){
						if(nnptes.nnp_.nnh(j).type().name()==struc_test[i].name(n)){
							struc_test[i].chi(n)=nnptes.nnp_.nnh(j).type().chi().val();
							break;
						}
					}
				}
			}
		}
		
		//======== set the idempotentials ========
		if(atomT.eta){
			if(WORLD.rank()==0) std::cout<<"setting idempotentials\n";
			//==== set idempotentials - training ====
			for(int i=0; i<dist_train.size(); ++i){
				for(int n=0; n<struc_train[i].nAtoms(); ++n){
					for(int j=0; j<nnptes.nnp_.ntypes(); ++j){
						if(nnptes.nnp_.nnh(j).type().name()==struc_train[i].name(n)){
							struc_train[i].eta(n)=nnptes.nnp_.nnh(j).type().eta().val();
							break;
						}
					}
				}
			}
			//==== set idempotentials - validation ====
			for(int i=0; i<dist_val.size(); ++i){
				for(int n=0; n<struc_val[i].nAtoms(); ++n){
					for(int j=0; j<nnptes.nnp_.ntypes(); ++j){
						if(nnptes.nnp_.nnh(j).type().name()==struc_val[i].name(n)){
							struc_val[i].eta(n)=nnptes.nnp_.nnh(j).type().eta().val();
							break;
						}
					}
				}
			}
			//==== set idempotentials - testing ====
			for(int i=0; i<dist_test.size(); ++i){
				for(int n=0; n<struc_test[i].nAtoms(); ++n){
					for(int j=0; j<nnptes.nnp_.ntypes(); ++j){
						if(nnptes.nnp_.nnh(j).type().name()==struc_test[i].name(n)){
							struc_test[i].eta(n)=nnptes.nnp_.nnh(j).type().eta().val();
							break;
						}
					}
				}
			}
		}
		
		//************************************************************************************
		// INITIALIZE OPTIMIZER
		//************************************************************************************
		
		//======== set optimization data ========
		if(WORLD.rank()==0) std::cout<<"setting optimization data\n";
		if(WORLD.rank()==0){
			//opt - data
			Opt::read(nnptes.data_,paramfile);
			//opt - model
			switch(nnptes.data_.algo()){
				case Opt::Algo::SGD:{
					Opt::SGD& nnModel_=static_cast<Opt::SGD&>(*nnptes.model());
					Opt::SGD& pModel_=static_cast<Opt::SGD&>(*model_param_);
					if(pModel_.gamma()>0) nnModel_.gamma()=pModel_.gamma();
					if(pModel_.alpha()>0) nnModel_.alpha()=pModel_.alpha();
				}break;
				case Opt::Algo::SDM:{
					Opt::SDM& nnModel_=static_cast<Opt::SDM&>(*nnptes.model());
					Opt::SDM& pModel_=static_cast<Opt::SDM&>(*model_param_);
					if(pModel_.gamma()>0) nnModel_.gamma()=pModel_.gamma();
					if(pModel_.alpha()>0) nnModel_.alpha()=pModel_.alpha();
					if(pModel_.eta()>0) nnModel_.eta()=pModel_.eta();
				}break;
				case Opt::Algo::NAG:{
					Opt::NAG& nnModel_=static_cast<Opt::NAG&>(*nnptes.model());
					Opt::NAG& pModel_=static_cast<Opt::NAG&>(*model_param_);
					if(pModel_.gamma()>0) nnModel_.gamma()=pModel_.gamma();
					if(pModel_.alpha()>0) nnModel_.alpha()=pModel_.alpha();
					if(pModel_.eta()>0) nnModel_.eta()=pModel_.eta();
				}break;
				case Opt::Algo::ADAGRAD:{
					Opt::ADAGRAD& nnModel_=static_cast<Opt::ADAGRAD&>(*nnptes.model());
					Opt::ADAGRAD& pModel_=static_cast<Opt::ADAGRAD&>(*model_param_);
					if(pModel_.gamma()>0) nnModel_.gamma()=pModel_.gamma();
					if(pModel_.alpha()>0) nnModel_.alpha()=pModel_.alpha();
				}break;
				case Opt::Algo::ADADELTA:{
					Opt::ADADELTA& nnModel_=static_cast<Opt::ADADELTA&>(*nnptes.model());
					Opt::ADADELTA& pModel_=static_cast<Opt::ADADELTA&>(*model_param_);
					if(pModel_.gamma()>0) nnModel_.gamma()=pModel_.gamma();
					if(pModel_.alpha()>0) nnModel_.alpha()=pModel_.alpha();
					if(pModel_.eta()>0) nnModel_.eta()=pModel_.eta();
				}break;
				case Opt::Algo::RMSPROP:{
					Opt::RMSPROP& nnModel_=static_cast<Opt::RMSPROP&>(*nnptes.model());
					Opt::RMSPROP& pModel_=static_cast<Opt::RMSPROP&>(*model_param_);
					if(pModel_.gamma()>0) nnModel_.gamma()=pModel_.gamma();
					if(pModel_.alpha()>0) nnModel_.alpha()=pModel_.alpha();
				}break;
				case Opt::Algo::ADAM:{
					Opt::ADAM& nnModel_=static_cast<Opt::ADAM&>(*nnptes.model());
					Opt::ADAM& pModel_=static_cast<Opt::ADAM&>(*model_param_);
					if(pModel_.gamma()>0) nnModel_.gamma()=pModel_.gamma();
					if(pModel_.alpha()>0) nnModel_.alpha()=pModel_.alpha();
					nnModel_.decay()=pModel_.decay();
				}break;
				case Opt::Algo::NADAM:{
					Opt::NADAM& nnModel_=static_cast<Opt::NADAM&>(*nnptes.model());
					Opt::NADAM& pModel_=static_cast<Opt::NADAM&>(*model_param_);
					if(pModel_.gamma()>0) nnModel_.gamma()=pModel_.gamma();
					if(pModel_.alpha()>0) nnModel_.alpha()=pModel_.alpha();
					nnModel_.decay()=pModel_.decay();
				}break;
				case Opt::Algo::AMSGRAD:{
					Opt::AMSGRAD& nnModel_=static_cast<Opt::AMSGRAD&>(*nnptes.model());
					Opt::AMSGRAD& pModel_=static_cast<Opt::AMSGRAD&>(*model_param_);
					if(pModel_.gamma()>0) nnModel_.gamma()=pModel_.gamma();
					if(pModel_.alpha()>0) nnModel_.alpha()=pModel_.alpha();
					nnModel_.decay()=pModel_.decay();
				}break;
				case Opt::Algo::BFGS:{
					Opt::BFGS& nnModel_=static_cast<Opt::BFGS&>(*nnptes.model());
					Opt::BFGS& pModel_=static_cast<Opt::BFGS&>(*model_param_);
					if(pModel_.gamma()>0) nnModel_.gamma()=pModel_.gamma();
					if(pModel_.alpha()>0) nnModel_.alpha()=pModel_.alpha();
				}break;
				case Opt::Algo::CG:{
					Opt::CG& nnModel_=static_cast<Opt::CG&>(*nnptes.model());
					Opt::CG& pModel_=static_cast<Opt::CG&>(*model_param_);
					if(pModel_.gamma()>0) nnModel_.gamma()=pModel_.gamma();
					if(pModel_.alpha()>0) nnModel_.alpha()=pModel_.alpha();
					nnModel_.decay()=pModel_.decay();
				}break;
				case Opt::Algo::RPROP:
					//no parameters
				break;
			}
		}
		
		//======== print optimization data ========
		if(WORLD.rank()==0){
			Opt::Model::print(std::cout,nnptes.model().get());
			std::cout<<nnptes.data_<<"\n";
		}
		MPI_Barrier(WORLD.mpic());
		
		//======== bcast the optimization data ========
		if(WORLD.rank()==0) std::cout<<"bcasting optimization data\n";
		thread::bcast(WORLD.mpic(),0,nnptes.data_);
		if(WORLD.rank()==0) std::cout<<"bcasting optimization model\n";
		switch(nnptes.data_.algo()){
			case Opt::Algo::SGD:
				if(WORLD.rank()!=0) nnptes.model().reset(new Opt::SGD());
				thread::bcast(WORLD.mpic(),0,static_cast<Opt::SGD&>(*nnptes.model()));
			break;
			case Opt::Algo::SDM:
				if(WORLD.rank()!=0) nnptes.model().reset(new Opt::SDM());
				thread::bcast(WORLD.mpic(),0,static_cast<Opt::SDM&>(*nnptes.model()));
			break;
			case Opt::Algo::NAG:
				if(WORLD.rank()!=0) nnptes.model().reset(new Opt::NAG());
				thread::bcast(WORLD.mpic(),0,static_cast<Opt::NAG&>(*nnptes.model()));
			break;
			case Opt::Algo::ADAGRAD:
				if(WORLD.rank()!=0) nnptes.model().reset(new Opt::ADAGRAD());
				thread::bcast(WORLD.mpic(),0,static_cast<Opt::ADAGRAD&>(*nnptes.model()));
			break;
			case Opt::Algo::ADADELTA:
				if(WORLD.rank()!=0) nnptes.model().reset(new Opt::ADADELTA());
				thread::bcast(WORLD.mpic(),0,static_cast<Opt::ADADELTA&>(*nnptes.model()));
			break;
			case Opt::Algo::RMSPROP:
				if(WORLD.rank()!=0) nnptes.model().reset(new Opt::RMSPROP());
				thread::bcast(WORLD.mpic(),0,static_cast<Opt::RMSPROP&>(*nnptes.model()));
			break;
			case Opt::Algo::ADAM:
				if(WORLD.rank()!=0) nnptes.model().reset(new Opt::ADAM());
				thread::bcast(WORLD.mpic(),0,static_cast<Opt::ADAM&>(*nnptes.model()));
			break;
			case Opt::Algo::NADAM:
				if(WORLD.rank()!=0) nnptes.model().reset(new Opt::NADAM());
				thread::bcast(WORLD.mpic(),0,static_cast<Opt::NADAM&>(*nnptes.model()));
			break;
			case Opt::Algo::AMSGRAD:
				if(WORLD.rank()!=0) nnptes.model().reset(new Opt::AMSGRAD());
				thread::bcast(WORLD.mpic(),0,static_cast<Opt::AMSGRAD&>(*nnptes.model()));
			break;
			case Opt::Algo::BFGS:
				if(WORLD.rank()!=0) nnptes.model().reset(new Opt::BFGS());
				thread::bcast(WORLD.mpic(),0,static_cast<Opt::BFGS&>(*nnptes.model()));
			break;
			case Opt::Algo::RPROP:
				if(WORLD.rank()!=0) nnptes.model().reset(new Opt::RPROP());
				thread::bcast(WORLD.mpic(),0,static_cast<Opt::RPROP&>(*nnptes.model()));
			break;
			case Opt::Algo::CG:
				if(WORLD.rank()!=0) nnptes.model().reset(new Opt::CG());
				thread::bcast(WORLD.mpic(),0,static_cast<Opt::CG&>(*nnptes.model()));
			break;
			
		}
		
		//************************************************************************************
		// SET INPUTS
		//************************************************************************************
		
		if(nnptes.symm_){
			
			//======== initialize the symmetry functions ========
			if(NNPTES_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"initializing symmetry functions - training set\n";
			for(int i=0; i<dist_train.size(); ++i) NNP::init(nnptes.nnp_,struc_train[i]);
			if(NNPTES_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"initializing symmetry functions - validation set\n";
			for(int i=0; i<dist_val.size(); ++i) NNP::init(nnptes.nnp_,struc_val[i]);
			if(NNPTES_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"initializing symmetry functions - test set\n";
			for(int i=0; i<dist_test.size(); ++i) NNP::init(nnptes.nnp_,struc_test[i]);
			
			//======== compute the symmetry functions ========
			//==== training ====
			clock.begin();
			if(dist_train.size()>0){
				if(NNPTES_PRINT_STATUS>-1) std::cout<<"setting the inputs (symmetry functions) - training - color "<<BATCH.color()<<" rank "<<BATCH.rank()<<"\n";
				//compute symmetry functions
				for(int n=BATCH.rank(); n<dist_train.size(); n+=BATCH.size()){
					if(NNPTES_PRINT_STATUS>0) std::cout<<"structure-train["<<n<<"]\n";
					NeighborList nlist(struc_train[n],nnptes.nnp_.rc());
					NNP::symm(nnptes.nnp_,struc_train[n],nlist);
				}
				MPI_Barrier(BATCH.mpic());
				//bcast symmetry functions
				for(int i=0; i<BATCH.size(); ++i){
					const int root=i;
					for(int n=root; n<dist_train.size(); n+=BATCH.size()){
						thread::bcast(BATCH.mpic(),root,struc_train[n]);
					}
				}
				MPI_Barrier(BATCH.mpic());
			}
			clock.end();
			time_symm_train=clock.duration();
			//==== validation ====
			clock.begin();
			if(dist_val.size()>0){
				if(NNPTES_PRINT_STATUS>-1) std::cout<<"setting the inputs (symmetry functions) - validation - color "<<BATCH.color()<<" rank "<<BATCH.rank()<<"\n";
				//compute symmetry functions
				for(int n=BATCH.rank(); n<dist_val.size(); n+=BATCH.size()){
					if(NNPTES_PRINT_STATUS>0) std::cout<<"structure-val["<<n<<"]\n";
					NeighborList nlist(struc_val[n],nnptes.nnp_.rc());
					NNP::symm(nnptes.nnp_,struc_val[n],nlist);
				}
				MPI_Barrier(BATCH.mpic());
				//bcast symmetry functions
				for(int i=0; i<BATCH.size(); ++i){
					const int root=i;
					for(int n=root; n<dist_val.size(); n+=BATCH.size()){
						thread::bcast(BATCH.mpic(),root,struc_val[n]);
					}
				}
				MPI_Barrier(BATCH.mpic());
			}
			clock.end();
			time_symm_val=clock.duration();
			//==== testing ====
			clock.begin();
			if(dist_test.size()>0){
				if(NNPTES_PRINT_STATUS>-1) std::cout<<"setting the inputs (symmetry functions) - testing - color "<<BATCH.color()<<" rank "<<BATCH.rank()<<"\n";
				//compute symmetry functions
				for(int n=BATCH.rank(); n<dist_test.size(); n+=BATCH.size()){
					if(NNPTES_PRINT_STATUS>0) std::cout<<"structure-test["<<n<<"]\n";
					NeighborList nlist(struc_test[n],nnptes.nnp_.rc());
					NNP::symm(nnptes.nnp_,struc_test[n],nlist);
				}
				MPI_Barrier(BATCH.mpic());
				for(int i=0; i<BATCH.size(); ++i){
					const int root=i;
					for(int n=root; n<dist_test.size(); n+=BATCH.size()){
						thread::bcast(BATCH.mpic(),root,struc_test[n]);
					}
				}
				MPI_Barrier(BATCH.mpic());
			}
			clock.end();
			time_symm_test=clock.duration();
			MPI_Barrier(WORLD.mpic());
			
			//======== write the inputs (symmetry functions) ========
			if(NNPTES_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"writing symmetry function inputs\n";
			if(write_symm){
				if(BATCH.rank()==0){
					// structures - training
					for(int j=0; j<dist_train.size(); ++j){
						std::string file_t=files_train[indices_train[dist_train.index(j)]];
						std::string filename=file_t.substr(0,file_t.find_last_of('.'));
						filename=filename+".dat";
						Structure::write_binary(struc_train[j],filename.c_str());
					}
					// structures - validation
					for(int j=0; j<dist_val.size(); ++j){
						std::string file_v=files_val[indices_val[dist_val.index(j)]];
						std::string filename=file_v.substr(0,file_v.find_last_of('.'));
						filename=filename+".dat";
						Structure::write_binary(struc_val[j],filename.c_str());
						
					}
					// structures - testing
					for(int j=0; j<dist_test.size(); ++j){
						std::string file_t=files_test[indices_test[dist_test.index(j)]];
						std::string filename=file_t.substr(0,file_t.find_last_of('.'));
						filename=filename+".dat";
						Structure::write_binary(struc_test[j],filename.c_str());
					}
				}
			}
		}
		
		//======== print the memory ========
		{
			//compute memory
			int mem_train_l=0;
			for(int i=0; i<dist_train.size(); ++i){
				mem_train_l+=serialize::nbytes(struc_train[i]);
			}
			int mem_val_l=0;
			for(int i=0; i<dist_val.size(); ++i){
				mem_val_l+=serialize::nbytes(struc_val[i]);
			}
			int mem_test_l=0;
			for(int i=0; i<dist_test.size(); ++i){
				mem_test_l+=serialize::nbytes(struc_test[i]);
			}
			//allocate arrays
			int* mem_train=new int[WORLD.size()];
			int* mem_val=new int[WORLD.size()];
			int* mem_test=new int[WORLD.size()];
			//gather memory
			MPI_Gather(&mem_train_l,1,MPI_INT,mem_train,1,MPI_INT,0,WORLD.mpic());
			MPI_Gather(&mem_val_l,1,MPI_INT,mem_val,1,MPI_INT,0,WORLD.mpic());
			MPI_Gather(&mem_test_l,1,MPI_INT,mem_test,1,MPI_INT,0,WORLD.mpic());
			//compute total
			double mem_train_t=0;
			double mem_val_t=0;
			double mem_test_t=0;
			for(int i=0; i<WORLD.size(); ++i) mem_train_t+=mem_train[i];
			for(int i=0; i<WORLD.size(); ++i) mem_val_t+=mem_val[i];
			for(int i=0; i<WORLD.size(); ++i) mem_test_t+=mem_test[i];
			//print
			if(WORLD.rank()==0){
				std::cout<<print::buf(strbuf)<<"\n";
				std::cout<<print::title("MEMORY",strbuf)<<"\n";
				std::cout<<"memory unit - MB\n";
				std::cout<<"mem - train - loc = "; for(int i=0; i<WORLD.size(); ++i) std::cout<<(1.0*mem_train[i])/1e6<<" "; std::cout<<"\n";
				std::cout<<"mem - val   - loc = "; for(int i=0; i<WORLD.size(); ++i) std::cout<<(1.0*mem_val[i])/1e6<<" "; std::cout<<"\n";
				std::cout<<"mem - test  - loc = "; for(int i=0; i<WORLD.size(); ++i) std::cout<<(1.0*mem_test[i])/1e6<<" "; std::cout<<"\n";
				std::cout<<"mem - train - tot = "<<(1.0*mem_train_t)/1e6<<"\n";
				std::cout<<"mem - val   - tot = "<<(1.0*mem_val_t)/1e6<<"\n";
				std::cout<<"mem - test  - tot = "<<(1.0*mem_test_t)/1e6<<"\n";
				std::cout<<print::title("MEMORY",strbuf)<<"\n";
				std::cout<<print::buf(strbuf)<<"\n";
			}
			//free arrays
			delete[] mem_train;
			delete[] mem_val;
			delete[] mem_test;
		}
		
		//************************************************************************************
		// TRAINING
		//************************************************************************************
		
		//======== subtract ground-state energies ========
		if(NNPTES_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"subtracting ground-state energies\n";
		for(int i=0; i<dist_train.size(); ++i){
			for(int n=0; n<struc_train[i].nAtoms(); ++n){
				struc_train[i].energy()-=nnptes.nnp().nnh(struc_train[i].type(n)).type().energy().val();
			}
		}
		for(int i=0; i<dist_val.size(); ++i){
			for(int n=0; n<struc_val[i].nAtoms(); ++n){
				struc_val[i].energy()-=nnptes.nnp().nnh(struc_val[i].type(n)).type().energy().val();
			}
		}
		for(int i=0; i<dist_test.size(); ++i){
			for(int n=0; n<struc_test[i].nAtoms(); ++n){
				struc_test[i].energy()-=nnptes.nnp().nnh(struc_test[i].type(n)).type().energy().val();
			}
		}
		
		//======== train the nn potential ========
		if(mode==Mode::TRAIN){
			if(NNPTES_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"training the nn potential\n";
			nnptes.train(dist_batch.size(),struc_train,struc_val);
		}
		MPI_Barrier(WORLD.mpic());
		
		//======== add ground-state energies ========
		if(NNPTES_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"adding ground-state energies\n";
		for(int i=0; i<dist_train.size(); ++i){
			for(int n=0; n<struc_train[i].nAtoms(); ++n){
				struc_train[i].energy()+=nnptes.nnp().nnh(struc_train[i].type(n)).type().energy().val();
			}
		}
		for(int i=0; i<dist_val.size(); ++i){
			for(int n=0; n<struc_val[i].nAtoms(); ++n){
				struc_val[i].energy()+=nnptes.nnp().nnh(struc_val[i].type(n)).type().energy().val();
			}
		}
		for(int i=0; i<dist_test.size(); ++i){
			for(int n=0; n<struc_test[i].nAtoms(); ++n){
				struc_test[i].energy()+=nnptes.nnp().nnh(struc_test[i].type(n)).type().energy().val();
			}
		}
		
		//************************************************************************************
		// EVALUATION
		//************************************************************************************
		
		//======== statistical data - energies/forces/errors ========
		//data - train
			Reduce<1> r1_energy_train;
			Reduce<2> r2_energy_train;
			Reduce<1> r1_force_train;
			std::vector<Reduce<2> > r2_force_train(3);
			std::vector<Reduce<1> > r1_charge_train;
			std::vector<Reduce<1> > r1_chi_train;
		//data - val
			Reduce<1> r1_energy_val;
			Reduce<2> r2_energy_val;
			Reduce<1> r1_force_val;
			std::vector<Reduce<2> > r2_force_val(3);
			std::vector<Reduce<1> > r1_charge_val;
			std::vector<Reduce<1> > r1_chi_val;
		//data - test
			Reduce<1> r1_energy_test;
			Reduce<2> r2_energy_test;
			Reduce<1> r1_force_test;
			std::vector<Reduce<2> > r2_force_test(3);
			std::vector<Reduce<1> > r1_charge_test;
			std::vector<Reduce<1> > r1_chi_test;
		
		//======== compute the final energies ========
		if(NNPTES_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"computing final energies\n";
		//==== train ====
		if(dist_train.size()>0){
			std::vector<double> energy_n(nTrain,std::numeric_limits<double>::max());
			std::vector<double> energy_r(nTrain,std::numeric_limits<double>::max());
			std::vector<double> energy_n_t(nTrain,std::numeric_limits<double>::max());
			std::vector<double> energy_r_t(nTrain,std::numeric_limits<double>::max());
			std::vector<int> natoms(nTrain,0); std::vector<int> natoms_t(nTrain,0);
			//compute energies
			clock.begin();
			for(int n=0; n<dist_train.size(); ++n){
				if(NNPTES_PRINT_STATUS>0) std::cout<<"structure-train["<<WORLD.rank()<<"]["<<n<<"]\n";
				energy_r[dist_train.index(n)]=struc_train[n].energy();
				const double energyS=0.0;
				const double energyV=NNP::energy(nnptes.nnp_,struc_train[n]);
				energy_n[dist_train.index(n)]=energyS+energyV;
				natoms[dist_train.index(n)]=struc_train[n].nAtoms();
			}
			clock.end();
			time_energy_train=clock.duration();
			MPI_Reduce(energy_r.data(),energy_r_t.data(),nTrain,MPI_DOUBLE,MPI_MIN,0,WORLD.mpic());
			MPI_Reduce(energy_n.data(),energy_n_t.data(),nTrain,MPI_DOUBLE,MPI_MIN,0,WORLD.mpic());
			MPI_Reduce(natoms.data(),natoms_t.data(),nTrain,MPI_INT,MPI_MAX,0,WORLD.mpic());
			//accumulate statistics
			for(int n=0; n<nTrain; ++n){
				r1_energy_train.push(std::fabs(energy_r_t[n]-energy_n_t[n])/natoms_t[n]);
				r2_energy_train.push(energy_r_t[n]/natoms_t[n],energy_n_t[n]/natoms_t[n]);
			}
			//normalize
			if(nnptes.norm()){
				for(int n=0; n<nTrain; ++n) energy_r_t[n]/=natoms_t[n];
				for(int n=0; n<nTrain; ++n) energy_n_t[n]/=natoms_t[n];
			}
			//write energies
			if(write_energy && WORLD.rank()==0){
				const char* file="nnp_energy_train.dat";
				FILE* writer=fopen(file,"w");
				if(writer==NULL) std::cout<<"WARNING: Could not open file: \""<<file<<"\"\n";
				else{
					std::vector<std::pair<int,double> > energy_r_pair(nTrain);
					std::vector<std::pair<int,double> > energy_n_pair(nTrain);
					for(int n=0; n<nTrain; ++n){
						energy_r_pair[n].first=indices_train[n];
						energy_r_pair[n].second=energy_r_t[n];
						energy_n_pair[n].first=indices_train[n];
						energy_n_pair[n].second=energy_n_t[n];
					}
					std::sort(energy_r_pair.begin(),energy_r_pair.end(),compare_pair);
					std::sort(energy_n_pair.begin(),energy_n_pair.end(),compare_pair);
					fprintf(writer,"#STRUCTURE ENERGY_REF ENERGY_NN\n");
					for(int n=0; n<nTrain; ++n){
						fprintf(writer,"%s %f %f\n",files_train[n].c_str(),energy_r_pair[n].second,energy_n_pair[n].second);
					}
					fclose(writer); writer=NULL;
				}
			}
		}
		//==== validation ====
		if(dist_val.size()>0){
			std::vector<double> energy_n(nVal,std::numeric_limits<double>::max());
			std::vector<double> energy_r(nVal,std::numeric_limits<double>::max());
			std::vector<double> energy_n_t(nVal,std::numeric_limits<double>::max());
			std::vector<double> energy_r_t(nVal,std::numeric_limits<double>::max());
			std::vector<int> natoms(nVal,0); std::vector<int> natoms_t(nVal,0);
			//compute energies
			clock.begin();
			for(int n=0; n<dist_val.size(); ++n){
				if(NNPTES_PRINT_STATUS>0) std::cout<<"structure-val["<<WORLD.rank()<<"]["<<n<<"]\n";
				energy_r[dist_val.index(n)]=struc_val[n].energy();
				const double energyS=0.0;
				const double energyV=NNP::energy(nnptes.nnp_,struc_val[n]);
				energy_n[dist_val.index(n)]=energyS+energyV;
				natoms[dist_val.index(n)]=struc_val[n].nAtoms();
			}
			clock.end();
			time_energy_train=clock.duration();
			MPI_Reduce(energy_r.data(),energy_r_t.data(),nVal,MPI_DOUBLE,MPI_MIN,0,WORLD.mpic());
			MPI_Reduce(energy_n.data(),energy_n_t.data(),nVal,MPI_DOUBLE,MPI_MIN,0,WORLD.mpic());
			MPI_Reduce(natoms.data(),natoms_t.data(),nVal,MPI_INT,MPI_MAX,0,WORLD.mpic());
			//accumulate statistics
			for(int n=0; n<nVal; ++n){
				r1_energy_val.push(std::fabs(energy_r_t[n]-energy_n_t[n])/natoms_t[n]);
				r2_energy_val.push(energy_r_t[n]/natoms_t[n],energy_n_t[n]/natoms_t[n]);
			}
			//normalize
			if(nnptes.norm()){
				for(int n=0; n<nVal; ++n) energy_r_t[n]/=natoms_t[n];
				for(int n=0; n<nVal; ++n) energy_n_t[n]/=natoms_t[n];
			}
			//write energies
			if(write_energy && WORLD.rank()==0){
				const char* file="nnp_energy_val.dat";
				FILE* writer=fopen(file,"w");
				if(writer==NULL) std::cout<<"WARNING: Could not open file: \""<<file<<"\"\n";
				else{
					std::vector<std::pair<int,double> > energy_r_pair(nVal);
					std::vector<std::pair<int,double> > energy_n_pair(nVal);
					for(int n=0; n<nVal; ++n){
						energy_r_pair[n].first=indices_val[n];
						energy_r_pair[n].second=energy_r_t[n];
						energy_n_pair[n].first=indices_val[n];
						energy_n_pair[n].second=energy_n_t[n];
					}
					std::sort(energy_r_pair.begin(),energy_r_pair.end(),compare_pair);
					std::sort(energy_n_pair.begin(),energy_n_pair.end(),compare_pair);
					fprintf(writer,"#STRUCTURE ENERGY_REF ENERGY_NN\n");
					for(int n=0; n<nVal; ++n){
						fprintf(writer,"%s %f %f\n",files_val[n].c_str(),energy_r_pair[n].second,energy_n_pair[n].second);
					}
					fclose(writer); writer=NULL;
				}
			}
		}
		//==== test ====
		if(dist_test.size()>0){
			std::vector<double> energy_n(nTest,std::numeric_limits<double>::max());
			std::vector<double> energy_r(nTest,std::numeric_limits<double>::max());
			std::vector<double> energy_n_t(nTest,std::numeric_limits<double>::max());
			std::vector<double> energy_r_t(nTest,std::numeric_limits<double>::max());
			std::vector<int> natoms(nTest,0); std::vector<int> natoms_t(nTest,0);
			//compute energies
			clock.begin();
			for(int n=0; n<dist_test.size(); ++n){
				if(NNPTES_PRINT_STATUS>0) std::cout<<"structure-test["<<WORLD.rank()<<"]["<<n<<"]\n";
				energy_r[dist_test.index(n)]=struc_test[n].energy();
				const double energyS=0.0;
				const double energyV=NNP::energy(nnptes.nnp_,struc_test[n]);
				energy_n[dist_test.index(n)]=energyS+energyV;
				natoms[dist_test.index(n)]=struc_test[n].nAtoms();
			}
			clock.end();
			time_energy_test=clock.duration();
			MPI_Reduce(energy_r.data(),energy_r_t.data(),nTest,MPI_DOUBLE,MPI_MIN,0,WORLD.mpic());
			MPI_Reduce(energy_n.data(),energy_n_t.data(),nTest,MPI_DOUBLE,MPI_MIN,0,WORLD.mpic());
			MPI_Reduce(natoms.data(),natoms_t.data(),nTest,MPI_INT,MPI_MAX,0,WORLD.mpic());
			//accumulate statistics
			for(int n=0; n<nTest; ++n){
				r1_energy_test.push(std::fabs(energy_r_t[n]-energy_n_t[n])/natoms_t[n]);
				r2_energy_test.push(energy_r_t[n]/natoms_t[n],energy_n_t[n]/natoms_t[n]);
			}
			//normalize
			if(nnptes.norm()){
				for(int n=0; n<nTest; ++n) energy_r_t[n]/=natoms_t[n];
				for(int n=0; n<nTest; ++n) energy_n_t[n]/=natoms_t[n];
			}
			//write energies
			if(write_energy && WORLD.rank()==0){
				const char* file="nnp_energy_test.dat";
				FILE* writer=fopen(file,"w");
				if(writer==NULL) std::cout<<"WARNING: Could not open file: \""<<file<<"\"\n";
				else{
					std::vector<std::pair<int,double> > energy_r_pair(nTest);
					std::vector<std::pair<int,double> > energy_n_pair(nTest);
					for(int n=0; n<nTest; ++n){
						energy_r_pair[n].first=indices_test[n];
						energy_r_pair[n].second=energy_r_t[n];
						energy_n_pair[n].first=indices_test[n];
						energy_n_pair[n].second=energy_n_t[n];
					}
					std::sort(energy_r_pair.begin(),energy_r_pair.end(),compare_pair);
					std::sort(energy_n_pair.begin(),energy_n_pair.end(),compare_pair);
					fprintf(writer,"#STRUCTURE ENERGY_REF ENERGY_NN\n");
					for(int n=0; n<nTest; ++n){
						fprintf(writer,"%s %f %f\n",files_test[n].c_str(),energy_r_pair[n].second,energy_n_pair[n].second);
					}
					fclose(writer); writer=NULL;
				}
			}
		}
		
		//======== compute the final forces ========
		if(NNPTES_PRINT_STATUS>-1 && WORLD.rank()==0 && nnptes.force_) std::cout<<"computing final forces\n";
		//==== training structures ====
		if(dist_train.size()>0 && nnptes.force_){
			//compute forces
			clock.begin();
			for(int n=0; n<dist_train.size(); ++n){
				if(NNPTES_PRINT_STATUS>0) std::cout<<"structure-train["<<n<<"]\n";
				Structure& struc=struc_train[n];
				//compute exact forces
				std::vector<Eigen::Vector3d> f_r(struc.nAtoms());
				for(int i=0; i<struc.nAtoms(); ++i) f_r[i]=struc.force(i);
				//compute nn forces
				NeighborList nlist(struc,nnptes.nnp_.rc());
				//NNP::force(nnptes.nnp_,struc,nlist);
				std::vector<Eigen::Vector3d> f_n(struc.nAtoms());
				for(int i=0; i<struc.nAtoms(); ++i) f_n[i]=struc.force(i);
				//compute statistics
				if(BATCH.rank()==0){
					for(int i=0; i<struc.nAtoms(); ++i){
						r1_force_train.push((f_r[i]-f_n[i]).norm());
						r2_force_train[0].push(f_r[i][0],f_n[i][0]);
						r2_force_train[1].push(f_r[i][1],f_n[i][1]);
						r2_force_train[2].push(f_r[i][2],f_n[i][2]);
					}
				}
			}
			clock.end();
			time_force_train=clock.duration();
			//accumulate statistics
			std::vector<Reduce<1> > r1fv(WORLD.size());
			thread::gather(r1_force_train,r1fv,WORLD.mpic());
			if(WORLD.rank()==0) for(int i=1; i<WORLD.size(); ++i) r1_force_train+=r1fv[i];
			for(int n=0; n<3; ++n){
				std::vector<Reduce<2> > r2fv(WORLD.size());
				thread::gather(r2_force_train[n],r2fv,WORLD.mpic());
				if(WORLD.rank()==0) for(int i=1; i<WORLD.size(); ++i) r2_force_train[n]+=r2fv[i];
			}
		}
		//==== validation structures ====
		if(dist_val.size()>0 && nnptes.force_){
			//compute forces
			clock.begin();
			for(int n=0; n<dist_val.size(); ++n){
				if(NNPTES_PRINT_STATUS>0) std::cout<<"structure-val["<<n<<"]\n";
				Structure& struc=struc_val[n];
				//compute exact forces
				std::vector<Eigen::Vector3d> f_r(struc.nAtoms());
				for(int i=0; i<struc.nAtoms(); ++i) f_r[i]=struc.force(i);
				//compute nn forces
				NeighborList nlist(struc,nnptes.nnp_.rc());
				//NNP::force(nnptes.nnp_,struc,nlist);
				std::vector<Eigen::Vector3d> f_n(struc.nAtoms());
				for(int i=0; i<struc.nAtoms(); ++i) f_n[i]=struc.force(i);
				//compute statistics
				if(BATCH.rank()==0){
					for(int i=0; i<struc.nAtoms(); ++i){
						r1_force_val.push((f_r[i]-f_n[i]).norm());
						r2_force_val[0].push(f_r[i][0],f_n[i][0]);
						r2_force_val[1].push(f_r[i][1],f_n[i][1]);
						r2_force_val[2].push(f_r[i][2],f_n[i][2]);
					}
				}
			}
			clock.end();
			time_force_val=clock.duration();
			//accumulate statistics
			std::vector<Reduce<1> > r1fv(WORLD.size());
			thread::gather(r1_force_val,r1fv,WORLD.mpic());
			if(WORLD.rank()==0) for(int i=1; i<WORLD.size(); ++i) r1_force_val+=r1fv[i];
			for(int n=0; n<3; ++n){
				std::vector<Reduce<2> > r2fv(WORLD.size());
				thread::gather(r2_force_val[n],r2fv,WORLD.mpic());
				if(WORLD.rank()==0) for(int i=1; i<WORLD.size(); ++i) r2_force_val[n]+=r2fv[i];
			}
		}
		//==== test structures ====
		if(dist_test.size()>0 && nnptes.force_){
			//compute forces
			clock.begin();
			for(int n=0; n<dist_test.size(); ++n){
				if(NNPTES_PRINT_STATUS>0) std::cout<<"structure-test["<<n<<"]\n";
				Structure& struc=struc_test[n];
				//compute exact forces
				std::vector<Eigen::Vector3d> f_r(struc.nAtoms());
				for(int i=0; i<struc.nAtoms(); ++i) f_r[i]=struc.force(i);
				//compute nn forces
				NeighborList nlist(struc,nnptes.nnp_.rc());
				//NNP::force(nnptes.nnp_,struc,nlist);
				std::vector<Eigen::Vector3d> f_n(struc.nAtoms());
				for(int i=0; i<struc.nAtoms(); ++i) f_n[i]=struc.force(i);
				//compute statistics
				if(BATCH.rank()==0){
					for(int i=0; i<struc.nAtoms(); ++i){
						r1_force_test.push((f_r[i]-f_n[i]).norm());
						r2_force_test[0].push(f_r[i][0],f_n[i][0]);
						r2_force_test[1].push(f_r[i][1],f_n[i][1]);
						r2_force_test[2].push(f_r[i][2],f_n[i][2]);
					}
				}
			}
			clock.end();
			time_force_test=clock.duration();
			//accumulate statistics
			std::vector<Reduce<1> > r1fv(WORLD.size());
			thread::gather(r1_force_test,r1fv,WORLD.mpic());
			if(WORLD.rank()==0) for(int i=1; i<WORLD.size(); ++i) r1_force_test+=r1fv[i];
			for(int n=0; n<3; ++n){
				std::vector<Reduce<2> > r2fv(WORLD.size());
				thread::gather(r2_force_test[n],r2fv,WORLD.mpic());
				if(WORLD.rank()==0) for(int i=1; i<WORLD.size(); ++i) r2_force_test[n]+=r2fv[i];
			}
		}
		
		//======== write the inputs ========
		//==== training structures ====
		if(dist_train.size()>0 && write_input){
			const char* file_inputs_train="nnp_inputs_train.dat";
			for(int ii=0; ii<WORLD.size(); ++ii){
				if(WORLD.rank()==ii){
					FILE* writer=NULL;
					if(ii==0) writer=fopen(file_inputs_train,"w");
					else writer=fopen(file_inputs_train,"a");
					if(writer!=NULL){
						for(int n=0; n<dist_train.size(); ++n){
							for(int i=0; i<struc_train[n].nAtoms(); ++i){
								fprintf(writer,"%s%i ",struc_train[n].name(i).c_str(),i);
								for(int j=0; j<struc_train[n].symm(i).size(); ++j){
									fprintf(writer,"%f ",struc_train[n].symm(i)[j]);
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
		//==== validation structures ====
		if(dist_val.size()>0 && write_input){
			const char* file_inputs_val="nnp_inputs_val.dat";
			for(int ii=0; ii<WORLD.size(); ++ii){
				if(WORLD.rank()==ii){
					FILE* writer=NULL;
					if(ii==0) writer=fopen(file_inputs_val,"w");
					else writer=fopen(file_inputs_val,"a");
					if(writer!=NULL){
						for(int n=0; n<dist_val.size(); ++n){
							for(int i=0; i<struc_val[n].nAtoms(); ++i){
								fprintf(writer,"%s%i ",struc_val[n].name(i).c_str(),i);
								for(int j=0; j<struc_val[n].symm(i).size(); ++j){
									fprintf(writer,"%f ",struc_val[n].symm(i)[j]);
								}
								fprintf(writer,"\n");
							}
						}
						fclose(writer); writer=NULL;
					} else std::cout<<"WARNING: Could not open inputs file for validation structures\n";
				}
				MPI_Barrier(WORLD.mpic());
			}
		}
		//==== testing structures ====
		if(dist_test.size()>0 && write_input){
			const char* file_inputs_test="nnp_inputs_test.dat";
			for(int ii=0; ii<WORLD.size(); ++ii){
				if(WORLD.rank()==ii){
					FILE* writer=NULL;
					if(ii==0) writer=fopen(file_inputs_test,"w");
					else writer=fopen(file_inputs_test,"a");
					if(writer!=NULL){
						for(int n=0; n<dist_test.size(); ++n){
							for(int i=0; i<struc_test[n].nAtoms(); ++i){
								fprintf(writer,"%s%i ",struc_test[n].name(i).c_str(),i);
								for(int j=0; j<struc_test[n].symm(i).size(); ++j){
									fprintf(writer,"%f ",struc_test[n].symm(i)[j]);
								}
								fprintf(writer,"\n");
							}
						}
						fclose(writer); writer=NULL;
					} else std::cout<<"WARNING: Could not open inputs file for testing structures\n";
				}
				MPI_Barrier(WORLD.mpic());
			}
		}
		
		//======== stop the wall clock ========
		if(WORLD.rank()==0) clock_wall.end();
		if(WORLD.rank()==0) time_wall=clock_wall.duration();
		
		//************************************************************************************
		// OUTPUT
		//************************************************************************************
		
		//======== print the timing info ========
		{
			double tmp;
			MPI_Reduce(&time_symm_train,&tmp,1,MPI_DOUBLE,MPI_SUM,0,WORLD.mpic()); if(WORLD.rank()==0) time_symm_train=tmp/WORLD.size();
			MPI_Reduce(&time_energy_train,&tmp,1,MPI_DOUBLE,MPI_SUM,0,WORLD.mpic()); if(WORLD.rank()==0) time_energy_train=tmp/WORLD.size();
			MPI_Reduce(&time_force_train,&tmp,1,MPI_DOUBLE,MPI_SUM,0,WORLD.mpic()); if(WORLD.rank()==0) time_force_train=tmp/WORLD.size();
			MPI_Reduce(&time_symm_val,&tmp,1,MPI_DOUBLE,MPI_SUM,0,WORLD.mpic()); if(WORLD.rank()==0) time_symm_val=tmp/WORLD.size();
			MPI_Reduce(&time_energy_val,&tmp,1,MPI_DOUBLE,MPI_SUM,0,WORLD.mpic()); if(WORLD.rank()==0) time_energy_val=tmp/WORLD.size();
			MPI_Reduce(&time_force_val,&tmp,1,MPI_DOUBLE,MPI_SUM,0,WORLD.mpic()); if(WORLD.rank()==0) time_force_val=tmp/WORLD.size();
			MPI_Reduce(&time_symm_test,&tmp,1,MPI_DOUBLE,MPI_SUM,0,WORLD.mpic()); if(WORLD.rank()==0) time_symm_test=tmp/WORLD.size();
			MPI_Reduce(&time_energy_test,&tmp,1,MPI_DOUBLE,MPI_SUM,0,WORLD.mpic()); if(WORLD.rank()==0) time_energy_test=tmp/WORLD.size();
			MPI_Reduce(&time_force_test,&tmp,1,MPI_DOUBLE,MPI_SUM,0,WORLD.mpic()); if(WORLD.rank()==0) time_force_test=tmp/WORLD.size();
		}
		if(WORLD.rank()==0){
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("TIMING (S)",strbuf)<<"\n";
		if(struc_train.size()>0){
		std::cout<<"time - symm   - train = "<<time_symm_train<<"\n";
		std::cout<<"time - energy - train = "<<time_energy_train<<"\n";
		std::cout<<"time - force  - train = "<<time_force_train<<"\n";
		}
		if(struc_val.size()>0){
		std::cout<<"time - symm   - val   = "<<time_symm_val<<"\n";
		std::cout<<"time - energy - val   = "<<time_energy_val<<"\n";
		std::cout<<"time - force  - val   = "<<time_force_val<<"\n";
		}
		if(struc_test.size()>0){
		std::cout<<"time - symm   - test  = "<<time_symm_test<<"\n";
		std::cout<<"time - energy - test  = "<<time_energy_test<<"\n";
		std::cout<<"time - force  - test  = "<<time_force_test<<"\n";
		}
		std::cout<<"time - wall           = "<<time_wall<<"\n";
		std::cout<<print::title("TIMING (S)",strbuf)<<"\n";
		std::cout<<print::buf(strbuf)<<"\n";
		}
		
		//======== print the error statistics - training ========
		if(WORLD.rank()==0 && mode==Mode::TRAIN){
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("STATISTICS - ERROR - TRAINING",strbuf)<<"\n";
		std::cout<<"\tERROR - AVG - ENERGY/ATOM = "<<r1_energy_train.avg()<<"\n";
		std::cout<<"\tERROR - DEV - ENERGY/ATOM = "<<r1_energy_train.dev()<<"\n";
		std::cout<<"\tERROR - MAX - ENERGY/ATOM = "<<r1_energy_train.max()<<"\n";
		std::cout<<"\tM/R2 - ENERGY/ATOM = "<<r2_energy_train.m()<<" "<<r2_energy_train.r2()<<"\n";
		if(nnptes.force_){
		std::cout<<"FORCE:\n";
		std::cout<<"\tERROR - AVG - FORCE = "<<r1_force_train.avg()<<"\n";
		std::cout<<"\tERROR - DEV - FORCE = "<<r1_force_train.dev()<<"\n";
		std::cout<<"\tERROR - MAX - FORCE = "<<r1_force_train.max()<<"\n";
		std::cout<<"\tM  (FX,FY,FZ) = "<<r2_force_train[0].m() <<" "<<r2_force_train[1].m() <<" "<<r2_force_train[2].m() <<"\n";
		std::cout<<"\tR2 (FX,FY,FZ) = "<<r2_force_train[0].r2()<<" "<<r2_force_train[1].r2()<<" "<<r2_force_train[2].r2()<<"\n";
		}
		std::cout<<print::title("STATISTICS - ERROR - TRAINING",strbuf)<<"\n";
		std::cout<<print::buf(strbuf)<<"\n";
		}
		
		//======== print the error statistics - validation ========
		if(WORLD.rank()==0 && dist_val.size()>0 && mode==Mode::TRAIN){
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("STATISTICS - ERROR - VALIDATION",strbuf)<<"\n";
		std::cout<<"\tERROR - AVG - ENERGY/ATOM = "<<r1_energy_val.avg()<<"\n";
		std::cout<<"\tERROR - DEV - ENERGY/ATOM = "<<r1_energy_val.dev()<<"\n";
		std::cout<<"\tERROR - MAX - ENERGY/ATOM = "<<r1_energy_val.max()<<"\n";
		std::cout<<"\tM/R2 - ENERGY/ATOM = "<<r2_energy_val.m()<<" "<<r2_energy_val.r2()<<"\n";
		if(nnptes.force_){
		std::cout<<"FORCE:\n";
		std::cout<<"\tERROR - AVG - FORCE = "<<r1_force_val.avg()<<"\n";
		std::cout<<"\tERROR - DEV - FORCE = "<<r1_force_val.dev()<<"\n";
		std::cout<<"\tERROR - MAX - FORCE = "<<r1_force_val.max()<<"\n";
		std::cout<<"\tM  (FX,FY,FZ) = "<<r2_force_val[0].m() <<" "<<r2_force_val[1].m() <<" "<<r2_force_val[2].m() <<"\n";
		std::cout<<"\tR2 (FX,FY,FZ) = "<<r2_force_val[0].r2()<<" "<<r2_force_val[1].r2()<<" "<<r2_force_val[2].r2()<<"\n";
		}
		std::cout<<print::title("STATISTICS - ERROR - VALIDATION",strbuf)<<"\n";
		std::cout<<print::buf(strbuf)<<"\n";
		}
		
		//======== print the error statistics - test ========
		if(WORLD.rank()==0 && dist_test.size()>0){
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("STATISTICS - ERROR - TEST",strbuf)<<"\n";
		std::cout<<"ENERGY:\n";
		std::cout<<"\tERROR - AVG - ENERGY/ATOM = "<<r1_energy_test.avg()<<"\n";
		std::cout<<"\tERROR - DEV - ENERGY/ATOM = "<<r1_energy_test.dev()<<"\n";
		std::cout<<"\tERROR - MAX - ENERGY/ATOM = "<<r1_energy_test.max()<<"\n";
		std::cout<<"\tM/R2 - ENERGY/ATOM = "<<r2_energy_test.m()<<" "<<r2_energy_test.r2()<<"\n";
		if(nnptes.force_){
		std::cout<<"FORCE:\n";
		std::cout<<"\tERROR - AVG - FORCE = "<<r1_force_test.avg()<<"\n";
		std::cout<<"\tERROR - DEV - FORCE = "<<r1_force_test.dev()<<"\n";
		std::cout<<"\tERROR - MAX - FORCE = "<<r1_force_test.max()<<"\n";
		std::cout<<"\tM  (FX,FY,FZ) = "<<r2_force_test[0].m() <<" "<<r2_force_test[1].m() <<" "<<r2_force_test[2].m() <<"\n";
		std::cout<<"\tR2 (FX,FY,FZ) = "<<r2_force_test[0].r2()<<" "<<r2_force_test[1].r2()<<" "<<r2_force_test[2].r2()<<"\n";
		}
		std::cout<<print::title("STATISTICS - ERROR - TEST",strbuf)<<"\n";
		std::cout<<print::buf(strbuf)<<"\n";
		}
		
		//======== write the nn's ========
		if(NNPTES_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"writing the nn's\n";
		if(WORLD.rank()==0){
			NNP::write(nnptes.file_ann_.c_str(),nnptes.nnp_);
		}
		//======== write restart file ========
		if(NNPTES_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"writing restart file\n";
		if(WORLD.rank()==0){
			nnptes.write_restart(nnptes.file_restart_.c_str());
		}
		
		//======== finalize mpi ========
		if(NNPTES_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"finalizing mpi\n";
		std::cout<<std::flush;
		MPI_Comm_free(&BATCH.mpic());
		MPI_Barrier(WORLD.mpic());
		MPI_Finalize();
	}catch(std::exception& e){
		std::cout<<"ERROR in nnptes::main(int,char**):\n";
		std::cout<<e.what()<<"\n";
	}
	
	//======== free local variables ========
	delete[] paramfile;
	delete[] input;
	delete[] strbuf;
	
	return 0;
}
