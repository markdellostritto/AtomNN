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
// ann - structure
#include "src/struc/structure.hpp"
#include "src/struc/neighbor.hpp"
// ann - format
#include "src/format/file.hpp"
#include "src/format/format.hpp"
#include "src/format/vasp_struc.hpp"
#include "src/format/qe_struc.hpp"
#include "src/format/xyz_struc.hpp"
#include "src/format/cp2k_struc.hpp"
// ann - math
#include "src/math/accumulator.hpp"
#include "src/math/func.hpp"
// ann - string
#include "src/str/string.hpp"
#include "src/str/print.hpp"
// ann - chem
#include "src/chem/units.hpp"
#include "src/chem/ewald3D.hpp"
// ann - thread
#include "src/thread/parallel.hpp"
// ann - util
#include "src/util/compiler.hpp"
#include "src/util/time.hpp"
// ann - nnp - train - mpi
#include "src/nnp/nnp_train.hpp"

//#define unlikely(expr) __builtin_expect(!!(expr), 0)
//#define likely(expr) __builtin_expect(!!(expr), 1)

static bool compare_pair(const std::pair<int,double>& p1, const std::pair<int,double>& p2){
	return p1.first<p2.first;
}

//************************************************************
// MPI Communicators
//************************************************************

parallel::Comm WORLD;//all processors
parallel::Comm BATCH;//group of nproc/nBatch processors handling each element of the batch

//************************************************************
// serialization
//************************************************************

namespace serialize{

//**********************************************
// byte measures
//**********************************************

template <> int nbytes(const NNPTrain& obj){
   if(NNP_TRAIN_PRINT_FUNC>0) std::cout<<"nbytes(const NNPTrain&)\n";
	int size=0;
	//input/output
		size+=nbytes(obj.file_error_);
		size+=nbytes(obj.file_ann_);
		size+=nbytes(obj.file_restart_);
	//flags
		size+=sizeof(bool);//restart
		size+=sizeof(bool);//pre-conditioning
		size+=sizeof(bool);//force
		size+=sizeof(bool);//symm
		size+=sizeof(bool);//norm
		size+=sizeof(bool);//charge
	//nnp
		size+=nbytes(obj.nnp_);
	//optimization
		size+=sizeof(Opt::Loss);
		size+=sizeof(obj.w_);
		size+=sizeof(obj.w2_);
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

template <> int pack(const NNPTrain& obj, char* arr){
	if(NNP_TRAIN_PRINT_FUNC>0) std::cout<<"pack(const NNPTrain&,char*)\n";
	int pos=0;
	//input/output
		pos+=pack(obj.file_error_,arr+pos);
		pos+=pack(obj.file_ann_,arr+pos);
		pos+=pack(obj.file_restart_,arr+pos);
	//flags
		std::memcpy(arr+pos,&obj.restart_,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(arr+pos,&obj.preCond_,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(arr+pos,&obj.force_,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(arr+pos,&obj.symm_,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(arr+pos,&obj.norm_,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(arr+pos,&obj.charge_,sizeof(bool)); pos+=sizeof(bool);
	//nnp
		pos+=pack(obj.nnp_,arr+pos);
	//optimization
		std::memcpy(arr+pos,&obj.loss_,sizeof(Opt::Loss)); pos+=sizeof(Opt::Loss);
		pos+=pack(obj.w_,arr+pos);
		pos+=pack(obj.w2_,arr+pos);
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
				default: throw std::runtime_error("pack(const NNPTrain&,char*): Invalid optimization method."); break;
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

template <> int unpack(NNPTrain& obj, const char* arr){
	if(NNP_TRAIN_PRINT_FUNC>0) std::cout<<"unpack(const NNPTrain&,char*)\n";
	int pos=0;
	//input/output
		pos+=unpack(obj.file_error_,arr+pos);
		pos+=unpack(obj.file_ann_,arr+pos);
		pos+=unpack(obj.file_restart_,arr+pos);
	//flags
		std::memcpy(&obj.restart_,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(&obj.preCond_,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(&obj.force_,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(&obj.symm_,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(&obj.norm_,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(&obj.charge_,arr+pos,sizeof(bool)); pos+=sizeof(bool);
	//nnp
		pos+=unpack(obj.nnp_,arr+pos);
	//optimization
		std::memcpy(&obj.loss_,arr+pos,sizeof(Opt::Loss)); pos+=sizeof(Opt::Loss);
		pos+=unpack(obj.w_,arr+pos);
		pos+=unpack(obj.w2_,arr+pos);
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
					throw std::runtime_error("unpack(NNPTrain&,const char*): Invalid optimization method.");
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
// NNPTrain - Neural Network Potential - Optimization
//************************************************************

std::ostream& operator<<(std::ostream& out, const NNPTrain& nnp_train){
	char* str=new char[print::len_buf];
	out<<print::buf(str)<<"\n";
	out<<print::title("NNP - TRAIN",str)<<"\n";
	out<<"PRE-COND     = "<<nnp_train.preCond_<<"\n";
	out<<"RESTART      = "<<nnp_train.restart_<<"\n";
	out<<"FORCE        = "<<nnp_train.force_<<"\n";
	out<<"SYMM         = "<<nnp_train.symm_<<"\n";
	out<<"HUBERW       = "<<nnp_train.w_<<"\n";
	out<<"CHARGE       = "<<nnp_train.charge_<<"\n";
	out<<"NORM         = "<<nnp_train.norm_<<"\n";
	out<<"BATCH        = "<<nnp_train.batch_<<"\n";
	out<<"LOSS         = "<<nnp_train.loss_<<"\n";
	out<<"ERROR_S      = "<<nnp_train.error_scale_<<"\n";
	out<<"FILE_ANN     = "<<nnp_train.file_ann_<<"\n";
	out<<"FILE_ERROR   = "<<nnp_train.file_error_<<"\n";
	out<<"FILE_RESTART = "<<nnp_train.file_restart_<<"\n";
	out<<print::title("NNP - TRAIN",str)<<"\n";
	out<<print::buf(str);
	delete[] str;
	return out;
}

NNPTrain::NNPTrain(){
	if(NNP_TRAIN_PRINT_FUNC>0) std::cout<<"NNP::NNPTrain():\n";
	defaults();
};

void NNPTrain::defaults(){
	if(NNP_TRAIN_PRINT_FUNC>0) std::cout<<"NNP::defaults():\n";
	//elements
		nElements_=0;
		gElement_.clear();
		pElement_.clear();
		gTotal_.clear();
		gLocal_.clear();
	//nn
		error_scale_=1.0;
		nParams_=0;
		nnp_.clear();
		preCond_=false;
		charge_=false;
		loss_=Opt::Loss::MSE;
	//input/output
		force_=true;
		symm_=true;
		restart_=false;
		norm_=false;
		file_error_="nnp_error.dat";
		file_restart_="nnp_train.restart";
		file_ann_="ann";
	//optimization
		identity_=Eigen::VectorXd::Identity(1,1);
	//error
		error_train_=0;
		error_val_=0;	
}

void NNPTrain::clear(){
	if(NNP_TRAIN_PRINT_FUNC>0) std::cout<<"NNP::clear():\n";
	//elements
		nElements_=0;
		gElement_.clear();
		pElement_.clear();
		gTotal_.clear();
		gLocal_.clear();
	//nn
		nParams_=0;
		nnp_.clear();
	//optimization
		batch_.clear();
		data_.clear();
		identity_=Eigen::VectorXd::Identity(1,1);
	//error
		error_train_=0;
		error_val_=0;
}

void NNPTrain::write_restart(const char* file){
	if(NNP_TRAIN_PRINT_FUNC>1) std::cout<<"NNPTrain::write_restart(const char*):\n";
	//local variables
	char* arr=NULL;
	FILE* writer=NULL;
	bool error=false;
	try{
		//open file
		writer=fopen(file,"wb");
		if(writer==NULL) throw std::runtime_error(std::string("NNPTrain::write_restart(const char*): Could not open file: ")+file);
		//allocate buffer
		const int nBytes=serialize::nbytes(*this);
		arr=new char[nBytes];
		if(arr==NULL) throw std::runtime_error("NNPTrain::write_restart(const char*): Could not allocate memory.");
		//write to buffer
		serialize::pack(*this,arr);
		//write to file
		const int nWrite=fwrite(arr,sizeof(char),nBytes,writer);
		if(nWrite!=nBytes) throw std::runtime_error("NNPTrain::write_restart(const char*): Write error.");
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
	if(error) throw std::runtime_error("NNPTrain::write_restart(const char*): Failed to write");
}

void NNPTrain::read_restart(const char* file){
	if(NNP_TRAIN_PRINT_FUNC>0) std::cout<<"NNPTrain::read_restart(const char*):\n";
	//local variables
	char* arr=NULL;
	FILE* reader=NULL;
	bool error=false;
	try{
		//open file
		reader=fopen(file,"rb");
		if(reader==NULL) throw std::runtime_error(std::string("NNPTrain::read_restart(const char*): Could not open file: ")+std::string(file));
		//find size
		std::fseek(reader,0,SEEK_END);
		const int nBytes=std::ftell(reader);
		std::fseek(reader,0,SEEK_SET);
		//allocate buffer
		arr=new char[nBytes];
		if(arr==NULL) throw std::runtime_error("NNPTrain::read_restart(const char*): Could not allocate memory.");
		//read from file
		const int nRead=fread(arr,sizeof(char),nBytes,reader);
		if(nRead!=nBytes) throw std::runtime_error("NNPTrain::read_restart(const char*): Read error.");
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
	if(error) throw std::runtime_error("NNPTrain::read_restart(const char*): Failed to read");
}

void NNPTrain::train(int batchSize, const std::vector<Structure>& struc_train, const std::vector<Structure>& struc_val){
	if(NNP_TRAIN_PRINT_FUNC>0) std::cout<<"NNPTrain::train(NNP&,std::vector<Structure>&,int):\n";
	//====== local function variables ======
	//statistics
		std::vector<int> N;//total number of inputs for each element
		std::vector<Eigen::VectorXd> avg_in;//average of the inputs for each element (nnp_.nSpecies_ x nInput_)
		std::vector<Eigen::VectorXd> max_in;//max of the inputs for each element (nnp_.nSpecies_ x nInput_)
		std::vector<Eigen::VectorXd> min_in;//min of the inputs for each element (nnp_.nSpecies_ x nInput_)
		std::vector<Eigen::VectorXd> dev_in;//average of the stddev for each element (nnp_.nSpecies_ x nInput_)
	//timing
		Clock clock;
		
	if(NNP_TRAIN_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"training NN potential\n";
	
	//====== check the parameters ======
	if(batchSize<=0) throw std::invalid_argument("NNPTrain::train(int): Invalid batch size.");
	if(struc_train.size()==0) throw std::invalid_argument("NNPTrain::train(int): No training data provided.");
	if(struc_val.size()==0) throw std::invalid_argument("NNPTrain::train(int): No validation data provided.");
	
	//====== get the number of structures ======
	int nBatch=0;
	int nTrain=0;
	int nVal=0;
	{
		double nBatchL=(1.0*batchSize)/BATCH.size();
		double nTrainL=(1.0*struc_train.size())/BATCH.size();
		double nValL=(1.0*struc_val.size())/BATCH.size();
		double nBatchG=0;
		double nTrainG=0;
		double nValG=0;
		MPI_Allreduce(&nBatchL,&nBatchG,1,MPI_DOUBLE,MPI_SUM,WORLD.label());
		MPI_Allreduce(&nTrainL,&nTrainG,1,MPI_DOUBLE,MPI_SUM,WORLD.label());
		MPI_Allreduce(&nValL,&nValG,1,MPI_DOUBLE,MPI_SUM,WORLD.label());
		nBatch=std::round(nBatchG);
		nTrain=std::round(nTrainG);
		nVal=std::round(nValG);
	}
	
	//====== set the distributions over the atoms ======
	dist_atomt.resize(struc_train.size());
	dist_atomv.resize(struc_val.size());
	for(int i=0; i<struc_train.size(); ++i) dist_atomt[i].init(BATCH.size(),BATCH.rank(),struc_train[i].nAtoms());
	for(int i=0; i<struc_val.size(); ++i) dist_atomv[i].init(BATCH.size(),BATCH.rank(),struc_val[i].nAtoms());
	
	//====== initialize the random number generator ======
	rngen_=std::mt19937(std::chrono::system_clock::now().time_since_epoch().count());
	
	//====== compute the number of atoms of each element ======
	if(NNP_TRAIN_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"computing the number of atoms of each element\n";
	if(nElements_<=0) nElements_=nnp_.nspecies();
	else if(nElements_!=nnp_.nspecies()) throw std::invalid_argument("NNPTrain::train(int): Invalid number of elements in the potential.");
	//compute the number of atoms of each species, set the type
	double nAtoms_[nElements_];
	for(int i=0; i<nElements_; ++i) nAtoms_[i]=0;
	for(int i=0; i<struc_train.size(); ++i){
		const Structure& strucl=struc_train[i];
		for(int j=0; j<strucl.nAtoms(); ++j){
			++nAtoms_[strucl.type(j)];
		}
	}
	//normalize the number of atoms based on the size of the BATCH communicator
	for(int i=0; i<nElements_; ++i) nAtoms_[i]/=BATCH.size();
	//consolidate and bcast total number of atoms
	double tempv_[nElements_];
	for(int i=0; i<nElements_; ++i) tempv_[i]=0;
	MPI_Allreduce(nAtoms_,tempv_,nElements_,MPI_DOUBLE,MPI_SUM,WORLD.label());
	for(int i=0; i<nElements_; ++i) nAtoms_[i]=tempv_[i];
	if(NNP_TRAIN_PRINT_DATA>-1 && WORLD.rank()==0){
		char* str=new char[print::len_buf];
		std::cout<<print::buf(str)<<"\n";
		std::cout<<print::title("ATOM - DATA",str)<<"\n";
		for(int i=0; i<nElements_; ++i){
			std::cout
				<<nnp_.nnh(i).atom().name()
				<<"("<<nnp_.index(nnp_.nnh(i).atom().name())<<") - "
				<<(int)nAtoms_[i]<<"\n";
		}
		std::cout<<print::title("ATOM - DATA",str)<<"\n";
		std::cout<<print::buf(str)<<"\n";
		delete[] str;
	}
	
	//====== set the indices and batch size ======
	if(NNP_TRAIN_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"setting indices and batch\n";
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
	if(NNP_TRAIN_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"compute the total number\n";
	for(int n=0; n<struc_train.size(); ++n){
		const Structure& strucl=struc_train[n];
		//loop over all species
		for(int i=0; i<strucl.nAtoms(); ++i){
			//increment the count
			++N[strucl.type(i)];
		}
	}
	//accumulate the number
	for(int i=0; i<nElements_; ++i){
		double Nloc=(1.0*N[i])/BATCH.size();//normalize by the size of the BATCH group
		double tmp=0;
		MPI_Allreduce(&Nloc,&tmp,1,MPI_DOUBLE,MPI_SUM,WORLD.label());
		N[i]=static_cast<int>(std::round(tmp));
	}
	//compute the max/min
	if(NNP_TRAIN_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"compute the max/min\n";
	for(int n=0; n<struc_train.size(); ++n){
		const Structure& strucl=struc_train[n];
		//loop over all species
		for(int i=0; i<strucl.nAtoms(); ++i){
			//find the index of the current species
			const int index=strucl.type(i);
			//set limits for max and min
			max_in[index].noalias()-=strucl.symm(i);
			min_in[index].noalias()+=strucl.symm(i);
		}
	}
	for(int n=0; n<struc_train.size(); ++n){
		const Structure& strucl=struc_train[n];
		//loop over all species
		for(int i=0; i<strucl.nAtoms(); ++i){
			//find the index of the current species
			const int index=strucl.type(i);
			//loop over all inputs
			for(int k=0; k<nnp_.nnh(index).nInput(); ++k){
				//find the max and min
				if(strucl.symm(i)[k]>max_in[index][k]) max_in[index][k]=strucl.symm(i)[k];
				if(strucl.symm(i)[k]<min_in[index][k]) min_in[index][k]=strucl.symm(i)[k];
			}
		}
	}
	//accumulate the min/max
	for(int i=0; i<min_in.size(); ++i){
		for(int j=0; j<min_in[i].size(); ++j){
			double tmp=0;
			MPI_Allreduce(&min_in[i][j],&tmp,1,MPI_DOUBLE,MPI_MIN,WORLD.label());
			min_in[i][j]=tmp;
		}
	}
	for(int i=0; i<max_in.size(); ++i){
		for(int j=0; j<max_in[i].size(); ++j){
			double tmp=0;
			MPI_Allreduce(&max_in[i][j],&tmp,1,MPI_DOUBLE,MPI_MAX,WORLD.label());
			max_in[i][j]=tmp;
		}
	}
	//compute the average - loop over all structures
	if(NNP_TRAIN_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"compute the average\n";
	for(int n=0; n<struc_train.size(); ++n){
		const Structure& strucl=struc_train[n];
		//loop over all species
		for(int i=0; i<strucl.nAtoms(); ++i){
			//find the index of the current species
			const int index=strucl.type(i);
			//add the inputs to the average
			avg_in[index].noalias()+=strucl.symm(i);
		}
	}
	//accumulate the average
	for(int i=0; i<avg_in.size(); ++i){
		for(int j=0; j<avg_in[i].size(); ++j){
			double tmp=0;
			avg_in[i][j]/=BATCH.size();//normalize by the size of the BATCH group
			MPI_Allreduce(&avg_in[i][j],&tmp,1,MPI_DOUBLE,MPI_SUM,WORLD.label());
			avg_in[i][j]=tmp/N[i];
		}
	}
	//compute the stddev - loop over all structures
	if(NNP_TRAIN_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"compute the stddev\n";
	for(int n=0; n<struc_train.size(); ++n){
		const Structure& strucl=struc_train[n];
		//loop over all species
		for(int i=0; i<strucl.nAtoms(); ++i){
			//find the index of the current species
			const int index=strucl.type(i);
			//update the variance
			dev_in[index].noalias()+=(avg_in[index]-strucl.symm(i)).cwiseProduct(avg_in[index]-strucl.symm(i));
		}
	}
	//accumulate the stddev
	for(int i=0; i<dev_in.size(); ++i){
		for(int j=0; j<dev_in[i].size(); ++j){
			double tmp=0;
			dev_in[i][j]/=BATCH.size();//normalize by the size of the BATCH group
			MPI_Allreduce(&dev_in[i][j],&tmp,1,MPI_DOUBLE,MPI_SUM,WORLD.label());
			dev_in[i][j]=std::sqrt(tmp/(N[i]-1.0));
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
		if(NNP_TRAIN_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"pre-conditioning input\n";
		//set the preconditioning vectors - bias
		for(int i=0; i<inb_.size(); ++i){
			for(int j=0; j<inb_[i].size(); ++j){
				inb_[i][j]=-1*avg_in[i][j];
			}
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
	if(NNP_TRAIN_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"setting the bias for each species\n";
	for(int n=0; n<nElements_; ++n){
		NN::ANN& nn_=nnp_.nnh(n).nn();
		for(int i=0; i<nn_.nIn(); ++i) nn_.inb()[i]=inb_[n][i];
		for(int i=0; i<nn_.nIn(); ++i) nn_.inw()[i]=inw_[n][i];
		nn_.outb()[0]=0.0;
		nn_.outw()[0]=1.0;
	}
	
	//====== resize the optimization data ======
	if(NNP_TRAIN_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"initializing the optimization data\n";
	//resize per-element arrays
	pElement_.resize(nElements_);
	gElement_.resize(nElements_);
	gTotal_.resize(nElements_);
	gLocal_.resize(nElements_);
	nParams_=0;
	for(int n=0; n<nElements_; ++n){
		const int nn_size=nnp_.nnh(n).nn().size();
		pElement_[n]=Eigen::VectorXd::Zero(nn_size);
		gElement_[n]=Eigen::VectorXd::Zero(nn_size);
		gTotal_[n]=Eigen::VectorXd::Zero(nn_size);
		gLocal_[n]=Eigen::VectorXd::Zero(nn_size);
		nParams_+=nn_size;
	}
	//resize gradient objects
	if(NNP_TRAIN_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"resizing gradient data\n";
	cost_.resize(nElements_);
	for(int i=0; i<nElements_; ++i){
		cost_[i].resize(nnp_.nnh(i).nn());
	}
	
	//====== initialize the optimization data ======
	if(restart_){
		//restart
		if(WORLD.rank()==0) std::cout<<"restarting optimization\n";
		if(nParams_!=data_.dim()) throw std::runtime_error(
			std::string("NNPTrain::train(int): Network has ")
			+std::to_string(nParams_)+std::string(" while opt has ")
			+std::to_string(data_.dim())+std::string(" parameters.")
		);
	} else {
		//from scratch
		if(WORLD.rank()==0) std::cout<<"starting from scratch\n";
		//resize the optimization objects
		data_.init(nParams_);
		model_->init(nParams_);
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
	if(NNP_TRAIN_PRINT_DATA>-1 && WORLD.rank()==0){
		char* str=new char[print::len_buf];
		std::cout<<print::buf(str)<<"\n";
		std::cout<<print::title("OPT - DATA",str)<<"\n";
		std::cout<<"N-PARAMS    = \n\t"<<nParams_<<"\n";
		std::cout<<"AVG - INPUT = \n"; for(int i=0; i<avg_in.size(); ++i) std::cout<<"\t"<<avg_in[i].transpose()<<"\n";
		std::cout<<"MAX - INPUT = \n"; for(int i=0; i<max_in.size(); ++i) std::cout<<"\t"<<max_in[i].transpose()<<"\n";
		std::cout<<"MIN - INPUT = \n"; for(int i=0; i<min_in.size(); ++i) std::cout<<"\t"<<min_in[i].transpose()<<"\n";
		std::cout<<"DEV - INPUT = \n"; for(int i=0; i<dev_in.size(); ++i) std::cout<<"\t"<<dev_in[i].transpose()<<"\n";
		std::cout<<"PRE-BIAS    = \n"; for(int i=0; i<inb_.size(); ++i) std::cout<<"\t"<<inb_[i].transpose()<<"\n";
		std::cout<<"PRE-SCALE   = \n"; for(int i=0; i<inw_.size(); ++i) std::cout<<"\t"<<inw_[i].transpose()<<"\n";
		std::cout<<print::title("OPT - DATA",str)<<"\n";
		std::cout<<print::buf(str)<<"\n";
		delete[] str;
	}
	
	//====== execute the optimization ======
	if(NNP_TRAIN_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"executing the optimization\n";
	//optimization variables
	bool fbreak=false;
	identity_=Eigen::VectorXd::Constant(1,1);
	const double nBatchi_=1.0/nBatch;
	const double nVali_=1.0/nVal;
	//bcast parameters
	MPI_Bcast(data_.p().data(),data_.p().size(),MPI_DOUBLE,0,WORLD.label());
	//allocate status vectors
	std::vector<int> step;
	std::vector<double> gamma,err_t,err_v;
	if(WORLD.rank()==0){
		int size=data_.max()/data_.nPrint();
		if(size==0) ++size;
		step.resize(size);
		gamma.resize(size);
		err_t.resize(size);
		err_v.resize(size);
	}
	//weight mask (for regularization)
	Eigen::VectorXd maskWeight;
	if(WORLD.rank()==0 && model_->lambda()>0){
		int count=0;
		maskWeight.resize(nParams_);
		for(int n=0; n<nElements_; ++n){
			for(int i=0; i<nnp_.nnh(n).nn().nBias(); ++i) maskWeight[count++]=0.0;
			for(int i=0; i<nnp_.nnh(n).nn().nWeight(); ++i) maskWeight[count++]=1.0;
		}
	}
	MPI_Barrier(WORLD.label());
	//print status header to standard output
	if(WORLD.rank()==0) printf("opt gamma err_t err_v\n");
	//start the clock
	clock.begin();
	//begin optimization
	for(int iter=0; iter<data_.max(); ++iter){
		double error_train_sum_=0,error_val_sum_=0;
		//compute the error and gradient
		error_energy(data_.p(),struc_train,struc_val);
		//accumulate error
		MPI_Reduce(&error_train_,&error_train_sum_,1,MPI_DOUBLE,MPI_SUM,0,WORLD.label());
		MPI_Reduce(&error_val_,&error_val_sum_,1,MPI_DOUBLE,MPI_SUM,0,WORLD.label());
		//accumulate gradient
		for(int n=0; n<nElements_; ++n){
			gTotal_[n].setZero();
			MPI_Reduce(gElement_[n].data(),gTotal_[n].data(),gElement_[n].size(),MPI_DOUBLE,MPI_SUM,0,WORLD.label());
		}
		if(WORLD.rank()==0){
			//compute error averaged over the batch
			error_train_=error_train_sum_*nBatchi_;
			error_val_=error_val_sum_*nVali_;
			//compute gradient averaged over the batch
			for(int n=0; n<nElements_; ++n) gElement_[n].noalias()=gTotal_[n]*nBatchi_;
			//pack the gradient
			int count=0;
			for(int n=0; n<nElements_; ++n){
				std::memcpy(data_.g().data()+count,gElement_[n].data(),gElement_[n].size()*sizeof(double));
				count+=gElement_[n].size();
			}
			//print/write error
			if(data_.step()%data_.nPrint()==0){
				const int t=iter/data_.nPrint();
				step[t]=data_.count();
				gamma[t]=model_->gamma();
				err_t[t]=std::sqrt(2.0*error_train_);
				err_v[t]=std::sqrt(2.0*error_val_);
				printf("%8i %12.10f %12.10f %12.10f\n",step[t],gamma[t],err_t[t],err_v[t]);
			}
			//write the basis and potentials
			if(data_.step()%data_.nWrite()==0){
				if(NNP_TRAIN_PRINT_STATUS>1) std::cout<<"writing the restart file and potentials\n";
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
		MPI_Bcast(data_.p().data(),data_.p().size(),MPI_DOUBLE,0,WORLD.label());
		//bcast break condition
		MPI_Bcast(&fbreak,1,MPI_C_BOOL,0,WORLD.label());
		if(fbreak) break;
		//increment step
		++data_.step();
		++data_.count();
	}
	//compute the training time
	clock.end();
	const double time_train=clock.duration();
	double time_train_avg=0;
	MPI_Allreduce(&time_train,&time_train_avg,1,MPI_DOUBLE,MPI_SUM,WORLD.label());
	time_train_avg/=WORLD.size();
	MPI_Barrier(WORLD.label());
	
	//====== write the error ======
	if(WORLD.rank()==0){
		FILE* writer_error_=NULL;
		if(!restart_){
			writer_error_=fopen(file_error_.c_str(),"w");
			fprintf(writer_error_,"#STEP GAMMA ERROR_RMS_TRAIN ERROR_RMS_VAL\n");
		} else {
			writer_error_=fopen(file_error_.c_str(),"a");
		}
		if(writer_error_==NULL) throw std::runtime_error("NNPTrain::train(int): Could not open error record file.");
		for(int t=0; t<step.size(); ++t){
			fprintf(writer_error_,"%6i %12.10f %12.10f %12.10f\n",step[t],gamma[t],err_t[t],err_v[t]);
		}
		fclose(writer_error_);
		writer_error_=NULL;
	}
	
	//====== unpack final parameters ======
	if(NNP_TRAIN_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"packing final parameters into neural network\n";
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
	
	if(NNP_TRAIN_PRINT_DATA>-1 && WORLD.rank()==0){
		char* str=new char[print::len_buf];
		std::cout<<print::buf(str)<<"\n";
		std::cout<<print::title("TRAIN - SUMMARY",str)<<"\n";
		std::cout<<"N-STEP = "<<data_.step()<<"\n";
		std::cout<<"TIME   = "<<time_train_avg<<"\n";
		if(NNP_TRAIN_PRINT_DATA>1){
			std::cout<<"p = "; for(int i=0; i<data_.p().size(); ++i) std::cout<<data_.p()[i]<<" "; std::cout<<"\n";
		}
		std::cout<<print::title("TRAIN - SUMMARY",str)<<"\n";
		std::cout<<print::buf(str)<<"\n";
		delete[] str;
	}
}

double NNPTrain::error_energy(const Eigen::VectorXd& x, const std::vector<Structure>& struc_train, const std::vector<Structure>& struc_val){
	if(NNP_TRAIN_PRINT_FUNC>0) std::cout<<"NNPTrain::error(const Eigen::VectorXd&):\n";
	
	//====== reset the error ======
	error_train_=0; //error - training
	error_val_=0;   //error - validation
	
	//====== unpack total parameters into element arrays ======
	if(NNP_TRAIN_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"unpacking total parameters into element arrays\n";
	int count=0;
	for(int n=0; n<nElements_; ++n){
		std::memcpy(pElement_[n].data(),x.data()+count,pElement_[n].size()*sizeof(double));
		count+=pElement_[n].size();
	}
	
	//====== unpack arrays into element nn's ======
	if(NNP_TRAIN_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"unpacking arrays into element nn's\n";
	for(int n=0; n<nElements_; ++n) nnp_.nnh(n).nn()<<pElement_[n];
	
	//====== reset the gradients ======
	if(NNP_TRAIN_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"resetting gradients\n";
	for(int n=0; n<nElements_; ++n) gElement_[n].setZero();
	
	//====== randomize the batch ======
	if(NNP_TRAIN_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"randomizing the batch\n";
	for(int i=0; i<batch_.size(); ++i) batch_[i]=batch_.data((batch_.count()++)%batch_.capacity());
	std::sort(batch_.elements(),batch_.elements()+batch_.size());
	if(batch_.count()>=batch_.capacity()){
		std::shuffle(batch_.data(),batch_.data()+batch_.capacity(),rngen_);
		MPI_Bcast(batch_.data(),batch_.capacity(),MPI_INT,0,BATCH.label());
		batch_.count()=0;
	}
	if(NNP_TRAIN_PRINT_DATA>1 && WORLD.rank()==0){std::cout<<"batch = "; for(int i=0; i<batch_.size(); ++i) std::cout<<batch_[i]<<" "; std::cout<<"\n";}
	
	//====== compute training error and gradient ======
	if(NNP_TRAIN_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"computing training error and gradient\n";
	for(int i=0; i<batch_.size(); ++i){
		const int ii=batch_[i];
		//reset the gradients
		for(int j=0; j<nElements_; ++j) gTotal_[j].setZero();
		for(int j=0; j<nElements_; ++j) gLocal_[j].setZero();
		//compute the energy
		double energyl=0;
		for(int n=0; n<dist_atomt[ii].size(); ++n){
			//get the index of the atom within the local processor subset
			const int m=dist_atomt[ii].index(n);
			//find the element index in the nn potential
			const int type=struc_train[ii].type(m);
			//execute the network
			nnp_.nnh(type).nn().execute(struc_train[ii].symm(m));
			//add the atom energy to the total
			energyl+=nnp_.nnh(type).nn().out()[0]+nnp_.nnh(type).atom().energy();
			//compute the gradient - here dcda (2nd argument) is one, as dcda is pulled out and multiplied later
			gLocal_[type].noalias()+=cost_[type].grad(nnp_.nnh(type).nn(),identity_);
		}
		//accumulate energy across BATCH communicator
		double energyt=0;
		MPI_Allreduce(&energyl,&energyt,1,MPI_DOUBLE,MPI_SUM,BATCH.label());
		//accumulate gradient across the BATCH communicator
		for(int j=0; j<nElements_; ++j){
			MPI_Allreduce(gLocal_[j].data(),gTotal_[j].data(),gTotal_[j].size(),MPI_DOUBLE,MPI_SUM,BATCH.label());
		}
		//compute the energy difference normalized by number of atoms
		const double scale=error_scale_/struc_train[ii].nAtoms();
		const double ediff=scale*(energyt-struc_train[ii].energy());
		//compute the error and parameter gradients
		switch(loss_){
			case Opt::Loss::MSE:{
				error_train_+=0.5*ediff*ediff;
				for(int j=0; j<nElements_; ++j){
					gElement_[j].noalias()+=gTotal_[j]*ediff*scale;
				}
			} break;
			case Opt::Loss::MAE:{
				error_train_+=std::fabs(ediff);
				for(int j=0; j<nElements_; ++j){
					gElement_[j].noalias()+=gTotal_[j]*math::func::sgn(ediff)*scale;
				}
			} break;
			case Opt::Loss::HUBER:{
				const double sqrtf=sqrt(ediff*ediff/w2_+1.0);
				error_train_+=w2_*(sqrtf-1.0);
				for(int j=0; j<nElements_; ++j){
					gElement_[j].noalias()+=gTotal_[j]*ediff*scale/sqrtf;
				}
			} break;
			default: break;
		}
	}
	
	//====== compute validation error and gradient ======
	if(data_.step()%data_.nPrint()==0 || data_.step()%data_.nWrite()==0){
		if(NNP_TRAIN_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"computing validation error and gradient\n";
		for(int i=0; i<struc_val.size(); ++i){
			//compute the energy
			double energyl=0;
			for(int n=0; n<dist_atomv[i].size(); ++n){
				//get the index of the atom within the local processor subset
				const int m=dist_atomv[i].index(n);
				//find the element index in the nn potential
				const int type=struc_val[i].type(m);
				//execute the network
				nnp_.nnh(type).nn().execute(struc_val[i].symm(m));
				//add the energy to the total
				energyl+=nnp_.nnh(type).nn().out()[0]+nnp_.nnh(type).atom().energy();
			}
			//accumulate energy
			double energyt=0;
			MPI_Allreduce(&energyl,&energyt,1,MPI_DOUBLE,MPI_SUM,BATCH.label());
			//compute the energy difference normalized by number of atoms
			const double ediff=error_scale_*(energyt-struc_val[i].energy())/struc_val[i].nAtoms();
			switch(loss_){
				case Opt::Loss::MSE:{
					error_val_+=0.5*ediff*ediff;
				} break;
				case Opt::Loss::MAE:{
					error_val_+=std::fabs(ediff);
				} break;
				case Opt::Loss::HUBER:{
					error_val_+=w2_*(sqrt(1.0+(ediff*ediff/w2_))-1.0);
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
	if(NNP_TRAIN_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"returning the error\n";
	return error_train_;
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
		atomT.name=true; atomT.an=false; atomT.type=true; atomT.index=false;
		atomT.posn=true; atomT.force=false; atomT.symm=true; atomT.charge=false;
		FILE_FORMAT::type format;//format of training data
	//nn potential - opt
		int nBatch=-1;
		std::vector<Atom> atoms;//unique atomic species
		NNPTrain nnp_train;//nn potential optimization data
		Opt::Model* model_param_=NULL;//optimization model
		std::vector<std::vector<int> > nh;//hidden layer configuration
		NN::Transfer tf;//transfer function
		NN::ANNInit annInit;//neural network initialization parameters
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
	//ewald
		Ewald3D::Coulomb ewald;//ewald object
		double prec=0;//ewald precision
	//mpi data distribution
		parallel::Dist dist_batch; //data distribution - batch
		parallel::Dist dist_train; //data distribution - training
		parallel::Dist dist_val;   //data distribution - validation
		parallel::Dist dist_test;  //data distribution - testing
		int* rank_batch=NULL;  //the BATCH ranks for each processor in WORLD
		int* rank_head=NULL;   //the WORLD rank of the processors with rank 0 in BATCH
		MPI_Group group_world; //the group associated with the WORLD communicator
		MPI_Group group_head;  //the group associated with all the head ranks
		MPI_Comm comm_head;    //the communicator between all the head ranks
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
		std::vector<std::string> strlist;
		char* paramfile=new char[string::M];
		char* input=new char[string::M];
		char* strbuf=new char[print::len_buf];
		bool read_pot=false;
		std::string file_pot;
		std::vector<std::string> files_basis;//file - stores basis
	//writing
		bool write_basis=false;  //writing - basis functions
		bool write_energy=false; //writing - energies
		bool write_ewald=false;  //writing - ewald energies
		bool write_corr=false;   //writing - input correlation
		bool write_input=false;  //writing - inputs
		bool write_symm=false;   //writing - symmetry functions
		bool write_force=false;  //writing - forces
		
	try{
		//************************************************************************************
		// LOADING/INITIALIZATION
		//************************************************************************************
		
		//======== initialize mpi ========
		MPI_Init(&argc,&argv);
		WORLD.label()=MPI_COMM_WORLD;
		MPI_Comm_size(WORLD.label(),&WORLD.size());
		MPI_Comm_rank(WORLD.label(),&WORLD.rank());
		
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
			MPI_Gather(&WORLD.rank(),1,MPI_INT,ranks,1,MPI_INT,0,WORLD.label());
			if(WORLD.rank()==0){
				std::cout<<print::buf(strbuf)<<"\n";
				std::cout<<print::title("MPI",strbuf)<<"\n";
				std::cout<<"world - size = "<<WORLD.size()<<"\n"<<std::flush;
				//for(int i=0; i<WORLD.size(); ++i) std::cout<<"reporting from process "<<ranks[i]<<" out of "<<WORLD.size()-1<<"\n"<<std::flush;
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
			if(NNP_TRAIN_PRINT_STATUS>0) std::cout<<"reading parameter file\n";
			std::strcpy(paramfile,argv[1]);
			
			//======== open the parameter file ========
			if(NNP_TRAIN_PRINT_STATUS>0) std::cout<<"opening parameter file\n";
			reader=fopen(paramfile,"r");
			if(reader==NULL) throw std::runtime_error(std::string("I/O Error: Could not open parameter file: ")+paramfile);
			
			//======== read in the parameters ========
			if(NNP_TRAIN_PRINT_STATUS>-1) std::cout<<"reading parameters\n";
			while(fgets(input,string::M,reader)!=NULL){
				string::trim_right(input,string::COMMENT);//trim comments
				if(string::split(input,string::WS,strlist)==0) continue;//skip if empty
				string::to_upper(strlist.at(0));//convert tag to upper case
				if(strlist.size()<2) throw std::runtime_error("Parameter tag without corresponding value.");
				//general
				if(strlist.at(0)=="UNITS"){//units
					unitsys=units::System::read(string::to_upper(strlist.at(1)).c_str());
				} else if(strlist.at(0)=="FORMAT"){//simulation format
					format=FILE_FORMAT::read(string::to_upper(strlist.at(1)).c_str());
				} 
				//data and execution mode
				if(strlist.at(0)=="Mode"){//mode of calculation
					mode=Mode::read(string::to_upper(strlist.at(1)).c_str());
				} else if(strlist.at(0)=="DATA_TRAIN"){//data - training
					data_train.push_back(strlist.at(1));
				} else if(strlist.at(0)=="DATA_VAL"){//data - validation
					data_val.push_back(strlist.at(1));
				} else if(strlist.at(0)=="DATA_TEST"){//data - testing
					data_test.push_back(strlist.at(1));
				} else if(strlist.at(0)=="ATOM"){//atom - name/mass/energy
					//process the string
					const std::string name=strlist.at(1);
					const std::string tag=string::to_upper(strlist.at(2));
					const std::string val=strlist.at(3);
					const int id=string::hash(name);
					//look for the atom name in the existing list of atom names
					int index=-1;
					for(int i=0; i<atoms.size(); ++i){
						if(name==atoms[i].name()){index=i;break;}
					}
					//if atom is not found, add it
					if(index<0){
						index=atoms.size();
						atoms.push_back(Atom());
						atoms.back().name()=name;
						atoms.back().id()=id;
						files_basis.resize(files_basis.size()+1);
						nh.resize(nh.size()+1);
					}
					//set tag value
					if(tag=="MASS") atoms[index].mass()=std::atof(val.c_str());
					else if(tag=="CHARGE") atoms[index].charge()=std::atof(val.c_str());
					else if(tag=="ENERGY") atoms[index].energy()=std::atof(val.c_str());
					else if(tag=="BASIS") files_basis[index]=val;
					else if(tag=="NH"){
						const int nl=strlist.size()-3;
						if(nl<=0) throw std::invalid_argument("Invalid hidden layer configuration.");
						nh[index].resize(nl);
						for(int i=0; i<nl; ++i){
							nh[index][i]=std::atoi(strlist.at(i+3).c_str());
							if(nh[index][i]<=0) throw std::invalid_argument("Invalid hidden layer configuration.");
						}
					} else throw std::invalid_argument("Invalid atom tag.");
				} 
				//neural network potential
				if(strlist.at(0)=="R_CUT"){//distance cutoff
					nnp_train.nnp().rc()=std::atof(strlist.at(1).c_str());
				} else if(strlist.at(0)=="HUBERW"){//distance cutoff
					const double w=std::atof(strlist.at(1).c_str());
					if(w<=0) throw std::invalid_argument("Invalid Huber width.");
					nnp_train.w()=w;
					nnp_train.w2()=w*w;
				}  else if(strlist.at(0)=="SEED"){//random seed
					annInit.seed()=std::atoi(strlist.at(1).c_str());
				} else if(strlist.at(0)=="SIGMA"){//initialization deviation
					annInit.sigma()=std::atof(strlist.at(1).c_str());
				} else if(strlist.at(0)=="DIST"){//initialization distribution
					annInit.dist()=rng::dist::Name::read(string::to_upper(strlist.at(1)).c_str());
				} else if(strlist.at(0)=="INIT"){//initialization
					annInit.init()=NN::Init::read(string::to_upper(strlist.at(1)).c_str());
				} else if(strlist.at(0)=="W_INIT"){//initialization
					annInit.wInit()=std::atof(strlist.at(1).c_str());
				} else if(strlist.at(0)=="B_INIT"){//initialization
					std::cout<<"B_INIT = "<<std::atof(strlist.at(1).c_str())<<"\n";
					annInit.bInit()=std::atof(strlist.at(1).c_str());
				} else if(strlist.at(0)=="TRANSFER"){//transfer function
					tf=NN::Transfer::read(string::to_upper(strlist.at(1)).c_str());
				} else if(strlist.at(0)=="READ_POT"){
					file_pot=strlist.at(1);
					read_pot=true;
				}
				//ewald
				if(strlist.at(0)=="PREC"){//precision of ewald calculation
					prec=std::atof(strlist.at(1).c_str());
				} else if(strlist.at(0)=="EPS"){//dielectric constant of bounding medium
					ewald.eps()=std::atof(strlist.at(1).c_str());
				}
				//nnp train - flags
				if(strlist.at(0)=="FILE_ERROR"){
					nnp_train.file_error()=strlist.at(1);//restart file
				} else if(strlist.at(0)=="FILE_ANN"){
					nnp_train.file_ann()=strlist.at(1);//file storing the ann
				} else if(strlist.at(0)=="FILE_RESTART"){
					nnp_train.file_restart()=strlist.at(1);//restart file
				} else if(strlist.at(0)=="RESTART"){//read restart file
					nnp_train.restart_=string::boolean(strlist.at(1).c_str());//restarting
				} else if(strlist.at(0)=="PRE_COND"){//whether to precondition the inputs
					nnp_train.preCond()=string::boolean(strlist.at(1).c_str());
				} else if(strlist.at(0)=="CALC_FORCE"){//compute force at end
					nnp_train.force()=string::boolean(strlist.at(1).c_str());
					atomT.force=nnp_train.force();
				} else if(strlist.at(0)=="NORM"){//normalize energy
					nnp_train.norm()=string::boolean(strlist.at(1).c_str());
				} else if(strlist.at(0)=="CHARGE"){//whether charge contributions to energy are included (n.y.i.)
					nnp_train.charge()=string::boolean(strlist.at(1).c_str());
					atomT.charge=nnp_train.charge();
				} else if(strlist.at(0)=="N_BATCH"){//size of the batch
					nBatch=std::atoi(strlist.at(1).c_str());
				} 
				//nnp train - optimization
				if(strlist.at(0)=="ERROR_SCALE"){
					nnp_train.error_scale()=std::atof(strlist.at(1).c_str());
				} else if(strlist.at(0)=="LOSS"){
					nnp_train.loss()=Opt::Loss::read(string::to_upper(strlist.at(1)).c_str());
				} 
				//writing
				if(strlist.at(0)=="WRITE_BASIS"){//whether to write the basis (function of distance/angle)
					write_basis=string::boolean(strlist.at(1).c_str());
				} else if(strlist.at(0)=="WRITE_ENERGY"){//whether to write the final energies
					write_energy=string::boolean(strlist.at(1).c_str());
				} else if(strlist.at(0)=="WRITE_INPUT"){//whether to write the final energies
					write_input=string::boolean(strlist.at(1).c_str());
					atomT.index=write_input;
				} else if(strlist.at(0)=="WRITE_FORCE"){//whether to write the final forces
					write_force=string::boolean(strlist.at(1).c_str());
				} else if(strlist.at(0)=="WRITE_EWALD"){//whether to write the final ewald energies
					write_ewald=string::boolean(strlist.at(1).c_str());
				} else if(strlist.at(0)=="WRITE_CORR"){//whether to write the final ewald energies
					write_corr=string::boolean(strlist.at(1).c_str());
				} else if(strlist.at(0)=="WRITE_SYMM"){//print symmetry functions
					write_symm=string::boolean(strlist.at(1).c_str());
				} 
			}
			
			//======== read optimization data =========
			if(NNP_TRAIN_PRINT_STATUS>0) std::cout<<"reading optimization data\n";
			Opt::read(nnp_train.data_,reader);
			
			//======== read optimization model ========
			if(NNP_TRAIN_PRINT_STATUS>0) std::cout<<"reading optimization model\n";
			switch(nnp_train.data().algo()){
				case Opt::Algo::SGD:
					nnp_train.model().reset(new Opt::SGD());
					Opt::read(static_cast<Opt::SGD&>(*nnp_train.model()),reader);
					model_param_=new Opt::SGD(static_cast<const Opt::SGD&>(*nnp_train.model()));
				break;
				case Opt::Algo::SDM:
					nnp_train.model().reset(new Opt::SDM());
					Opt::read(static_cast<Opt::SDM&>(*nnp_train.model()),reader);
					model_param_=new Opt::SDM(static_cast<const Opt::SDM&>(*nnp_train.model()));
				break;
				case Opt::Algo::NAG:
					nnp_train.model().reset(new Opt::NAG());
					Opt::read(static_cast<Opt::NAG&>(*nnp_train.model()),reader);
					model_param_=new Opt::NAG(static_cast<const Opt::NAG&>(*nnp_train.model()));
				break;
				case Opt::Algo::ADAGRAD:
					nnp_train.model().reset(new Opt::ADAGRAD());
					Opt::read(static_cast<Opt::ADAGRAD&>(*nnp_train.model()),reader);
					model_param_=new Opt::ADAGRAD(static_cast<const Opt::ADAGRAD&>(*nnp_train.model()));
				break;
				case Opt::Algo::ADADELTA:
					nnp_train.model().reset(new Opt::ADADELTA());
					Opt::read(static_cast<Opt::ADADELTA&>(*nnp_train.model()),reader);
					model_param_=new Opt::ADADELTA(static_cast<const Opt::ADADELTA&>(*nnp_train.model()));
				break;
				case Opt::Algo::RMSPROP:
					nnp_train.model().reset(new Opt::RMSPROP());
					Opt::read(static_cast<Opt::RMSPROP&>(*nnp_train.model()),reader);
					model_param_=new Opt::RMSPROP(static_cast<const Opt::RMSPROP&>(*nnp_train.model()));
				break;
				case Opt::Algo::ADAM:
					nnp_train.model().reset(new Opt::ADAM());
					Opt::read(static_cast<Opt::ADAM&>(*nnp_train.model()),reader);
					model_param_=new Opt::ADAM(static_cast<const Opt::ADAM&>(*nnp_train.model()));
				break;
				case Opt::Algo::NADAM:
					nnp_train.model().reset(new Opt::NADAM());
					Opt::read(static_cast<Opt::NADAM&>(*nnp_train.model()),reader);
					model_param_=new Opt::NADAM(static_cast<const Opt::NADAM&>(*nnp_train.model()));
				break;
				case Opt::Algo::AMSGRAD:
					nnp_train.model().reset(new Opt::AMSGRAD());
					Opt::read(static_cast<Opt::AMSGRAD&>(*nnp_train.model()),reader);
					model_param_=new Opt::AMSGRAD(static_cast<const Opt::AMSGRAD&>(*nnp_train.model()));
				break;
				case Opt::Algo::BFGS:
					nnp_train.model().reset(new Opt::BFGS());
					Opt::read(static_cast<Opt::BFGS&>(*nnp_train.model()),reader);
					model_param_=new Opt::BFGS(static_cast<const Opt::BFGS&>(*nnp_train.model()));
				break;
				case Opt::Algo::RPROP:
					nnp_train.model().reset(new Opt::RPROP());
					Opt::read(static_cast<Opt::RPROP&>(*nnp_train.model()),reader);
					model_param_=new Opt::RPROP(static_cast<const Opt::RPROP&>(*nnp_train.model()));
				break;
				case Opt::Algo::CG:
					nnp_train.model().reset(new Opt::CG());
					Opt::read(static_cast<Opt::CG&>(*nnp_train.model()),reader);
					model_param_=new Opt::CG(static_cast<const Opt::CG&>(*nnp_train.model()));
				break;
				default:
					throw std::invalid_argument("Invalid optimization algorithm.");
				break;
			}
			
			//======== close parameter file ========
			if(NNP_TRAIN_PRINT_STATUS>0) std::cout<<"closing parameter file\n";
			fclose(reader);
			reader=NULL;
			
			//======== check if we compute symmetry functions ========
			if(mode==Mode::SYMM) nnp_train.symm()=true;
			else if(mode==Mode::TRAIN || mode==Mode::TEST){
				if(format==FILE_FORMAT::BINARY) nnp_train.symm()=false;
				else nnp_train.symm()=true;
			}
			
			//========= check the data =========
			if(mode==Mode::TRAIN && data_train.size()==0) throw std::invalid_argument("No training data provided.");
			if(mode==Mode::TRAIN && data_val.size()==0) throw std::invalid_argument("No validation data provided.");
			if(mode==Mode::TEST && data_test.size()==0) throw std::invalid_argument("No test data provided.");
		}
		
		//======== bcast the paramters ========
		if(NNP_TRAIN_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"broadcasting parameters\n";
		//general parameters
		MPI_Bcast(&unitsys,1,MPI_INT,0,WORLD.label());
		//nnp_opt
		MPI_Bcast(&nBatch,1,MPI_INT,0,WORLD.label());
		parallel::bcast(WORLD.label(),0,nnp_train);
		//file i/o
		MPI_Bcast(&format,1,MPI_INT,0,WORLD.label());
		//writing
		MPI_Bcast(&write_basis,1,MPI_C_BOOL,0,WORLD.label());
		MPI_Bcast(&write_energy,1,MPI_C_BOOL,0,WORLD.label());
		MPI_Bcast(&write_force,1,MPI_C_BOOL,0,WORLD.label());
		MPI_Bcast(&write_ewald,1,MPI_C_BOOL,0,WORLD.label());
		MPI_Bcast(&write_corr,1,MPI_C_BOOL,0,WORLD.label());
		MPI_Bcast(&write_input,1,MPI_C_BOOL,0,WORLD.label());
		MPI_Bcast(&write_symm,1,MPI_C_BOOL,0,WORLD.label());
		//mode
		MPI_Bcast(&mode,1,MPI_INT,0,WORLD.label());
		//ewald
		MPI_Bcast(&prec,1,MPI_DOUBLE,0,WORLD.label());
		//atom type
		parallel::bcast(WORLD.label(),0,atomT);
		parallel::bcast(WORLD.label(),0,annInit);
		
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
			std::cout<<"WRITE_BASIS  = "<<write_basis<<"\n";
			std::cout<<"WRITE_ENERGY = "<<write_energy<<"\n";
			std::cout<<"WRITE_EWALD  = "<<write_ewald<<"\n";
			std::cout<<"WRITE_CORR   = "<<write_corr<<"\n";
			std::cout<<"WRITE_INPUTS = "<<write_input<<"\n";
			std::cout<<"WRITE_SYMM   = "<<write_symm<<"\n";
			std::cout<<"WRITE_FORCE  = "<<write_force<<"\n";
			std::cout<<print::title("WRITING",strbuf)<<"\n";
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("ATOMS",strbuf)<<"\n";
			for(int i=0; i<atoms.size(); ++i){
				std::cout<<atoms[i]<<"\n";
			}
			std::cout<<print::title("ATOMS",strbuf)<<"\n";
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<annInit<<"\n";
			std::cout<<nnp_train<<"\n";
		}
		MPI_Barrier(WORLD.label());
		
		//======== check the parameters ========
		if(mode==Mode::UNKNOWN) throw std::invalid_argument("Invalid calculation mode");
		if(format==FILE_FORMAT::UNKNOWN) throw std::invalid_argument("Invalid file format.");
		if(unitsys==units::System::UNKNOWN) throw std::invalid_argument("Invalid unit system.");
		if(nnp_train.loss_==Opt::Loss::UNKNOWN) throw std::invalid_argument("Invalid loss function.");
		if(nnp_train.error_scale_<=0) throw std::invalid_argument("Invalid error scaling.");
		
		//======== set the unit system ========
		if(NNP_TRAIN_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"setting the unit system\n";
		units::consts::init(unitsys);
		
		//************************************************************************************
		// READ DATA
		//************************************************************************************
		
		//======== rank 0 reads the data files (lists of structure files) ========
		if(WORLD.rank()==0){
			//==== read the training data ====
			if(NNP_TRAIN_PRINT_STATUS>-1) std::cout<<"reading data - training\n";
			for(int i=0; i<data_train.size(); ++i){
				//open the data file
				if(NNP_TRAIN_PRINT_DATA>0) std::cout<<"data file "<<i<<": "<<data_train[i]<<"\n";
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
			if(NNP_TRAIN_PRINT_STATUS>-1) std::cout<<"reading data - validation\n";
			for(int i=0; i<data_val.size(); ++i){
				//open the data file
				if(NNP_TRAIN_PRINT_DATA>0) std::cout<<"data file "<<i<<": "<<data_val[i]<<"\n";
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
			if(NNP_TRAIN_PRINT_STATUS>-1) std::cout<<"reading data - testing\n";
			for(int i=0; i<data_test.size(); ++i){
				//open the data file
				if(NNP_TRAIN_PRINT_DATA>0) std::cout<<"data file "<<i<<": "<<data_test[i]<<"\n";
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
			if(NNP_TRAIN_PRINT_DATA>1){
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
		if(NNP_TRAIN_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"bcasting file names\n";
		//bcast names
		parallel::bcast(WORLD.label(),0,files_train);
		parallel::bcast(WORLD.label(),0,files_val);
		parallel::bcast(WORLD.label(),0,files_test);
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
		parallel::Comm::split(WORLD,BATCH,nBatch);//create the BATCH groups
		MPI_Comm_split(WORLD.label(),BATCH.color(),WORLD.rank(),&BATCH.label());//split using the BATCH color and key (WORLD rank), store label
		MPI_Comm_rank(BATCH.label(),&BATCH.rank());//set the rank within the BATCH communicator
		MPI_Comm_size(BATCH.label(),&BATCH.size());//set the size of the BATCH communicator
		MPI_Allreduce(&BATCH.color(),&BATCH.ngroup(),1,MPI_INT,MPI_MAX,WORLD.label()); ++BATCH.ngroup();//compute the number of groups
		//gather the batch ranks and head ranks
		rank_batch=new int[WORLD.size()];
		rank_head=new int[BATCH.ngroup()];
		MPI_Allgather(&BATCH.rank(),1,MPI_INT,rank_batch,1,MPI_INT,WORLD.label());
		{
			int count=0;
			for(int i=0; i<WORLD.size(); ++i){
				if(rank_batch[i]==0) rank_head[count++]=i;
			}
		}
		if(WORLD.rank()==0 && NNP_TRAIN_PRINT_STATUS>0){
			std::cout<<"ngroup = "<<BATCH.ngroup()<<"\n";
			for(int i=0; i<WORLD.size(); ++i){
				std::cout<<"rank_world "<<i<<" rank_batch "<<rank_batch[i]<<"\n";
			}
			for(int i=0; i<BATCH.ngroup(); ++i){
				std::cout<<"rank_head["<<i<<"] = "<<rank_head[i]<<"\n";
			}
		}
		//create the HEAD communicator
		MPI_Comm_group(WORLD.label(),&group_world);
		MPI_Group_incl(group_world,BATCH.ngroup(),rank_head,&group_head);
		MPI_Comm_create_group(WORLD.label(),group_head,0,&comm_head);
		int rank_head=-1,size_head=-1;
		if(comm_head!=MPI_COMM_NULL){
			MPI_Comm_rank(comm_head, &rank_head);
			MPI_Comm_size(comm_head, &size_head);
		}
		MPI_Barrier(WORLD.label());
		//print batch communicators
		{
			if(WORLD.rank()==0){
				std::cout<<print::buf(strbuf)<<"\n";
				std::cout<<print::title("BATCH Communicators",strbuf)<<"\n";
				std::cout<<std::flush;
			}
			MPI_Barrier(WORLD.label());
			const int sizeb=serialize::nbytes(BATCH);
			const int sizet=WORLD.size()*serialize::nbytes(BATCH);
			char* arrb=new char[sizeb];
			char* arrt=new char[sizet];
			serialize::pack(BATCH,arrb);
			MPI_Gather(arrb,sizeb,MPI_CHAR,arrt,sizeb,MPI_CHAR,0,WORLD.label());
			if(WORLD.rank()==0){
				for(int i=0; i<WORLD.size(); ++i){
					parallel::Comm tmp;
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
			MPI_Barrier(WORLD.label());
		}
		//thread dist - divide structures equally among the batch groups
		dist_batch.init(BATCH.ngroup(),BATCH.color(),nBatch);
		dist_train.init(BATCH.ngroup(),BATCH.color(),nTrain);
		dist_val.init(BATCH.ngroup(),BATCH.color(),nVal);
		dist_test.init(BATCH.ngroup(),BATCH.color(),nTest);
		//print
		{
			//allocate arrays
			int* thread_dist_batch=new int[BATCH.ngroup()];
			int* thread_dist_train=new int[BATCH.ngroup()];
			int* thread_dist_val=new int[BATCH.ngroup()];
			int* thread_dist_test=new int[BATCH.ngroup()];
			int* thread_offset_train=new int[BATCH.ngroup()];
			int* thread_offset_val=new int[BATCH.ngroup()];
			int* thread_offset_test=new int[BATCH.ngroup()];
			//assign arrays
			parallel::Dist::size(BATCH.ngroup(),nBatch,thread_dist_batch);
			parallel::Dist::size(BATCH.ngroup(),nTrain,thread_dist_train);
			parallel::Dist::size(BATCH.ngroup(),nVal,thread_dist_val);
			parallel::Dist::size(BATCH.ngroup(),nTest,thread_dist_test);
			parallel::Dist::offset(BATCH.ngroup(),nTrain,thread_offset_train);
			parallel::Dist::offset(BATCH.ngroup(),nVal,thread_offset_val);
			parallel::Dist::offset(BATCH.ngroup(),nTest,thread_offset_test);
			//print
			if(WORLD.rank()==0){
				std::cout<<"thread_dist_batch   = "; for(int i=0; i<BATCH.ngroup(); ++i) std::cout<<thread_dist_batch[i]<<" "; std::cout<<"\n";
				std::cout<<"thread_dist_train   = "; for(int i=0; i<BATCH.ngroup(); ++i) std::cout<<thread_dist_train[i]<<" "; std::cout<<"\n";
				std::cout<<"thread_dist_val     = "; for(int i=0; i<BATCH.ngroup(); ++i) std::cout<<thread_dist_val[i]<<" "; std::cout<<"\n";
				std::cout<<"thread_dist_test    = "; for(int i=0; i<BATCH.ngroup(); ++i) std::cout<<thread_dist_test[i]<<" "; std::cout<<"\n";
				std::cout<<"thread_offset_train = "; for(int i=0; i<BATCH.ngroup(); ++i) std::cout<<thread_offset_train[i]<<" "; std::cout<<"\n";
				std::cout<<"thread_offset_val   = "; for(int i=0; i<BATCH.ngroup(); ++i) std::cout<<thread_offset_val[i]<<" "; std::cout<<"\n";
				std::cout<<"thread_offset_test  = "; for(int i=0; i<BATCH.ngroup(); ++i) std::cout<<thread_offset_test[i]<<" "; std::cout<<"\n";
				std::cout<<std::flush;
			}
			//free arrays
			delete[] thread_dist_batch;
			delete[] thread_dist_train;
			delete[] thread_dist_val;
			delete[] thread_dist_test;
			delete[] thread_offset_train;
			delete[] thread_offset_val;
			delete[] thread_offset_test;
		}
		
		//======== gen indices (random shuffle) ========
		std::vector<int> indices_train(nTrain,0);
		std::vector<int> indices_val(nVal,0);
		std::vector<int> indices_test(nTest,0);
		if(WORLD.rank()==0){
			for(int i=0; i<indices_train.size(); ++i) indices_train[i]=i;
			for(int i=0; i<indices_val.size(); ++i) indices_val[i]=i;
			for(int i=0; i<indices_test.size(); ++i) indices_test[i]=i;
			std::random_shuffle(indices_train.begin(),indices_train.end());
			std::random_shuffle(indices_val.begin(),indices_val.end());
			std::random_shuffle(indices_test.begin(),indices_test.end());
		}
		//======== bcast randomized indices ========
		parallel::bcast(WORLD.label(),0,indices_train);
		parallel::bcast(WORLD.label(),0,indices_val);
		parallel::bcast(WORLD.label(),0,indices_test);
		
		//======== read the structures ========
		//==== training structures ====
		if(NNP_TRAIN_PRINT_STATUS>-1) std::cout<<"reading structures - training - color "<<BATCH.color()<<" rank "<<BATCH.rank()<<"\n"<<std::flush;
		if(files_train.size()>0){
			struc_train.resize(dist_train.size());
			//rank 0 of batch group reads structures
			if(BATCH.rank()==0){
				for(int i=0; i<dist_train.size(); ++i){
					const std::string& file=files_train[indices_train[dist_train.index(i)]];
					read_struc(file.c_str(),format,atomT,struc_train[i]);
					if(NNP_TRAIN_PRINT_DATA>1) std::cout<<"\t"<<file<<" "<<struc_train[i].energy()<<"\n";
				}
			}
			//broadcast structures to all other procs in the BATCH group
			for(int i=0; i<dist_train.size(); ++i){
				parallel::bcast(BATCH.label(),0,struc_train[i]);
			}
		}
		//==== validation structures ====
		if(NNP_TRAIN_PRINT_STATUS>-1) std::cout<<"reading structures - validation - color "<<BATCH.color()<<" rank "<<BATCH.rank()<<"\n"<<std::flush;
		if(files_val.size()>0){
			struc_val.resize(dist_val.size());
			//rank 0 of batch group reads structures
			if(BATCH.rank()==0){
				for(int i=0; i<dist_val.size(); ++i){
					const std::string& file=files_val[indices_val[dist_val.index(i)]];
					read_struc(file.c_str(),format,atomT,struc_val[i]);
					if(NNP_TRAIN_PRINT_DATA>1) std::cout<<"\t"<<file<<" "<<struc_val[i].energy()<<"\n";
				}
			}
			//broadcast structures to all other procs in BATCH group
			for(int i=0; i<dist_val.size(); ++i){
				parallel::bcast(BATCH.label(),0,struc_val[i]);
			}
		}
		//==== testing structures ====
		if(NNP_TRAIN_PRINT_STATUS>-1) std::cout<<"reading structures - testing - color "<<BATCH.color()<<" rank "<<BATCH.rank()<<"\n"<<std::flush;
		if(files_test.size()>0){
			struc_test.resize(dist_test.size());
			//rank 0 of batch group reads structures
			if(BATCH.rank()==0){
				for(int i=0; i<dist_test.size(); ++i){
					const std::string& file=files_test[indices_test[dist_test.index(i)]];
					read_struc(file.c_str(),format,atomT,struc_test[i]);
					if(NNP_TRAIN_PRINT_DATA>1) std::cout<<"\t"<<file<<" "<<struc_test[i].energy()<<"\n";
				}
			}
			//broadcast structures to all other procs in group
			for(int i=0; i<dist_test.size(); ++i){
				parallel::bcast(BATCH.label(),0,struc_test[i]);
			}
		}
		MPI_Barrier(WORLD.label());
		
		//======== check the structures ========
		if(NNP_TRAIN_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"checking the structures\n";
		//==== training structures ====
		if(BATCH.rank()==0){
			for(int i=0; i<dist_train.size(); ++i){
				const std::string filename=files_train[indices_train[dist_train.index(i)]];
				const Structure& strucl=struc_train[i];
				if(strucl.nAtoms()==0) throw std::runtime_error(std::string("ERROR: Structure \"")+filename+std::string(" has zero atoms."));
				if(std::isinf(strucl.energy())) throw std::runtime_error(std::string("ERROR: Structure \"")+filename+std::string(" has inf energy."));
				if(strucl.energy()!=strucl.energy()) throw std::runtime_error(std::string("ERROR: Structure \"")+filename+std::string(" has nan energy."));
				if(std::fabs(strucl.energy())<math::constant::ZERO) std::cout<<"*********** WARNING: Structure \""<<filename<<"\" has ZERO energy. ***********";
				if(nnp_train.force_){
					for(int n=0; n<strucl.nAtoms(); ++n){
						const double force=strucl.force(n).squaredNorm();
						if(std::isinf(force)) std::cout<<"WARNING: Atom \""<<strucl.name(n)<<strucl.index(n)<<"\" in \""<<filename<<" has inf force.\n";
						if(force!=force) std::cout<<"WARNING: Atom \""<<strucl.name(n)<<strucl.index(n)<<"\" in \""<<filename<<" has nan force.\n";
					}
				}
				if(NNP_TRAIN_PRINT_DATA>1) std::cout<<"\t"<<filename<<" "<<strucl.energy()<<" "<<WORLD.rank()<<"\n";
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
				if(nnp_train.force_){
					for(int n=0; n<strucl.nAtoms(); ++n){
						const double force=strucl.force(n).squaredNorm();
						if(std::isinf(force)) std::cout<<"WARNING: Atom \""<<strucl.name(n)<<strucl.index(n)<<"\" in \""<<filename<<" has inf force.\n";
						if(force!=force) std::cout<<"WARNING: Atom \""<<strucl.name(n)<<strucl.index(n)<<"\" in \""<<filename<<" has nan force.\n";
					}
				}
			}
			if(NNP_TRAIN_PRINT_DATA>1) for(int i=0; i<dist_val.size(); ++i) std::cout<<"\t"<<files_val[dist_val.index(i)]<<" "<<struc_val[i].energy()<<"\n";
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
				if(nnp_train.force_){
					for(int n=0; n<strucl.nAtoms(); ++n){
						const double force=strucl.force(n).squaredNorm();
						if(std::isinf(force)) std::cout<<"WARNING: Atom \""<<strucl.name(n)<<strucl.index(n)<<"\" in \""<<filename<<" has inf force.\n";
						if(force!=force) std::cout<<"WARNING: Atom \""<<strucl.name(n)<<strucl.index(n)<<"\" in \""<<filename<<" has nan force.\n";
					}
				}
			}
			if(NNP_TRAIN_PRINT_DATA>1) for(int i=0; i<dist_test.size(); ++i) std::cout<<"\t"<<files_test[indices_test[dist_test.index(i)]]<<" "<<struc_test[i].energy()<<"\n";
		}
		MPI_Barrier(WORLD.label());
		
		//************************************************************************************
		// READ/INITIALIZE NN-POT
		//************************************************************************************
		
		//======== initialize the potential (rank 0) ========
		if(NNP_TRAIN_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"initializing the potential\n";
		if(WORLD.rank()==0){
			//======== read the basis (if not restarting) ========
			if(!nnp_train.restart_){
				if(!read_pot){
					//resize the potential
					if(NNP_TRAIN_PRINT_STATUS>-1) std::cout<<"resizing potential\n";
					nnp_train.nnp_.resize(atoms);
					//read basis files
					if(NNP_TRAIN_PRINT_STATUS>-1) std::cout<<"reading basis files\n";
					if(files_basis.size()!=nnp_train.nnp_.nspecies()) throw std::runtime_error("main(int,char**): invalid number of basis files.");
					for(int i=0; i<nnp_train.nnp_.nspecies(); ++i){
						const char* file=files_basis[i].c_str();
						const char* atomName=atoms[i].name().c_str();
						NNP::read_basis(file,nnp_train.nnp_,atomName);
					}
					//initialize the neural network hamiltonians
					if(NNP_TRAIN_PRINT_STATUS>-1) std::cout<<"initializing neural network hamiltonians\n";
					for(int i=0; i<nnp_train.nnp_.nspecies(); ++i){
						NNH& nnhl=nnp_train.nnp_.nnh(i);
						nnhl.atom()=atoms[i];
						nnhl.nn().tf()=tf;
						nnhl.nn().resize(annInit,nnhl.nInput(),nh[i],1);
						nnhl.dOutDVal().resize(nnhl.nn());
					}
				} else {
					if(NNP_TRAIN_PRINT_STATUS>-1) std::cout<<"reading potential\n";
					NNP::read(file_pot.c_str(),nnp_train.nnp_);
				}
			}
			//======== read restart file (if restarting) ========
			if(nnp_train.restart_){
				if(NNP_TRAIN_PRINT_STATUS>-1) std::cout<<"reading restart file\n";
				const std::string file=nnp_train.file_restart_;
				nnp_train.read_restart(file.c_str());
				nnp_train.restart()=true;
			}
			//======== print the potential ========
			if(WORLD.rank()==0) std::cout<<"printing the potential\n";
			std::cout<<nnp_train.nnp_<<"\n";
		}
		
		//======== bcast the potential ========
		if(WORLD.rank()==0) std::cout<<"bcasting the potential\n";
		parallel::bcast(WORLD.label(),0,nnp_train.nnp_);
		
		//======== set the types ========
		if(WORLD.rank()==0) std::cout<<"setting the types\n";
		for(int i=0; i<dist_train.size(); ++i){
			for(int n=0; n<struc_train[i].nAtoms(); ++n){
				struc_train[i].type(n)=nnp_train.nnp_.index(struc_train[i].name(n));
			}
		}
		for(int i=0; i<dist_val.size(); ++i){
			for(int n=0; n<struc_val[i].nAtoms(); ++n){
				struc_val[i].type(n)=nnp_train.nnp_.index(struc_val[i].name(n));
			}
		}
		for(int i=0; i<dist_test.size(); ++i){
			for(int n=0; n<struc_test[i].nAtoms(); ++n){
				struc_test[i].type(n)=nnp_train.nnp_.index(struc_test[i].name(n));
			}
		}
		
		//************************************************************************************
		// INITIALIZE OPTIMIZER
		//************************************************************************************
		
		//======== set optimization data ========
		if(WORLD.rank()==0) std::cout<<"setting optimization data\n";
		if(WORLD.rank()==0){
			//opt - data
			Opt::read(nnp_train.data_,paramfile);
			//opt - model
			switch(nnp_train.data_.algo()){
				case Opt::Algo::SGD:{
					Opt::SGD& nnModel_=static_cast<Opt::SGD&>(*nnp_train.model());
					Opt::SGD& pModel_=static_cast<Opt::SGD&>(*model_param_);
					if(pModel_.gamma()>0) nnModel_.gamma()=pModel_.gamma();
				}break;
				case Opt::Algo::SDM:{
					Opt::SDM& nnModel_=static_cast<Opt::SDM&>(*nnp_train.model());
					Opt::SDM& pModel_=static_cast<Opt::SDM&>(*model_param_);
					if(pModel_.gamma()>0) nnModel_.gamma()=pModel_.gamma();
					if(pModel_.eta()>0) nnModel_.eta()=pModel_.eta();
				}break;
				case Opt::Algo::NAG:{
					Opt::NAG& nnModel_=static_cast<Opt::NAG&>(*nnp_train.model());
					Opt::NAG& pModel_=static_cast<Opt::NAG&>(*model_param_);
					if(pModel_.gamma()>0) nnModel_.gamma()=pModel_.gamma();
					if(pModel_.eta()>0) nnModel_.eta()=pModel_.eta();
				}break;
				case Opt::Algo::ADAGRAD:{
					Opt::ADAGRAD& nnModel_=static_cast<Opt::ADAGRAD&>(*nnp_train.model());
					Opt::ADAGRAD& pModel_=static_cast<Opt::ADAGRAD&>(*model_param_);
					if(pModel_.gamma()>0) nnModel_.gamma()=pModel_.gamma();
				}break;
				case Opt::Algo::ADADELTA:{
					Opt::ADADELTA& nnModel_=static_cast<Opt::ADADELTA&>(*nnp_train.model());
					Opt::ADADELTA& pModel_=static_cast<Opt::ADADELTA&>(*model_param_);
					if(pModel_.gamma()>0) nnModel_.gamma()=pModel_.gamma();
					if(pModel_.eta()>0) nnModel_.eta()=pModel_.eta();
				}break;
				case Opt::Algo::RMSPROP:{
					Opt::RMSPROP& nnModel_=static_cast<Opt::RMSPROP&>(*nnp_train.model());
					Opt::RMSPROP& pModel_=static_cast<Opt::RMSPROP&>(*model_param_);
					if(pModel_.gamma()>0) nnModel_.gamma()=pModel_.gamma();
				}break;
				case Opt::Algo::ADAM:{
					Opt::ADAM& nnModel_=static_cast<Opt::ADAM&>(*nnp_train.model());
					Opt::ADAM& pModel_=static_cast<Opt::ADAM&>(*model_param_);
					if(pModel_.gamma()>0) nnModel_.gamma()=pModel_.gamma();
					nnModel_.decay()=pModel_.decay();
				}break;
				case Opt::Algo::NADAM:{
					Opt::NADAM& nnModel_=static_cast<Opt::NADAM&>(*nnp_train.model());
					Opt::NADAM& pModel_=static_cast<Opt::NADAM&>(*model_param_);
					if(pModel_.gamma()>0) nnModel_.gamma()=pModel_.gamma();
					nnModel_.decay()=pModel_.decay();
				}break;
				case Opt::Algo::AMSGRAD:{
					Opt::AMSGRAD& nnModel_=static_cast<Opt::AMSGRAD&>(*nnp_train.model());
					Opt::AMSGRAD& pModel_=static_cast<Opt::AMSGRAD&>(*model_param_);
					if(pModel_.gamma()>0) nnModel_.gamma()=pModel_.gamma();
					nnModel_.decay()=pModel_.decay();
				}break;
				case Opt::Algo::BFGS:{
					Opt::BFGS& nnModel_=static_cast<Opt::BFGS&>(*nnp_train.model());
					Opt::BFGS& pModel_=static_cast<Opt::BFGS&>(*model_param_);
					if(pModel_.gamma()>0) nnModel_.gamma()=pModel_.gamma();
				}break;
				case Opt::Algo::CG:{
					Opt::CG& nnModel_=static_cast<Opt::CG&>(*nnp_train.model());
					Opt::CG& pModel_=static_cast<Opt::CG&>(*model_param_);
					if(pModel_.gamma()>0) nnModel_.gamma()=pModel_.gamma();
					nnModel_.decay()=pModel_.decay();
				}break;
				case Opt::Algo::RPROP:
					//no parameters
				break;
			}
		}
		
		//======== print optimization data ========
		if(WORLD.rank()==0){
			Opt::Model::print(std::cout,nnp_train.model().get());
			std::cout<<nnp_train.data_<<"\n";
		}
		MPI_Barrier(WORLD.label());
		
		//======== bcast the optimization data ========
		if(WORLD.rank()==0) std::cout<<"bcasting optimization data\n";
		parallel::bcast(WORLD.label(),0,nnp_train.data_);
		if(WORLD.rank()==0) std::cout<<"bcasting optimization model\n";
		switch(nnp_train.data_.algo()){
			case Opt::Algo::SGD:
				if(WORLD.rank()!=0) nnp_train.model().reset(new Opt::SGD());
				parallel::bcast(WORLD.label(),0,static_cast<Opt::SGD&>(*nnp_train.model()));
			break;
			case Opt::Algo::SDM:
				if(WORLD.rank()!=0) nnp_train.model().reset(new Opt::SDM());
				parallel::bcast(WORLD.label(),0,static_cast<Opt::SDM&>(*nnp_train.model()));
			break;
			case Opt::Algo::NAG:
				if(WORLD.rank()!=0) nnp_train.model().reset(new Opt::NAG());
				parallel::bcast(WORLD.label(),0,static_cast<Opt::NAG&>(*nnp_train.model()));
			break;
			case Opt::Algo::ADAGRAD:
				if(WORLD.rank()!=0) nnp_train.model().reset(new Opt::ADAGRAD());
				parallel::bcast(WORLD.label(),0,static_cast<Opt::ADAGRAD&>(*nnp_train.model()));
			break;
			case Opt::Algo::ADADELTA:
				if(WORLD.rank()!=0) nnp_train.model().reset(new Opt::ADADELTA());
				parallel::bcast(WORLD.label(),0,static_cast<Opt::ADADELTA&>(*nnp_train.model()));
			break;
			case Opt::Algo::RMSPROP:
				if(WORLD.rank()!=0) nnp_train.model().reset(new Opt::RMSPROP());
				parallel::bcast(WORLD.label(),0,static_cast<Opt::RMSPROP&>(*nnp_train.model()));
			break;
			case Opt::Algo::ADAM:
				if(WORLD.rank()!=0) nnp_train.model().reset(new Opt::ADAM());
				parallel::bcast(WORLD.label(),0,static_cast<Opt::ADAM&>(*nnp_train.model()));
			break;
			case Opt::Algo::NADAM:
				if(WORLD.rank()!=0) nnp_train.model().reset(new Opt::NADAM());
				parallel::bcast(WORLD.label(),0,static_cast<Opt::NADAM&>(*nnp_train.model()));
			break;
			case Opt::Algo::AMSGRAD:
				if(WORLD.rank()!=0) nnp_train.model().reset(new Opt::AMSGRAD());
				parallel::bcast(WORLD.label(),0,static_cast<Opt::AMSGRAD&>(*nnp_train.model()));
			break;
			case Opt::Algo::BFGS:
				if(WORLD.rank()!=0) nnp_train.model().reset(new Opt::BFGS());
				parallel::bcast(WORLD.label(),0,static_cast<Opt::BFGS&>(*nnp_train.model()));
			break;
			case Opt::Algo::RPROP:
				if(WORLD.rank()!=0) nnp_train.model().reset(new Opt::RPROP());
				parallel::bcast(WORLD.label(),0,static_cast<Opt::RPROP&>(*nnp_train.model()));
			break;
			case Opt::Algo::CG:
				if(WORLD.rank()!=0) nnp_train.model().reset(new Opt::CG());
				parallel::bcast(WORLD.label(),0,static_cast<Opt::CG&>(*nnp_train.model()));
			break;
			
		}
		
		//************************************************************************************
		// EWALD
		//************************************************************************************
		
		//======== compute ewald energies ========
		if(nnp_train.charge_){
			if(WORLD.rank()==0) std::cout<<ewald<<"\n";
			//==== set charges - training ====
			if(WORLD.rank()==0) std::cout<<"setting charges - training\n";
			for(int i=0; i<dist_train.size(); ++i){
				for(int n=0; n<struc_train[i].nAtoms(); ++n){
					for(int j=0; j<nnp_train.nnp_.nspecies(); ++j){
						if(nnp_train.nnp_.nnh(j).atom().name()==struc_train[i].name(n)){
							struc_train[i].charge(n)=nnp_train.nnp_.nnh(j).atom().charge();
							break;
						}
					}
				}
			}
			//==== set charges - validation ====
			if(WORLD.rank()==0) std::cout<<"setting charges - validation\n";
			for(int i=0; i<dist_val.size(); ++i){
				for(int n=0; n<struc_val[i].nAtoms(); ++n){
					for(int j=0; j<nnp_train.nnp_.nspecies(); ++j){
						if(nnp_train.nnp_.nnh(j).atom().name()==struc_val[i].name(n)){
							struc_val[i].charge(n)=nnp_train.nnp_.nnh(j).atom().charge();
							break;
						}
					}
				}
			}
			//==== set charges - testing ====
			if(WORLD.rank()==0) std::cout<<"setting charges - testing\n";
			for(int i=0; i<dist_test.size(); ++i){
				for(int n=0; n<struc_test[i].nAtoms(); ++n){
					for(int j=0; j<nnp_train.nnp_.nspecies(); ++j){
						if(nnp_train.nnp_.nnh(j).atom().name()==struc_test[i].name(n)){
							struc_test[i].charge(n)=nnp_train.nnp_.nnh(j).atom().charge();
							break;
						}
					}
				}
			}
			//==== compute energies - training ====
			if(WORLD.rank()==0) std::cout<<"computing ewald energies - training\n";
			for(int i=0; i<dist_train.size(); ++i){
				ewald.init(struc_train[i],prec);
				struc_train[i].ewald()=ewald.energy(struc_train[i]);
				struc_train[i].energy()-=struc_train[i].ewald();
			}
			//==== compute energies - validation ====
			if(WORLD.rank()==0) std::cout<<"computing ewald energies - validation\n";
			for(int i=0; i<dist_val.size(); ++i){
				ewald.init(struc_val[i],prec);
				struc_val[i].ewald()=ewald.energy(struc_val[i]);
				struc_val[i].energy()-=struc_val[i].ewald();
			}
			//==== compute energies - testing ====
			if(WORLD.rank()==0) std::cout<<"computing ewald energies - testing\n";
			for(int i=0; i<dist_test.size(); ++i){
				ewald.init(struc_test[i],prec);
				struc_test[i].ewald()=ewald.energy(struc_test[i]);
				struc_test[i].energy()-=struc_test[i].ewald();
			}
		}
		
		//************************************************************************************
		// SET INPUTS
		//************************************************************************************
		
		if(nnp_train.symm_){
			
			//======== initialize the symmetry functions ========
			if(NNP_TRAIN_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"initializing symmetry functions - training set\n";
			for(int i=0; i<dist_train.size(); ++i) NNP::init(nnp_train.nnp_,struc_train[i]);
			if(NNP_TRAIN_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"initializing symmetry functions - validation set\n";
			for(int i=0; i<dist_val.size(); ++i) NNP::init(nnp_train.nnp_,struc_val[i]);
			if(NNP_TRAIN_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"initializing symmetry functions - test set\n";
			for(int i=0; i<dist_test.size(); ++i) NNP::init(nnp_train.nnp_,struc_test[i]);
			
			//======== compute the symmetry functions ========
			//==== training ====
			clock.begin();
			if(dist_train.size()>0){
				if(NNP_TRAIN_PRINT_STATUS>-1) std::cout<<"setting the inputs (symmetry functions) - training - color "<<BATCH.color()<<" rank "<<BATCH.rank()<<"\n";
				//compute symmetry functions
				for(int n=BATCH.rank(); n<dist_train.size(); n+=BATCH.size()){
					if(NNP_TRAIN_PRINT_STATUS>0) std::cout<<"structure-train["<<n<<"]\n";
					NeighborList nlist(struc_train[n],nnp_train.nnp_.rc());
					NNP::symm(nnp_train.nnp_,struc_train[n],nlist);
				}
				MPI_Barrier(BATCH.label());
				//bcast symmetry functions
				for(int i=0; i<BATCH.size(); ++i){
					const int root=i;
					for(int n=root; n<dist_train.size(); n+=BATCH.size()){
						parallel::bcast(BATCH.label(),root,struc_train[n]);
					}
				}
				MPI_Barrier(BATCH.label());
			}
			clock.end();
			time_symm_train=clock.duration();
			//==== validation ====
			clock.begin();
			if(dist_val.size()>0){
				if(NNP_TRAIN_PRINT_STATUS>-1) std::cout<<"setting the inputs (symmetry functions) - validation - color "<<BATCH.color()<<" rank "<<BATCH.rank()<<"\n";
				//compute symmetry functions
				for(int n=BATCH.rank(); n<dist_val.size(); n+=BATCH.size()){
					if(NNP_TRAIN_PRINT_STATUS>0) std::cout<<"structure-val["<<n<<"]\n";
					NeighborList nlist(struc_val[n],nnp_train.nnp_.rc());
					NNP::symm(nnp_train.nnp_,struc_val[n],nlist);
				}
				MPI_Barrier(BATCH.label());
				//bcast symmetry functions
				for(int i=0; i<BATCH.size(); ++i){
					const int root=i;
					for(int n=root; n<dist_val.size(); n+=BATCH.size()){
						parallel::bcast(BATCH.label(),root,struc_val[n]);
					}
				}
				MPI_Barrier(BATCH.label());
			}
			clock.end();
			time_symm_val=clock.duration();
			//==== testing ====
			clock.begin();
			if(dist_test.size()>0){
				if(NNP_TRAIN_PRINT_STATUS>-1) std::cout<<"setting the inputs (symmetry functions) - testing - color "<<BATCH.color()<<" rank "<<BATCH.rank()<<"\n";
				//compute symmetry functions
				for(int n=BATCH.rank(); n<dist_test.size(); n+=BATCH.size()){
					if(NNP_TRAIN_PRINT_STATUS>0) std::cout<<"structure-test["<<n<<"]\n";
					NeighborList nlist(struc_test[n],nnp_train.nnp_.rc());
					NNP::symm(nnp_train.nnp_,struc_test[n],nlist);
				}
				MPI_Barrier(BATCH.label());
				for(int i=0; i<BATCH.size(); ++i){
					const int root=i;
					for(int n=root; n<dist_test.size(); n+=BATCH.size()){
						parallel::bcast(BATCH.label(),root,struc_test[n]);
					}
				}
				MPI_Barrier(BATCH.label());
			}
			clock.end();
			time_symm_test=clock.duration();
			MPI_Barrier(WORLD.label());
			
			//======== write the inputs (symmetry functions) ========
			if(NNP_TRAIN_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"writing symmetry function inputs\n";
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
			MPI_Gather(&mem_train_l,1,MPI_INT,mem_train,1,MPI_INT,0,WORLD.label());
			MPI_Gather(&mem_val_l,1,MPI_INT,mem_val,1,MPI_INT,0,WORLD.label());
			MPI_Gather(&mem_test_l,1,MPI_INT,mem_test,1,MPI_INT,0,WORLD.label());
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
		
		//======== train the nn potential ========
		if(mode==Mode::TRAIN){
			if(NNP_TRAIN_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"training the nn potential\n";
			nnp_train.train(dist_batch.size(),struc_train,struc_val);
		}
		MPI_Barrier(WORLD.label());
		
		//************************************************************************************
		// EVALUTION
		//************************************************************************************
		
		//======== statistical data - energies/forces/errors ========
		//data - train
			Accumulator1D<Max,Avg,Var> acc1d_energy_train_n;
			Accumulator1D<Max,Avg,Var> acc1d_force_train_a;
			Accumulator2D<LinReg> acc2d_energy_train;
			Accumulator2D<LinReg> acc2d_forcex_train;
			Accumulator2D<LinReg> acc2d_forcey_train;
			Accumulator2D<LinReg> acc2d_forcez_train;
		//data - val	
			Accumulator1D<Max,Avg,Var> acc1d_energy_val_n;
			Accumulator1D<Max,Avg,Var> acc1d_force_val_a;
			Accumulator2D<LinReg> acc2d_energy_val;
			Accumulator2D<LinReg> acc2d_forcex_val;
			Accumulator2D<LinReg> acc2d_forcey_val;
			Accumulator2D<LinReg> acc2d_forcez_val;
		//data - test
			Accumulator1D<Max,Avg,Var> acc1d_energy_test_n;
			Accumulator1D<Max,Avg,Var> acc1d_force_test_a;
			Accumulator2D<LinReg> acc2d_energy_test;
			Accumulator2D<LinReg> acc2d_forcex_test;
			Accumulator2D<LinReg> acc2d_forcey_test;
			Accumulator2D<LinReg> acc2d_forcez_test;
		
		//======== correlation - inputs ========
		std::vector<std::vector<std::vector<Accumulator2D<PCorr> > > > acc2d_inp_train;
		std::vector<LMat<Accumulator2D<PCorr,Covar,LinReg> > > acc2d_inp_val;
		std::vector<LMat<Accumulator2D<PCorr,Covar,LinReg> > > acc2d_inp_test;
		
		//======== compute the final energies ========
		//==== training systems ====
		if(dist_train.size()>0){
			if(WORLD.rank()==0) std::cout<<"final energies - training set\n";
			//global energy arrays
			double* energy_nn_t=NULL;
			double* energy_exact_t=NULL;
			int* natoms_t=NULL;
			if(WORLD.rank()==0){
				energy_nn_t=new double[nTrain];
				energy_exact_t=new double[nTrain];
				natoms_t=new int[nTrain];
			}
			//local energy arrays
			double* energy_nn=new double[dist_train.size()];
			double* energy_exact=new double[dist_train.size()];
			int* natoms=new int[dist_train.size()];
			//compute energies
			clock.begin();
			for(int n=0; n<dist_train.size(); ++n){
				if(NNP_TRAIN_PRINT_STATUS>0) std::cout<<"structure-train["<<WORLD.rank()<<"]["<<n<<"]\n";
				energy_exact[n]=struc_train[n].energy();
				energy_nn[n]=NNP::energy(nnp_train.nnp_,struc_train[n]);
				natoms[n]=struc_train[n].nAtoms();
			}
			clock.end();
			time_energy_train=clock.duration();
			MPI_Barrier(WORLD.label());
			if(comm_head!=MPI_COMM_NULL){
				//gather energies
				int* dist_size=new int[BATCH.ngroup()];
				int* dist_offset=new int[BATCH.ngroup()];
				//compute dist
				MPI_Gather(&dist_train.size(),1,MPI_INT,dist_size,1,MPI_INT,0,comm_head);
				MPI_Gather(&dist_train.offset(),1,MPI_INT,dist_offset,1,MPI_INT,0,comm_head);
				//gather energies
				MPI_Gatherv(energy_exact,dist_train.size(),MPI_DOUBLE,energy_exact_t,dist_size,dist_offset,MPI_DOUBLE,0,comm_head);
				MPI_Gatherv(energy_nn,dist_train.size(),MPI_DOUBLE,energy_nn_t,dist_size,dist_offset,MPI_DOUBLE,0,comm_head);
				MPI_Gatherv(natoms,dist_train.size(),MPI_INT,natoms_t,dist_size,dist_offset,MPI_INT,0,comm_head);
				if(WORLD.rank()==0){
					//accumulate statistics
					for(int n=0; n<nTrain; ++n){
						acc1d_energy_train_n.push(std::fabs(energy_exact_t[n]-energy_nn_t[n])/natoms_t[n]);
						acc2d_energy_train.push(energy_exact_t[n]/natoms_t[n],energy_nn_t[n]/natoms_t[n]);
					}
					//normalize
					if(nnp_train.norm_){
						for(int n=0; n<nTrain; ++n){
							energy_exact_t[n]/=natoms_t[n];
							energy_nn_t[n]/=natoms_t[n];
						}
					}
					//write energies
					if(write_energy){
						const char* file="nnp_energy_train.dat";
						FILE* writer=fopen(file,"w");
						if(writer==NULL) std::cout<<"WARNING: Could not open file: \""<<file<<"\"\n";
						else{
							std::vector<std::pair<int,double> > energy_exact_pair(nTrain);
							std::vector<std::pair<int,double> > energy_nn_pair(nTrain);
							for(int n=0; n<nTrain; ++n){
								energy_exact_pair[n].first=indices_train[n];
								energy_exact_pair[n].second=energy_exact_t[n];
								energy_nn_pair[n].first=indices_train[n];
								energy_nn_pair[n].second=energy_nn_t[n];
							}
							std::sort(energy_exact_pair.begin(),energy_exact_pair.end(),compare_pair);
							std::sort(energy_nn_pair.begin(),energy_nn_pair.end(),compare_pair);
							fprintf(writer,"#STRUCTURE ENERGY_EXACT ENERGY_NN\n");
							for(int n=0; n<nTrain; ++n){
								fprintf(writer,"%s %f %f\n",files_train[n].c_str(),energy_exact_pair[n].second,energy_nn_pair[n].second);
							}
							fclose(writer);
							writer=NULL;
						}
					}
				}
				//free memory
				delete[] dist_size;
				delete[] dist_offset;
			}
			//free memory
			delete[] energy_nn;
			delete[] energy_exact;
			delete[] natoms;
			if(WORLD.rank()==0){
				delete[] energy_nn_t;
				delete[] energy_exact_t;
				delete[] natoms_t;
			}
			MPI_Barrier(WORLD.label());
		}
		//==== validation systems ====
		if(dist_val.size()>0){
			if(WORLD.rank()==0) std::cout<<"final energies - validation set\n";
			//global energy arrays
			double* energy_nn_t=NULL;
			double* energy_exact_t=NULL;
			int* natoms_t=NULL;
			if(WORLD.rank()==0){
				energy_nn_t=new double[nVal];
				energy_exact_t=new double[nVal];
				natoms_t=new int[nVal];
			}
			//local energy arrays
			double* energy_nn=new double[dist_val.size()];
			double* energy_exact=new double[dist_val.size()];
			int* natoms=new int[dist_val.size()];
			//compute energies
			clock.begin();
			for(int n=0; n<dist_val.size(); ++n){
				if(NNP_TRAIN_PRINT_STATUS>0) std::cout<<"structure-val["<<WORLD.rank()<<"]["<<n<<"]\n";
				energy_exact[n]=struc_val[n].energy();
				energy_nn[n]=NNP::energy(nnp_train.nnp_,struc_val[n]);
				natoms[n]=struc_val[n].nAtoms();
			}
			clock.end();
			time_energy_val=clock.duration();
			MPI_Barrier(WORLD.label());
			if(comm_head!=MPI_COMM_NULL){
				//gather energies
				int* dist_size=new int[BATCH.ngroup()];
				int* dist_offset=new int[BATCH.ngroup()];
				//compute dist
				MPI_Gather(&dist_val.size(),1,MPI_INT,dist_size,1,MPI_INT,0,comm_head);
				MPI_Gather(&dist_val.offset(),1,MPI_INT,dist_offset,1,MPI_INT,0,comm_head);
				//gather energies
				MPI_Gatherv(energy_exact,dist_val.size(),MPI_DOUBLE,energy_exact_t,dist_size,dist_offset,MPI_DOUBLE,0,comm_head);
				MPI_Gatherv(energy_nn,dist_val.size(),MPI_DOUBLE,energy_nn_t,dist_size,dist_offset,MPI_DOUBLE,0,comm_head);
				MPI_Gatherv(natoms,dist_val.size(),MPI_INT,natoms_t,dist_size,dist_offset,MPI_INT,0,comm_head);
				if(WORLD.rank()==0){
					//accumulate statistics
					for(int n=0; n<nVal; ++n){
						acc1d_energy_val_n.push(std::fabs(energy_exact_t[n]-energy_nn_t[n])/natoms_t[n]);
						acc2d_energy_val.push(energy_exact_t[n]/natoms_t[n],energy_nn_t[n]/natoms_t[n]);
					}
					//normalize
					if(nnp_train.norm_){
						for(int n=0; n<nVal; ++n){
							energy_exact_t[n]/=natoms_t[n];
							energy_nn_t[n]/=natoms_t[n];
						}
					}
					//write energies
					if(write_energy){
						const char* file="nnp_energy_val.dat";
						FILE* writer=fopen(file,"w");
						if(writer==NULL) std::cout<<"WARNING: Could not open file: \""<<file<<"\"\n";
						else{
							std::vector<std::pair<int,double> > energy_exact_pair(nVal);
							std::vector<std::pair<int,double> > energy_nn_pair(nVal);
							for(int n=0; n<nVal; ++n){
								energy_exact_pair[n].first=indices_val[n];
								energy_exact_pair[n].second=energy_exact_t[n];
								energy_nn_pair[n].first=indices_val[n];
								energy_nn_pair[n].second=energy_nn_t[n];
							}
							std::sort(energy_exact_pair.begin(),energy_exact_pair.end(),compare_pair);
							std::sort(energy_nn_pair.begin(),energy_nn_pair.end(),compare_pair);
							fprintf(writer,"#STRUCTURE ENERGY_EXACT ENERGY_NN\n");
							for(int n=0; n<nVal; ++n){
								fprintf(writer,"%s %f %f\n",files_val[n].c_str(),energy_exact_pair[n].second,energy_nn_pair[n].second);
							}
							fclose(writer);
							writer=NULL;
						}
					}
				}
				//free memory
				delete[] dist_size;
				delete[] dist_offset;
			}
			//free memory
			delete[] energy_nn;
			delete[] energy_exact;
			delete[] natoms;
			if(WORLD.rank()==0){
				delete[] energy_nn_t;
				delete[] energy_exact_t;
				delete[] natoms_t;
			}
			MPI_Barrier(WORLD.label());
		}
		//==== test systems ====
		if(dist_test.size()>0){
			if(WORLD.rank()==0) std::cout<<"final energies - test set\n";
			//global energy arrays
			double* energy_nn_t=NULL;
			double* energy_exact_t=NULL;
			int* natoms_t=NULL;
			if(WORLD.rank()==0){
				energy_nn_t=new double[nTest];
				energy_exact_t=new double[nTest];
				natoms_t=new int[nTest];
			}
			//local energy arrays
			double* energy_nn=new double[dist_test.size()];
			double* energy_exact=new double[dist_test.size()];
			int* natoms=new int[dist_test.size()];
			//compute energies
			clock.begin();
			for(int n=0; n<dist_test.size(); ++n){
				if(NNP_TRAIN_PRINT_STATUS>0) std::cout<<"structure-test["<<WORLD.rank()<<"]["<<n<<"]\n";
				energy_exact[n]=struc_test[n].energy();
				energy_nn[n]=NNP::energy(nnp_train.nnp_,struc_test[n]);
				natoms[n]=struc_test[n].nAtoms();
			}
			clock.end();
			time_energy_test=clock.duration();
			MPI_Barrier(WORLD.label());
			if(comm_head!=MPI_COMM_NULL){
				//gather energies
				int* dist_size=new int[BATCH.ngroup()];
				int* dist_offset=new int[BATCH.ngroup()];
				//compute dist
				MPI_Gather(&dist_test.size(),1,MPI_INT,dist_size,1,MPI_INT,0,comm_head);
				MPI_Gather(&dist_test.offset(),1,MPI_INT,dist_offset,1,MPI_INT,0,comm_head);
				//gather energies
				MPI_Gatherv(energy_exact,dist_test.size(),MPI_DOUBLE,energy_exact_t,dist_size,dist_offset,MPI_DOUBLE,0,comm_head);
				MPI_Gatherv(energy_nn,dist_test.size(),MPI_DOUBLE,energy_nn_t,dist_size,dist_offset,MPI_DOUBLE,0,comm_head);
				MPI_Gatherv(natoms,dist_test.size(),MPI_INT,natoms_t,dist_size,dist_offset,MPI_INT,0,comm_head);
				if(WORLD.rank()==0){
					//accumulate statistics
					for(int n=0; n<nTest; ++n){
						acc1d_energy_test_n.push(std::fabs(energy_exact_t[n]-energy_nn_t[n])/natoms_t[n]);
						acc2d_energy_test.push(energy_exact_t[n]/natoms_t[n],energy_nn_t[n]/natoms_t[n]);
					}
					//normalize
					if(nnp_train.norm_){
						for(int n=0; n<nTest; ++n){
							energy_exact_t[n]/=natoms_t[n];
							energy_nn_t[n]/=natoms_t[n];
						}
					}
					//write energies
					if(write_energy){
						const char* file="nnp_energy_test.dat";
						FILE* writer=fopen(file,"w");
						if(writer==NULL) std::cout<<"WARNING: Could not open file: \""<<file<<"\"\n";
						else{
							std::vector<std::pair<int,double> > energy_exact_pair(nTest);
							std::vector<std::pair<int,double> > energy_nn_pair(nTest);
							for(int n=0; n<nTest; ++n){
								energy_exact_pair[n].first=indices_test[n];
								energy_exact_pair[n].second=energy_exact_t[n];
								energy_nn_pair[n].first=indices_test[n];
								energy_nn_pair[n].second=energy_nn_t[n];
							}
							std::sort(energy_exact_pair.begin(),energy_exact_pair.end(),compare_pair);
							std::sort(energy_nn_pair.begin(),energy_nn_pair.end(),compare_pair);
							fprintf(writer,"#STRUCTURE ENERGY_EXACT ENERGY_NN\n");
							for(int n=0; n<nTest; ++n){
								fprintf(writer,"%s %f %f\n",files_test[n].c_str(),energy_exact_pair[n].second,energy_nn_pair[n].second);
							}
							fclose(writer);
							writer=NULL;
						}
					}
				}
				//free memory
				delete[] dist_size;
				delete[] dist_offset;
			}
			//free memory
			delete[] energy_nn;
			delete[] energy_exact;
			delete[] natoms;
			if(WORLD.rank()==0){
				delete[] energy_nn_t;
				delete[] energy_exact_t;
				delete[] natoms_t;
			}
			MPI_Barrier(WORLD.label());
		}
		
		//======== write the ewald energies ========
		//==== training systems ====
		if(dist_train.size()>0 && nnp_train.charge_ && write_ewald){
			if(WORLD.rank()==0) std::cout<<"writing ewald - training set\n";
			//global energy arrays
			double* ewald_t=NULL;
			if(WORLD.rank()==0) ewald_t=new double[nTrain];
			//local energy arrays
			double* ewald_l=new double[dist_train.size()];
			for(int i=0; i<dist_train.size(); ++i) ewald_l[i]=struc_train[i].ewald();
			//gather energies
			if(comm_head!=MPI_COMM_NULL){
				//compute dist
				int* dist_size=new int[BATCH.ngroup()];
				int* dist_offset=new int[BATCH.ngroup()];
				MPI_Gather(&dist_train.size(),1,MPI_INT,dist_size,1,MPI_INT,0,comm_head);
				MPI_Gather(&dist_train.offset(),1,MPI_INT,dist_offset,1,MPI_INT,0,comm_head);
				//gather energies
				MPI_Gatherv(ewald_l,dist_train.size(),MPI_DOUBLE,ewald_t,dist_size,dist_offset,MPI_DOUBLE,0,comm_head);
				//write energies
				if(WORLD.rank()==0){
					const char* file="nnp_ewald_train.dat";
					FILE* writer=fopen(file,"w");
					if(writer==NULL) std::cout<<"WARNING: Could not open file: \""<<file<<"\"\n";
					else{
						std::vector<std::pair<int,double> > ewald_pair(nTrain);
						for(int n=0; n<nTrain; ++n){
							ewald_pair[n].first=indices_train[n];
							ewald_pair[n].second=ewald_t[n];
						}
						std::sort(ewald_pair.begin(),ewald_pair.end(),compare_pair);
						fprintf(writer,"#STRUCTURE ENERGY_EWALD\n");
						for(int n=0; n<nTrain; ++n){
							fprintf(writer,"%s %f\n",files_train[n].c_str(),ewald_pair[n].second);
						}
						fclose(writer);
						writer=NULL;
					}
				}
				//free memory
				delete[] dist_size;
				delete[] dist_offset;
			}
			//free memory
			delete[] ewald_l;
			if(WORLD.rank()==0) delete[] ewald_t;
		}
		//==== validation systems ====
		if(dist_val.size()>0 && nnp_train.charge_ && write_ewald){
			if(WORLD.rank()==0) std::cout<<"writing ewald - validation set\n";
			//global energy arrays
			double* ewald_t=NULL;
			if(WORLD.rank()==0) ewald_t=new double[nVal];
			//local energy arrays
			double* ewald_l=new double[dist_val.size()];
			for(int i=0; i<dist_val.size(); ++i) ewald_l[i]=struc_val[i].ewald();
			//gather energies
			if(comm_head!=MPI_COMM_NULL){
				//compute dist
				int* dist_size=new int[BATCH.ngroup()];
				int* dist_offset=new int[BATCH.ngroup()];
				MPI_Gather(&dist_val.size(),1,MPI_INT,dist_size,1,MPI_INT,0,comm_head);
				MPI_Gather(&dist_val.offset(),1,MPI_INT,dist_offset,1,MPI_INT,0,comm_head);
				//gather energies
				MPI_Gatherv(ewald_l,dist_val.size(),MPI_DOUBLE,ewald_t,dist_size,dist_offset,MPI_DOUBLE,0,comm_head);
				//write energies
				if(WORLD.rank()==0){
					const char* file="nnp_ewald_val.dat";
					FILE* writer=fopen(file,"w");
					if(writer==NULL) std::cout<<"WARNING: Could not open file: \""<<file<<"\"\n";
					else{
						std::vector<std::pair<int,double> > ewald_pair(nVal);
						for(int n=0; n<nVal; ++n){
							ewald_pair[n].first=indices_val[n];
							ewald_pair[n].second=ewald_t[n];
						}
						std::sort(ewald_pair.begin(),ewald_pair.end(),compare_pair);
						fprintf(writer,"#STRUCTURE ENERGY_EWALD\n");
						for(int n=0; n<nVal; ++n){
							fprintf(writer,"%s %f\n",files_val[n].c_str(),ewald_pair[n].second);
						}
						fclose(writer);
						writer=NULL;
					}
				}
				//free memory
				delete[] dist_size;
				delete[] dist_offset;
			}
			//free memory
			delete[] ewald_l;
			if(WORLD.rank()==0) delete[] ewald_t;
		}
		//==== test systems ====
		if(dist_test.size()>0 && nnp_train.charge_ && write_ewald){
			if(WORLD.rank()==0) std::cout<<"writing ewald - test set\n";
			//global energy arrays
			double* ewald_t=NULL;
			if(WORLD.rank()==0) ewald_t=new double[nTest];
			//local energy arrays
			double* ewald_l=new double[dist_test.size()];
			for(int i=0; i<dist_test.size(); ++i) ewald_l[i]=struc_test[i].ewald();
			//gather energies
			if(comm_head!=MPI_COMM_NULL){
				//compute dist
				int* dist_size=new int[BATCH.ngroup()];
				int* dist_offset=new int[BATCH.ngroup()];
				MPI_Gather(&dist_test.size(),1,MPI_INT,dist_size,1,MPI_INT,0,comm_head);
				MPI_Gather(&dist_test.offset(),1,MPI_INT,dist_offset,1,MPI_INT,0,comm_head);
				//gather energies
				MPI_Gatherv(ewald_l,dist_test.size(),MPI_DOUBLE,ewald_t,dist_size,dist_offset,MPI_DOUBLE,0,comm_head);
				//write energies
				if(WORLD.rank()==0){
					const char* file="nnp_ewald_test.dat";
					FILE* writer=fopen(file,"w");
					if(writer==NULL) std::cout<<"WARNING: Could not open file: \""<<file<<"\"\n";
					else{
						std::vector<std::pair<int,double> > ewald_pair(nTest);
						for(int n=0; n<nTest; ++n){
							ewald_pair[n].first=indices_test[n];
							ewald_pair[n].second=ewald_t[n];
						}
						std::sort(ewald_pair.begin(),ewald_pair.end(),compare_pair);
						fprintf(writer,"#STRUCTURE ENERGY_EWALD\n");
						for(int n=0; n<nTest; ++n){
							fprintf(writer,"%s %f\n",files_test[n].c_str(),ewald_pair[n].second);
						}
						fclose(writer);
						writer=NULL;
					}
				}
				//free memory
				delete[] dist_size;
				delete[] dist_offset;
			}
			//free memory
			delete[] ewald_l;
			if(WORLD.rank()==0) delete[] ewald_t;
		}
		
		//======== compute the final forces ========
		//==== training structures ====
		if(dist_train.size()>0 && nnp_train.force_){
			if(WORLD.rank()==0) std::cout<<"computing final forces - training set\n";
			//global force arrays
			double* forces_nn_t=NULL;
			double* forces_exact_t=NULL;
			int* natoms_t=NULL;
			if(WORLD.rank()==0) natoms_t=new int[nTrain];
			//local force arrays
			int count=0,ndata=0,ndata_t=0;
			int* natoms=new int[dist_train.size()];
			double* forces_nn=NULL;
			double* forces_exact=NULL;
			std::vector<std::vector<Eigen::Vector3d> > forces_exact_v(dist_train.size());
			std::vector<std::vector<Eigen::Vector3d> > forces_nn_v(dist_train.size());
			//compute forces
			for(int n=0; n<dist_train.size(); ++n){
				forces_exact_v[n].resize(struc_train[n].nAtoms());
				for(int j=0; j<struc_train[n].nAtoms(); ++j) forces_exact_v[n][j]=struc_train[n].force(j);
				ndata+=forces_exact_v[n].size()*3;
			}
			clock.begin();
			for(int n=0; n<dist_train.size(); ++n){
				if(NNP_TRAIN_PRINT_STATUS>0) std::cout<<"structure-train["<<n<<"]\n";
				NeighborList nlist(struc_train[n],nnp_train.nnp_.rc());
				NNP::force(nnp_train.nnp_,struc_train[n],nlist);
			}
			clock.end();
			time_force_train=clock.duration();
			for(int n=0; n<dist_train.size(); ++n){
				forces_nn_v[n].resize(struc_train[n].nAtoms());
				for(int j=0; j<struc_train[n].nAtoms(); ++j) forces_nn_v[n][j]=struc_train[n].force(j);
			}
			MPI_Barrier(WORLD.label());
			//gather forces
			if(comm_head!=MPI_COMM_NULL){
				//compute dist
				int* dist_size=new int[BATCH.ngroup()];//the number of structures in each batch group
				int* dist_offset=new int[BATCH.ngroup()];//the offset of each group of structurs in each batch group
				MPI_Gather(&dist_train.size(),1,MPI_INT,dist_size,1,MPI_INT,0,comm_head);
				MPI_Gather(&dist_train.offset(),1,MPI_INT,dist_offset,1,MPI_INT,0,comm_head);
				//gather ndata
				int* dist_data_size=new int[BATCH.ngroup()];
				int* dist_data_offset=new int[BATCH.ngroup()];
				MPI_Reduce(&ndata,&ndata_t,1,MPI_INT,MPI_SUM,0,comm_head);
				MPI_Gather(&ndata,1,MPI_INT,dist_data_size,1,MPI_INT,0,comm_head);
				dist_data_offset[0]=0;
				for(int i=1; i<BATCH.ngroup(); ++i) dist_data_offset[i]=dist_data_offset[i-1]+dist_data_size[i-1];
				//pack forces into 1D arrays
				forces_nn=new double[ndata];
				forces_exact=new double[ndata];
				if(WORLD.rank()==0){
					forces_nn_t=new double[ndata_t];
					forces_exact_t=new double[ndata_t];
				}
				count=0;
				for(int n=0; n<forces_exact_v.size(); ++n){
					for(int j=0; j<forces_exact_v[n].size(); ++j){
						forces_exact[count++]=forces_exact_v[n][j][0];
						forces_exact[count++]=forces_exact_v[n][j][1];
						forces_exact[count++]=forces_exact_v[n][j][2];
					}
				}
				count=0;
				for(int n=0; n<forces_nn_v.size(); ++n){
					for(int j=0; j<forces_nn_v[n].size(); ++j){
						forces_nn[count++]=forces_nn_v[n][j][0];
						forces_nn[count++]=forces_nn_v[n][j][1];
						forces_nn[count++]=forces_nn_v[n][j][2];
					}
				}
				//gather 1D force arrays
				MPI_Gatherv(forces_nn,ndata,MPI_DOUBLE,forces_nn_t,dist_data_size,dist_data_offset,MPI_DOUBLE,0,comm_head);
				MPI_Gatherv(forces_exact,ndata,MPI_DOUBLE,forces_exact_t,dist_data_size,dist_data_offset,MPI_DOUBLE,0,comm_head);
				MPI_Gatherv(natoms,dist_train.size(),MPI_INT,natoms_t,dist_size,dist_offset,MPI_INT,0,comm_head);
				//accumulate statistics
				if(WORLD.rank()==0){
					for(int i=0; i<ndata_t; i+=3){
						Eigen::Vector3d f_exact,f_nn;
						f_exact[0]=forces_exact_t[i+0];
						f_exact[1]=forces_exact_t[i+1];
						f_exact[2]=forces_exact_t[i+2];
						f_nn[0]=forces_nn_t[i+0];
						f_nn[1]=forces_nn_t[i+1];
						f_nn[2]=forces_nn_t[i+2];
						acc1d_force_train_a.push((f_exact-f_nn).norm());
						acc2d_forcex_train.push(f_exact[0],f_nn[0]);
						acc2d_forcey_train.push(f_exact[1],f_nn[1]);
						acc2d_forcez_train.push(f_exact[2],f_nn[2]);
					}
					//write forces
					if(write_force){
						const char* file="nnp_force_train.dat";
						FILE* writer=fopen(file,"w");
						if(writer==NULL) std::cout<<"WARNING: Could not open file: \""<<file<<"\"\n";
						else{
							for(int i=0; i<ndata_t; i+=3){
								Eigen::Vector3d f_exact,f_nn;
								f_exact[0]=forces_exact_t[i+0];
								f_exact[1]=forces_exact_t[i+1];
								f_exact[2]=forces_exact_t[i+2];
								f_nn[0]=forces_nn_t[i+0];
								f_nn[1]=forces_nn_t[i+1];
								f_nn[2]=forces_nn_t[i+2];
								fprintf(writer,"%f %f %f %f %f %f\n",
									f_exact[0],f_exact[1],f_exact[2],
									f_nn[0],f_nn[1],f_nn[2]
								);
							}
							fclose(writer); writer=NULL;
						}
					}
				}
				//free memory
				delete[] dist_size;
				delete[] dist_offset;
				delete[] dist_data_size;
				delete[] dist_data_offset;
				delete[] forces_exact;
				delete[] forces_nn;
				if(WORLD.rank()==0){
					delete[] forces_nn_t;
					delete[] forces_exact_t;
				}
			}
			MPI_Barrier(WORLD.label());
			if(WORLD.rank()==0) delete[] natoms_t;
			delete[] natoms;
		}
		//==== validation structures ====
		if(dist_val.size()>0 && nnp_train.force_){
			if(WORLD.rank()==0) std::cout<<"computing final forces - validation set\n";
			//global force arrays
			double* forces_nn_t=NULL;
			double* forces_exact_t=NULL;
			//local force arrays
			int count=0,ndata=0,ndata_t=0;
			int* natoms=new int[dist_val.size()];
			int* natoms_t=NULL;
			double* forces_nn=NULL;
			double* forces_exact=NULL;
			std::vector<std::vector<Eigen::Vector3d> > forces_exact_v(dist_val.size());
			std::vector<std::vector<Eigen::Vector3d> > forces_nn_v(dist_val.size());
			//compute forces
			for(int n=0; n<dist_val.size(); ++n){
				forces_exact_v[n].resize(struc_val[n].nAtoms());
				for(int j=0; j<struc_val[n].nAtoms(); ++j) forces_exact_v[n][j]=struc_val[n].force(j);
				ndata+=forces_exact_v[n].size()*3;
			}
			clock.begin();
			for(int n=0; n<dist_val.size(); ++n){
				if(NNP_TRAIN_PRINT_STATUS>0) std::cout<<"structure-val["<<n<<"]\n";
				NeighborList nlist(struc_val[n],nnp_train.nnp_.rc());
				NNP::force(nnp_train.nnp_,struc_val[n],nlist);
			}
			clock.end();
			time_force_val=clock.duration();
			for(int n=0; n<dist_val.size(); ++n){
				forces_nn_v[n].resize(struc_val[n].nAtoms());
				for(int j=0; j<struc_val[n].nAtoms(); ++j) forces_nn_v[n][j]=struc_val[n].force(j);
			}
			MPI_Barrier(WORLD.label());
			//gather forces
			if(comm_head!=MPI_COMM_NULL){
				//compute dist
				int* dist_size=new int[BATCH.ngroup()];//the number of structures in each batch
				int* dist_offset=new int[BATCH.ngroup()];//the offset of each group of structurs in each batch
				MPI_Gather(&dist_val.size(),1,MPI_INT,dist_size,1,MPI_INT,0,comm_head);
				MPI_Gather(&dist_val.offset(),1,MPI_INT,dist_offset,1,MPI_INT,0,comm_head);
				//gather forces
				int* dist_data_size=new int[BATCH.ngroup()];
				int* dist_data_offset=new int[BATCH.ngroup()];
				MPI_Reduce(&ndata,&ndata_t,1,MPI_INT,MPI_SUM,0,comm_head);
				MPI_Gather(&ndata,1,MPI_INT,dist_data_size,1,MPI_INT,0,comm_head);
				dist_data_offset[0]=0;
				for(int i=1; i<BATCH.ngroup(); ++i) dist_data_offset[i]=dist_data_offset[i-1]+dist_data_size[i-1];
				forces_nn=new double[ndata];
				forces_exact=new double[ndata];
				if(WORLD.rank()==0){
					forces_nn_t=new double[ndata_t];
					forces_exact_t=new double[ndata_t];
					natoms_t=new int[nTrain];
				}
				count=0;
				for(int n=0; n<forces_exact_v.size(); ++n){
					for(int j=0; j<forces_exact_v[n].size(); ++j){
						forces_exact[count++]=forces_exact_v[n][j][0];
						forces_exact[count++]=forces_exact_v[n][j][1];
						forces_exact[count++]=forces_exact_v[n][j][2];
					}
				}
				count=0;
				for(int n=0; n<forces_nn_v.size(); ++n){
					for(int j=0; j<forces_nn_v[n].size(); ++j){
						forces_nn[count++]=forces_nn_v[n][j][0];
						forces_nn[count++]=forces_nn_v[n][j][1];
						forces_nn[count++]=forces_nn_v[n][j][2];
					}
				}
				MPI_Gatherv(forces_nn,ndata,MPI_DOUBLE,forces_nn_t,dist_data_size,dist_data_offset,MPI_DOUBLE,0,comm_head);
				MPI_Gatherv(forces_exact,ndata,MPI_DOUBLE,forces_exact_t,dist_data_size,dist_data_offset,MPI_DOUBLE,0,comm_head);
				MPI_Gatherv(natoms,dist_val.size(),MPI_INT,natoms_t,dist_size,dist_offset,MPI_INT,0,comm_head);
				//accumulate statistics
				if(WORLD.rank()==0){
					for(int i=0; i<ndata_t; i+=3){
						Eigen::Vector3d f_exact,f_nn;
						f_exact[0]=forces_exact_t[i+0];
						f_exact[1]=forces_exact_t[i+1];
						f_exact[2]=forces_exact_t[i+2];
						f_nn[0]=forces_nn_t[i+0];
						f_nn[1]=forces_nn_t[i+1];
						f_nn[2]=forces_nn_t[i+2];
						acc1d_force_val_a.push((f_exact-f_nn).norm());
						acc2d_forcex_val.push(f_exact[0],f_nn[0]);
						acc2d_forcey_val.push(f_exact[1],f_nn[1]);
						acc2d_forcez_val.push(f_exact[2],f_nn[2]);
					}
					//write forces
					if(write_force){
						const char* file="nnp_force_val.dat";
						FILE* writer=fopen(file,"w");
						if(writer==NULL) std::cout<<"WARNING: Could not open file: \""<<file<<"\"\n";
						else{
							for(int i=0; i<ndata_t; i+=3){
								Eigen::Vector3d f_exact,f_nn;
								f_exact[0]=forces_exact_t[i+0];
								f_exact[1]=forces_exact_t[i+1];
								f_exact[2]=forces_exact_t[i+2];
								f_nn[0]=forces_nn_t[i+0];
								f_nn[1]=forces_nn_t[i+1];
								f_nn[2]=forces_nn_t[i+2];
								fprintf(writer,"%f %f %f %f %f %f\n",
									f_exact[0],f_exact[1],f_exact[2],
									f_nn[0],f_nn[1],f_nn[2]
								);
							}
							fclose(writer); writer=NULL;
						}
					}
				}
				//free memory
				delete[] dist_size;
				delete[] dist_offset;
				delete[] dist_data_size;
				delete[] dist_data_offset;
				delete[] forces_exact;
				delete[] forces_nn;
				if(WORLD.rank()==0){
					delete[] forces_nn_t;
					delete[] forces_exact_t;
					delete[] natoms_t;
				}
			}
			MPI_Barrier(WORLD.label());
			delete[] natoms;
		}
		//==== test structures ====
		if(dist_test.size()>0 && nnp_train.force_){
			if(WORLD.rank()==0) std::cout<<"computing final forces - test set\n";
			//global force arrays
			double* forces_nn_t=NULL;
			double* forces_exact_t=NULL;
			//local force arrays
			int count=0,ndata=0,ndata_t=0;
			int* natoms=new int[dist_test.size()];
			int* natoms_t=NULL;
			double* forces_nn=NULL;
			double* forces_exact=NULL;
			std::vector<std::vector<Eigen::Vector3d> > forces_exact_v(dist_test.size());
			std::vector<std::vector<Eigen::Vector3d> > forces_nn_v(dist_test.size());
			//compute forces
			for(int n=0; n<dist_test.size(); ++n){
				forces_exact_v[n].resize(struc_test[n].nAtoms());
				for(int j=0; j<struc_test[n].nAtoms(); ++j) forces_exact_v[n][j]=struc_test[n].force(j);
				ndata+=forces_exact_v[n].size()*3;
			}
			clock.begin();
			for(int n=0; n<dist_test.size(); ++n){
				if(NNP_TRAIN_PRINT_STATUS>0) std::cout<<"structure-val["<<n<<"]\n";
				NeighborList nlist(struc_test[n],nnp_train.nnp_.rc());
				NNP::force(nnp_train.nnp_,struc_test[n],nlist);
			}
			clock.end();
			time_force_test=clock.duration();
			for(int n=0; n<dist_test.size(); ++n){
				forces_nn_v[n].resize(struc_test[n].nAtoms());
				for(int j=0; j<struc_test[n].nAtoms(); ++j) forces_nn_v[n][j]=struc_test[n].force(j);
			}
			MPI_Barrier(WORLD.label());
			//gather forces
			if(comm_head!=MPI_COMM_NULL){
				//compute dist
				int* dist_size=new int[BATCH.ngroup()];//the number of structures in each batch
				int* dist_offset=new int[BATCH.ngroup()];//the offset of each group of structurs in each batch
				MPI_Gather(&dist_test.size(),1,MPI_INT,dist_size,1,MPI_INT,0,comm_head);
				MPI_Gather(&dist_test.offset(),1,MPI_INT,dist_offset,1,MPI_INT,0,comm_head);
				//gather forces
				int* dist_data_size=new int[BATCH.ngroup()];
				int* dist_data_offset=new int[BATCH.ngroup()];
				MPI_Reduce(&ndata,&ndata_t,1,MPI_INT,MPI_SUM,0,comm_head);
				MPI_Gather(&ndata,1,MPI_INT,dist_data_size,1,MPI_INT,0,comm_head);
				dist_data_offset[0]=0;
				for(int i=1; i<BATCH.ngroup(); ++i) dist_data_offset[i]=dist_data_offset[i-1]+dist_data_size[i-1];
				forces_nn=new double[ndata];
				forces_exact=new double[ndata];
				if(WORLD.rank()==0){
					forces_nn_t=new double[ndata_t];
					forces_exact_t=new double[ndata_t];
					natoms_t=new int[nTrain];
				}
				count=0;
				for(int n=0; n<forces_exact_v.size(); ++n){
					for(int j=0; j<forces_exact_v[n].size(); ++j){
						forces_exact[count++]=forces_exact_v[n][j][0];
						forces_exact[count++]=forces_exact_v[n][j][1];
						forces_exact[count++]=forces_exact_v[n][j][2];
					}
				}
				count=0;
				for(int n=0; n<forces_nn_v.size(); ++n){
					for(int j=0; j<forces_nn_v[n].size(); ++j){
						forces_nn[count++]=forces_nn_v[n][j][0];
						forces_nn[count++]=forces_nn_v[n][j][1];
						forces_nn[count++]=forces_nn_v[n][j][2];
					}
				}
				MPI_Gatherv(forces_nn,ndata,MPI_DOUBLE,forces_nn_t,dist_data_size,dist_data_offset,MPI_DOUBLE,0,comm_head);
				MPI_Gatherv(forces_exact,ndata,MPI_DOUBLE,forces_exact_t,dist_data_size,dist_data_offset,MPI_DOUBLE,0,comm_head);
				MPI_Gatherv(natoms,dist_test.size(),MPI_INT,natoms_t,dist_size,dist_offset,MPI_INT,0,comm_head);
				//accumulate statistics
				if(WORLD.rank()==0){
					for(int i=0; i<ndata_t; i+=3){
						Eigen::Vector3d f_exact,f_nn;
						f_exact[0]=forces_exact_t[i+0];
						f_exact[1]=forces_exact_t[i+1];
						f_exact[2]=forces_exact_t[i+2];
						f_nn[0]=forces_nn_t[i+0];
						f_nn[1]=forces_nn_t[i+1];
						f_nn[2]=forces_nn_t[i+2];
						acc1d_force_test_a.push((f_exact-f_nn).norm());
						acc2d_forcex_test.push(f_exact[0],f_nn[0]);
						acc2d_forcey_test.push(f_exact[1],f_nn[1]);
						acc2d_forcez_test.push(f_exact[2],f_nn[2]);
					}
					//write forces
					if(write_force){
						const char* file="nnp_force_test.dat";
						FILE* writer=fopen(file,"w");
						if(writer==NULL) std::cout<<"WARNING: Could not open file: \""<<file<<"\"\n";
						else{
							for(int i=0; i<ndata_t; i+=3){
								Eigen::Vector3d f_exact,f_nn;
								f_exact[0]=forces_exact_t[i+0];
								f_exact[1]=forces_exact_t[i+1];
								f_exact[2]=forces_exact_t[i+2];
								f_nn[0]=forces_nn_t[i+0];
								f_nn[1]=forces_nn_t[i+1];
								f_nn[2]=forces_nn_t[i+2];
								fprintf(writer,"%f %f %f %f %f %f\n",
									f_exact[0],f_exact[1],f_exact[2],
									f_nn[0],f_nn[1],f_nn[2]
								);
							}
							fclose(writer); writer=NULL;
						}
					}
				}
				//free memory
				delete[] dist_size;
				delete[] dist_offset;
				delete[] dist_data_size;
				delete[] dist_data_offset;
				delete[] forces_exact;
				delete[] forces_nn;
				if(WORLD.rank()==0){
					delete[] forces_nn_t;
					delete[] forces_exact_t;
					delete[] natoms_t;
				}
			}
			MPI_Barrier(WORLD.label());
			delete[] natoms;
		}
		
		//======== write the inputs ========
		//==== training structures ====
		if(dist_train.size()>0 && write_input){
			const char* file_inputs_train="nnp_inputs_train.dat";
			if(WORLD.rank()==0){
				FILE* writer=fopen(file_inputs_train,"w");
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
					fclose(writer);
					writer=NULL;
				} else std::cout<<"WARNING: Could not open inputs file for training structures\n";
			}
			MPI_Barrier(WORLD.label());
			for(int ii=1; ii<WORLD.size(); ++ii){
				if(WORLD.rank()==ii){
					FILE* writer=fopen(file_inputs_train,"a");
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
						fclose(writer);
						writer=NULL;
					} else std::cout<<"WARNING: Could not open inputs file for training structures\n";
				}
				MPI_Barrier(WORLD.label());
			}
		}
		//==== validation structures ====
		if(dist_val.size()>0 && write_input){
			const char* file_inputs_val="nnp_inputs_val.dat";
			if(WORLD.rank()==0){
				FILE* writer=fopen(file_inputs_val,"w");
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
					fclose(writer);
					writer=NULL;
				} else std::cout<<"WARNING: Could not open inputs file for validation structures\n";
			}
			MPI_Barrier(WORLD.label());
			for(int ii=1; ii<WORLD.size(); ++ii){
				if(WORLD.rank()==ii){
					FILE* writer=fopen(file_inputs_val,"a");
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
						fclose(writer);
						writer=NULL;
					} else std::cout<<"WARNING: Could not open inputs file for validation structures\n";
				}
				MPI_Barrier(WORLD.label());
			}
		}
		//==== testing structures ====
		if(dist_test.size()>0 && write_input){
			const char* file_inputs_test="nnp_inputs_test.dat";
			if(WORLD.rank()==0){
				FILE* writer=fopen(file_inputs_test,"w");
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
					fclose(writer);
					writer=NULL;
				} else std::cout<<"WARNING: Could not open inputs file for testing structures\n";
			}
			MPI_Barrier(WORLD.label());
			for(int ii=1; ii<WORLD.size(); ++ii){
				if(WORLD.rank()==ii){
					FILE* writer=fopen(file_inputs_test,"a");
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
						fclose(writer);
						writer=NULL;
					} else std::cout<<"WARNING: Could not open inputs file for testing structures\n";
				}
				MPI_Barrier(WORLD.label());
			}
		}
		
		//======== compute the correlation between inputs ========
		//==== training structures ====
		if(dist_train.size()>0 && write_corr){
			if(WORLD.rank()==0){
				acc2d_inp_train.resize(nnp_train.nnp_.nspecies());
				for(int i=0; i<nnp_train.nnp_.nspecies(); ++i){
					acc2d_inp_train[i].resize(nnp_train.nnp_.nnh(i).nInput());
					for(int j=0; j<nnp_train.nnp_.nnh(i).nInput(); ++j){
						acc2d_inp_train[i][j].resize(nnp_train.nnp_.nnh(i).nInput());
					}
				}
				for(int n=0; n<dist_train.size(); ++n){
					for(int i=0; i<struc_train[n].nAtoms(); ++i){
						const int id=nnp_train.nnp_.index(struc_train[n].name(i));
						for(int j=0; j<struc_train[n].symm(i).size(); ++j){
							for(int k=0; k<struc_train[n].symm(i).size(); ++k){
								acc2d_inp_train[id][j][k].push(struc_train[n].symm(i)[j],struc_train[n].symm(i)[k]);
							}
						}
					}
				}
			}
			for(int nn=1; nn<WORLD.size(); ++nn){
				if(WORLD.rank()==0 || WORLD.rank()==nn){
					//make sure "nn" and "0" have the same number of structures to loop over
					int subset_train_loc=dist_train.size();
					if(WORLD.rank()==nn) MPI_Send(&subset_train_loc,1,MPI_INT,0,0,WORLD.label());
					else if(WORLD.rank()==0) MPI_Recv(&subset_train_loc,1,MPI_INT,nn,0,WORLD.label(),MPI_STATUS_IGNORE);
					//loop over all structures on "nn"
					for(int n=0; n<subset_train_loc; ++n){
						//send the structure on "nn" to "0"
						Structure struc_loc;
						char* arr=NULL;
						int size=0;
						//send the size to "0"
						if(WORLD.rank()==nn){
							struc_loc=struc_train[n];
							size=serialize::nbytes(struc_loc);
						}
						if(WORLD.rank()==nn) MPI_Send(&size,1,MPI_INT,0,0,WORLD.label());
						else if(WORLD.rank()==0) MPI_Recv(&size,1,MPI_INT,nn,0,WORLD.label(),MPI_STATUS_IGNORE);
						arr=new char[size];
						if(WORLD.rank()==nn) serialize::pack(struc_loc,arr);
						if(WORLD.rank()==nn) MPI_Send(arr,size,MPI_CHAR,0,0,WORLD.label());
						else if(WORLD.rank()==0) MPI_Recv(arr,size,MPI_CHAR,nn,0,WORLD.label(),MPI_STATUS_IGNORE);
						if(WORLD.rank()==0) serialize::unpack(struc_loc,arr);
						delete[] arr;
						//compute correlation
						if(WORLD.rank()==0){
							for(int i=0; i<struc_loc.nAtoms(); ++i){
								const int id=nnp_train.nnp_.index(struc_loc.name(i));
								for(int j=0; j<struc_loc.symm(i).size(); ++j){
									for(int k=0; k<struc_loc.symm(i).size(); ++k){
										acc2d_inp_train[id][j][k].push(struc_loc.symm(i)[j],struc_loc.symm(i)[k]);
									}
								}
							}
						}
					}
				}
				MPI_Barrier(WORLD.label());
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
			MPI_Reduce(&time_symm_train,&tmp,1,MPI_DOUBLE,MPI_SUM,0,WORLD.label()); if(WORLD.rank()==0) time_symm_train=tmp/WORLD.size();
			MPI_Reduce(&time_energy_train,&tmp,1,MPI_DOUBLE,MPI_SUM,0,WORLD.label()); if(WORLD.rank()==0) time_energy_train=tmp/WORLD.size();
			MPI_Reduce(&time_force_train,&tmp,1,MPI_DOUBLE,MPI_SUM,0,WORLD.label()); if(WORLD.rank()==0) time_force_train=tmp/WORLD.size();
			MPI_Reduce(&time_symm_val,&tmp,1,MPI_DOUBLE,MPI_SUM,0,WORLD.label()); if(WORLD.rank()==0) time_symm_val=tmp/WORLD.size();
			MPI_Reduce(&time_energy_val,&tmp,1,MPI_DOUBLE,MPI_SUM,0,WORLD.label()); if(WORLD.rank()==0) time_energy_val=tmp/WORLD.size();
			MPI_Reduce(&time_force_val,&tmp,1,MPI_DOUBLE,MPI_SUM,0,WORLD.label()); if(WORLD.rank()==0) time_force_val=tmp/WORLD.size();
			MPI_Reduce(&time_symm_test,&tmp,1,MPI_DOUBLE,MPI_SUM,0,WORLD.label()); if(WORLD.rank()==0) time_symm_test=tmp/WORLD.size();
			MPI_Reduce(&time_energy_test,&tmp,1,MPI_DOUBLE,MPI_SUM,0,WORLD.label()); if(WORLD.rank()==0) time_energy_test=tmp/WORLD.size();
			MPI_Reduce(&time_force_test,&tmp,1,MPI_DOUBLE,MPI_SUM,0,WORLD.label()); if(WORLD.rank()==0) time_force_test=tmp/WORLD.size();
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
		std::cout<<"\tERROR - AVG - ENERGY/ATOM = "<<acc1d_energy_train_n.avg()<<"\n";
		std::cout<<"\tERROR - DEV - ENERGY/ATOM = "<<std::sqrt(acc1d_energy_train_n.var())<<"\n";
		std::cout<<"\tERROR - MAX - ENERGY/ATOM = "<<acc1d_energy_train_n.max()<<"\n";
		std::cout<<"\tM/R2 - ENERGY/ATOM = "<<acc2d_energy_train.m()<<" "<<acc2d_energy_train.r2()<<"\n";
		if(nnp_train.force_){
		std::cout<<"FORCE:\n";
		std::cout<<"\tERROR - AVG - FORCE = "<<acc1d_force_train_a.avg()<<"\n";
		std::cout<<"\tERROR - DEV - FORCE = "<<std::sqrt(acc1d_force_train_a.var())<<"\n";
		std::cout<<"\tERROR - MAX - FORCE = "<<acc1d_force_train_a.max()<<"\n";
		std::cout<<"\tM  (FX,FY,FZ) = "<<acc2d_forcex_train.m() <<" "<<acc2d_forcey_train.m() <<" "<<acc2d_forcez_train.m() <<"\n";
		std::cout<<"\tR2 (FX,FY,FZ) = "<<acc2d_forcex_train.r2()<<" "<<acc2d_forcey_train.r2()<<" "<<acc2d_forcez_train.r2()<<"\n";
		}
		std::cout<<print::title("STATISTICS - ERROR - TRAINING",strbuf)<<"\n";
		std::cout<<print::buf(strbuf)<<"\n";
		}
		
		//======== print the error statistics - validation ========
		if(WORLD.rank()==0 && dist_val.size()>0 && mode==Mode::TRAIN){
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("STATISTICS - ERROR - VALIDATION",strbuf)<<"\n";
		std::cout<<"\tERROR - AVG - ENERGY/ATOM = "<<acc1d_energy_val_n.avg()<<"\n";
		std::cout<<"\tERROR - DEV - ENERGY/ATOM = "<<std::sqrt(acc1d_energy_val_n.var())<<"\n";
		std::cout<<"\tERROR - MAX - ENERGY/ATOM = "<<acc1d_energy_val_n.max()<<"\n";
		std::cout<<"\tM/R2 - ENERGY/ATOM = "<<acc2d_energy_val.m()<<" "<<acc2d_energy_val.r2()<<"\n";
		if(nnp_train.force_){
		std::cout<<"FORCE:\n";
		std::cout<<"\tERROR - AVG - FORCE = "<<acc1d_force_val_a.avg()<<"\n";
		std::cout<<"\tERROR - DEV - FORCE = "<<std::sqrt(acc1d_force_val_a.var())<<"\n";
		std::cout<<"\tERROR - MAX - FORCE = "<<acc1d_force_val_a.max()<<"\n";
		std::cout<<"\tM  (FX,FY,FZ) = "<<acc2d_forcex_val.m() <<" "<<acc2d_forcey_val.m() <<" "<<acc2d_forcez_val.m() <<"\n";
		std::cout<<"\tR2 (FX,FY,FZ) = "<<acc2d_forcex_val.r2()<<" "<<acc2d_forcey_val.r2()<<" "<<acc2d_forcez_val.r2()<<"\n";
		}
		std::cout<<print::title("STATISTICS - ERROR - VALIDATION",strbuf)<<"\n";
		std::cout<<print::buf(strbuf)<<"\n";
		}
		
		//======== print the error statistics - test ========
		if(WORLD.rank()==0 && dist_test.size()>0){
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("STATISTICS - ERROR - TEST",strbuf)<<"\n";
		std::cout<<"ENERGY:\n";
		std::cout<<"\tERROR - AVG - ENERGY/ATOM = "<<acc1d_energy_test_n.avg()<<"\n";
		std::cout<<"\tERROR - DEV - ENERGY/ATOM = "<<std::sqrt(acc1d_energy_test_n.var())<<"\n";
		std::cout<<"\tERROR - MAX - ENERGY/ATOM = "<<acc1d_energy_test_n.max()<<"\n";
		std::cout<<"\tM/R2 - ENERGY/ATOM = "<<acc2d_energy_test.m()<<" "<<acc2d_energy_test.r2()<<"\n";
		if(nnp_train.force_){
		std::cout<<"FORCE:\n";
		std::cout<<"\tERROR - AVG - FORCE = "<<acc1d_force_test_a.avg()<<"\n";
		std::cout<<"\tERROR - DEV - FORCE = "<<std::sqrt(acc1d_force_test_a.var())<<"\n";
		std::cout<<"\tERROR - MAX - FORCE = "<<acc1d_force_test_a.max()<<"\n";
		std::cout<<"\tM  (FX,FY,FZ) = "<<acc2d_forcex_test.m() <<" "<<acc2d_forcey_test.m() <<" "<<acc2d_forcez_test.m() <<"\n";
		std::cout<<"\tR2 (FX,FY,FZ) = "<<acc2d_forcex_test.r2()<<" "<<acc2d_forcey_test.r2()<<" "<<acc2d_forcez_test.r2()<<"\n";
		}
		std::cout<<print::title("STATISTICS - ERROR - TEST",strbuf)<<"\n";
		std::cout<<print::buf(strbuf)<<"\n";
		}
		
		//======== print the correlation - training ========
		if(WORLD.rank()==0 && write_corr && dist_train.size()>0){
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("CORRELATION - INPUTS - TRAIN",strbuf)<<"\n";
			for(int n=0; n<nnp_train.nnp_.nspecies(); ++n){
				std::cout<<"PCORR - SPECIES - "<<nnp_train.nnp_.nnh(n).atom().name()<<"\n";
				for(int j=0; j<nnp_train.nnp_.nnh(n).nInput(); ++j){
					for(int k=0; k<nnp_train.nnp_.nnh(n).nInput(); ++k){
						std::cout<<acc2d_inp_train[n][j][k].pcorr()<<" ";
					}
					std::cout<<"\n";
				}
			}
			std::cout<<print::title("CORRELATION - INPUTS - TRAIN",strbuf)<<"\n";
			std::cout<<print::buf(strbuf)<<"\n";
		}
		
		//======== write the basis functions ========
		if(WORLD.rank()==0 && write_basis){
			int N=200;
			for(int n=0; n<nnp_train.nnp_.nspecies(); ++n){
				for(int m=0; m<nnp_train.nnp_.nspecies(); ++m){
					std::string filename="basisR_"+nnp_train.nnp_.nnh(n).atom().name()+"_"+nnp_train.nnp_.nnh(m).atom().name()+".dat";
					FILE* writer=fopen(filename.c_str(),"w");
					if(writer!=NULL){
						const BasisR& basisR=nnp_train.nnp_.nnh(n).basisR(m);
						for(int i=0; i<N; ++i){
							const double dr=nnp_train.nnp_.rc()*i/(N-1.0);
							fprintf(writer,"%f ",dr);
							for(int j=0; j<basisR.nfR(); ++j){
								fprintf(writer,"%f ",basisR.fR(j).val(dr,basisR.cutoff()->val(dr)));
							}
							fprintf(writer,"\n");
						}
						fclose(writer);
						writer=NULL;
					} else std::cout<<"WARNING: Could not open: \""<<filename<<"\"\n";
				}
			}
			for(int n=0; n<nnp_train.nnp_.nspecies(); ++n){
				for(int m=0; m<nnp_train.nnp_.nspecies(); ++m){
					for(int l=m; l<nnp_train.nnp_.nspecies(); ++l){
						std::string filename="basisA_"
							+nnp_train.nnp_.nnh(n).atom().name()+"_"
							+nnp_train.nnp_.nnh(m).atom().name()+"_"
							+nnp_train.nnp_.nnh(l).atom().name()+".dat";
						FILE* writer=fopen(filename.c_str(),"w");
						if(writer!=NULL){
							const BasisA basisA=nnp_train.nnp_.nnh(n).basisA(m,l);
							for(int i=0; i<N; ++i){
								const double angle=math::constant::PI*i/(N-1.0);
								fprintf(writer,"%f ",angle);
								for(int j=0; j<basisA.nfA(); ++j){
									double tvec[3]={0,0,0};
									double cvec[3]={0,0,0};
									fprintf(writer,"%f ",basisA.fA(j).val(std::cos(angle),tvec,cvec));
								}
								fprintf(writer,"\n");
							}
							fclose(writer);
							writer=NULL;
						} else std::cout<<"WARNING: Could not open: \""<<filename<<"\"\n";
					}
				}
			}
		}
		MPI_Barrier(WORLD.label());
		
		//======== write the nn's ========
		if(NNP_TRAIN_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"writing the nn's\n";
		if(WORLD.rank()==0){
			NNP::write(nnp_train.file_ann_.c_str(),nnp_train.nnp_);
		}
		//======== write restart file ========
		if(NNP_TRAIN_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"writing restart file\n";
		if(WORLD.rank()==0){
			nnp_train.write_restart(nnp_train.file_restart_.c_str());
		}
		
		//======== finalize mpi ========
		if(NNP_TRAIN_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"finalizing mpi\n";
		std::cout<<std::flush;
		MPI_Group_free(&group_world);
		MPI_Group_free(&group_head);
		if(MPI_COMM_NULL!=comm_head) MPI_Comm_free(&comm_head);
		MPI_Comm_free(&BATCH.label());
		MPI_Barrier(WORLD.label());
		MPI_Finalize();
	}catch(std::exception& e){
		std::cout<<"ERROR in nnp_train::main(int,char**):\n";
		std::cout<<e.what()<<"\n";
	}
	
	//======== free local variables ========
	delete[] paramfile;
	delete[] input;
	delete[] strbuf;
	if(rank_batch!=NULL) delete[] rank_batch;
	if(rank_head!=NULL) delete[] rank_head;
	
	return 0;
}
