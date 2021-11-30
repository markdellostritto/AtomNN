// c++ libraries
#include <iostream>
#include <chrono>
// c libraries
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <algorithm>
// ann - str
#include "src/str/string.hpp"
#include "src/str/print.hpp"
// ann - util
#include "src/util/time.hpp"
// ann - ml
#include "src/ml/nn_train.hpp"

//***********************************************************************
// NN Optimization
//***********************************************************************

//==== operators ====

std::ostream& operator<<(std::ostream& out, const NNOpt& nnopt){
	char* str=new char[print::len_buf];
	out<<print::buf(str)<<"\n";
	out<<print::title("NN - OPT",str)<<"\n";
	out<<"seed      = "<<nnopt.seed_<<"\n";
	out<<"pre-cond  = "<<nnopt.preCond_<<"\n";
	out<<"post-cond = "<<nnopt.postCond_<<"\n";
	out<<"restart   = "<<nnopt.restart_<<"\n";
	out<<"batch     = "<<nnopt.batch_<<"\n";
	out<<"loss      = "<<nnopt.loss_<<"\n";
	out<<nnopt.data_<<"\n";
	if(nnopt.model_.use_count()>0) Opt::Model::print(out,nnopt.model_.get());
	out<<print::title("NN - OPT",str)<<"\n";
	out<<print::buf(str);
	delete[] str;
	return out;
}

//==== member functions ====

/**
* set defaults
*/
void NNOpt::defaults(){
	if(NN_TRAIN_PRINT_FUNC>0) std::cout<<"NNOpt::defaults():\n";
	//optimization
		seed_=-1;
		batch_.clear();
		data_.clear();
		err_train_=0;
		err_val_=0;
	//conditioning
		preCond_=false;
		postCond_=false;
	//file i/o
		restart_=false;
		file_error_="nn_train_error.dat";
}

/**
* clear data
*/
void NNOpt::clear(){
	if(NN_TRAIN_PRINT_FUNC>0) std::cout<<"NNOpt::clear():\n";
	//neural network
		nn_.clear();
	//optimization
		batch_.clear();
		data_.clear();
		err_train_=0;
		err_val_=0;
	//conditioning
		preCond_=true;
		postCond_=true;
	//file i/o
		restart_=false;
}

/**
* train a neural network given a set of training and validation data
* @param nn - pointer to neural network which will be optimized
* @param nbatchl - the size of the local batch (number of elements in the batch handled by a single processor)
*/
void NNOpt::train(int nbatchl, const MLData& data_train, const MLData& data_val){
	if(NN_TRAIN_PRINT_FUNC>0) std::cout<<"NNOpt::train(std::shared_ptr<NN::ANN>&,nbatchl)):\n";
	
	//==== set MPI variables ====
	int nprocs=1,rank=0;
	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	
	//==== initialize the randomizer ====
	rngen_=std::mt19937(seed_<0?std::chrono::system_clock::now().time_since_epoch().count():seed_);
	
	//==== check the data ====
	if(data_train.size()<=0) throw std::invalid_argument("NNOpt::train(std::shared_ptr<NN::ANN>&,int): no training data provided.");
	if(data_val.size()<=0) throw std::invalid_argument("NNOpt::train(std::shared_ptr<NN::ANN>&,int): no training data provided.");
	for(int n=0; n<data_train.size(); ++n){
		if(data_train.in(n).size()!=nn_.nIn()) throw std::invalid_argument("NNOpt::train(std::shared_ptr<NN::ANN>&,int): Invalid inputs - training: incompatible with network size.");
		if(data_train.out(n).size()!=nn_.nOut()) throw std::invalid_argument("NNOpt::train(std::shared_ptr<NN::ANN>&,int): Invalid inputs - training: incompatible with network size.");
	}
	for(int n=0; n<data_val.size(); ++n){
		if(data_val.in(n).size()!=nn_.nIn()) throw std::invalid_argument("NNOpt::train(std::shared_ptr<NN::ANN>&,int): Invalid inputs - validation: incompatible with network size.");
		if(data_val.out(n).size()!=nn_.nOut()) throw std::invalid_argument("NNOpt::train(std::shared_ptr<NN::ANN>&,int): Invalid inputs - validation: incompatible with network size.");
	}
	int nTrain_=0;
	int nVal_=0;
	MPI_Allreduce(&data_train.size(),&nTrain_,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
	MPI_Allreduce(&data_val.size(),&nVal_,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
	
	//==== resize the batch and indices ====
	batch_.resize(nbatchl,data_train.size());
	std::cout<<batch_<<"\n";
	
	//==== compute input statistics ====
	if(NN_TRAIN_PRINT_STATUS>0 && rank==0) std::cout<<"computing input statistics\n";
	Eigen::VectorXd avg_in_loc=Eigen::VectorXd::Zero(nn_.nIn());
	Eigen::VectorXd avg_in_tot=Eigen::VectorXd::Zero(nn_.nIn());
	Eigen::VectorXd dev_in_loc=Eigen::VectorXd::Zero(nn_.nIn());
	Eigen::VectorXd dev_in_tot=Eigen::VectorXd::Zero(nn_.nIn());
	//compute average
	for(int n=0; n<data_train.size(); ++n){
		avg_in_loc.noalias()+=data_train.in(n);
	}
	MPI_Allreduce(avg_in_loc.data(),avg_in_tot.data(),nn_.nIn(),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
	avg_in_tot/=nTrain_;
	//compute deviation
	for(int n=0; n<data_train.size(); ++n){
		const Eigen::VectorXd diff=data_train.in(n)-avg_in_tot;
		dev_in_loc.noalias()+=diff.cwiseProduct(diff);
	}
	MPI_Allreduce(dev_in_loc.data(),dev_in_tot.data(),nn_.nIn(),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
	for(int i=0; i<nn_.nIn(); ++i) dev_in_tot[i]=std::sqrt(dev_in_tot[i]/(nTrain_-1.0));
	//print
	if(NN_TRAIN_PRINT_STATUS>0 && rank==0){
		std::cout<<"avg - in = "<<avg_in_tot.transpose()<<"\n";
		std::cout<<"dev - in = "<<dev_in_tot.transpose()<<"\n";
	}
	
	//==== compute output statistics ====
	if(NN_TRAIN_PRINT_STATUS>0 && rank==0) std::cout<<"computing output statistics\n";
	Eigen::VectorXd avg_out_loc=Eigen::VectorXd::Zero(nn_.nOut());
	Eigen::VectorXd avg_out_tot=Eigen::VectorXd::Zero(nn_.nOut());
	Eigen::VectorXd dev_out_loc=Eigen::VectorXd::Zero(nn_.nOut());
	Eigen::VectorXd dev_out_tot=Eigen::VectorXd::Zero(nn_.nOut());
	//compute average
	for(int n=0; n<data_train.size(); ++n){
		avg_out_loc.noalias()+=data_train.out(n);
	}
	MPI_Allreduce(avg_out_loc.data(),avg_out_tot.data(),nn_.nOut(),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
	//compute deviation
	for(int n=0; n<data_train.size(); ++n){
		const Eigen::VectorXd diff=data_train.out(n)-avg_out_tot;
		dev_out_loc.noalias()+=diff.cwiseProduct(diff);
	}
	MPI_Allreduce(dev_out_loc.data(),dev_out_tot.data(),nn_.nOut(),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
	for(int i=0; i<nn_.nOut(); ++i) dev_out_tot[i]=std::sqrt(dev_out_tot[i]/(nTrain_-1.0));
	//print
	if(NN_TRAIN_PRINT_STATUS>0 && rank==0){
		std::cout<<"avg - out = "<<avg_out_tot.transpose()<<"\n";
		std::cout<<"dev - out = "<<dev_out_tot.transpose()<<"\n";
	}
	
	//==== precondition the input ====
	if(NN_TRAIN_PRINT_STATUS>0 && rank==0) std::cout<<"pre-conditioning input\n";
	//set bias and weight
	if(preCond_){
		for(int i=0; i<nn_.nIn(); ++i) nn_.inb()[i]=-1.0*avg_in_tot[i];
		for(int i=0; i<nn_.nIn(); ++i) nn_.inw()[i]=1.0/dev_in_tot[i];
	} else {
		for(int i=0; i<nn_.nIn(); ++i) nn_.inb()[i]=0.0;
		for(int i=0; i<nn_.nIn(); ++i) nn_.inw()[i]=1.0;
	}
	//print
	if(NN_TRAIN_PRINT_STATUS>0 && rank==0){
		std::cout<<"bias-in  = "<<nn_.inb().transpose()<<"\n";
		std::cout<<"scale-in = "<<nn_.inw().transpose()<<"\n";
	}
	
	//==== postcondition the output ====
	if(NN_TRAIN_PRINT_STATUS>0 && rank==0) std::cout<<"post-conditioning output\n";
	if(postCond_){
		for(int i=0; i<nn_.nOut(); ++i) nn_.outb()[i]=avg_out_tot[i];
		for(int i=0; i<nn_.nOut(); ++i) nn_.outw()[i]=dev_out_tot[i];
	} else {
		for(int i=0; i<nn_.nOut(); ++i) nn_.outb()[i]=0.0;
		for(int i=0; i<nn_.nOut(); ++i) nn_.outw()[i]=1.0;
	}
	//print
	if(NN_TRAIN_PRINT_STATUS>0 && rank==0){
		std::cout<<"bias-out  = "<<nn_.outb().transpose()<<"\n";
		std::cout<<"scale-out = "<<nn_.outw().transpose()<<"\n";
	}
	
	//==== initalize the optimizer ====
	if(NN_TRAIN_PRINT_STATUS>0 && rank==0) std::cout<<"initializing the optimizer\n";
	data_.init(nn_.size());
	model_->init(nn_.size());
	
	//==== set the initial values ====
	if(NN_TRAIN_PRINT_STATUS>0 && rank==0) std::cout<<"setting the initial values\n";
	//set values
	nn_>>data_.p();
	nn_>>data_.pOld();
	data_.g().setZero();
	data_.gOld().setZero();
	
	//==== allocate status vectors ====
	std::vector<int> step;
	std::vector<double> gamma,err_t,err_v;
	if(rank==0){
		step.resize(data_.max()/data_.nPrint());
		gamma.resize(data_.max()/data_.nPrint());
		err_v.resize(data_.max()/data_.nPrint());
		err_t.resize(data_.max()/data_.nPrint());
	}
	
	//==== execute the optimization ====
	if(NN_TRAIN_PRINT_STATUS>0 && rank==0) std::cout<<"executing the optimization\n";
	//bcast parameters
	MPI_Bcast(data_.p().data(),data_.dim(),MPI_DOUBLE,0,MPI_COMM_WORLD);
	Eigen::VectorXd gSum_(data_.dim());
	bool fbreak=false;
	const double nbatchi_=1.0/batch_.size();
	const double nVali_=1.0/nVal_;
	cost_.resize(nn_);
	Clock clock;
	clock.begin();
	for(int iter=0; iter<data_.max(); ++iter){
		double err_train_sum_=0,err_val_sum_=0;
		//set the parameters of the network
		nn_<<data_.p();
		//compute the error and gradient
		error(data_train,data_val);
		//accumulate error
		MPI_Reduce(&err_train_,&err_train_sum_,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
		MPI_Reduce(&err_val_,&err_val_sum_,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
		//accumulate gradient
		MPI_Reduce(data_.g().data(),gSum_.data(),data_.dim(),MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
		//compute step
		if(rank==0){
			//normalize error and gradient
			err_train_=err_train_sum_*nbatchi_;
			err_val_=err_val_sum_*nVali_;
			data_.g().noalias()=gSum_*nbatchi_;
			//compute regularization error and gradient
			//print error
			if(data_.step()%data_.nPrint()==0){
				const int t=iter/data_.nPrint();
				step[t]=data_.step();
				gamma[t]=model_->gamma();
				err_t[t]=std::sqrt(2.0*err_train_);
				err_v[t]=std::sqrt(2.0*err_val_);
				printf("opt %8i gamma %12.10f err_t %12.10f err_v %12.10f\n",step[t],gamma[t],err_t[t],err_v[t]);
			}
			//compute the new position
			data_.val()=err_train_;
			model_->step(data_);
			//compute the difference
			data_.dv()=std::fabs(data_.val()-data_.valOld());
			data_.dp()=(data_.p()-data_.pOld()).norm();
			//set the new "old" values
			data_.pOld()=data_.p();//set "old" p value
			data_.gOld()=data_.g();//set "old" g value
			data_.valOld()=data_.val();//set "old" value
			//check the break condition
			switch(data_.stop()){
				case Opt::Stop::FABS: fbreak=(data_.val()<data_.tol()); break;
				case Opt::Stop::FREL: fbreak=(data_.dv()<data_.tol()); break;
				case Opt::Stop::XREL: fbreak=(data_.dp()<data_.tol()); break;
			}
		}
		//bcast parameters
		MPI_Bcast(data_.p().data(),data_.p().size(),MPI_DOUBLE,0,MPI_COMM_WORLD);
		//bcast break condition
		MPI_Bcast(&fbreak,1,MPI_C_BOOL,0,MPI_COMM_WORLD);
		if(fbreak) break;
		//increment step
		++data_.step();
		++data_.count();
	}
	clock.end();
	MPI_Barrier(MPI_COMM_WORLD);
	
	//==== write the error ====
	if(rank==0){
		FILE* writer_error_=NULL;
		if(!restart_){
			writer_error_=fopen(file_error_.c_str(),"w");
			fprintf(writer_error_,"#STEP GAMMA ERROR_RMS_TRAIN ERROR_RMS_VAL\n");
		} else writer_error_=fopen(file_error_.c_str(),"a");
		if(writer_error_==NULL) throw std::runtime_error("I/O Error: Could not open error file.");
		for(int t=0; t<step.size(); ++t){
			fprintf(writer_error_,"%6i %12.10f %12.10f %12.10f\n",step[t],gamma[t],err_t[t],err_v[t]);
		}
		fclose(writer_error_);
		writer_error_=NULL;
	}
	
	if(NN_TRAIN_PRINT_STATUS>-1 && rank==0){
		char* str=new char[print::len_buf];
		std::cout<<print::buf(str)<<"\n";
		std::cout<<print::title("OPT - SUMMARY",str)<<"\n";
		std::cout<<"n-steps = "<<data_.step()<<"\n";
		std::cout<<"opt-val = "<<data_.val()<<"\n";
		std::cout<<"time    = "<<clock.duration()<<"\n";
		std::cout<<print::title("OPT - SUMMARY",str)<<"\n";
		std::cout<<print::buf(str)<<"\n";
		delete[] str;
	}
}

/**
* compute the error associated with the neural network with respect to the training data
* @param data_train - training data
* @param data_val - validation data
* @return the error associated with the training data
*/
double NNOpt::error(const MLData& data_train, const MLData& data_val){
	if(NN_TRAIN_PRINT_FUNC>0) std::cout<<"NNOpt::error(const MLData&,const MLData&):\n";
	Eigen::VectorXd dcdo=Eigen::VectorXd::Zero(nn_.nOut());
	
	//reset the gradient
	Eigen::VectorXd& grad=data_.g();
	grad.setZero();
	
	//randomize the batch
	for(int i=0; i<batch_.size(); ++i) batch_[i]=batch_.data((batch_.count()++)%batch_.capacity());
	std::sort(batch_.elements(),batch_.elements()+batch_.size());
	if(batch_.count()>=batch_.capacity()){
		std::shuffle(batch_.data(),batch_.data()+batch_.capacity(),rngen_);
		batch_.count()=0;
	}
	
	//compute the training error
	err_train_=0;
	for(int i=0; i<batch_.size(); ++i){
		const int ii=batch_[i];
		//set the inputs
		nn_.in()=data_train.in(ii);
		//execute the network
		nn_.execute();
		//compute the error and dcdo
		err_train_+=Opt::Loss::error(loss_,nn_.out(),data_train.out(ii),dcdo);
		//compute the gradient
		cost_.grad(nn_,dcdo);
		//add the gradient
		grad.noalias()+=cost_.grad();
	}
	
	//compute the validation error
	err_val_=0;
	if(data_.step()%data_.nPrint()==0 || data_.step()%data_.nWrite()==0){
		for(int i=0; i<data_val.size(); ++i){
			//set the inputs
			nn_.in()=data_val.in(i);
			//execute the network
			nn_.execute();
			//compute the error
			err_val_+=Opt::Loss::error(loss_,nn_.out(),data_val.out(i));
		}
	}
	
	//return the error
	return err_train_;
}

namespace serialize{
		
	//**********************************************
	// byte measures
	//**********************************************

	template <> int nbytes(const NNOpt& obj){
		if(NN_TRAIN_PRINT_FUNC>0) std::cout<<"serialize::nbytes(const NNOpt&):\n";
		int size=0;
		size+=sizeof(bool);//preCond_
		size+=sizeof(bool);//postCond_
		size+=sizeof(int);//loss_
		size+=nbytes(obj.batch());
		size+=nbytes(obj.data());
		size+=sizeof(Opt::Loss);
		size+=nbytes(obj.nn());
		switch(obj.data().algo()){
			case Opt::Algo::SGD:      size+=nbytes(static_cast<const Opt::SGD&>(*obj.model())); break;
			case Opt::Algo::SDM:      size+=nbytes(static_cast<const Opt::SDM&>(*obj.model())); break;
			case Opt::Algo::NAG:      size+=nbytes(static_cast<const Opt::NAG&>(*obj.model())); break;
			case Opt::Algo::ADAGRAD:  size+=nbytes(static_cast<const Opt::ADAGRAD&>(*obj.model())); break;
			case Opt::Algo::ADADELTA: size+=nbytes(static_cast<const Opt::ADADELTA&>(*obj.model())); break;
			case Opt::Algo::RMSPROP:  size+=nbytes(static_cast<const Opt::RMSPROP&>(*obj.model())); break;
			case Opt::Algo::ADAM:     size+=nbytes(static_cast<const Opt::ADAM&>(*obj.model())); break;
			case Opt::Algo::NADAM:    size+=nbytes(static_cast<const Opt::NADAM&>(*obj.model())); break;
			case Opt::Algo::BFGS:     size+=nbytes(static_cast<const Opt::BFGS&>(*obj.model())); break;
			case Opt::Algo::RPROP:    size+=nbytes(static_cast<const Opt::RPROP&>(*obj.model())); break;
			default: throw std::runtime_error("nbytes(const NNPotOpt&): Invalid optimization method."); break;
		}
		size+=sizeof(bool);//restart_
		size+=nbytes(obj.file_error());//file_error_
		size+=nbytes(obj.file_restart());//file_restart_
		return size;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const NNOpt& obj, char* arr){
		if(NN_TRAIN_PRINT_FUNC>0) std::cout<<"serialize::pack(const NNOpt&,char*):\n";
		int pos=0;
		std::memcpy(arr+pos,&obj.preCond(),sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(arr+pos,&obj.postCond(),sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(arr+pos,&obj.loss(),sizeof(int)); pos+=sizeof(int);
		pos+=pack(obj.batch(),arr+pos);
		pos+=pack(obj.data(),arr+pos);
		pos+=pack(obj.nn(),arr+pos);
		switch(obj.data().algo()){
			case Opt::Algo::SGD:      pos+=pack(static_cast<const Opt::SGD&>(*obj.model()),arr+pos); break;
			case Opt::Algo::SDM:      pos+=pack(static_cast<const Opt::SDM&>(*obj.model()),arr+pos); break;
			case Opt::Algo::NAG:      pos+=pack(static_cast<const Opt::NAG&>(*obj.model()),arr+pos); break;
			case Opt::Algo::ADAGRAD:  pos+=pack(static_cast<const Opt::ADAGRAD&>(*obj.model()),arr+pos); break;
			case Opt::Algo::ADADELTA: pos+=pack(static_cast<const Opt::ADADELTA&>(*obj.model()),arr+pos); break;
			case Opt::Algo::RMSPROP:  pos+=pack(static_cast<const Opt::RMSPROP&>(*obj.model()),arr+pos); break;
			case Opt::Algo::ADAM:     pos+=pack(static_cast<const Opt::ADAM&>(*obj.model()),arr+pos); break;
			case Opt::Algo::NADAM:    pos+=pack(static_cast<const Opt::NADAM&>(*obj.model()),arr+pos); break;
			case Opt::Algo::BFGS:     pos+=pack(static_cast<const Opt::BFGS&>(*obj.model()),arr+pos); break;
			case Opt::Algo::RPROP:    pos+=pack(static_cast<const Opt::RPROP&>(*obj.model()),arr+pos); break;
			default: throw std::runtime_error("pack(const NNPotOpt&,char*): Invalid optimization method."); break;
		}
		std::memcpy(arr+pos,&obj.restart(),sizeof(bool)); pos+=sizeof(bool);
		pos+=pack(obj.file_error(),arr+pos);
		pos+=pack(obj.file_restart(),arr+pos);
		return pos;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(NNOpt& obj, const char* arr){
		if(NN_TRAIN_PRINT_FUNC>0) std::cout<<"serialize::unpack(NNPot&,const char*):\n";
		int pos=0;
		std::memcpy(&obj.preCond(),arr+pos,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(&obj.postCond(),arr+pos,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(&obj.loss(),arr+pos,sizeof(int)); pos+=sizeof(int);
		pos+=unpack(obj.batch(),arr+pos);
		pos+=unpack(obj.data(),arr+pos);
		pos+=unpack(obj.nn(),arr+pos);
		switch(obj.data().algo()){
			case Opt::Algo::SGD:
				obj.model().reset(new Opt::SGD());
				pos+=unpack(static_cast<Opt::SGD&>(*obj.model()),arr+pos);
			break;
			case Opt::Algo::SDM:
				obj.model().reset(new Opt::SDM());
				pos+=unpack(static_cast<Opt::SDM&>(*obj.model()),arr+pos);
			break;
			case Opt::Algo::NAG:
				obj.model().reset(new Opt::NAG());
				pos+=unpack(static_cast<Opt::NAG&>(*obj.model()),arr+pos);
			break;
			case Opt::Algo::ADAGRAD:
				obj.model().reset(new Opt::ADAGRAD());
				pos+=unpack(static_cast<Opt::ADAGRAD&>(*obj.model()),arr+pos);
			break;
			case Opt::Algo::ADADELTA:
				obj.model().reset(new Opt::ADADELTA());
				pos+=unpack(static_cast<Opt::ADADELTA&>(*obj.model()),arr+pos);
			break;
			case Opt::Algo::RMSPROP:
				obj.model().reset(new Opt::RMSPROP());
				pos+=unpack(static_cast<Opt::RMSPROP&>(*obj.model()),arr+pos);
			break;
			case Opt::Algo::ADAM:
				obj.model().reset(new Opt::ADAM());
				pos+=unpack(static_cast<Opt::ADAM&>(*obj.model()),arr+pos);
			break;
			case Opt::Algo::NADAM:
				obj.model().reset(new Opt::NADAM());
				pos+=unpack(static_cast<Opt::NADAM&>(*obj.model()),arr+pos);
			break;
			case Opt::Algo::BFGS:
				obj.model().reset(new Opt::BFGS());
				pos+=unpack(static_cast<Opt::BFGS&>(*obj.model()),arr+pos);
			break;
			case Opt::Algo::RPROP:
				obj.model().reset(new Opt::RPROP());
				pos+=unpack(static_cast<Opt::RPROP&>(*obj.model()),arr+pos);
			break;
			default:
				throw std::runtime_error("unpack(NNOpt&,const char*): Invalid optimization method.");
			break;
		}
		std::memcpy(&obj.restart(),arr+pos,sizeof(bool)); pos+=sizeof(bool);
		pos+=unpack(obj.file_error(),arr+pos);
		pos+=unpack(obj.file_restart(),arr+pos);
		return pos;
	}
	
}