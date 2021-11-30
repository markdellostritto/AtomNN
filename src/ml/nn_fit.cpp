// c libraries
#include <cstdio>
// c++ libraries
#include <iostream>
// ann - str
#include "src/str/string.hpp"
#include "src/str/print.hpp"
#include "src/str/input.hpp"
// ann - thread
#include "src/thread/parallel.hpp"
// ann - util
#include "src/util/time.hpp"
#include "src/util/compiler.hpp"
// ann - math
#include "src/math/eigen.hpp"
// ann - neural network
#include "src/ml/nn.hpp"
#include "src/ml/nn_train.hpp"
// ann - optimization
#include "src/opt/optimize.hpp"

//************************************************************
// MPI Communicators
//************************************************************

parallel::Comm WORLD;//all processors

int main(int argc, char* argv[]){
	//==== global variables ====
	//file i/o
		char* input=new char[string::M];
		char* strbuf=new char[print::len_buf];
		std::string file_param;
		std::vector<std::string> file_train;
		std::vector<std::string> file_val;
		std::vector<std::string> file_test;
		std::string file_out="nn_fit.dat";
		FILE* reader=NULL;
		FILE* writer=NULL;
	//function data
		MLData data_train_g,data_val_g,data_test_g;
		MLData data_train_l,data_val_l,data_test_l;
		int dIn=0,dOut=0;
	//batch
		int nbatch=0;
	//neural network
		std::vector<int> nh;
		std::shared_ptr<NN::ANN> nn(new NN::ANN());
		NNOpt nnopt;
		NN::ANNInit annInit;
	//mpi
		MPI_Group group_world; //the group associated with the WORLD communicator
		parallel::Dist dist_batch; //data distribution - batch
		parallel::Dist dist_train; //data distribution - training
		parallel::Dist dist_val;   //data distribution - validation
		parallel::Dist dist_test;  //data distribution - testing
	//timing
		Clock clock_wall;
		
	try{
		
		//************************************************************************************
		// INITIALIZATION
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
			std::cout<<print::title("NN_FIT",strbuf,' ')<<"\n";
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
			std::cout<<print::title("MATH CONSTANTS",strbuf)<<"\n";
			std::printf("PI    = %.15f\n",math::constant::PI);
			std::printf("RadPI = %.15f\n",math::constant::RadPI);
			std::printf("Rad2  = %.15f\n",math::constant::Rad2);
			std::cout<<print::title("MATH CONSTANTS",strbuf)<<"\n";
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
				std::cout<<print::title("MPI",strbuf)<<"\n";
				std::cout<<print::buf(strbuf)<<"\n";
				std::cout<<std::flush;
			}
			delete[] ranks;
		}
		
		//************************************************************************************
		// PARAMETER DETERMINATION
		//************************************************************************************
		
		//==== read parameters ====
		if(WORLD.rank()==0){
			
			//======== check the arguments ========
			if(argc!=2) throw std::invalid_argument("Invalid number of arguments.");
			
			//======== load the parameter file ========
			std::cout<<"reading parameter file\n";
			file_param=argv[1];
			
			//==== open parameter file ====
			reader=fopen(file_param.c_str(),"r");
			if(reader==NULL) throw std::runtime_error("Unable to open parameter file.");
			
			//==== read parameters ====
			std::cout<<"reading parameters\n";
			while(fgets(input,string::M,reader)!=NULL)
			{
				std::vector<std::string> strlist;
				string::trim_right(input,string::COMMENT);
				string::split(input,string::WS,strlist);
				if(strlist.size()==0) continue;
				string::to_upper(strlist.at(0));
				//optimization
				if(strlist.at(0)=="PRE_COND"){
					nnopt.preCond()=string::boolean(strlist.at(1).c_str());
				} else if(strlist.at(0)=="POST_COND"){
					nnopt.postCond()=string::boolean(strlist.at(1).c_str());
				} else if(strlist.at(0)=="NBATCH"){
					nbatch=std::atoi(strlist.at(1).c_str());
				} else if(strlist.at(0)=="READ_RESTART"){
					nnopt.restart()=true;
					nnopt.file_restart()=strlist.at(1);
				} else if(strlist.at(0)=="FILE_ERROR"){
					nnopt.file_error()=strlist.at(1);
				}
				//data
				if(strlist.at(0)=="DATA_TRAIN"){//data - training
					file_train.push_back(strlist.at(1));
				} else if(strlist.at(0)=="DATA_VAL"){//data - validation
					file_val.push_back(strlist.at(1));
				} else if(strlist.at(0)=="DATA_TEST"){//data - testing
					file_test.push_back(strlist.at(1));
				} else if(strlist.at(0)=="DIM_IN"){
					dIn=std::atoi(strlist.at(1).c_str());
				} else if(strlist.at(0)=="DIM_OUT"){
					dOut=std::atoi(strlist.at(1).c_str());
				}
				//neural network
				if(strlist.at(0)=="N_HIDDEN"){
					int nl=strlist.size()-1;
					if(nl<=0) throw std::invalid_argument("Invalid hidden layer configuration.");
					nh.resize(nl);
					for(int i=0; i<nl; ++i){
						nh.at(i)=std::atoi(strlist.at(i+1).c_str());
						if(nh.at(i)==0) throw std::invalid_argument("Invalid hidden layer configuration.");
					}
				} else if(strlist.at(0)=="SIGMA"){//initialization deviation
					annInit.sigma()=std::atof(strlist.at(1).c_str());
				} else if(strlist.at(0)=="INIT"){//initialization
					annInit.init()=NN::Init::read(string::to_upper(strlist.at(1)).c_str());
				} else if(strlist.at(0)=="SEED"){//initialization
					annInit.seed()=std::atof(strlist.at(1).c_str());
				} else if(strlist.at(0)=="TRANSFER"){//transfer function
					nnopt.nn().tf()=NN::Transfer::read(string::to_upper(strlist.at(1)).c_str());
				} else if(strlist.at(0)=="LOSS"){//loss function
					nnopt.loss()=Opt::Loss::read(string::to_upper(strlist.at(1)).c_str());
				}
			}
			
			//==== read optimization data ====
			Opt::read(nnopt.data(),reader);
			
			//==== read optimization object ====
			switch(nnopt.data().algo()){
				case Opt::Algo::SGD:
					nnopt.model().reset(new Opt::SGD());
					read(static_cast<Opt::SGD&>(*nnopt.model()),reader);
				break;
				case Opt::Algo::SDM:
					nnopt.model().reset(new Opt::SDM());
					read(static_cast<Opt::SDM&>(*nnopt.model()),reader);
				break;
				case Opt::Algo::NAG:
					nnopt.model().reset(new Opt::NAG());
					read(static_cast<Opt::NAG&>(*nnopt.model()),reader);
				break;
				case Opt::Algo::ADAGRAD:
					nnopt.model().reset(new Opt::ADAGRAD());
					read(static_cast<Opt::ADAGRAD&>(*nnopt.model()),reader);
				break;
				case Opt::Algo::ADADELTA:
					nnopt.model().reset(new Opt::ADADELTA());
					read(static_cast<Opt::ADADELTA&>(*nnopt.model()),reader);
				break;
				case Opt::Algo::RMSPROP:
					nnopt.model().reset(new Opt::RMSPROP());
					read(static_cast<Opt::RMSPROP&>(*nnopt.model()),reader);
				break;
				case Opt::Algo::ADAM:
					nnopt.model().reset(new Opt::ADAM());
					read(static_cast<Opt::ADAM&>(*nnopt.model()),reader);
				break;
				case Opt::Algo::NADAM:
					nnopt.model().reset(new Opt::NADAM());
					read(static_cast<Opt::NADAM&>(*nnopt.model()),reader);
				break;
				case Opt::Algo::BFGS:
					nnopt.model().reset(new Opt::BFGS());
					read(static_cast<Opt::BFGS&>(*nnopt.model()),reader);
				break;
				case Opt::Algo::RPROP:
					nnopt.model().reset(new Opt::RPROP());
					read(static_cast<Opt::RPROP&>(*nnopt.model()),reader);
				break;
			}
		
			//==== close parameter file ====
			fclose(reader);
			reader=NULL;
			
			//==== check the parameters ====
			if(dIn<=0) throw std::invalid_argument("Invalid input dimension");
			if(dOut<=0) throw std::invalid_argument("Invalid output dimension");
			if(nnopt.loss()==Opt::Loss::UNKNOWN) throw std::invalid_argument("Invalid loss function.");
			
			//==== initialize neural network ====
			if(!nnopt.restart()){
				std::cout<<"initializing neural network\n";
				nnopt.nn().resize(annInit,dIn,nh,dOut);
			} else {
				std::cout<<"reading restart file\n";
				//nnopt.read_restart(nnopt.file_restart.c_str());
			}
			
			//==== print data ====
			std::cout<<nnopt<<"\n";
			std::cout<<nnopt.nn()<<"\n";
			std::cout<<nnopt.data()<<"\n";
			Opt::Model::print(std::cout,nnopt.model().get());
			
			//==== read training data ====
			std::cout<<"reading training data\n";
			data_train_g.resize(dIn,dOut);
			for(int i=0; i<file_train.size(); ++i){
				std::cout<<"\t\""<<file_train[i]<<"\"\n";
				reader=fopen(file_train[i].c_str(),"r");
				if(reader==NULL) throw std::runtime_error("Unable to open file - data - training.");
				Eigen::VectorXd x(dIn),y(dOut);
				fgets(input,string::M,reader);//skip first line
				while(fgets(input,string::M,reader)){
					std::vector<std::string> strlist;
					string::split(input,string::WS,strlist);
					if(strlist.size()!=dIn+dOut) throw std::runtime_error("Invalid data - incorrect dimension.\n");
					for(int i=0; i<dIn; ++i) x[i]=std::atof(strlist[i].c_str());
					for(int i=0; i<dOut; ++i) y[i]=std::atof(strlist[i+dIn].c_str());
					data_train_g.push(x,y);
				}
				fclose(reader);
				reader=NULL;
			}
		
			//==== read validation data ====
			std::cout<<"reading validation data\n";
			data_val_g.resize(dIn,dOut);
			for(int i=0; i<file_val.size(); ++i){
				std::cout<<"\t\""<<file_val[i]<<"\"\n";
				reader=fopen(file_val[i].c_str(),"r");
				if(reader==NULL) throw std::runtime_error("Unable to open file - data - validation.");
				Eigen::VectorXd x(dIn),y(dOut);
				fgets(input,string::M,reader);//skip first line
				while(fgets(input,string::M,reader)){
					std::vector<std::string> strlist;
					string::split(input,string::WS,strlist);
					if(strlist.size()!=dIn+dOut) throw std::runtime_error("Invalid data - incorrect dimension.\n");
					for(int i=0; i<dIn; ++i) x[i]=std::atof(strlist[i].c_str());
					for(int i=0; i<dOut; ++i) y[i]=std::atof(strlist[i+dIn].c_str());
					data_val_g.push(x,y);
				}
				fclose(reader);
				reader=NULL;
			}
		
			//==== read testing data ====
			std::cout<<"reading testing data\n";
			data_test_g.resize(dIn,dOut);
			for(int i=0; i<file_test.size(); ++i){
				std::cout<<"\t\""<<file_test[i]<<"\"\n";
				reader=fopen(file_test[i].c_str(),"r");
				if(reader==NULL) throw std::runtime_error("Unable to open file - data - testing.");
				Eigen::VectorXd x(dIn),y(dOut);
				fgets(input,string::M,reader);//skip first line
				while(fgets(input,string::M,reader)){
					std::vector<std::string> strlist;
					string::split(input,string::WS,strlist);
					if(strlist.size()!=dIn+dOut) throw std::runtime_error("Invalid data - incorrect dimension.\n");
					for(int i=0; i<dIn; ++i) x[i]=std::atof(strlist[i].c_str());
					for(int i=0; i<dOut; ++i) y[i]=std::atof(strlist[i+dIn].c_str());
					data_test_g.push(x,y);
				}
				fclose(reader);
				reader=NULL;
			}
		}
		
		//==== gen thread dist + offset ====
		int ntrain=data_train_g.size();
		int nval=data_val_g.size();
		int ntest=data_test_g.size();
		MPI_Bcast(&nbatch,1,MPI_INT,0,MPI_COMM_WORLD);
		MPI_Bcast(&ntrain,1,MPI_INT,0,MPI_COMM_WORLD);
		MPI_Bcast(&nval  ,1,MPI_INT,0,MPI_COMM_WORLD);
		MPI_Bcast(&ntest ,1,MPI_INT,0,MPI_COMM_WORLD);
		dist_batch.init(WORLD.size(),WORLD.rank(),nbatch);
		dist_train.init(WORLD.size(),WORLD.rank(),ntrain);
		dist_val.init(WORLD.size(),WORLD.rank(),nval);
		dist_test.init(WORLD.size(),WORLD.rank(),ntest);
		//print
		if(WORLD.rank()==0){
			//thread dist
			int* thread_dist_batch=new int[WORLD.size()];
			int* thread_dist_train=new int[WORLD.size()];
			int* thread_dist_val  =new int[WORLD.size()];
			int* thread_dist_test =new int[WORLD.size()];
			MPI_Gather(&dist_batch.size(),1,MPI_INT,thread_dist_batch,1,MPI_INT,0,WORLD.label());
			MPI_Gather(&dist_train.size(),1,MPI_INT,thread_dist_train,1,MPI_INT,0,WORLD.label());
			MPI_Gather(&dist_val.size(),1,MPI_INT,thread_dist_val,1,MPI_INT,0,WORLD.label());
			MPI_Gather(&dist_test.size(),1,MPI_INT,thread_dist_test,1,MPI_INT,0,WORLD.label());
			//thread offset
			int* thread_offset_train=new int[WORLD.size()];
			int* thread_offset_val  =new int[WORLD.size()];
			int* thread_offset_test =new int[WORLD.size()];
			MPI_Gather(&dist_train.offset(),1,MPI_INT,thread_offset_train,1,MPI_INT,0,WORLD.label());
			MPI_Gather(&dist_val.offset(),1,MPI_INT,thread_offset_val,1,MPI_INT,0,WORLD.label());
			MPI_Gather(&dist_test.offset(),1,MPI_INT,thread_offset_test,1,MPI_INT,0,WORLD.label());
			//print
			std::cout<<"n-batch             = "<<nbatch<<"\n";
			std::cout<<"n-train             = "<<ntrain<<"\n";
			std::cout<<"n-val               = "<<nval<<"\n";
			std::cout<<"n-test              = "<<ntest<<"\n";
			std::cout<<"thread_dist_batch   = "; for(int i=0; i<WORLD.size(); ++i) std::cout<<thread_dist_batch[i]<<" "; std::cout<<"\n";
			std::cout<<"thread_dist_train   = "; for(int i=0; i<WORLD.size(); ++i) std::cout<<thread_dist_train[i]<<" "; std::cout<<"\n";
			std::cout<<"thread_dist_val     = "; for(int i=0; i<WORLD.size(); ++i) std::cout<<thread_dist_val[i]<<" "; std::cout<<"\n";
			std::cout<<"thread_dist_test    = "; for(int i=0; i<WORLD.size(); ++i) std::cout<<thread_dist_test[i]<<" "; std::cout<<"\n";
			std::cout<<"thread_offset_train = "; for(int i=0; i<WORLD.size(); ++i) std::cout<<thread_offset_train[i]<<" "; std::cout<<"\n";
			std::cout<<"thread_offset_val   = "; for(int i=0; i<WORLD.size(); ++i) std::cout<<thread_offset_val[i]<<" "; std::cout<<"\n";
			std::cout<<"thread_offset_test  = "; for(int i=0; i<WORLD.size(); ++i) std::cout<<thread_offset_test[i]<<" "; std::cout<<"\n";
			//free
			delete[] thread_dist_batch;
			delete[] thread_dist_train;
			delete[] thread_dist_val;
			delete[] thread_dist_test;
			delete[] thread_offset_train;
			delete[] thread_offset_val;
			delete[] thread_offset_test;
		}
		
		//==== b-cast data ====
		if(WORLD.rank()==0) std::cout<<"broadcasting data\n";
		parallel::bcast(WORLD.label(),0,nnopt);
		MPI_Bcast(&dIn,1,MPI_INT,0,MPI_COMM_WORLD);
		MPI_Bcast(&dOut,1,MPI_INT,0,MPI_COMM_WORLD);
		if(WORLD.rank()==0) std::cout<<"broadcasting train\n";
		if(ntrain>0){
			int* thread_dist=new int[WORLD.size()];
			int* thread_offset=new int[WORLD.size()];
			MPI_Gather(&dist_train.size(),1,MPI_INT,thread_dist,1,MPI_INT,0,WORLD.label());
			MPI_Gather(&dist_train.offset(),1,MPI_INT,thread_offset,1,MPI_INT,0,WORLD.label());
			data_train_l.resize(thread_dist[WORLD.rank()],dIn,dOut);
			parallel::scatterv(data_train_g.in(),data_train_l.in(),thread_dist,thread_offset);
			parallel::scatterv(data_train_g.out(),data_train_l.out(),thread_dist,thread_offset);
			delete[] thread_dist;
			delete[] thread_offset;
		}
		if(WORLD.rank()==0) std::cout<<"broadcasting val\n";
		if(nval>0){
			int* thread_dist=new int[WORLD.size()];
			int* thread_offset=new int[WORLD.size()];
			MPI_Gather(&dist_val.size(),1,MPI_INT,thread_dist,1,MPI_INT,0,WORLD.label());
			MPI_Gather(&dist_val.offset(),1,MPI_INT,thread_offset,1,MPI_INT,0,WORLD.label());
			data_val_l.resize(thread_dist[WORLD.rank()],dIn,dOut);
			parallel::scatterv(data_val_g.in(),data_val_l.in(),thread_dist,thread_offset);
			parallel::scatterv(data_val_g.out(),data_val_l.out(),thread_dist,thread_offset);
			delete[] thread_dist;
			delete[] thread_offset;
		}
		if(WORLD.rank()==0) std::cout<<"broadcasting test\n";
		if(ntest>0){
			int* thread_dist=new int[WORLD.size()];
			int* thread_offset=new int[WORLD.size()];
			MPI_Gather(&dist_test.size(),1,MPI_INT,thread_dist,1,MPI_INT,0,WORLD.label());
			MPI_Gather(&dist_test.offset(),1,MPI_INT,thread_offset,1,MPI_INT,0,WORLD.label());
			data_test_l.resize(thread_dist[WORLD.rank()],dIn,dOut);
			parallel::scatterv(data_test_g.in(),data_test_l.in(),thread_dist,thread_offset);
			parallel::scatterv(data_test_g.out(),data_test_l.out(),thread_dist,thread_offset);
			delete[] thread_dist;
			delete[] thread_offset;
		}
		
		//==== execute optimization ====
		if(WORLD.rank()==0) std::cout<<"executing optimization\n";
		nnopt.train(dist_batch.size(),data_train_l,data_val_l);
		
		//==== write the data ====
		if(WORLD.rank()==0){
			std::cout<<"writing data\n";
			//test
			writer=fopen("nn_fit_train.dat","w");
			if(writer!=NULL){
				//print header
				fprintf(writer,"#");
				for(int i=0; i<dIn; ++i) fprintf(writer,"X%i ",i);
				for(int i=0; i<dOut; ++i) fprintf(writer,"Y%i ",i);
				fprintf(writer,"\n");
				for(int i=0; i<data_train_g.size(); ++i){
					const Eigen::VectorXd& in=data_train_g.in(i);
					const Eigen::VectorXd& out=nnopt.nn().execute(data_train_g.in(i));
					for(int j=0; j<dIn; ++j) fprintf(writer,"%f ",in[j]);
					for(int j=0; j<dOut; ++j) fprintf(writer,"%f ",out[j]);
					fprintf(writer,"\n");
				}
				fclose(writer);
				writer=NULL;
			} else std::cout<<"could not open data file\n";
			//test
			writer=fopen("nn_fit_val.dat","w");
			if(writer!=NULL){
				//print header
				fprintf(writer,"#");
				for(int i=0; i<dIn; ++i) fprintf(writer,"X%i ",i);
				for(int i=0; i<dOut; ++i) fprintf(writer,"Y%i ",i);
				fprintf(writer,"\n");
				for(int i=0; i<data_val_g.size(); ++i){
					const Eigen::VectorXd& in=data_val_g.in(i);
					const Eigen::VectorXd& out=nnopt.nn().execute(data_val_g.in(i));
					for(int j=0; j<dIn; ++j) fprintf(writer,"%f ",in[j]);
					for(int j=0; j<dOut; ++j) fprintf(writer,"%f ",out[j]);
					fprintf(writer,"\n");
				}
				fclose(writer);
				writer=NULL;
			} else std::cout<<"could not open data file\n";
			//test
			writer=fopen("nn_fit_test.dat","w");
			if(writer!=NULL){
				//print header
				fprintf(writer,"#");
				for(int i=0; i<dIn; ++i) fprintf(writer,"X%i ",i);
				for(int i=0; i<dOut; ++i) fprintf(writer,"Y%i ",i);
				fprintf(writer,"\n");
				for(int i=0; i<data_test_g.size(); ++i){
					const Eigen::VectorXd& in=data_test_g.in(i);
					const Eigen::VectorXd& out=nnopt.nn().execute(data_test_g.in(i));
					for(int j=0; j<dIn; ++j) fprintf(writer,"%f ",in[j]);
					for(int j=0; j<dOut; ++j) fprintf(writer,"%f ",out[j]);
					fprintf(writer,"\n");
				}
				fclose(writer);
				writer=NULL;
			} else std::cout<<"could not open data file\n";
		}
		
		//==== stop wall clock ====
		if(WORLD.rank()==0) clock_wall.end();
		if(WORLD.rank()==0) std::cout<<"duration = "<<clock_wall.duration()<<"\n";
		
	} catch(std::exception& e){
		std::cout<<"ERROR in nn_fit::main(int,char**):\n";
		std::cout<<e.what()<<"\n";
	}
	
	//==== free memory ====
	delete[] input;
	delete[] strbuf;
}