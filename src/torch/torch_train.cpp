// omp
#include "src/thread/openmp.hpp"
// c++
#include <iostream>
#include <stdexcept>
#include <chrono>
#include <random>
// nnp
#include "src/nnp/type.hpp"
#include "src/nnp/nnp.hpp"
#include "src/ml/nn.hpp"
// torch
#include "src/torch/engine.hpp"
// chem
#include "src/chem/units.hpp"
#include "src/chem/ptable.hpp"
// struc
#include "src/struc/structure.hpp"
// format
#include "src/format/format.hpp"
#include "src/format/file_struc.hpp"
// str
#include "src/str/string.hpp"
#include "src/str/token.hpp"
#include "src/str/print.hpp"
// torch
#include "src/torch/integrator.hpp"
#include "src/torch/pot_read.hpp"
#include "src/torch/monte_carlo.hpp"
#include "src/torch/dump.hpp"
// util
#include "src/util/time.hpp"

int main(int argc, char* argv[]){
	//units
		units::System unitsys=units::System::UNKNOWN;
	//files
		std::string fparam;
		std::string fstruc;
		char* input=new char[string::M];
		char* strbuf=new char[print::len_buf];
		Token token;
	//struc
		AtomType atomT;
		Structure struc;
		FILE_FORMAT::type format;
		Eigen::Vector3i nlat;
		bool super=false;
		std::vector<Type> types;
		double Tinit=0;
		int nstep=0;
		int tau=0;
		double T=0;
		double dt=0;
		double beta=0;
		double gamma=0;
	//engine
		Engine engine;
		Dump dump;
	//nnp
		NNP nnp;
		std::vector<NN::DODP> dodp;
	//rand
		std::srand(std::time(NULL));
		int seed=std::chrono::system_clock::now().time_since_epoch().count();
		std::default_random_engine generator(seed);
	//time
		Clock clock;
	//misc
		bool error=false;
	
	try{
		//==== check the arguments ====
		if(argc!=2) throw std::invalid_argument("Torch::main(int,char**): Invalid number of arguments.");
		
		//==== omp ====
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("OMP",strbuf)<<"\n";
		#pragma omp parallel
		{if(omp_get_thread_num()==0) std::cout<<"num threads = "<<omp_get_num_threads()<<"\n";}
		std::cout<<print::title("OMP",strbuf)<<"\n";
		std::cout<<print::buf(strbuf)<<"\n";
		
		//==== open the parameter file ==== 
		fparam=argv[1];
		FILE* reader=fopen(fparam.c_str(),"r");
		if(reader==NULL) throw std::runtime_error("Torch::main(int,char**): Could not open parameter file.");
		
		//==== read the parameter file ==== 
		std::cout<<"reading general parameters\n";
		while(fgets(input,string::M,reader)!=NULL){
			string::trim_right(input,string::COMMENT);//trim comments
			Token token(input,string::WS); //split line into tokens
			if(token.end()) continue; //skip empty lines
			std::string tag=string::to_upper(token.next());
			if(tag=="FSTRUC"){
				fstruc=token.next();
			} else if(tag=="UNITS"){
				unitsys=units::System::read(string::to_upper(token.next()).c_str());
			} else if(tag=="FORMAT"){//simulation format
				format=FILE_FORMAT::read(string::to_upper(token.next()).c_str());
			} else if(tag=="SUPER"){
				nlat[0]=std::atoi(token.next().c_str());
				nlat[1]=std::atoi(token.next().c_str());
				nlat[2]=std::atoi(token.next().c_str());
				super=true;
			} else if(tag=="ENGINE"){
				engine.read(token);
			} else if(tag=="POT"){
				std::shared_ptr<ptnl::Pot> pot;
				ptnl::read(pot,token);
				engine.pots().push_back(pot);
			} else if(tag=="DUMP"){
				dump.read(token);
			} else if(tag=="TYPE"){
				types.push_back(Type());
				Type::read(input,types.back());
			} else if(tag=="ATOM_TYPE"){
				atomT=AtomType::read(token);
			} else if(tag=="TINIT"){
				Tinit=std::atof(token.next().c_str());
			} else if(tag=="NSTEP"){
				nstep=std::atoi(token.next().c_str());
			} else if(tag=="TEMP"){
				T=std::atof(token.next().c_str());
			} else if(tag=="TAU"){
				tau=std::atoi(token.next().c_str());
			} else if(tag=="DT"){
				dt=std::atof(token.next().c_str());
			} else if(tag=="BETA"){
				beta=std::atof(token.next().c_str());
			} else if(tag=="GAMMA"){
				gamma=std::atof(token.next().c_str());
			} else if(tag=="NNP"){
				NNP::read(token.next().c_str(),nnp);
			}
		}
		const int ntypes=types.size();
		
		//==== print ====
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("TORCH",strbuf)<<"\n";
		std::cout<<"ATOMT  = "<<atomT<<"\n";
		std::cout<<"TEMP   = "<<T<<"\n";
		std::cout<<"TAU    = "<<tau<<"\n";
		std::cout<<"DT     = "<<dt<<"\n";
		std::cout<<"BETA   = "<<beta<<"\n";
		std::cout<<"GAMMA  = "<<gamma<<"\n";
		std::cout<<"UNITS  = "<<unitsys<<"\n";
		std::cout<<"TINIT  = "<<Tinit<<"\n";
		std::cout<<"NSTEP  = "<<nstep<<"\n";
		std::cout<<"FSTRUC = "<<fstruc<<"\n";
		std::cout<<"FORMAT = "<<format<<"\n";
		std::cout<<"DUMP   = "<<dump<<"\n";
		if(super) std::cout<<"NLAT   = "<<nlat.transpose()<<"\n";
		for(int i=0; i<types.size(); ++i){
			std::cout<<"TYPE["<<i<<"] = "<<types[i]<<"\n";
		}
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<engine<<"\n";
		std::cout<<nnp<<"\n";
		
		//==== check the parameters ====
		if(format==FILE_FORMAT::UNKNOWN) throw std::invalid_argument("Invalid file format.");
		if(atomT.posn==false || atomT.mass==false || atomT.type==false || atomT.index==false) throw std::invalid_argument("Atom type missing basic data.");
		
		//==== set the unit system ====
		std::cout<<"setting the unit system\n";
		units::consts::init(unitsys);
		std::cout<<"ke = "<<units::consts::ke()<<"\n";
		std::cout<<"kb = "<<units::consts::kb()<<"\n";
		
		//==== read the structure ====
		std::cout<<"reading the structure\n";
		read_struc(fstruc.c_str(),format,atomT,struc);
		std::cout<<struc<<"\n";
		
		//==== set the indices ====
		std::cout<<"setting the indices\n";
		for(int i=0; i<struc.nAtoms(); ++i){
			struc.index(i)=i;
		}
		
		//==== set the types ====
		std::cout<<"setting the types\n";
		for(int i=0; i<struc.nAtoms(); ++i){
			bool match=false;
			for(int j=0; j<types.size(); ++j){
				if(types[j].name()==struc.name(i)){
					struc.type(i)=j; 
					match=true; break;
				}
			}
			if(!match) throw std::invalid_argument("Found no match for type.");
		}
		
		//==== set atomic properites ====
		
		//charge
		if(atomT.mass){
			std::cout<<"setting the mass\n";
			for(int i=0; i<struc.nAtoms(); ++i){
				for(int j=0; j<types.size(); ++j){
					if(types[j].name()==struc.name(i)){
						struc.mass(i)=types[j].mass().val();
						break;
					}
				}
			}
		}
		
		//charge
		if(atomT.charge){
			std::cout<<"setting the charge\n";
			for(int i=0; i<struc.nAtoms(); ++i){
				for(int j=0; j<types.size(); ++j){
					if(types[j].name()==struc.name(i)){
						struc.charge(i)=types[j].charge().val();
						break;
					}
				}
			}
		}
		
		//electronegativity
		if(atomT.chi){
			std::cout<<"setting the eletronegativity\n";
			for(int i=0; i<struc.nAtoms(); ++i){
				for(int j=0; j<types.size(); ++j){
					if(types[j].name()==struc.name(i)){
						struc.chi(i)=types[j].chi().val();
						break;
					}
				}
			}
		}
		
		//idempotential
		if(atomT.eta){
			std::cout<<"setting the idempotential\n";
			for(int i=0; i<struc.nAtoms(); ++i){
				for(int j=0; j<types.size(); ++j){
					if(types[j].name()==struc.name(i)){
						struc.eta(i)=types[j].eta().val();
						break;
					}
				}
			}
		}
		
		//==== make supercell ====
		if(super){
			std::cout<<"making supercell\n";
			Structure struc_super;
			Structure::super(struc,struc_super,nlat);
			struc=struc_super;
		}
		
		//==== resize the engine ====
		std::cout<<"resizing the engine\n";
		engine.resize(ntypes);
		
		//==== read the coefficients ====
		std::cout<<"reading coefficients\n";
		std::rewind(reader);
		while(fgets(input,string::M,reader)!=NULL){
			string::trim_right(input,string::COMMENT);//trim comments
			Token token(input,string::WS); //split line into tokens
			if(token.end()) continue; //skip empty lines
			std::string tag=string::to_upper(token.next());
			if(tag=="COEFF"){
				ptnl::coeff(engine.pots(),token);
			} 
		}
		
		//==== initialize the engine ====
		std::cout<<"initializing the engine\n";
		engine.init();
		
		//==== close parameter file ==== 
		std::fclose(reader);
		reader=NULL;
		
		//==== init velocities ====
		std::cout<<"initializing velocities\n";
		//randomize velocities
		for(int i=0; i<struc.nAtoms(); ++i){
			struc.force(i).setZero();
			struc.vel(i)=Eigen::Vector3d::Random();
		}
		//compute KE/T
		struc.ke()=0;
		for(int i=0; i<struc.nAtoms(); ++i){
			struc.ke()+=struc.mass(i)*struc.vel(i).squaredNorm();
		}
		struc.ke()*=0.5;
		struc.T()=struc.ke()*(2.0/3.0)/(struc.nAtoms()*units::consts::kb());
		//rescale velocities
		const double fac=sqrt(Tinit/(struc.T()+1e-6));
		for(int i=0; i<struc.nAtoms(); ++i){
			struc.vel(i)*=fac;
		}
		//compute KE/T
		struc.ke()=0;
		for(int i=0; i<struc.nAtoms(); ++i){
			struc.ke()+=struc.mass(i)*struc.vel(i).squaredNorm();
		}
		struc.ke()*=0.5;
		struc.T()=struc.ke()*(2.0/3.0)/(struc.nAtoms()*units::consts::kb());
		
		//==== compute ==== 
		Structure strucc=struc;
		Structure strucn=struc;
		NNP::init(nnp,strucn);
		dodp.resize(nnp.ntypes());
		std::vector<Eigen::VectorXd> grad;
		grad.resize(nnp.ntypes());
		for(int i=0; i<nnp.ntypes(); ++i){
			dodp[i].resize(nnp.nnh(i).nn());
			grad[i]=Eigen::VectorXd::Zero(nnp.nnh(i).nn().size());
		}
		clock.begin();
		switch(engine.job()){
			case Job::MD:{
				std::cout<<"JOB - MD\n";
				FILE* writer=fopen("out.dump","w");
				if(writer==NULL) throw std::runtime_error("Could not open dump file.");
				printf("N T KE PE PEC PEN TE EXP\n");
				for(int t=0; t<nstep; ++t){
					struc.t()=t;
					engine.nlist().build(struc);
					//verlet - first half-step
					for(int n=0; n<struc.nAtoms(); ++n){
						struc.vel(n).noalias()+=0.5*dt*struc.force(n)/struc.mass(n);
						struc.posn(n).noalias()+=struc.vel(n)*dt;
						Cell::returnToCell(struc.posn(n),struc.posn(n),struc.R(),struc.RInv());
					}
					//reset force/energy
					struc.pe()=0; 
					for(int n=0; n<struc.nAtoms(); ++n) struc.force(n).setZero();
					//compute - classical
					strucc=struc;
					const double pec=engine.compute(strucc);
					//compute - nnp
					strucn=struc;
					NNP::init(nnp,strucn);
					NNP::symm(nnp,strucn,engine.nlist());
					NNP::compute(nnp,strucn,engine.nlist());
					const double pen=strucn.energy();
					//combine
					const double deltaE=(pen-pec);
					const double expf=exp(-beta*beta*deltaE*deltaE);
					struc.pe()=(1.0-expf)*pec+expf*pen;
					for(int n=0; n<struc.nAtoms(); ++n){
						struc.force(n).noalias()=(1.0-expf)*strucc.force(n)+expf*strucn.force(n);
					}
					//second half-step
					for(int n=0; n<struc.nAtoms(); ++n){
						struc.vel(n).noalias()+=0.5*dt*struc.force(n)/struc.mass(n);
					}
					//update parameters
					for(int i=0; i<nnp.ntypes(); ++i){
						grad[i].setZero();
					}
					for(int n=0; n<struc.nAtoms(); ++n){
						const int type=nnp.index(strucn.name(n));
						nnp.nnh(type).nn().execute(strucn.symm(n));
						dodp[type].grad(nnp.nnh(type).nn());
						const double fac=deltaE/std::fabs(deltaE)*(1.0-expf)/strucn.nAtoms();
						int c=0;
						for(int l=0; l<nnp.nnh(type).nn().nlayer(); ++l){
							for(int i=0; i<nnp.nnh(type).nn().b(l).size(); ++i){
								grad[type][c++]+=fac*dodp[type].dodb()[0][l][i];
							}
						}
						for(int l=0; l<nnp.nnh(type).nn().nlayer(); ++l){
							for(int i=0; i<nnp.nnh(type).nn().w(l).size(); ++i){
								grad[type][c++]+=fac*dodp[type].dodw()[0][l](i);
							}
						}
					}
					for(int j=0; j<nnp.ntypes(); ++j){
						int c=0;
						for(int l=0; l<nnp.nnh(j).nn().nlayer(); ++l){
							for(int i=0; i<nnp.nnh(j).nn().b(l).size(); ++i){
								nnp.nnh(j).nn().b(l)(i)-=gamma*grad[j][c++];
							}
						}
						for(int l=0; l<nnp.nnh(j).nn().nlayer(); ++l){
							for(int i=0; i<nnp.nnh(j).nn().w(l).size(); ++i){
								nnp.nnh(j).nn().w(l)(i)-=gamma*grad[j][c++];
							}
						}
					}
					//increment
					++struc.t();
					//compute KE, T
					struc.ke()=0;
					for(int n=0; n<struc.nAtoms(); ++n){
						struc.ke()+=struc.mass(n)*struc.vel(n).squaredNorm();
					}
					struc.ke()*=0.5;
					struc.T()=struc.ke()*(2.0/3.0)/(struc.nAtoms()*units::consts::kb());
					//alter velocities
					if(struc.t()%tau==0){
						const double fac=sqrt(T/(struc.T()+1e-6));
						for(int i=0; i<struc.nAtoms(); ++i){
							struc.vel(i)*=fac;
						}
					}
					//compute KE, T
					struc.ke()=0;
					for(int n=0; n<struc.nAtoms(); ++n){
						struc.ke()+=struc.mass(n)*struc.vel(n).squaredNorm();
					}
					struc.ke()*=0.5;
					struc.T()=struc.ke()*(2.0/3.0)/(struc.nAtoms()*units::consts::kb());
					//print
					if(t%dump.nprint()==0) printf("%i %4.5f %4.5f %4.5f %4.5f %4.5f %4.5f %4.5f\n",t,struc.T(),struc.ke(),struc.pe(),pec,pen,struc.ke()+struc.pe(),expf);
					//write
					if(t%dump.nwrite()==0) Dump::write(struc,writer);
				}
				fclose(writer); writer=NULL;
			}break;
			default:{
				std::cout<<"WARNING: Invalid job.";
			}break;
		}
		clock.end();
		std::cout<<"time = "<<clock.duration()<<"\n";
		
	}catch(std::exception& e){
		std::cout<<"ERROR in Torch::main(int,char**):\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	//free memory
	std::cout<<"freeing memory\n";
	delete[] input;
	delete[] strbuf;
	
	if(error) return 1;
	else return 0;
}