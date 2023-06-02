// mpi
#include <mpi.h>
// c libraries
#include <cstdio>
#include <ctime>
// c++ libraries
#include <iostream>
#include <exception>
#include <algorithm>
// structure
#include "src/struc/structure.hpp"
#include "src/struc/neighbor.hpp"
// format
#include "src/format/file_struc.hpp"
#include "src/format/format.hpp"
#include "src/format/vasp_struc.hpp"
#include "src/format/qe_struc.hpp"
#include "src/format/xyz_struc.hpp"
#include "src/format/cp2k_struc.hpp"
// chem
#include "src/chem/units.hpp"
// ml
#include "src/ml/nn.hpp"
#include "src/nnp/nnp.hpp"
// string
#include "src/str/string.hpp"
#include "src/str/token.hpp"
#include "src/str/print.hpp"
// thread
#include "src/thread/comm.hpp"
#include "src/thread/dist.hpp"
#include "src/thread/mpif.hpp"
// util
#include "src/util/compiler.hpp"
#include "src/util/time.hpp"
// torch
#include "src/torch/qeq.hpp"
#include "src/torch/pot.hpp"
#include "src/torch/pot_factory.hpp"
#include "src/torch/pot_gauss_cut.hpp"
#include "src/torch/pot_gauss_long.hpp"

static const double FCHI=1.0;
thread::Comm WORLD;//all processors

int main(int argc, char* argv[]){
	//==== file i/o ====
		FILE* reader=NULL;
		char* paramfile=new char[string::M];
		char* input    =new char[string::M];
		char* simstr   =new char[string::M];
		char* potstr   =new char[string::M];
		char* strbuf   =new char[print::len_buf];
		Token token;
	//==== simulation variables ====
		AtomType atomT;
		atomT.name=true; atomT.an=true; atomT.type=true; atomT.index=true;
		atomT.posn=true; atomT.force=true; atomT.symm=true; atomT.charge=true;
		atomT.eta=true; atomT.chi=true;
		FILE_FORMAT::type format;//format of training data
		std::vector<Structure> strucs;
		std::vector<std::string> data;
		std::vector<std::string> files;
		bool calc_force=false;
		bool norm=false;
		bool print=false;
	//==== qeq ====
		QEQ qeq;
	//==== nn potential ====
		NNP nnp;
		bool force=false;
	//==== timing ====
		Clock clock;
	//==== units ====
		units::System unitsys=units::System::UNKNOWN;
	//==== thread ====
		thread::Dist dist;
	
	//==== initialize mpi ====
	MPI_Init(&argc,&argv);
	WORLD.mpic()=MPI_COMM_WORLD;
	MPI_Comm_size(WORLD.mpic(),&WORLD.size());
	MPI_Comm_rank(WORLD.mpic(),&WORLD.rank());
	
	try{
		if(argc!=2) throw std::invalid_argument("Invalid number of command-line arguments.");
		
		//==== start wall clock ====
		if(WORLD.rank()==0) clock.begin();
		
		//==== print title ====
		if(WORLD.rank()==0){
			std::cout<<print::buf(strbuf,'*')<<"\n";
			std::cout<<print::buf(strbuf,'*')<<"\n";
			std::cout<<print::title("ANN - COMPUTE",strbuf,' ')<<"\n";
			std::cout<<print::buf(strbuf,'*')<<"\n";
			std::cout<<print::buf(strbuf,'*')<<"\n";
		}
		
		//==== print compiler information ====
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
		
		//==== print mathematical constants ====
		if(WORLD.rank()==0){
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("MATHEMATICAL CONSTANTS",strbuf)<<"\n";
			std::printf("PI    = %.15f\n",math::constant::PI);
			std::printf("RadPI = %.15f\n",math::constant::RadPI);
			std::printf("Rad2  = %.15f\n",math::constant::Rad2);
			std::cout<<print::title("MATHEMATICAL CONSTANTS",strbuf)<<"\n";
			std::cout<<print::buf(strbuf)<<"\n";
		}
		
		//==== print physical constants ====
		if(WORLD.rank()==0){
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("PHYSICAL CONSTANTS",strbuf)<<"\n";
			std::printf("bohr-r  (A)  = %.12f\n",units::BOHR);
			std::printf("hartree (eV) = %.12f\n",units::HARTREE);
			std::cout<<print::title("PHYSICAL CONSTANTS",strbuf)<<"\n";
			std::cout<<print::buf(strbuf)<<"\n";
		}
		
		//==== set mpi data ====
		{
			int* ranks=new int[WORLD.size()];
			MPI_Gather(&WORLD.rank(),1,MPI_INT,ranks,1,MPI_INT,0,WORLD.mpic());
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
		
		//==== rank 0 reads parameters ====
		if(WORLD.rank()==0){
		
			//==== copy the parameter file ====
			std::cout<<"reading parameter file\n";
			std::strcpy(paramfile,argv[1]);
			
			//==== read in the general parameters ====
			reader=fopen(paramfile,"r");
			if(reader==NULL) throw std::runtime_error("I/O Error: could not open parameter file.");
			while(fgets(input,string::M,reader)!=NULL){
				token.read(string::trim_right(input,string::COMMENT),string::WS);
				if(token.end()) continue; //skip empty lines
				const std::string tag=string::to_upper(token.next());
				if(tag=="DATA"){
					data.push_back(token.next());
				} else if(tag=="NNPOT"){
					std::strcpy(potstr,token.next().c_str());
				} else if(tag=="FORMAT"){
					format=FILE_FORMAT::read(string::to_upper(token.next()).c_str());
				} else if(tag=="UNITS"){
					unitsys=units::System::read(string::to_upper(token.next()).c_str());
				} else if(tag=="FORCE"){
					force=string::boolean(token.next().c_str());
				} else if(tag=="NORM"){
					norm=string::boolean(token.next().c_str());
				} else if(tag=="PRINT"){
					print=string::boolean(token.next().c_str());
				} else if(tag=="POT_QEQ"){
					ptnl::read(qeq.pot(),token);
				}
			}
			//close the file
			fclose(reader);
			reader=NULL;
			
			//==== print the parameters ====
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("GENERAL PARAMETERS",strbuf)<<"\n";
			std::cout<<"DATA   = \n"; for(int i=0; i<data.size(); ++i) std::cout<<"\t"<<data[i]<<"\n";
			std::cout<<"FORMAT = "<<format<<"\n";
			std::cout<<"UNITS  = "<<unitsys<<"\n";
			std::cout<<"FORCE  = "<<force<<"\n";
			std::cout<<"NORM   = "<<norm<<"\n";
			std::cout<<"NNPOT  = "<<potstr<<"\n";
			std::cout<<"PQEQ   = "<<qeq.pot()<<"\n";
			std::cout<<"QEQ    = "<<qeq<<"\n";
			std::cout<<print::buf(strbuf)<<"\n";
			
			//==== check the parameters ====
			if(unitsys==units::System::UNKNOWN) throw std::invalid_argument("Invalid unit system.");
			if(format==FILE_FORMAT::UNKNOWN) throw std::invalid_argument("Invalid file format.");
			if(data.size()==0) throw std::invalid_argument("No data");
		}
		
		//==== broadcast the parameters ====
		MPI_Bcast(potstr,string::M,MPI_CHAR,0,WORLD.mpic());
		MPI_Bcast(&unitsys,1,MPI_INT,0,WORLD.mpic());
		MPI_Bcast(&format,1,MPI_INT,0,WORLD.mpic());
		MPI_Bcast(&force,1,MPI_C_BOOL,0,WORLD.mpic());
		MPI_Bcast(&norm,1,MPI_C_BOOL,0,WORLD.mpic());
		thread::bcast(WORLD.mpic(),0,qeq);
		
		//==== initialize the unit system ====
		std::cout<<"initializing the unit system\n";
		units::consts::init(unitsys);
		
		//==== read the data ====
		if(WORLD.rank()==0){
			std::cout<<"reading data\n";
			for(int i=0; i<data.size(); ++i){
				//open the data file
				reader=fopen(data[i].c_str(),"r");
				if(reader==NULL) throw std::runtime_error(std::string("I/O Error: Could not open data file: ")+data[i]);
				//read in the data
				while(fgets(input,string::M,reader)!=NULL){
					if(!string::empty(input)) files.push_back(std::string(string::trim(input)));
					std::cout<<files.back()<<"\n";
				}
				//close the file
				fclose(reader);
				reader=NULL;
			}
		}
		
		//==== broadcast the files ====
		thread::bcast(WORLD.mpic(),0,files);
		//==== generate the distribution over the files
		dist.init(WORLD.size(),WORLD.rank(),files.size());
		
		//==== read the structure ====
		if(WORLD.rank()==0) std::cout<<"reading structures\n";
		if(dist.size()>0){
			strucs.resize(dist.size());
			if(format==FILE_FORMAT::QE){
				for(int i=0; i<dist.size(); ++i){
					QE::OUT::read(files[dist.index(i)].c_str(),atomT,strucs[i]);
				}
			} else if(format==FILE_FORMAT::POSCAR){
				for(int i=0; i<dist.size(); ++i){
					VASP::POSCAR::read(files[dist.index(i)].c_str(),atomT,strucs[i]);
				}
			} else if(format==FILE_FORMAT::XYZ){
				for(int i=0; i<dist.size(); ++i){
					XYZ::read(files[dist.index(i)].c_str(),atomT,strucs[i]);
				}
			} else throw std::invalid_argument("Invalid file format.");
		}
		
		//==== read the potential ====
		if(WORLD.rank()==0) std::cout<<"reading potential\n";
		NNP::read(potstr,nnp);
		if(WORLD.rank()==0) std::cout<<nnp<<"\n";
		
		//======== initialize the qeq potential ========
		if(WORLD.rank()==0) std::cout<<"initializing the qeq potential\n";
		qeq.pot()->resize(nnp.ntypes());
		if(qeq.pot()->name()==ptnl::Pot::Name::GAUSS_CUT){
			ptnl::PotGaussCut& pot=static_cast<ptnl::PotGaussCut&>(*qeq.pot());
			for(int i=0; i<nnp.ntypes(); ++i){
				pot.radius(i)=nnp.nnh(i).type().rcov().val();
				pot.f(i)=1;
			}
		}
		if(qeq.pot()->name()==ptnl::Pot::Name::GAUSS_LONG){
			ptnl::PotGaussLong& pot=static_cast<ptnl::PotGaussLong&>(*qeq.pot());
			for(int i=0; i<nnp.ntypes(); ++i){
				pot.radius(i)=nnp.nnh(i).type().rcov().val();
				pot.f(i)=1;
			}
		}
		qeq.pot()->init();
		
		//==== set atom properties ====
		if(WORLD.rank()==0) std::cout<<"setting atom properties\n";
		//index
		if(WORLD.rank()==0) std::cout<<"index\n";
		for(int i=0; i<strucs.size(); ++i){
			for(int n=0; n<strucs[i].nAtoms(); ++n){
				strucs[i].index(n)=n;
			}
		}
		//type
		if(WORLD.rank()==0) std::cout<<"type\n";
		for(int i=0; i<strucs.size(); ++i){
			for(int n=0; n<strucs[i].nAtoms(); ++n){
				strucs[i].type(n)=nnp.index(strucs[i].name(n));
			}
		}
		//charge
		if(WORLD.rank()==0) std::cout<<"charge\n";
		for(int i=0; i<strucs.size(); ++i){
			for(int n=0; n<strucs[i].nAtoms(); ++n){
				strucs[i].charge(n)=nnp.nnh(strucs[i].type(n)).type().charge().val();
			}
		}
		//chi
		if(WORLD.rank()==0) std::cout<<"chi\n";
		for(int i=0; i<strucs.size(); ++i){
			for(int n=0; n<strucs[i].nAtoms(); ++n){
				strucs[i].chi(n)=nnp.nnh(strucs[i].type(n)).type().chi().val();
			}
		}
		//eta
		if(WORLD.rank()==0) std::cout<<"eta\n";
		for(int i=0; i<strucs.size(); ++i){
			for(int n=0; n<strucs[i].nAtoms(); ++n){
				strucs[i].eta(n)=nnp.nnh(strucs[i].type(n)).type().eta().val();
			}
		}
		
		//==== compute ====
		if(WORLD.rank()==0) std::cout<<"initializing symmetry functions\n";
		for(int i=0; i<strucs.size(); ++i){
			NNP::init(nnp,strucs[i]);
		}
		if(WORLD.rank()==0) std::cout<<"computing\n";
		std::vector<double> energy_ref(strucs.size(),0.0);
		std::vector<double> energy_nnp(strucs.size(),0.0);
		std::vector<double> energy_q(strucs.size(),0.0);
		std::vector<double> energy_v(strucs.size(),0.0);
		for(int i=0; i<strucs.size(); ++i){
			energy_ref[i]=strucs[i].energy();
			NeighborList nlist;
			//nnp
			nlist.build(strucs[i],nnp.rc());
			NNP::symm(nnp,strucs[i],nlist);
			double energyV=0;
			for(int n=0; n<strucs[i].nAtoms(); ++n){
				//set the index
				const int ii=nnp.index(strucs[i].name(n));
				//execute the network
				nnp.nnh(ii).nn().execute(strucs[i].symm(n));
				//add to the energy
				energyV+=nnp.nnh(ii).nn().out()[0]+nnp.nnh(ii).type().energy().val();
				//compute chi
				strucs[i].chi(n)=nnp.nnh(ii).nn().out()[1]+nnp.nnh(ii).type().chi().val()*FCHI;
			}
			energy_v[i]=energyV;
			//qeq
			nlist.build(strucs[i],qeq.pot()->rc());
			qeq.qt(strucs[i],nlist);
			const double energyQ=-0.5*qeq.x().dot(qeq.b());
			energy_q[i]=energyQ;
			//force
			if(force) NNP::force(nnp,strucs[i],nlist);
			energy_nnp[i]=energyQ+energyV;
		}
		
		//==== print ====
		for(int n=0; n<WORLD.size(); ++n){
			if(n==WORLD.rank()){
				for(int i=0; i<strucs.size(); ++i){
					const double fnorm=(norm)?1.0/(1.0*strucs[i].nAtoms()):1.0;
					printf("energy %i %.6f %.6f %.6f %.6f\n",
						dist.index(i),energy_ref[i]*fnorm,energy_nnp[i]*fnorm,energy_q[i]*fnorm,energy_v[i]*fnorm
					);
				}
			}
			MPI_Barrier(WORLD.mpic());
		}
		if(print){
			for(int n=0; n<WORLD.size(); ++n){
				if(n==WORLD.rank()){
					for(int i=0; i<strucs.size(); ++i){
						std::cout<<"struc["<<dist.index(i)<<"]\n";
						for(int j=0; j<strucs[i].nAtoms(); ++j){
							std::cout<<strucs[i].name(j)<<" "<<strucs[i].type(j)<<" "<<
							strucs[i].posn(j)[0]<<" "<<strucs[i].posn(j)[1]<<" "<<strucs[i].posn(j)[2]<<" "<<
							strucs[i].charge(j)<<" "<<strucs[i].chi(j)<<" "<<strucs[i].eta(j)<<"\n";
						}
					}
				}
				std::cout<<std::flush;
				MPI_Barrier(WORLD.mpic());
			}
		}
		if(force){
			for(int n=0; n<WORLD.size(); ++n){
				if(n==WORLD.rank()){
					for(int i=0; i<strucs.size(); ++i){
						std::cout<<"struc["<<dist.index(i)<<"]\n";
						for(int j=0; j<strucs[i].nAtoms(); ++j){
							std::cout<<"force["<<i<<"] = "<<strucs[i].force(j).transpose()<<"\n";
						}
					}
				}
				MPI_Barrier(WORLD.mpic());
			}
		}
		
		//==== stop the wall clock ====
		if(WORLD.rank()==0) clock.end();
		if(WORLD.rank()==0) std::cout<<"time = "<<clock.duration()<<"\n";
		
	}catch(std::exception& e){
		std::cout<<"ERROR in nn_pot_compute::main(int,char**):\n";
		std::cout<<e.what()<<"\n";
	}
	
	//==== finalize mpi ====
	MPI_Finalize();
	
	
	//==== free memory ====
	delete[] paramfile;
	delete[] input;
	delete[] simstr;
	delete[] potstr;
	delete[] strbuf;
		
}