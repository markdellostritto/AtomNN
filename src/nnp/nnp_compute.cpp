// mpi
#include <mpi.h>
// c libraries
#include <cstdio>
#include <ctime>
// c++ libraries
#include <iostream>
#include <exception>
#include <algorithm>
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
// ann - chem
#include "src/chem/units.hpp"
// ann - ml
#include "src/ml/nn.hpp"
#include "src/nnp/nnp.hpp"
// ann - string
#include "src/str/string.hpp"
#include "src/str/print.hpp"
// ann - thread
#include "src/thread/parallel.hpp"
// ann - util
#include "src/util/compiler.hpp"
#include "src/util/time.hpp"

parallel::Comm WORLD;//all processors

int main(int argc, char* argv[]){
	//==== file i/o ====
		FILE* reader=NULL;
		char* paramfile=new char[string::M];
		char* input    =new char[string::M];
		char* simstr   =new char[string::M];
		char* potstr   =new char[string::M];
		char* strbuf   =new char[print::len_buf];
		std::vector<std::string> strlist;
	//==== simulation variables ====
		AtomType atomT;
		atomT.name=true; atomT.an=false; atomT.type=true; atomT.index=false;
		atomT.posn=true; atomT.force=true; atomT.symm=true; atomT.charge=false;
		FILE_FORMAT::type format;//format of training data
		std::vector<Structure> strucs;
		std::vector<std::string> data;
		std::vector<std::string> files;
		bool calc_force=false;
	//==== nn potential - opt ====
		NNP nnp;
	//==== timing ====
		Clock clock;
	//==== units ====
		units::System unitsys=units::System::UNKNOWN;
	//==== thread ====
		parallel::Dist dist;
	
	//==== initialize mpi ====
	MPI_Init(&argc,&argv);
	WORLD.label()=MPI_COMM_WORLD;
	MPI_Comm_size(WORLD.label(),&WORLD.size());
	MPI_Comm_rank(WORLD.label(),&WORLD.rank());
	
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
		
		//==== rank 0 reads parameters ====
		if(WORLD.rank()==0){
		
			//==== copy the parameter file ====
			std::cout<<"reading parameter file\n";
			std::strcpy(paramfile,argv[1]);
			
			//==== read in the general parameters ====
			reader=fopen(paramfile,"r");
			if(reader==NULL) throw std::runtime_error("I/O Error: could not open parameter file.");
			while(fgets(input,string::M,reader)!=NULL){
				string::trim_right(input,string::COMMENT);
				string::split(input,string::WS,strlist);
				if(strlist.size()==0) continue;
				string::to_upper(strlist.at(0));
				if(strlist.at(0)=="DATA"){
					data.push_back(strlist.at(1));
				} else if(strlist.at(0)=="NNPOT"){
					std::strcpy(potstr,strlist.at(1).c_str());
				} else if(strlist.at(0)=="FORMAT"){
					format=FILE_FORMAT::read(string::to_upper(strlist.at(1)).c_str());
				} else if(strlist.at(0)=="UNITS"){
					unitsys=units::System::read(string::to_upper(strlist.at(1)).c_str());
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
			std::cout<<"NNPOT  = "<<potstr<<"\n";
			std::cout<<print::title("GENERAL PARAMETERS",strbuf)<<"\n";
			std::cout<<print::buf(strbuf)<<"\n";
			
			//==== check the parameters ====
			if(unitsys==units::System::UNKNOWN) throw std::invalid_argument("Invalid unit system.");
			if(format==FILE_FORMAT::UNKNOWN) throw std::invalid_argument("Invalid file format.");
			if(data.size()==0) throw std::invalid_argument("No data");
		
		}
		
		//==== broadcast the parameters ====
		MPI_Bcast(potstr,string::M,MPI_CHAR,0,WORLD.label());
		MPI_Bcast(&unitsys,1,MPI_INT,0,WORLD.label());
		MPI_Bcast(&format,1,MPI_INT,0,WORLD.label());
		
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
		parallel::bcast(WORLD.label(),0,files);
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
		
		//==== compute ====
		if(WORLD.rank()==0) std::cout<<"set the type\n";
		for(int i=0; i<strucs.size(); ++i){
			for(int n=0; n<strucs[i].nAtoms(); ++n){
				strucs[i].type(n)=nnp.index(strucs[i].name(n));
			}
		}
		if(WORLD.rank()==0) std::cout<<"initializing symmetry functions\n";
		for(int i=0; i<strucs.size(); ++i){
			NNP::init(nnp,strucs[i]);
		}
		if(WORLD.rank()==0) std::cout<<"computing\n";
		std::vector<double> energy(strucs.size(),0.0);
		for(int i=0; i<strucs.size(); ++i){
			NeighborList nlist;
			nlist.build(strucs[i],nnp.rc());
			NNP::symm(nnp,strucs[i],nlist);
			NNP::energy(nnp,strucs[i]);
			//nnp.force(strucs[i],nlist);
			energy[i]=strucs[i].energy();
		}
		
		//==== print ====
		for(int n=0; n<WORLD.size(); ++n){
			if(n==WORLD.rank()){
				for(int i=0; i<strucs.size(); ++i){
					printf("energy %i = %.6f\n",dist.index(i),energy[i]);
				}
			}
			MPI_Barrier(WORLD.label());
		}
		for(int n=0; n<WORLD.size(); ++n){
			if(n==WORLD.rank()){
				for(int i=0; i<strucs.size(); ++i){
					printf("energy/atom %i  = %.6f\n",dist.index(i),energy[i]/strucs[i].nAtoms());
				}
			}
			MPI_Barrier(WORLD.label());
		}
		/*for(int n=0; n<WORLD.size(); ++n){
			if(n==WORLD.rank()){
				for(int i=0; i<strucs.size(); ++i){
					std::cout<<"struc["<<dist.index(i)<<"]\n";
					for(int j=0; j<strucs[i].nAtoms(); ++j){
						std::cout<<"force["<<i<<"] = "<<strucs[i].force(j).transpose()<<"\n";
					}
				}
			}
			MPI_Barrier(WORLD.label());
		}*/
		
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