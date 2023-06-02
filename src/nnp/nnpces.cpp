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
#include "src/struc/sim.hpp"
#include "src/struc/neighbor.hpp"
// format
#include "src/format/file_sim.hpp"
#include "src/format/format.hpp"
#include "src/format/vasp_sim.hpp"
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
// t
// util
#include "src/util/compiler.hpp"
#include "src/util/time.hpp"

thread::Comm WORLD;//all processors

int main(int argc, char* argv[]){
	//==== file i/o ====
		FILE* reader=NULL;
		char* paramfile=new char[string::M];
		char* input    =new char[string::M];
		char* strbuf   =new char[print::len_buf];
		std::string simstr;
		std::string potstr;
		Token token;
	//==== simulation variables ====
		AtomType atomT;
		atomT.name=true; atomT.an=true; atomT.type=true; atomT.index=true;
		atomT.posn=true; atomT.force=true; atomT.symm=true; atomT.charge=false;
		FILE_FORMAT::type format;//format of training data
		Simulation sim;
		Interval interval;
		bool calc_force=false;
		bool norm=false;
		bool force=false;
	//==== nn potential - opt ====
		NNP nnp;
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
			std::cout<<print::title("ANN - COMPUTE - SIM",strbuf,' ')<<"\n";
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
				if(tag=="SIM"){
					simstr=token.next();
				} else if(tag=="NNPOT"){
					potstr=token.next();
				} else if(tag=="INTERVAL"){
					Interval::read(token.next().c_str(),interval);
				} else if(tag=="FORMAT"){
					format=FILE_FORMAT::read(string::to_upper(token.next()).c_str());
				} else if(tag=="UNITS"){
					unitsys=units::System::read(string::to_upper(token.next()).c_str());
				} else if(tag=="NORM"){
					norm=string::boolean(token.next().c_str());
				}
			}
			//close the file
			fclose(reader);
			reader=NULL;
			
			//==== print the parameters ====
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("GENERAL PARAMETERS",strbuf)<<"\n";
			std::cout<<"SIM      = "<<simstr<<"\n";
			std::cout<<"FORMAT   = "<<format<<"\n";
			std::cout<<"INTERVAL = "<<interval<<"\n";
			std::cout<<"UNITS  = "<<unitsys<<"\n";
			std::cout<<"FORCE  = "<<force<<"\n";
			std::cout<<"NORM   = "<<norm<<"\n";
			std::cout<<"NNPOT    = "<<potstr<<"\n";
			std::cout<<print::title("GENERAL PARAMETERS",strbuf)<<"\n";
			std::cout<<print::buf(strbuf)<<"\n";
			
			//==== check the parameters ====
			if(unitsys==units::System::UNKNOWN) throw std::invalid_argument("Invalid unit system.");
			if(format==FILE_FORMAT::UNKNOWN) throw std::invalid_argument("Invalid file format.");
			
		}
		
		//==== broadcast the parameters ====
		if(WORLD.rank()==0) std::cout<<"broadcasting parameters\n";
		thread::bcast(WORLD.mpic(),0,interval);
		thread::bcast(WORLD.mpic(),0,simstr);
		thread::bcast(WORLD.mpic(),0,potstr);
		MPI_Bcast(&unitsys,1,MPI_INT,0,WORLD.mpic());
		MPI_Bcast(&format,1,MPI_INT,0,WORLD.mpic());
		MPI_Bcast(&force,1,MPI_C_BOOL,0,WORLD.mpic());
		MPI_Bcast(&norm,1,MPI_C_BOOL,0,WORLD.mpic());
		
		//==== initialize the unit system ====
		if(WORLD.rank()==0) std::cout<<"initializing the unit system\n";
		units::consts::init(unitsys);
		
		//==== split the interval ====
		if(WORLD.rank()==0) std::cout<<"splitting the interval\n";
		Interval intloc=Interval::split(interval,WORLD.rank(),WORLD.size());
		for(int n=0; n<WORLD.size(); ++n){
			if(WORLD.rank()==n) std::cout<<"intloc["<<n<<"] = "<<intloc<<"\n"<<std::flush;
			MPI_Barrier(WORLD.mpic());
		}
		
		//==== read the potential ====
		if(WORLD.rank()==0) std::cout<<"reading potential\n";
		NNP::read(potstr.c_str(),nnp);
		if(WORLD.rank()==0) std::cout<<nnp<<"\n";
		
		//==== read the data ====
		if(WORLD.rank()==0) std::cout<<"reading simulation\n";
		read_sim(simstr.c_str(),format,intloc,atomT,sim);
		if(WORLD.rank()==0) std::cout<<sim.frame(0)<<"\n";
		
		//==== compute ====
		if(WORLD.rank()==0) std::cout<<"set the indices\n";
		for(int t=0; t<sim.timesteps(); ++t){
			for(int n=0; n<sim.frame(t).nAtoms(); ++n){
				sim.frame(t).index(n)=n;
			}
		}
		if(WORLD.rank()==0) std::cout<<"set the types\n";
		for(int t=0; t<sim.timesteps(); ++t){
			for(int n=0; n<sim.frame(t).nAtoms(); ++n){
				sim.frame(t).type(n)=nnp.index(sim.frame(t).name(n));
			}
		}
		if(WORLD.rank()==0) std::cout<<"computing\n";
		double* energyloc=new double[sim.timesteps()];
		for(int t=0; t<sim.timesteps(); ++t){
			NeighborList nlist;
			nlist.build(sim.frame(t),nnp.rc());
			NNP::init(nnp,sim.frame(t));
			NNP::symm(nnp,sim.frame(t),nlist);
			energyloc[t]=NNP::energy(nnp,sim.frame(t))/sim.frame(t).nAtoms();
		}
		
		//==== print ====
		if(WORLD.rank()==0) std::cout<<"printing energy\n";
		int* ts=new int[WORLD.size()];
		int tsloc=sim.timesteps();
		MPI_Gather(&tsloc,1,MPI_INT,ts,1,MPI_INT,0,WORLD.mpic());
		int tstot=0;
		for(int n=0; n<WORLD.size(); ++n){
			tstot+=ts[n];
			if(WORLD.rank()==0) std::cout<<"ts["<<n<<"] = "<<ts[n]<<"\n";
		}
		MPI_Bcast(&tstot,1,MPI_INT,0,WORLD.mpic());
		MPI_Bcast(ts,WORLD.size(),MPI_INT,0,WORLD.mpic());
		if(WORLD.rank()==0) std::cout<<"tstot = "<<tstot<<"\n";
		int* offsets=new int[WORLD.size()];
		offsets[0]=0;
		if(WORLD.rank()==0) std::cout<<"offset["<<0<<"] = "<<offsets[0]<<"\n";
		for(int n=1; n<WORLD.size(); ++n){
			offsets[n]=offsets[n-1]+ts[n-1];
			if(WORLD.rank()==0) std::cout<<"offset["<<n<<"] = "<<offsets[n]<<"\n";
		}
		MPI_Bcast(offsets,WORLD.size(),MPI_INT,0,WORLD.mpic());
		double* energytot=new double[tstot];
		for(int t=0; t<tstot; ++t) energytot[t]=0;
		MPI_Gatherv(energyloc,tsloc,MPI_DOUBLE,energytot,ts,offsets,MPI_DOUBLE,0,WORLD.mpic());
		for(int t=0; t<tstot; ++t){
			if(WORLD.rank()==0) printf("energy/atom %i  = %.12f\n",t,energytot[t]);
		}
		
		//==== free memory ====
		delete[] energytot;
		delete[] energyloc;
		delete[] ts;
		delete[] offsets;
    	
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
	delete[] strbuf;
		
}