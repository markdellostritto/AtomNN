// c libraries
#include <cstdio>
#include <ctime>
// c++ libraries
#include <iostream>
#include <string>
#include <stdexcept>
// eigen libraries
#include <Eigen/Dense>
// ann - structure
#include "src/struc/structure.hpp"
// ann - strings
#include "src/str/string.hpp"
// ann - chem
#include "src/chem/units.hpp"
#include "src/chem/ptable.hpp"
// ann - vasp
#include "src/format/vasp_sim.hpp"

//*****************************************************
//XDATCAR
//*****************************************************

namespace VASP{
	
namespace XDATCAR{

void read(const char* file, const Interval& interval, const AtomType& atomT, Simulation& sim){
	static const char* funcName="read(const char*,const Interval&,Simulation&)";
	if(VASP_PRINT_FUNC>0) std::cout<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<funcName<<":\n";
	//==== local function variables ====
	//file i/o
		FILE* reader=NULL;
		char* input=new char[string::M];
		char* str_name=new char[string::M];
		char* str_number=new char[string::M];
		std::string str;
	//simulation flags
		bool direct;//whether the coordinates are in direct or Cartesian coordinates
	//time info
		int ts=0;//number of timesteps
		int tsint=0;//requested interval
	//cell info
		double scale=1;
		Eigen::Matrix3d lv;
	//units
		double s_len=0.0;
		if(units::consts::system()==units::System::AU) s_len=1.0/units::BOHR;
		else if(units::consts::system()==units::System::METAL) s_len=1.0;
		else throw std::runtime_error("Invalid units.");
	//misc
		bool error=false;
		
	try{
		//start the timer
		const clock_t start=std::clock();
		
		//open the file
		if(VASP_PRINT_STATUS>0) std::cout<<"Opening file\n";
		reader=fopen(file,"r");
		if(reader==NULL) throw std::runtime_error("I/O Error: Unable to open file.");
		
		//==== clear the simulation ====
		if(VASP_PRINT_STATUS>0) std::cout<<"clearing simulation\n";
		sim.clear();
		
		//==== read in the system name ====
		if(VASP_PRINT_STATUS>0) std::cout<<"reading system name\n";
		sim.name()=string::trim(fgets(input,string::M,reader));
		
		//==== read the simulation cell ====
		if(VASP_PRINT_STATUS>0) std::cout<<"reading cell\n";
		std::sscanf(fgets(input,string::M,reader),"%lf",&scale);
		std::sscanf(fgets(input,string::M,reader),"%lf %lf %lf",&lv(0,0),&lv(1,0),&lv(2,0));
		std::sscanf(fgets(input,string::M,reader),"%lf %lf %lf",&lv(0,1),&lv(1,1),&lv(2,1));
		std::sscanf(fgets(input,string::M,reader),"%lf %lf %lf",&lv(0,2),&lv(1,2),&lv(2,2));
		lv*=s_len*scale;
		
		//==== read species ====
		if(VASP_PRINT_STATUS>0) std::cout<<"read species\n";
		//read number of species
		fgets(str_name,string::M,reader);
		fgets(str_number,string::M,reader);
		const int nNames=string::substrN(str_name,string::WS);
		const int nNumbers=string::substrN(str_number,string::WS);
		const int nSpecies=string::substrN(str_name,string::WS);
		if(nNames!=nNumbers) throw std::runtime_error("Invalid number of species");
		if(nSpecies<=0) throw std::runtime_error("Invalid number of species");
		//read in the species names
		std::vector<std::string> species(nSpecies);
		species[0]=std::strtok(str_name,string::WS);
		for(int i=1; i<nSpecies; ++i) species[i]=std::strtok(NULL,string::WS);
		//read in the species numbers
		std::vector<int> nAtoms(nSpecies);
		nAtoms[0]=std::atoi(std::strtok(str_number,string::WS));
		for(int i=1; i<nSpecies; ++i) nAtoms[i]=std::atoi(std::strtok(NULL,string::WS));
		int nAtomsT=0;
		for(int i=0; i<nSpecies; ++i) nAtomsT+=nAtoms[i];
		
		//==== read coord ====
		if(VASP_PRINT_STATUS>0) std::cout<<"read coord\n";
		fgets(input, string::M, reader);
		if(input[0]=='D') direct=true;
		else direct=false;
		
		//==== check if the cell is variable or not ====
		if(VASP_PRINT_STATUS>0) std::cout<<"Checking whether cell is variable\n";
		for(int n=0; n<nAtomsT; ++n) fgets(input, string::M, reader);
		str=std::string(string::trim(fgets(input,string::M,reader)));
		if(str==sim.name()) sim.cell_fixed()=false;
		else sim.cell_fixed()=true;
		// reset the line position 
		if(!sim.cell_fixed()) for(int i=0; i<HEADER_SIZE; ++i) fgets(input,string::M,reader);
		
		//==== find the number of timesteps ====
		if(VASP_PRINT_STATUS>0) std::cout<<"reading the number of timesteps\n";
		std::rewind(reader);
		//find the total number of lines in the file
		int nLines=0;
		while(fgets(input, string::M, reader)!=NULL){++nLines;};
		if(sim.cell_fixed()) ts=std::floor((1.0*nLines-HEADER_SIZE)/(1.0*nAtomsT+1.0));
		else ts=std::floor((1.0*nLines)/(1.0*nAtomsT+1.0+HEADER_SIZE));
		
		//==== reset the line position ====
		if(VASP_PRINT_STATUS>0) std::cout<<"resetting the line position\n";
		std::rewind(reader);
		if(sim.cell_fixed()) for(int i=0; i<HEADER_SIZE; ++i) fgets(input,string::M,reader);
		
		//==== set the interval ====
		if(VASP_PRINT_STATUS>0) std::cout<<"setting the interval\n";
		if(interval.beg<0) throw std::invalid_argument("Invalid beginning timestep.");
		const int beg=interval.beg-1;
		int end=interval.end-1;
		if(interval.end<0) end=ts+interval.end;
		tsint=end-beg+1;
		
		//==== print data to screen ====
		if(VASP_PRINT_DATA>0){
			std::cout<<"NAME    = "<<sim.name()<<"\n";
			std::cout<<"ATOMT   = "<<atomT<<"\n";
			std::cout<<"DIRECT  = "<<(direct?"T":"F")<<"\n";
			std::cout<<"CELL    = \n"<<lv<<"\n";
			std::cout<<"SPECIES = "; for(int i=0; i<nSpecies; ++i) std::cout<<species[i]<<" "; std::cout<<"\n";
			std::cout<<"NUMBERS = "; for(int i=0; i<nSpecies; ++i) std::cout<<nAtoms[i]<<" "; std::cout<<"\n";
			std::cout<<"NATOMST = "<<nAtomsT<<"\n";
			std::cout<<"INTERVAL   = "<<interval<<"\n";
			std::cout<<"TIMESTEPS  = "<<ts<<"\n";
			std::cout<<"N_STEPS    = "<<tsint/interval.stride<<"\n";
		}
		
		//==== resize the simulation ====
		if(VASP_PRINT_STATUS>0) std::cout<<"allocating memory\n";
		sim.resize(tsint/interval.stride,nAtomsT,atomT);
		
		//==== read positions ====
		if(VASP_PRINT_STATUS>0) std::cout<<"reading positions\n";
		//skip timesteps until beg is reached
		for(int t=0; t<beg; ++t){
			if(!sim.cell_fixed()) for(int i=0; i<HEADER_SIZE; ++i) fgets(input,string::M,reader); //skip header
			fgets(input,string::M,reader);//skip single line
			for(int n=0; n<nAtomsT; ++n) fgets(input,string::M,reader);
		}
		//read the positions
		if(sim.cell_fixed()){
			for(int t=0; t<sim.timesteps(); ++t) static_cast<Cell&>(sim.frame(t)).init(lv);
			for(int t=0; t<sim.timesteps(); ++t){
				if(VASP_PRINT_STATUS>1) std::cout<<"T = "<<t<<"\n";
				else if(t%1000==0) std::cout<<"T = "<<t<<"\n";
				fgets(input,string::M,reader);//skip line
				for(int n=0; n<nAtomsT; ++n){
					std::sscanf(
						fgets(input,string::M,reader),"%lf %lf %lf",
						&sim.frame(t).posn(n)[0],&sim.frame(t).posn(n)[1],&sim.frame(t).posn(n)[2]
					);
				}
				//skip "stride-1" steps
				for(int tt=0; tt<interval.stride-1; ++tt){
					fgets(input,string::M,reader);//skip line
					for(int n=0; n<nAtomsT; ++n){
						fgets(input,string::M,reader);
					}
				}
			}
		} else {
			for(int t=0; t<sim.timesteps(); ++t){
				if(VASP_PRINT_STATUS>1) std::cout<<"T = "<<t<<"\n";
				else if(t%1000==0) std::cout<<"T = "<<t<<"\n";
				//read in lattice vectors
				fgets(input,string::M,reader);//name
				scale=std::atof(fgets(input,string::M,reader));//scale
				for(int i=0; i<3; ++i){
					fgets(input, string::M, reader);
					lv(0,i)=std::atof(std::strtok(input,string::WS));
					for(int j=1; j<3; ++j){
						lv(j,i)=std::atof(std::strtok(NULL,string::WS));
					}
				}
				static_cast<Cell&>(sim.frame(t)).init(s_len*scale*lv);
				fgets(input,string::M,reader);//skip line (atom names)
				fgets(input,string::M,reader);//skip line (atom numbers)
				fgets(input,string::M,reader);//skip line (Direct or Cart)
				for(int n=0; n<nAtomsT; ++n){
					std::sscanf(
						fgets(input,string::M,reader),"%lf %lf %lf",
						&sim.frame(t).posn(n)[0],&sim.frame(t).posn(n)[1],&sim.frame(t).posn(n)[2]
					);
				}
				//skip "stride-1" steps
				for(int tt=0; tt<interval.stride-1; ++tt){
					fgets(input,string::M,reader);//name
					fgets(input,string::M,reader);//scale
					for(int i=0; i<3; ++i){
						fgets(input,string::M,reader);//lv
					}
					fgets(input,string::M,reader);//skip line (atom names)
					fgets(input,string::M,reader);//skip line (atom numbers)
					fgets(input,string::M,reader);//skip line (Direct or Cart)
					for(int n=0; n<nAtomsT; ++n){
						fgets(input,string::M,reader);
					}
				}
			}
		}
		
		//==== convert to Cartesian coordinates if necessary ====
		if(direct){
			if(VASP_PRINT_STATUS>-1) std::cout<<"converting to cartesian coordinates\n";
			for(int t=0; t<sim.timesteps(); ++t){
				for(int n=0; n<nAtomsT; ++n){
					sim.frame(t).posn(n)=sim.frame(t).R()*sim.frame(t).posn(n);
				}
			}
		} else if(s_len!=1.0){
			for(int t=0; t<sim.timesteps(); ++t){
				for(int n=0; n<nAtomsT; ++n){
					sim.frame(t).posn(n)*=s_len;
				}
			}
		}
		
		//==== close the file ====
		fclose(reader);
		reader=NULL;
		
		//==== stop the timer ====
		const clock_t stop=std::clock();
		
		//==== print the time ====
		const double time=((double)(stop-start))/CLOCKS_PER_SEC;
		std::cout<<"Simulation loaded in "<<time<<" seconds.\n";
	}catch(std::exception& e){
		std::cout<<"ERROR in "<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<funcName<<":\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	//free all local variables
	if(reader!=NULL) fclose(reader);
	delete[] input;
	
	if(error) throw std::runtime_error("I/O Exception: Could not read data.");
}

void write(const char* file, const Interval& interval, const AtomType& atomT, const Simulation& sim){
	/*
	const char* funcName="write(const char*,const Interval&,const AtomType&,Simulation&)";
	if(VASP_PRINT_FUNC>0) std::cout<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<funcName<<":\n";
	FILE* writer=NULL;
	bool error=false;
	//units
		double s_len=0.0;
		if(units::consts::system()==units::System::AU) s_len=units::BOHR;
		else if(units::consts::system()==units::System::METAL) s_len=1.0;
		else throw std::runtime_error("Invalid units.");
	
	try{
		//==== open the file ====
		writer=fopen(file,"w");
		if(writer==NULL) throw std::runtime_error(std::string("I/O Error: Could not open file: \"")+std::string(file)+std::string("\"\n"));
		
		//==== check simulation ====
		if(sim.timesteps()==0) throw std::invalid_argument("Invalid simulation.");
		
		//==== check timing info ====
		int beg=interval.beg-1;
		int end;
		if(interval.end>=0) end=interval.end-1;
		else end=interval.end+sim.timesteps();
		if(beg<0) throw std::invalid_argument("Invalid beginning timestep.");
		if(end>=sim.timesteps()) throw std::invalid_argument("Invalid ending timestep.");
		if(end<beg) throw std::invalid_argument("Invalid timestep interval.");
		
		//==== write the coord ====
		std::string coord;
		if(atomT.frac) coord=std::string("Direct");
		else coord=std::string("Cart");
		
		if(sim.cell_fixed()){
			fprintf(writer,"%s\n",sim.name().c_str());
			fprintf(writer,"1.0\n");
			for(int i=0; i<3; ++i){
				for(int j=0; j<3; ++j){
					fprintf(writer,"%f ",sim.frame(0).R()(j,i)*s_len);
				}
				fprintf(writer,"\n");
			}
			for(int i=0; i<nSpecies; ++i) fprintf(writer,"%s ",species[i].c_str());
			fprintf(writer,"\n");
			for(int i=0; i<nSpecies; ++i) fprintf(writer,"%i ",nAtoms[i]);
			fprintf(writer,"\n");
			for(int t=beg; t<=end; ++t){
				fprintf(writer,"%s\n",coord.c_str());
				int count=0;
				for(int n=0; n<nSpecies; ++n){
					for(int m=0; m<nAtoms[n]; ++m){
						Eigen::Vector3d posn=sim.frame(t).RInv()*sim.frame(t).posn(count++);
						fprintf(writer,"%f %f %f\n",posn[0],posn[1],posn[2]);
					}
				}
			}
		} else {
			for(int t=beg; t<=end; ++t){
				fprintf(writer,"%s\n",sim.name().c_str());
				fprintf(writer,"1.0\n");
				for(int i=0; i<3; ++i){
					for(int j=0; j<3; ++j){
						fprintf(writer,"%f ",sim.frame(t).R()(j,i)*s_len);
					}
					fprintf(writer,"\n");
				}
				for(int i=0; i<nSpecies; ++i) fprintf(writer,"%s ",species[i].c_str());
				fprintf(writer,"\n");
				for(int i=0; i<nSpecies; ++i) fprintf(writer,"%i ",nAtoms[i]);
				fprintf(writer,"\n");
				fprintf(writer,"%s\n",coord.c_str());
				int count=0;
				for(int n=0; n<as.nSpecies(); ++n){
					for(int m=0; m<as.nAtoms(n); ++m){
						Eigen::Vector3d posn=sim.frame(t).RInv()*sim.frame(t).posn(count++);
						fprintf(writer,"%f %f %f\n",posn[0],posn[1],posn[2]);
					}
				}
			}
		}
		
		fclose(writer);
		writer=NULL;
	}catch(std::exception& e){
		std::cout<<"ERROR in "<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<funcName<<":\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	if(error) throw std::runtime_error("I/O Exception: Could not write data.");
	*/
}

}

}
