// c libraries
#include <cstdio>
#include <ctime>
// c++ libraries
#include <string>
#include <stdexcept>
#include <iostream>
// math
#include "src/math/func.hpp"
// str
#include "src/str/string.hpp"
// chem
#include "src/chem/units.hpp"
#include "src/chem/ptable.hpp"
// format
#include "src/format/xyz_sim.hpp"

namespace XYZ{

void read(const char* file, const Interval& interval, const AtomType& atomT, Simulation& sim){
	if(XYZ_PRINT_FUNC>0) std::cout<<"read(const char*,Interval&,const AtomType&,Simulation&):\n";
	//==== local function variables ====
	//file i/o
		FILE* reader=NULL;
		char* input=new char[string::M];
		char* name=new char[string::M];
		std::vector<std::string> strlist;
	//atom info	
		int nAtoms=0;
		std::vector<std::string> names;
		int nSpecies=0;
		std::vector<std::string> speciesNames;
		std::vector<int> speciesNumbers;
	//positions
		Eigen::Vector3d r;
	//units
		double s=0.0;
		if(units::consts::system()==units::System::AU) s=units::BOHRpANG;
		else if(units::consts::system()==units::System::METAL) s=1.0;
		else throw std::runtime_error("Invalid units.");
	
	//open file
	if(XYZ_PRINT_STATUS>0) std::cout<<"opening file\n";
	reader=fopen(file,"r");
	if(reader==NULL) throw std::runtime_error("Runtime Error: Could not open file.");
	
	//read natoms
	if(XYZ_PRINT_STATUS>0) std::cout<<"reading natoms\n";
	fgets(input,string::M,reader);
	nAtoms=std::atoi(input);
	if(nAtoms<=0) throw std::runtime_error("Runtime Error: found zero atoms.");
	if(XYZ_PRINT_DATA>0) std::cout<<"natoms = "<<nAtoms<<"\n";
	
	//find the total number of timesteps
	if(XYZ_PRINT_STATUS>0) std::cout<<"reading timesteps\n";
	std::rewind(reader);
	int nlines=0;
	while(fgets(input,string::M,reader)) ++nlines;
	int ts=nlines/(nAtoms+2);//natoms + natoms-line + comment-line
	if(XYZ_PRINT_DATA>0) std::cout<<"ts = "<<ts<<"\n";
	
	//set the interval
	if(XYZ_PRINT_STATUS>0) std::cout<<"setting interval\n";
	if(interval.beg()<0) throw std::invalid_argument("Invalid beginning timestep.");
	const int beg=interval.beg()-1;
	int end=interval.end()-1;
	if(interval.end()<0) end=ts+interval.end();
	const int tsint=end-beg+1;
	if(XYZ_PRINT_DATA>0) std::cout<<"interval = "<<beg<<":"<<end<<":"<<tsint<<"\n";
	
	//resize the simulation
	if(XYZ_PRINT_STATUS>0) std::cout<<"resizing simulation\n";
	sim.resize(tsint/interval.stride(),nAtoms,atomT);
	
	//read the simulation
	if(XYZ_PRINT_STATUS>0) std::cout<<"reading simulation\n";
	std::rewind(reader);
	for(int t=0; t<beg; ++t){
		fgets(input,string::M,reader);//natoms
		const int N=std::atoi(input);
		fgets(input,string::M,reader);//comment line
		for(int n=0; n<N; ++n){
			fgets(input,string::M,reader);
		}
	}
	for(int t=0; t<sim.timesteps(); ++t){
		//read natoms
		fgets(input,string::M,reader);//natoms
		const int N=std::atoi(input);
		if(sim.frame(t).nAtoms()!=N) throw std::invalid_argument("Invalid number of atoms.");
		//read in cell length and energy
		double energy=0;
		Eigen::Matrix3d lv=Eigen::Matrix3d::Zero();
		fgets(input,string::M,reader);
		string::split(input,string::WS,strlist);
		if(strlist.size()==1+1){
			energy=std::atof(strlist.at(1).c_str());
		} else if(strlist.size()==1+3){
			lv(0,0)=std::atof(strlist.at(1).c_str());
			lv(1,1)=std::atof(strlist.at(2).c_str());
			lv(2,2)=std::atof(strlist.at(3).c_str());
		}else if(strlist.size()==1+1+3){
			energy=std::atof(strlist.at(1).c_str());
			lv(0,0)=std::atof(strlist.at(2).c_str());
			lv(1,1)=std::atof(strlist.at(3).c_str());
			lv(2,2)=std::atof(strlist.at(4).c_str());
		}
		sim.frame(t).energy()=energy;
		//read positions
		for(int n=0; n<N; ++n){
			fgets(input,string::M,reader);
			std::sscanf(input,"%s %lf %lf %lf",name,&r[0],&r[1],&r[2]);
			sim.frame(t).posn(n).noalias()=s*r;
			if(atomT.name) sim.frame(t).name(n)=name;
		}
		//set the cell
		if(XYZ_PRINT_STATUS>0) std::cout<<"setting cell\n";
		if(lv.norm()>0) static_cast<Cell&>(sim.frame(t)).init(lv);
		//skip "stride-1" steps
		for(int tt=0; tt<interval.stride()-1; ++tt){
			fgets(input,string::M,reader);//natoms
			const int NN=std::atoi(input);
			fgets(input,string::M,reader);//comment line
			for(int n=0; n<NN; ++n) fgets(input,string::M,reader);
		}
	}
	
	//set an
	if(atomT.name && atomT.an){
		for(int t=0; t<sim.timesteps(); ++t){
			Structure& struc=sim.frame(t);
			for(int i=0; i<nAtoms; ++i){
				struc.an(i)=ptable::an(struc.name(i).c_str());
			}
		}
	}
	
	//set mass
	if(atomT.mass && atomT.an){
		for(int t=0; t<sim.timesteps(); ++t){
			Structure& struc=sim.frame(t);
			for(int i=0; i<nAtoms; ++i){
				struc.mass(i)=ptable::mass(struc.an(i));
			}
		}
	} else if(atomT.mass && atomT.name){
		for(int t=0; t<sim.timesteps(); ++t){
			Structure& struc=sim.frame(t);
			for(int i=0; i<nAtoms; ++i){
				const int an=ptable::an(struc.name(i).c_str());
				struc.mass(i)=ptable::mass(struc.mass(an));
			}
		}
	}
	
	//close file
	if(XYZ_PRINT_STATUS>0) std::cout<<"closing file\n";
	fclose(reader);
	reader=NULL;
	
	//free memory
	delete[] input;
	delete[] name;
}

void write(const char* file, const Interval& interval, const AtomType& atomT, const Simulation& sim){
	if(XYZ_PRINT_FUNC>0) std::cout<<"write(const char*,Interval&,const AtomType&,Simulation&):\n";
	FILE* writer=NULL;
	
	//open file
	if(XYZ_PRINT_STATUS>0) std::cout<<"opening file\n";
	writer=fopen(file,"w");
	if(writer==NULL) throw std::runtime_error("Runtime Error: Could not open file.");
	
	//check timing info
	if(XYZ_PRINT_STATUS>0) std::cout<<"setting interval\n";
	int beg=interval.beg()-1;
	int end;
	if(interval.end()>=0) end=interval.end()-1;
	else end=interval.end()+sim.timesteps();
	if(beg<0) throw std::invalid_argument("Invalid beginning timestep.");
	if(end>=sim.timesteps()) throw std::invalid_argument("Invalid ending timestep.");
	if(end<beg) throw std::invalid_argument("Invalid timestep interval.");
	
	//write simulation
	if(XYZ_PRINT_STATUS>0) std::cout<<"writing simulation\n";
	for(int t=beg; t<=end; ++t){
		Structure struc=sim.frame(t);
		//unwrap(struc);
		fprintf(writer,"%i\n",struc.nAtoms());
		if(struc.R().norm()>0){
			fprintf(writer,"%s %.10f %f %f %f\n",sim.name().c_str(),struc.energy(),struc.R()(0,0),struc.R()(1,1),struc.R()(2,2));
		} else {
			fprintf(writer,"%s\n",sim.name().c_str());
		}
		for(int i=0; i<struc.nAtoms(); ++i){
			fprintf(writer,"  %-2s %19.10f %19.10f %19.10f\n",struc.name(i).c_str(),
				struc.posn(i)[0],struc.posn(i)[1],struc.posn(i)[2]
			);
		}
	}
	
	//close file
	if(XYZ_PRINT_STATUS>0) std::cout<<"closing file\n";
	fclose(writer);
	writer=NULL;
}
	
}