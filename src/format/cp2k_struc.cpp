// c libraries
#include <cstdio>
#include <ctime>
// c++ libraries
#include <string>
#include <stdexcept>
#include <iostream>
// DAME - io
#include "src/str/string.hpp"
// DAME - chem
#include "src/chem/units.hpp"
// DAME - math
#include "src/math/const.hpp"
//struc
#include "src/struc/structure.hpp"
//cp2k
#include "src/format/cp2k_struc.hpp"

namespace CP2K{

//*****************************************************
//reading
//*****************************************************

void read(const char* file, const AtomType& atomT, Structure& struc){
	//file i/o
		FILE* reader=NULL;
		char* input=new char[string::M];
		std::vector<std::string> strlist;
	//units
		double s_posn=0.0,s_energy=0.0;
		if(units::consts::system()==units::System::AU){
			s_posn=1.0;
			s_energy=1.0;
		}
		else if(units::consts::system()==units::System::METAL){
			s_posn=1.0;
			s_energy=units::HARTREE;
		}
		else throw std::runtime_error("Invalid units.");
	//structure
		int natomst=0;
		std::vector<int> natoms;
		int nspecies=0;
		std::vector<std::string> species;
	//flags
		const char* flag_atom="ATOMIC COORDINATES";
		const char* flag_cell="CELL|";
		const char* flag_energy="ENERGY|";
		const char* flag_natoms="Atoms:";
	//misc
		bool error=false;
		
	try{
		//open the file
		if(CP2K_PRINT_STATUS>0) std::cout<<"opening the file: "<<file<<"\n";
		reader=fopen(file,"r");
		if(reader==NULL) throw std::runtime_error(std::string("ERROR: Could not open file: \"")+std::string(file)+std::string("\"\n"));
		
		//read data
		while(fgets(input,string::M,reader)!=NULL){
			if(std::strstr(input,flag_natoms)!=NULL){
				natomst=std::atoi(std::strstr(input,flag_natoms)+6);
				if(natomst==0) throw std::runtime_error("Found zero atoms");
				struc.resize(natomst,atomT);
			} else if(std::strstr(input,flag_atom)!=NULL){
				//skip three lines
				fgets(input,string::M,reader);
				fgets(input,string::M,reader);
				fgets(input,string::M,reader);
				for(int i=0; i<natomst; ++i){
					string::split(fgets(input,string::M,reader),string::WS,strlist);
					if(atomT.name) struc.name(i)=strlist[2];
					if(atomT.posn) struc.posn(i)<<std::atof(strlist[4].c_str()),std::atof(strlist[5].c_str()),std::atof(strlist[6].c_str());
					if(atomT.mass) struc.mass(i)=std::atof(strlist[8].c_str());
				}
			} else if(std::strstr(input,flag_energy)!=NULL){
				string::split(input,string::WS,strlist);
				struc.energy()=std::atof(strlist.back().c_str())*s_energy;
			} else if(std::strstr(input,flag_cell)!=NULL){
				Eigen::Matrix3d R;
				string::split(fgets(input,string::M,reader),string::WS,strlist);
				R(0,0)=std::atof(strlist[4].c_str());
				R(1,0)=std::atof(strlist[5].c_str());
				R(2,0)=std::atof(strlist[6].c_str());
				string::split(fgets(input,string::M,reader),string::WS,strlist);
				R(0,1)=std::atof(strlist[4].c_str());
				R(1,1)=std::atof(strlist[5].c_str());
				R(2,1)=std::atof(strlist[6].c_str());
				string::split(fgets(input,string::M,reader),string::WS,strlist);
				R(0,2)=std::atof(strlist[4].c_str());
				R(1,2)=std::atof(strlist[5].c_str());
				R(2,2)=std::atof(strlist[6].c_str());
				static_cast<Cell&>(struc).init(R);
				fgets(input,string::M,reader);
				fgets(input,string::M,reader);
				fgets(input,string::M,reader);
				fgets(input,string::M,reader);
			}
		}
	}catch(std::exception& e){
		std::cout<<"Error in CP2K::read()\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
}

}