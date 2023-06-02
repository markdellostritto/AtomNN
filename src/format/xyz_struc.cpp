// c libraries
#include <cstdio>
#include <ctime>
// c++ libraries
#include <string>
#include <stdexcept>
#include <iostream>
// ann - structure
#include "src/struc/structure.hpp"
// ann - strings
#include "src/str/string.hpp"
// ann - chem
#include "src/chem/units.hpp"
#include "src/chem/ptable.hpp"
// ann - format
#include "src/format/xyz_struc.hpp"

namespace XYZ{

//*****************************************************
//FORMAT struct
//*****************************************************

Format& Format::read(const std::vector<std::string>& strlist, Format& format){
	for(int i=0; i<strlist.size(); ++i){
		if(strlist[i]=="-xyz"){
			if(i==strlist.size()-1) throw std::invalid_argument("No file specified for \"-xdatcar\" option.");
			else format.xyz=strlist[i+1];
		}
	}
	return format;
}

//*****************************************************
//reading
//*****************************************************

void read(const char* xyzfile, const AtomType& atomT, Structure& struc){
	if(XYZ_PRINT_FUNC>0) std::cout<<"XYZ::read(const char*,const AtomType&,Structure&):\n";
	//==== local function variables ====
	//file i/o
		FILE* reader=NULL;
		char* input=new char[string::M];
		char* name=new char[string::M];
		std::vector<std::string> strlist;
	//atom info
		int nAtoms=0;
		Eigen::Vector3d posn;
		double energy=0;
		Eigen::Matrix3d lv=Eigen::Matrix3d::Zero();
	//units
		double s_len=0.0,s_energy=0.0;
		if(units::consts::system()==units::System::AU){
			s_len=units::BOHRpANG;
			s_energy=units::HARTREEpEV;
		} else if(units::consts::system()==units::System::METAL){
			s_len=1.0;
			s_energy=1.0;
		} else if(units::consts::system()==units::System::LJ){
			s_len=1.0;
			s_energy=1.0;
		}
		else throw std::runtime_error("Invalid units.");
		
	//open file
	if(XYZ_PRINT_STATUS>0) std::cout<<"opening file\n";
	reader=fopen(xyzfile,"r");
	if(reader==NULL) throw std::runtime_error(std::string("ERROR in XYZ::read(const char*,const AtomType&,Structure&): Could not open file: ")+std::string(xyzfile));
	
	//read natoms
	if(XYZ_PRINT_STATUS>0) std::cout<<"reading natoms\n";
	fgets(input,string::M,reader);
	nAtoms=std::atoi(input);
	if(nAtoms<=0) throw std::runtime_error("ERROR in XYZ::read(const char*,const AtomType&,Structure&): found zero atoms.");
	
	//read in cell length and energy
	fgets(input,string::M,reader);
	string::split(input,string::WS,strlist);
	if(strlist.size()==1+1){
		energy=std::atof(strlist.at(1).c_str());
	} else if(strlist.size()==1+1+3){
		energy=std::atof(strlist.at(1).c_str());
		lv(0,0)=std::atof(strlist.at(2).c_str());
		lv(1,1)=std::atof(strlist.at(3).c_str());
		lv(2,2)=std::atof(strlist.at(4).c_str());
	}
	lv*=s_len;
	
	//resize the structure
	if(XYZ_PRINT_STATUS>0) std::cout<<"resizing structure\n";
	struc.resize(nAtoms,atomT);
	
	//read in names and positions
	if(XYZ_PRINT_STATUS>0) std::cout<<"reading names and posns\n";
	for(int i=0; i<nAtoms; ++i){
		fgets(input,string::M,reader);
		std::sscanf(input,"%s %lf %lf %lf",name,&posn[0],&posn[1],&posn[2]);
		if(struc.atomType().name) struc.name(i)=name;
		if(struc.atomType().posn) struc.posn(i).noalias()=posn*s_len;
	}
	
	//close the file
	if(XYZ_PRINT_STATUS>0) std::cout<<"closing file\n";
	fclose(reader);
	reader=NULL;
	
	//set the cell
	if(XYZ_PRINT_STATUS>0) std::cout<<"setting cell\n";
	if(lv.norm()>0) static_cast<Cell&>(struc).init(lv);
	
	//set the energy
	if(XYZ_PRINT_STATUS>0) std::cout<<"setting energy\n";
	struc.energy()=s_energy*energy;
	
	//set an
	if(atomT.an && atomT.name){
		for(int i=0; i<nAtoms; ++i){
			struc.an(i)=ptable::an(struc.name(i).c_str());
		}
	}
	
	//set mass
	if(atomT.an && atomT.mass){
		for(int i=0; i<nAtoms; ++i){
			struc.mass(i)=ptable::mass(struc.an(i));
		}
	} else if(atomT.name && atomT.mass){
		for(int i=0; i<nAtoms; ++i){
			const int an=ptable::an(struc.name(i).c_str());
			struc.mass(i)=ptable::mass(an);
		}
	}
	
	//set radius
	if(atomT.an && atomT.radius){
		for(int i=0; i<nAtoms; ++i){
			struc.radius(i)=ptable::radius_covalent(struc.an(i));
		}
	} else if(atomT.name && atomT.radius){
		for(int i=0; i<nAtoms; ++i){
			const int an=ptable::an(struc.name(i).c_str());
			struc.radius(i)=ptable::radius_covalent(an);
		}
	}
	
	//free memory
	delete[] input;
	delete[] name;
}

//*****************************************************
//writing
//*****************************************************

void write(const char* file, const AtomType& atomT, const Structure& struc){
	if(XYZ_PRINT_FUNC>0) std::cout<<"write(const char*,const AtomType&,const Structure&):\n";
	FILE* writer=NULL;
	
	//open file
	if(XYZ_PRINT_STATUS>0) std::cout<<"opening file\n";
	writer=fopen(file,"w");
	if(writer==NULL) throw std::runtime_error("Runtime Error: Could not open file: \""+std::string(file)+"\"");
	
	//write xyz
	if(XYZ_PRINT_STATUS>0) std::cout<<"writing structure\n";
	fprintf(writer,"%i\n",struc.nAtoms());
	fprintf(writer,"SIMULATION %f\n",struc.energy());
	for(int i=0; i<struc.nAtoms(); ++i){
		fprintf(writer,"%-2s %19.10f %19.10f %19.10f\n",struc.name(i).c_str(),
			struc.posn(i)[0],struc.posn(i)[1],struc.posn(i)[2]
		);
	}
	
	//close file
	if(XYZ_PRINT_STATUS>0) std::cout<<"closing file\n";
	fclose(writer);
	writer=NULL;
}

	
}