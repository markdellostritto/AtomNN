// c libraries
#include <cstdio>
#include <ctime>
// c++ libraries
#include <string>
#include <stdexcept>
#include <iostream>
// ann - structure
#include "src/struc/structure.hpp"
// ann - math
#include "src/math/func.hpp"
// ann - strings
#include "src/str/string.hpp"
// ann - chem
#include "src/chem/units.hpp"
#include "src/chem/ptable.hpp"
// ann - format
#include "src/format/ame_struc.hpp"

namespace AME{

//*****************************************************
//reading
//*****************************************************

void read(const char* file, const AtomType& atomT, Structure& struc){
	if(AME_PRINT_FUNC>0) std::cout<<"AME::read(const char*,const AtomType&,Structure&):\n";
	//==== local function variables ====
	//file i/o
		FILE* reader=NULL;
		char* input=new char[string::M];
		char* name=new char[string::M];
		std::vector<std::string> strlist;
	//atom info
		int natoms=0;
		Eigen::Vector3d posn;
		double energy=0;
		Eigen::Matrix3d lv=Eigen::Matrix3d::Zero();
		FORMAT_ATOM format;
	//atom data
		std::vector<Eigen::Vector3d> r_;
		std::vector<Eigen::Vector3d> f_;
		std::vector<double> q_;
		std::vector<std::string> name_;
	//units
		units::System units;
		
	//open file
	if(AME_PRINT_STATUS>0) std::cout<<"opening file\n";
	reader=fopen(file,"r");
	if(reader==NULL) throw std::runtime_error(std::string("ERROR in AME::read(const char*,const AtomType&,Structure&): Could not open file: ")+std::string(file));
	
	//read lines
	while(fgets(input,string::M,reader)!=NULL){
		string::split(input,string::WS,strlist);
		const std::string& label=strlist.at(0);
		if(label=="units"){
			units=units::System::read(strlist.at(1).c_str());
		} else if(label=="energy"){
			energy=std::atof(strlist.at(1).c_str());
		} else if(label=="atoms"){
			natoms=std::atoi(strlist.at(1).c_str());
			//read format
			string::split(fgets(input,string::M,reader),string::WS,strlist);
			for(int i=0; i<strlist.size(); ++i){
				const char* tmpstr=strlist.at(i).c_str();
				if(std::strcmp(tmpstr,"name")==0) format.name=i;
				else if(std::strcmp(tmpstr,"q")==0) format.q=i;
				else if(std::strcmp(tmpstr,"mass")==0) format.m=i;
				else if(std::strcmp(tmpstr,"rx")==0) format.rx=i;
				else if(std::strcmp(tmpstr,"ry")==0) format.ry=i;
				else if(std::strcmp(tmpstr,"rz")==0) format.rz=i;
				else if(std::strcmp(tmpstr,"fx")==0) format.fx=i;
				else if(std::strcmp(tmpstr,"fy")==0) format.fy=i;
				else if(std::strcmp(tmpstr,"fz")==0) format.fz=i;
			}
			//read atom data
			r_.resize(natoms,Eigen::Vector3d::Zero());
			f_.resize(natoms,Eigen::Vector3d::Zero());
			q_.resize(natoms,0.0);
			name_.resize(natoms,"NULL");
			for(int i=0; i<natoms; ++i){
				string::split(fgets(input,string::M,reader),string::WS,strlist);
				if(format.name>=0) name_[i]=strlist[format.name].c_str();
				if(format.q>=0) q_[i]=std::atof(strlist[format.q].c_str());
				if(format.rx>=0) r_[i][0]=std::atof(strlist[format.rx].c_str());
				if(format.ry>=0) r_[i][1]=std::atof(strlist[format.ry].c_str());
				if(format.rz>=0) r_[i][2]=std::atof(strlist[format.rz].c_str());
				if(format.fx>=0) f_[i][0]=std::atof(strlist[format.fx].c_str());
				if(format.fy>=0) f_[i][1]=std::atof(strlist[format.fy].c_str());
				if(format.fz>=0) f_[i][2]=std::atof(strlist[format.fz].c_str());
			}
		}
	}
	//close the file
	if(AME_PRINT_STATUS>0) std::cout<<"closing file\n";
	fclose(reader);
	reader=NULL;
	
	//resize the structure
	if(AME_PRINT_STATUS>0) std::cout<<"resizing structure\n";
	struc.resize(natoms,atomT);
	
	//set the atom data
	if(AME_PRINT_STATUS>0) std::cout<<"setting atom data\n";
	struc.energy()=energy;
	for(int i=0; i<natoms; ++i){
		if(atomT.posn) struc.posn(i)=r_[i];
		if(atomT.force) struc.force(i)=f_[i];
		if(atomT.charge) struc.charge(i)=q_[i];
		if(atomT.name) struc.name(i)=name_[i];
	}
	
	//free memory
	delete[] input;
	delete[] name;
}

//*****************************************************
//writing
//*****************************************************

void write(const char* file, const AtomType& atomT, const Structure& struc){
	if(AME_PRINT_FUNC>0) std::cout<<"write(const char*,const AtomType&,const Structure&):\n";
	FILE* writer=NULL;
	
}

	
}