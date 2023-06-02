// c libraries
#include <ctime>
// c++ libraries
#include <iostream>
#include <string>
#include <stdexcept>
// eigen libraries
#include <Eigen/Dense>
// ann - chem
#include "src/chem/units.hpp"
#include "src/chem/ptable.hpp"
// ann - structure
#include "src/struc/structure.hpp"
// ann - string
#include "src/str/string.hpp"
// ann - math
#include "src/math/const.hpp"
// ann - format
#include "src/format/qe_struc.hpp"

namespace QE{

//*****************************************************
//FORMAT struct
//*****************************************************

Format& Format::read(const std::vector<std::string>& strlist, Format& format){
	for(int i=0; i<strlist.size(); ++i){
		if(strlist[i]=="-pos"){
			if(i==strlist.size()-1) throw std::invalid_argument("No file specified for \"-pos\" option.");
			else format.filePos=strlist[i+1];
		} else if(strlist[i]=="-in"){
			if(i==strlist.size()-1) throw std::invalid_argument("No file specified for \"-in\" option.");
			else format.fileIn=strlist[i+1];
		} else if(strlist[i]=="-cel"){
			if(i==strlist.size()-1) throw std::invalid_argument("No file specified for \"-cel\" option.");
			else format.fileCel=strlist[i+1];
		} else if(strlist[i]=="-evp"){
			if(i==strlist.size()-1) throw std::invalid_argument("No file specified for \"-evp\" option.");
			else format.fileEvp=strlist[i+1];
		} else if(strlist[i]=="-qeout"){
			if(i==strlist.size()-1) throw std::invalid_argument("No file specified for \"-evp\" option.");
			else format.fileOut=strlist[i+1];
		}
	}
	return format;
}

//*****************************************************
//OUT format
//*****************************************************

namespace OUT{
	
void read(const char* file, const AtomType& atomT, Structure& struc){
	const char* func_name="read(FILE*,const AtomType&,Structure&)";
	if(QE_PRINT_FUNC>0) std::cout<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<func_name<<":\n";
	//file i/o
		FILE* reader=NULL;
		char* input=new char[string::M];
	//structure
		int natomst=0;
		std::vector<int> natoms;
		int nspecies=0;
		std::vector<std::string> species;
		std::vector<int> zval;
	//cell
		double alat=0;
		Eigen::Matrix3d lv=Eigen::Matrix3d::Zero();
	//format strings
		const char* str_energy="!    total energy";
		const char* str_lv="crystal axes";
		const char* str_alat="lattice parameter";
		const char* str_posn="site n.";
		const char* str_force="Forces";
		const char* str_stress="total   stress";
		const char* str_natoms="number of atoms/cell";
		const char* str_nspecies="number of atomic types";
		const char* str_species="atomic species  ";
		const char* str_spin="Magnetic moment per site";
		const char* str_zval="Zval";
		const char* str_nelect="number of electrons";
	//units
		double s_posn=0.0,s_energy=0.0;
		if(units::consts::system()==units::System::AU){
			s_posn=1.0;
			s_energy=1.0;
		}
		else if(units::consts::system()==units::System::METAL){
			s_posn=units::ANGpBOHR;
			s_energy=0.5*units::EVpHARTREE;//QE energy: Rydberg
		}
		else throw std::runtime_error("Invalid units.");
	//charge
		int nelect=0;
		double qtot=0;
	//misc
		bool error=false;
	
	try{
		//open the file
		if(QE_PRINT_STATUS>0) std::cout<<"opening the file: "<<file<<"\n";
		reader=fopen(file,"r");
		if(reader==NULL) throw std::runtime_error(std::string("ERROR: Could not open file: \"")+std::string(file)+std::string("\"\n"));
		
		if(QE_PRINT_STATUS>0) std::cout<<"reading simulation info\n";
		while(fgets(input,string::M,reader)!=NULL){
			if(std::strstr(input,str_energy)!=NULL){
				struc.energy()=s_energy*std::atof(string::trim_right(string::trim_left(input,"="),"Ry"));
			} else if(std::strstr(input,str_lv)!=NULL){
				std::vector<std::string> strlist;
				fgets(input,string::M,reader); string::replace(input,')',' '); string::replace(input,'(',' ');
				string::split(input,string::WS,strlist);
				lv(0,0)=std::atof(strlist.at(3).c_str());
				lv(1,0)=std::atof(strlist.at(4).c_str());
				lv(2,0)=std::atof(strlist.at(5).c_str());
				fgets(input,string::M,reader); string::replace(input,')',' '); string::replace(input,'(',' ');
				string::split(input,string::WS,strlist);
				lv(0,1)=std::atof(strlist.at(3).c_str());
				lv(1,1)=std::atof(strlist.at(4).c_str());
				lv(2,1)=std::atof(strlist.at(5).c_str());
				fgets(input,string::M,reader); string::replace(input,')',' '); string::replace(input,'(',' ');
				string::split(input,string::WS,strlist);
				lv(0,2)=std::atof(strlist.at(3).c_str());
				lv(1,2)=std::atof(strlist.at(4).c_str());
				lv(2,2)=std::atof(strlist.at(5).c_str());
			} else if(std::strstr(input,str_alat)!=NULL){
				alat=s_posn*std::atof(string::trim_right(string::trim_left(input,"="),"a"));
			} else if(std::strstr(input,str_natoms)!=NULL){
				natomst=std::atoi(string::trim_left(input,"="));
			} else if(std::strstr(input,str_nspecies)!=NULL){
				nspecies=std::atoi(string::trim_left(input,"="));
				natoms.resize(nspecies,0);
			} else if(std::strstr(input,str_species)!=NULL){
				species.clear();
				for(int i=0; i<nspecies; ++i){
					fgets(input,string::M,reader);
					species.push_back(std::string(std::strtok(input,string::WS)));
				}
			} else if(std::strstr(input,str_zval)!=NULL){
				std::vector<std::string> strlist;
				string::split(input,"=",strlist);
				zval.push_back(std::round(std::atof(strlist.at(1).c_str())));
			} else if(std::strstr(input,str_nelect)!=NULL){
				std::vector<std::string> strlist;
				string::split(input,"=",strlist);
				nelect=std::round(std::atof(strlist.at(1).c_str()));
			} else if(std::strstr(input,str_posn)!=NULL){
				std::vector<std::string> strlist;
				for(int i=0; i<nspecies; ++i){
					natoms[i]=0;
				}
				for(int i=0; i<natomst; ++i){
					fgets(input,string::M,reader);
					string::split(input,string::WS,strlist);
					std::string name=strlist.at(1);
					for(int j=0; j<nspecies; ++j){
						if(name==species[j]){++natoms[j];break;}
					}
				}
			}
		}
		int ztot=0;
		for(int i=0; i<nspecies; ++i){
			ztot+=zval[i]*natoms[i];
		}
		qtot=ztot-nelect;
		
		//print parameters
		if(QE_PRINT_STATUS>0){
			std::cout<<"ATOM    = "<<atomT<<"\n";
			std::cout<<"NATOMST = "<<natomst<<"\n";
			std::cout<<"SPECIES = "; for(int i=0; i<nspecies; ++i) std::cout<<species[i]<<" "; std::cout<<"\n";
			std::cout<<"ZVAL    = "; for(int i=0; i<nspecies; ++i) std::cout<<zval[i]<<" "; std::cout<<"\n";
			std::cout<<"NATOMS  = "; for(int i=0; i<natoms.size(); ++i) std::cout<<natoms[i]<<" "; std::cout<<"\n";
			std::cout<<"ENERGY  = "<<struc.energy()<<"\n";
			std::cout<<"NELECT  = "<<nelect<<"\n";
			std::cout<<"QTOT    = "<<qtot<<"\n";
			std::cout<<"ALAT    = "<<alat<<"\n";
			std::cout<<"LV      = \n"<<lv<<"\n";
		}
		
		//check the parameters
		if(nspecies<=0) throw std::runtime_error("ERROR reading number of species.");
		if(natomst<=0) throw std::runtime_error("ERROR reading number of atoms.");
		if(species.size()!=nspecies) throw std::runtime_error("ERROR reading species names.");
		if(alat<=0) throw std::runtime_error("ERROR reading alat.");
		if(lv.norm()==0) throw std::runtime_error("ERROR reading alat.");
		if(std::fabs(lv.determinant())<math::constant::ZERO) throw std::runtime_error("Invalid lattice vector matrix.");
		
		//resize the structure
		if(QE_PRINT_STATUS>0) std::cout<<"resizing structure\n";
		struc.resize(natomst,atomT);
		static_cast<Cell&>(struc).init(alat*lv);
		struc.qtot()=qtot;
		
		//read in stress
		#ifdef INCLUDE_VIRIAL
		if(QE_PRINT_STATUS>0) std::cout<<"reading virial\n";
		std::rewind(reader);
		while(fgets(input,string::M,reader)!=NULL){
			if(std::strstr(input,str_stress)!=NULL){
				std::vector<std::string> strlist;
				const double scale=s_energy/(s_posn*s_posn*s_posn);
				fgets(input,string::M,reader);
				string::split(input,string::WS,strlist);
				struc.virial()(0,0)=std::atof(strlist[0].c_str())*scale;
				struc.virial()(0,1)=std::atof(strlist[1].c_str())*scale;
				struc.virial()(0,2)=std::atof(strlist[2].c_str())*scale;
				fgets(input,string::M,reader);
				string::split(input,string::WS,strlist);
				struc.virial()(1,0)=std::atof(strlist[0].c_str())*scale;
				struc.virial()(1,1)=std::atof(strlist[1].c_str())*scale;
				struc.virial()(1,2)=std::atof(strlist[2].c_str())*scale;
				fgets(input,string::M,reader);
				string::split(input,string::WS,strlist);
				struc.virial()(2,0)=std::atof(strlist[0].c_str())*scale;
				struc.virial()(2,1)=std::atof(strlist[1].c_str())*scale;
				struc.virial()(2,2)=std::atof(strlist[2].c_str())*scale;
			}
		}
		#endif
		
		//read atoms
		if(atomT.posn){
			std::vector<int> index(nspecies,0);
			if(QE_PRINT_STATUS>0) std::cout<<"reading atoms\n";
			std::rewind(reader);
			while(fgets(input,string::M,reader)!=NULL){
				if(std::strstr(input,str_posn)!=NULL){
					std::vector<std::string> strlist;
					for(int i=0; i<natomst; ++i){
						//read line
						fgets(input,string::M,reader);
						string::replace(input,'(',' ');
						string::replace(input,')',' ');
						string::split(input,string::WS,strlist);
						//read atom properties
						std::string name=strlist.at(1);
						const int an=ptable::an(name.c_str());
						Eigen::Vector3d posn;
						posn[0]=std::atof(strlist[5].c_str());
						posn[1]=std::atof(strlist[6].c_str());
						posn[2]=std::atof(strlist[7].c_str());
						//get the species
						int type=-1;
						for(int j=0; j<nspecies; ++j){
							if(name==species[j]){
								type=j;
								break;
							}
						}
						if(type<0) throw std::invalid_argument("Invalid species name in atomic positions");
						//set data
						if(struc.atomType().name) struc.name(i)=name;
						if(struc.atomType().an) struc.an(i)=an;
						if(struc.atomType().mass) struc.mass(i)=ptable::mass(an);
						if(struc.atomType().posn) struc.posn(i)=posn;
						if(struc.atomType().type) struc.type(i)=type;
						if(struc.atomType().index) struc.index(i)=index[type];
						index[type]++;
					}
				}
			}
			//convert to cartesian coordinates
			for(int i=0; i<natomst; ++i) struc.posn(i)*=alat;
			//return to cell
			for(int i=0; i<natomst; ++i) Cell::returnToCell(struc.posn(i),struc.posn(i),struc.R(),struc.RInv());
		}
		
		//read in forces
		if(atomT.force){
			if(QE_PRINT_STATUS>0) std::cout<<"reading forces\n";
			std::rewind(reader);
			while(fgets(input,string::M,reader)!=NULL){
				if(std::strstr(input,str_force)!=NULL){
					std::vector<std::string> strlist;
					fgets(input,string::M,reader);
					const double fac=s_energy/s_posn;
					for(int i=0; i<natomst; ++i){
						fgets(input,string::M,reader);
						string::split(input,string::WS,strlist);
						Eigen::Vector3d force;
						force[0]=std::atof(strlist.at(6).c_str())*fac;
						force[1]=std::atof(strlist.at(7).c_str())*fac;
						force[2]=std::atof(strlist.at(8).c_str())*fac;
						struc.force(i)=force;
					}
				}
			}
		}
		
		//read in spins
		if(atomT.spin){
			if(QE_PRINT_STATUS>0) std::cout<<"reading spins\n";
			std::rewind(reader);
			const double fac=1;
			while(fgets(input,string::M,reader)!=NULL){
				if(std::strstr(input,str_spin)!=NULL){
					std::vector<std::string> strlist;
					for(int i=0; i<natomst; ++i){
						fgets(input,string::M,reader);
						std::string istring=std::string(input);
						std::string str=istring.substr(istring.find_last_of("=")+1,istring.length());
						const double s=std::atof(str.c_str())*fac;
						struc.spin(i)<<0,0,s;//for now, assume only collinear spins
					}
				}
			}
		}
		
		//set the types
		if(atomT.type){
			for(int i=0; i<struc.nAtoms(); ++i){
				for(int j=0; j<species.size(); ++j){
					if(struc.name(i)==species[j]){
						struc.type(i)=j;
					}
				}
			}
		}
		
		//close the file
		fclose(reader);
		reader=NULL;
	}catch(std::exception& e){
		std::cout<<"Error in "<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<func_name<<":\n";
		std::cout<<"Failed to read file: \""<<file<<"\"\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	//free local variables
	delete[] input;
	
	if(error) throw std::runtime_error("I/O Exception: Could not read data.");
}

}

}