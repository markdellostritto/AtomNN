// mem
#include "src/mem/map.hpp"
// format
#include "src/format/file_struc.hpp"
// struc
#include "src/struc/structure.hpp"
// string
#include "src/str/string.hpp"
#include "src/str/print.hpp"
// torch
#include "src/torch/qtpie.hpp"
#include "src/torch/pot_factory.hpp"
// math
#include "src/math/const.hpp"
#include "src/math/reduce.hpp"
// nnp
#include "src/nnp/type.hpp"
// util
#include "src/util/time.hpp"

int main(int argc, char* argv[]){
	//file parameter
		char* fparam=new char[string::M];
		char* fstruc=new char[string::M];
		char* input=new char[string::M];
		char* str=new char[string::M];
		FILE* reader=NULL;
		std::vector<std::string> strlist;
	//structure
		AtomType atomT;
		atomT.name=true; atomT.type=true; atomT.index=true;
		atomT.posn=true; atomT.an=true; 
		atomT.charge=true; atomT.chi=true; atomT.eta=true;
		Structure struc;
		FILE_FORMAT::type format;//format of training data
	//qtpie
		QTPIE qtpie;
		double qtot=0;
		std::shared_ptr<ptnl::Pot> pot;
		std::vector<Type> types;
	//units
		units::System unitsys=units::System::UNKNOWN;
	//misc
		bool error=true;
		Clock clock;
	
	try{
		
		//==== check the arguments ====
		if(argc!=2) throw std::invalid_argument("Invalid number of arguments.");
		
		//==== load the parameter file ====
		std::cout<<"reading parameter file\n";
		std::strcpy(fparam,argv[1]);
		
		//==== open the parameter file ====
		std::cout<<"opening parameter file\n";
		FILE* reader=fopen(fparam,"r");
		if(reader==NULL) throw std::runtime_error(std::string("I/O Error: Could not open parameter file: ")+fparam);
		
		//==== read in the parameters ====
		std::cout<<"reading parameters\n";
		while(fgets(input,string::M,reader)!=NULL){
			string::trim_right(input,string::COMMENT);//trim comments
			Token token(input,string::WS); //split line into tokens
			if(token.end()) continue; //skip empty lines
			std::string tag=string::to_upper(token.next());
			if(tag=="UNITS"){//units
				unitsys=units::System::read(string::to_upper(token.next()).c_str());
			} else if(tag=="STRUC"){
				std::strcpy(fstruc,token.next().c_str());
			} else if(tag=="QTOT"){
				qtot=std::atof(token.next().c_str());
			} else if(tag=="FORMAT"){
				format=FILE_FORMAT::read(token.next().c_str());
			} else if(tag=="POT"){
				ptnl::read(pot,token);
			} else if(tag=="TYPE"){
				Type type;
				Type::read(type,token);
				types.push_back(type);
			}
		}
		
		//==== resize potential ====
		std::cout<<"resizing potential\n";
		pot->resize(types.size());
		
		//==== read coeffs ====
		std::cout<<"reading coefficients\n";
		std::rewind(reader);
		while(fgets(input,string::M,reader)!=NULL){
			string::trim_right(input,string::COMMENT);//trim comments
			Token token(input,string::WS); //split line into tokens
			if(token.end()) continue; //skip empty lines
			std::string tag=string::to_upper(token.next());
			if(tag=="COEFF"){
				pot->coeff(token);
			}
		}
		pot->init();
		
		//==== close the parameter file ====
		std::cout<<"closing parameter file\n";
		fclose(reader);
		reader=NULL;
		
		//==== print the parameters ====
		std::cout<<print::buf(str)<<"\n";
		std::cout<<print::title("GENERAL",str)<<"\n";
		std::cout<<"UNITS  = "<<unitsys<<"\n";
		std::cout<<"FSTRUC = "<<fstruc<<"\n";
		std::cout<<"QTOT   = "<<qtot<<"\n";
		std::cout<<print::buf(str)<<"\n";
		std::cout<<print::title("TYPES",str)<<"\n";
		for(int i=0; i<types.size(); ++i){
			std::cout<<types[i]<<"\n";
		}
		std::cout<<print::buf(str)<<"\n";
		
		//==== check the parameters ====
		if(unitsys==units::System::UNKNOWN) throw std::invalid_argument("Invalid unit system.");
		
		//==== set the unit system ====
		std::cout<<"setting the unit system\n";
		units::consts::init(unitsys);
		
		//==== read the structure ====
		std::cout<<"reading the structure\n";
		read_struc(fstruc,format,atomT,struc);
		struc.qtot()=qtot;
		std::cout<<struc<<"\n";
		
		//==== set atom data ====
		std::cout<<"setting atom data\n";
		//type
		for(int i=0; i<struc.nAtoms(); ++i){
			for(int j=0; j<types.size(); ++j){
				if(struc.name(i)==types[j].name()){struc.type(i)=j; break;}
			}
		}
		//charge
		for(int i=0; i<struc.nAtoms(); ++i){
			struc.charge(i)=types[struc.type(i)].charge().val();
		}
		//chi
		for(int i=0; i<struc.nAtoms(); ++i){
			struc.chi(i)=types[struc.type(i)].chi().val();
		}
		//eta
		for(int i=0; i<struc.nAtoms(); ++i){
			struc.eta(i)=types[struc.type(i)].eta().val();
		}
		//print
		for(int i=0; i<struc.nAtoms(); ++i){
			std::cout<<struc.name(i)<<" "<<struc.type(i)<<" "<<struc.eta(i)<<" "<<struc.chi(i)<<"\n";
		}
		
		//==== compute charges ====
		std::cout<<"computing charges\n";
		clock.start();
		qtpie.pot()=pot;
		NeighborList nlist(struc,pot->rc());
		qtpie.qt(struc,nlist);
		clock.stop();
		std::cout<<"energy = "<<-0.5*qtpie.x().dot(qtpie.b()*qtpie.x())<<"\n";
		std::cout<<"time   = "<<clock.time()<<"\n";
		
		//==== print charges ====
		std::vector<Reduce<1> > reduce(types.size());
		for(int i=0; i<struc.nAtoms(); ++i){
			reduce[struc.type(i)].push(struc.charge(i));
			std::cout<<struc.name(i)<<" "<<struc.charge(i)<<"\n";
		}
		for(int i=0; i<types.size(); ++i){
			std::cout<<types[i].name()<<" "<<reduce[i].avg()<<" "<<reduce[i].dev()<<" "<<reduce[i].max()<<" "<<reduce[i].min()<<"\n";
		}
	}catch(std::exception& e){
		std::cout<<"ERROR in QTPIEM::main(int,char**):\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	delete[] fparam;
	delete[] fstruc;
	delete[] input;
	delete[] str;
	
	if(error) return 1;
	else return 0;
}