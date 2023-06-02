// c++
#include <iostream>
#include <stdexcept>
// string
#include "src/str/string.hpp"
#include "src/str/token.hpp"
#include "src/str/print.hpp"
// basis - radial
#include "src/nnp/basis_radial.hpp"
// basis - angular
#include "src/nnp/basis_angular.hpp"

class Mix{
public:
	enum Type{
		MAX,
		AVG,
		HYPOT,
		UNKNOWN
	};
	//constructor
	Mix():t_(Type::UNKNOWN){}
	Mix(Type t):t_(t){}
	//operators
	operator Type()const{return t_;}
	//member functions
	static Mix read(const char* str);
	static const char* name(const Mix& mix);
private:
	Type t_;
	//prevent automatic conversion for other built-in types
	//template<typename T> operator T() const;
};

std::ostream& operator<<(std::ostream& out, const Mix& mix){
	switch(mix){
		case Mix::MAX: out<<"MAX"; break;
		case Mix::AVG: out<<"AVG"; break;
		case Mix::HYPOT: out<<"HYPOT"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

const char* Mix::name(const Mix& mix){
	switch(mix){
		case Mix::MAX: return "MAX";
		case Mix::AVG: return "AVG";
		case Mix::HYPOT: return "HYPOT";
		default: return "UNKNOWN";
	}
}

Mix Mix::read(const char* str){
	if(std::strcmp(str,"MAX")==0) return Mix::MAX;
	else if(std::strcmp(str,"AVG")==0) return Mix::AVG;
	else if(std::strcmp(str,"HYPOT")==0) return Mix::HYPOT;
	else return Mix::UNKNOWN;
}

int main(int argc, char* argv[]){
	//file i/o
		FILE* reader=NULL;
		char* pfile=new char[string::M];
		char* input=new char[string::M];
		char* strbuf=new char[print::len_buf];
		Token token;
	//cutoff
		double rcut=0;
		cutoff::Name cutname;
		cutoff::Norm cutnorm;
	//types
		int ntypes=0;
		std::vector<double> radr;
		std::vector<double> rada;
		std::vector<std::string> types;
	//basis - radial
		PhiRN phiRN;
		int nR=0;
		std::vector<double> eta;
	//basis - angular
		PhiAN phiAN;
		int nA=0;
		std::vector<double> zeta;
		std::vector<int> lambda;
		int alpha=0;
		Mix mix;
	//misc
		bool error=false;
	
	try{
		//======== print title ========
		std::cout<<print::buf(strbuf,'*')<<"\n";
		std::cout<<print::buf(strbuf,'*')<<"\n";
		std::cout<<print::title("MAKE BASIS",strbuf,' ')<<"\n";
		std::cout<<print::buf(strbuf,'*')<<"\n";
		std::cout<<print::buf(strbuf,'*')<<"\n";
		
		//======== check the arguments ========
		if(argc!=2) throw std::invalid_argument("Invalid number of arguments.");
		
		//======== load the parameter file ========
		std::cout<<"reading parameter file\n";
		std::strcpy(pfile,argv[1]);
		
		//======== open the parameter file ========
		std::cout<<"opening parameter file\n";
		reader=fopen(pfile,"r");
		if(reader==NULL) throw std::runtime_error(std::string("I/O Error: Could not open parameter file: ")+pfile);
		
		//======== read in the parameters ========
		std::cout<<"reading parameters\n";
		//cutoff
		token.read(string::trim_right(fgets(input,string::M,reader),string::COMMENT),string::WS);
		token.next();
		cutname=cutoff::Name::read(token.next().c_str());
		cutnorm=cutoff::Norm::read(token.next().c_str());
		rcut=std::atof(token.next().c_str());
		std::cout<<"cutoff = "<<cutname<<" "<<cutnorm<<" "<<rcut<<"\n";
		//mixing
		token.read(string::trim_right(fgets(input,string::M,reader),string::COMMENT),string::WS);
		token.next();
		mix=Mix::read(string::to_upper(token.next()).c_str());
		//types
		token.read(string::trim_right(fgets(input,string::M,reader),string::COMMENT),string::WS);
		token.next();
		ntypes=std::atoi(token.next().c_str());
		std::cout<<"ntypes = "<<ntypes<<"\n";
		for(int i=0; i<ntypes; ++i){
			token.read(string::trim_right(fgets(input,string::M,reader),string::COMMENT),string::WS);
			types.push_back(token.next());
			radr.push_back(std::atof(token.next().c_str()));
			rada.push_back(std::atof(token.next().c_str()));
			std::cout<<"type["<<i<<"] = "<<types.back()<<" "<<radr.back()<<" "<<rada.back()<<"\n";
		}
		//radial basis
		std::cout<<"reading radial basis\n";
		token.read(string::trim_right(fgets(input,string::M,reader),string::COMMENT),string::WS);
		token.next();
		phiRN=PhiRN::read(token.next().c_str());
		nR=std::atoi(token.next().c_str());
		for(int i=0; i<nR; ++i){
			token.read(string::trim_right(fgets(input,string::M,reader),string::COMMENT),string::WS);
			eta.push_back(std::atof(token.next().c_str()));
		}
		//angular basis
		std::cout<<"reading angular basis\n";
		token.read(string::trim_right(fgets(input,string::M,reader),string::COMMENT),string::WS);
		token.next();
		phiAN=PhiAN::read(token.next().c_str());
		nA=std::atoi(token.next().c_str());
		alpha=std::atoi(token.next().c_str());
		for(int i=0; i<nA; ++i){
			token.read(string::trim_right(fgets(input,string::M,reader),string::COMMENT),string::WS);
			zeta.push_back(std::atof(token.next().c_str()));
			lambda.push_back(std::atoi(token.next().c_str()));
		}
		
		//======== close file ========
		std::cout<<"closing parameter file\n";
		fclose(reader);
		reader=NULL;
		
		if(mix==Mix::UNKNOWN) throw std::invalid_argument("Invalid triplet radius mixing scheme.\n");
		
		//======== write basis files ========
		for(int i=0; i<ntypes; ++i){
			std::string fname="basis_"+types[i];
			FILE* writer=fopen(fname.c_str(),"w");
			if(writer==NULL) throw std::invalid_argument("Could not open basis file.");
			fprintf(writer,"cut %6.4f\n",rcut);
			fprintf(writer,"nspecies %i\n",ntypes);
			fprintf(writer,"basis %s\n",types[i].c_str());
			//radial basis
			for(int j=0; j<ntypes; ++j){
				fprintf(writer,"basis_radial %s\n",types[j].c_str());
				fprintf(writer,"BasisR %s %s %6.4f %s %i\n",
					cutoff::Norm::name(cutnorm),
					cutoff::Name::name(cutname),
					rcut,PhiRN::name(phiRN),nR
				);
				const double rs=0.5*(radr[i]+radr[j]);
				for(int n=0; n<nR; ++n){
					fprintf(writer,"\t%5.3f %5.3f\n",rs,eta[n]);
				}
			}
			//angular basis
			for(int j=0; j<ntypes; ++j){
				for(int k=j; k<ntypes; ++k){
					fprintf(writer,"basis_angular %s %s\n",types[j].c_str(),types[k].c_str());
					fprintf(writer,"BasisA %s %s %6.4f %s %i %i\n",
						cutoff::Norm::name(cutnorm),
						cutoff::Name::name(cutname),
						rcut,PhiAN::name(phiAN),nA,alpha
					);
					double reta=0;
					switch(mix){
						case Mix::MAX:{
							reta=rada[i]+(rada[j]>rada[k])?rada[j]:rada[k]; 
						} break;
						case Mix::AVG: {
							reta=rada[i]+0.5*(rada[j]+rada[k]); 
						} break;
						case Mix::HYPOT:{
							const double rij=0.5*(rada[i]+rada[j]);
							const double rik=0.5*(rada[i]+rada[k]);
							reta=sqrt(rij*rij+rik*rik);
						} break;
						default: reta=0.0;
					}
					for(int n=0; n<nA; ++n){
						fprintf(writer,"\t%5.3f %5.3f %i\n",reta,zeta[n],lambda[n]);
					}
				}
			}
			//close file
			fclose(writer);
			writer=NULL;
		}
		
	}catch(std::exception& e){
		std::cout<<"ERROR in make_basis(int,char**):\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	delete[] pfile;
	delete[] input;
	delete[] strbuf;
	
	if(error) return 1;
	return 0;
}