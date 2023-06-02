// c libraries
#include <cstdlib>
#include <cstdio>
// c++ libraries
#include <iostream>
// io
#include "src/str/token.hpp"
#include "src/str/parse.hpp"
#include "src/str/string.hpp"
// structure
#include "src/struc/sim.hpp"
// format
#include "src/format/file_sim.hpp"
#include "src/format/format.hpp"
// math
#include "src/math/const.hpp"

int main(int argc, char* argv[]){
	//simulation
		Simulation sim;
		Interval interval;
		FILE_FORMAT::type format_in;
		FILE_FORMAT::type format_out;
	//file i/o
		std::string file_in;
		std::string file_out;
	//arguments
		Parser parser;
		Parser::Arg arg;
	//atom type
		AtomType atomT;
		atomT.name=true; atomT.an=true; atomT.type=true; atomT.index=true;
		atomT.posn=true; atomT.force=true; atomT.frac=true;
	//offset
		std::string offset_str;
		Eigen::Vector3d offset=Eigen::Vector3d::Zero();
	//misc
		bool error=false;
		
	try{
		//check the number of arguments
		if(argc==1) throw std::invalid_argument("No arguments provided.");
		
		//parse the arguments
		parser.read(argc,argv);
		
		//read the arguments
		if(parser.arg("fin",arg) && arg.nvals()==1) format_in=FILE_FORMAT::read(arg.val(0).c_str());
		else throw std::invalid_argument("Invalid input specification.");
		if(parser.arg("fout",arg) && arg.nvals()==1) format_out=FILE_FORMAT::read(arg.val(0).c_str());
		else throw std::invalid_argument("Invalid output specification.");
		if(parser.arg("in",arg) && arg.nvals()==1) file_in=arg.val(0);
		else throw std::invalid_argument("No output file.");
		if(parser.arg("out",arg) && arg.nvals()==1) file_out=arg.val(0);
		else throw std::invalid_argument("No output file.");
		if(parser.arg("cart",arg)) atomT.frac=false;
		if(parser.arg("frac",arg)) atomT.frac=true;
		if(parser.arg("interval",arg) && arg.nvals()==1) Interval::read(arg.val(0).c_str(),interval);
		else throw std::invalid_argument("No interval.");
		if(parser.arg("offset",arg) && arg.nvals()==1) offset_str=arg.val(0);
		else throw std::invalid_argument("No offset.");
		
		//read offset
		if(!offset_str.empty()){
			Token token(offset_str.c_str(),":");
			offset[0]=std::atof(token.next().c_str());
			offset[1]=std::atof(token.next().c_str());
			offset[2]=std::atof(token.next().c_str());
		}
		
		//print parameters
		std::cout<<"format-in  = "<<format_in<<"\n";
		std::cout<<"format-out = "<<format_out<<"\n";
		std::cout<<"file-in    = "<<file_in<<"\n";
		std::cout<<"file-out   = "<<file_out<<"\n";
		std::cout<<"interval   = "<<interval<<"\n";
		std::cout<<"offset     = "<<offset.transpose()<<"\n";
		
		//check parameters
		if(format_in==FILE_FORMAT::UNKNOWN) throw std::invalid_argument("Invalid input format.");
		if(format_out==FILE_FORMAT::UNKNOWN) throw std::invalid_argument("Invalid output format.");
		
		//read
		std::cout<<"reading structure\n";
		read_sim(file_in.c_str(),format_in,interval,atomT,sim);
		std::cout<<"reading completed\n";
		
		//apply offset
		if(offset.norm()>1e-6){
			std::cout<<"setting offset\n";
			Eigen::Vector3d r;
			for(int t=0; t<sim.timesteps(); ++t){
				for(int n=0; n<sim.frame(t).nAtoms(); ++n){
					sim.frame(t).posn(n).noalias()+=offset;
					sim.frame(t).modv(sim.frame(t).posn(n),r);
					sim.frame(t).posn(n)=r;
				}
			}
		}
		
		//write
		std::cout<<"writing structure\n";
		interval.beg()=1; interval.end()=-1;
		write_sim(file_out.c_str(),format_out,interval,atomT,sim);
		std::cout<<"writing completed\n";
		
	}catch(std::exception& e){
		std::cout<<"ERROR in convert_sim::main(int,char**):\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	if(error) return 1;
	else return 0;
}