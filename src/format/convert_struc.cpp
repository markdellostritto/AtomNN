// c libraries
#include <cstdlib>
#include <cstdio>
// c++ libraries
#include <iostream>
// io
#include "src/str/parse.hpp"
#include "src/str/string.hpp"
// structure
#include "src/struc/structure.hpp"
// format
#include "src/format/file_struc.hpp"
#include "src/format/format.hpp"
// math
#include "src/math/const.hpp"

int main(int argc, char* argv[]){
	//simulation
		Structure struc;
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
		
		//print parameters
		std::cout<<"format-in  = "<<format_in<<"\n";
		std::cout<<"format-out = "<<format_out<<"\n";
		std::cout<<"file-in    = "<<file_in<<"\n";
		std::cout<<"file-out   = "<<file_out<<"\n";
		std::cout<<"frac       = "<<atomT.frac<<"\n";
		
		//check parameters
		if(format_in==FILE_FORMAT::UNKNOWN) throw std::invalid_argument("Invalid input format.");
		if(format_out==FILE_FORMAT::UNKNOWN) throw std::invalid_argument("Invalid output format.");
		
		//read
		std::cout<<"reading structure\n";
		read_struc(file_in.c_str(),format_in,atomT,struc);
		
		//write
		std::cout<<"writing structure\n";
		write_struc(file_out.c_str(),format_out,atomT,struc);
		
	}catch(std::exception& e){
		std::cout<<"ERROR in convert_struc::main(int,char**):\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	if(error) return 1;
	else return 0;
}