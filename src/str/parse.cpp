// c libraries
#include <cstring>
#include <iostream>
// parse
#include "src/str/parse.hpp"

namespace input{
	
	void Arg::clear(){
		key_.clear();
		vals_.clear();
	}
	
	void parse(int argc, char* argv[], std::vector<Arg>& args){
		args.clear();
		//loop over all arguments
		for(int i=1; i<argc; ++i){
			//check for argument
			if(argv[i][0]=='-'){
				if(std::strlen(argv[i])==1) continue;//skip "-"
				args.push_back(Arg());
				args.back().key()=std::string(argv[i]+1);
				for(int j=i+1; j<argc; ++j){
					if(argv[j][0]=='-') break;
					else args.back().vals().push_back(argv[j]);
				}
			}
		}
	}
	
} //end namespace input
