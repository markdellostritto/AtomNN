//c++
#include <iostream>
#include <stdexcept>
// str
#include "src/str/string.hpp"
#include "src/str/token.hpp"
// struc
#include "src/struc/structure.hpp"
#include "src/struc/pair.hpp"

//=== operators ====

std::ostream& operator<<(std::ostream& out, const Pair& pair){
	return out<<"pair rc "<<pair.rcut_<<" stride "<<pair.stride_;
}

//==== member functions ====

void Pair::clear(){
	neigh_.clear();
}

void Pair::read(const Token& token){
	Token token_=token;
	int stride=0;
	double rcut=0;
	//pair rc 6.0 stride 10
	while(!token_.end()){
		const std::string tag=string::to_upper(token_.next());
		if(tag=="STRIDE"){
			stride=std::atoi(token_.next().c_str());
		} else if(tag=="RC"){
			rcut=std::atof(token_.next().c_str());
		} 
	}
	if(stride<=0) throw std::invalid_argument("Pair::read(const Token&): invalid stride.");
	if(rcut<=0) throw std::invalid_argument("Pair::read(const Token&): invalid rcut.");
	stride_=stride;
	rcut_=rcut;
	rcut2_=rcut*rcut;
}

void Pair::build(const Structure& struc, double rcut){
	if(PAIR_PRINT_FUNC>0) std::cout<<"Pair::build(const Structure&,double):\n";
	Eigen::Vector3d r;
	const int natoms=struc.nAtoms();
	rcut_=rcut;
	rcut2_=rcut_*rcut_;
	neigh_.resize(natoms);
	for(int i=0; i<natoms; ++i){
		neigh_[i].clear();
		for(int j=0; j<natoms; ++j){
			const double dr2=struc.dist2(struc.posn(i),struc.posn(j),r);
			if(1e-6<dr2 && dr2<rcut2_){
				neigh_[i].push_back(j);
			}
		}
	}
}