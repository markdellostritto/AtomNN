//c++ libraries
#include <iostream>
// ann - str
#include "src/str/print.hpp"
// ann - structure
#include "src/struc/structure.hpp"
// ann - math
#include "src/math/const.hpp"
// ann - neighbor
#include "src/struc/neighbor.hpp"

//**********************************************************************************************
// Neighbor
//**********************************************************************************************

void Neighbor::clear(){
	r_.setZero();
	dr_=0;
	type_=-1;
	index_=-1;
}

//**********************************************************************************************
// NeigborList
//**********************************************************************************************

//==== operators ====

std::ostream& operator<<(std::ostream& out, const NeighborList& obj){
	char* str=new char[print::len_buf];
	out<<print::buf(str)<<"\n";
	out<<print::title("NEIGHBOR-LIST",str)<<"\n";
	out<<"RC     = "<<obj.rc_<<"\n";
	out<<print::title("NEIGHBOR-LIST",str)<<"\n";
	out<<print::buf(str);
	delete[] str;
	return out;
}

void NeighborList::clear(){
	rc_=0.0;
	neigh_.clear();
}

void NeighborList::build(const Structure& struc, double rc){
	rc_=rc;
	//resize the neighbor list
	neigh_.resize(struc.nAtoms());
	//local variables
	const double rc2=rc*rc;
	//lattice vector shifts
	if(struc.R().norm()>0){
		Eigen::Vector3d tmp;
		const int shellx=floor(2.0*rc/struc.R().row(0).lpNorm<Eigen::Infinity>());//number of repeated unit cells needed in the x-dir.
		const int shelly=floor(2.0*rc/struc.R().row(1).lpNorm<Eigen::Infinity>());//number of repeated unit cells needed in the y-dir.
		const int shellz=floor(2.0*rc/struc.R().row(2).lpNorm<Eigen::Infinity>());//number of repeated unit cells needed in the z-dir.
		const int Rsize=(2*shellx+1)*(2*shelly+1)*(2*shellz+1);
		if(STRUC_PRINT_DATA>0) std::cout<<"Rsize = "<<Rsize<<"\n";
		if(STRUC_PRINT_DATA>0) std::cout<<"shell = ("<<shellx<<","<<shelly<<","<<shellz<<") = "<<(2*shellx+1)*(2*shelly+1)*(2*shellz+1)<<"\n";
		std::vector<Eigen::Vector3d> R_(Rsize);
		int count=0;
		for(int ix=-shellx; ix<=shellx; ++ix){
			for(int iy=-shelly; iy<=shelly; ++iy){
				for(int iz=-shellz; iz<=shellz; ++iz){
					R_[count++].noalias()=ix*struc.R().col(0)+iy*struc.R().col(1)+iz*struc.R().col(2);
				}
			}
		}
		//loop over all atoms
		if(STRUC_PRINT_DATA>0) std::cout<<"computing neighbor list\n";
		for(int i=0; i<struc.nAtoms(); ++i){
			//clear the neighbor list
			neigh_[i].clear();
			//loop over all atoms
			for(int j=0; j<struc.nAtoms(); ++j){
				const Eigen::Vector3d rIJ_=struc.diff(struc.posn(i),struc.posn(j),tmp);
				//loop over lattice vector shifts - atom j
				for(int n=0; n<Rsize; ++n){
					//alter the rIJ_ distance by a lattice vector shift
					const Eigen::Vector3d rIJt_=rIJ_-R_[n];
					const double dIJ2=rIJt_.squaredNorm();
					if(math::constant::ZERO<dIJ2 && dIJ2<=rc2){
						neigh_[i].push_back(Neighbor());
						neigh_[i].back().r()=rIJt_;
						neigh_[i].back().dr()=std::sqrt(dIJ2);
						neigh_[i].back().type()=struc.type(j);
						const Eigen::Vector3d rIJf_=struc.RInv()*rIJt_;
						if(
							-0.5<=rIJf_[0] && rIJf_[0]<=0.5 &&
							-0.5<=rIJf_[1] && rIJf_[1]<=0.5 &&
							-0.5<=rIJf_[2] && rIJf_[2]<=0.5
						) neigh_[i].back().index()=j;
						//if(neigh_[i].back().index()>=0) std::cout<<"neigh index "<<neigh_[i].back().index()<<"\n";
					}
				}
			}
		}
	} else {
		//loop over all atoms
		if(STRUC_PRINT_DATA>0) std::cout<<"computing neighbor list\n";
		for(int i=0; i<struc.nAtoms(); ++i){
			//clear the neighbor list
			neigh_[i].clear();
			//loop over all atoms
			for(int j=0; j<struc.nAtoms(); ++j){
				const Eigen::Vector3d rIJ_=struc.posn(i)-struc.posn(j);
				//loop over lattice vector shifts - atom j
				const double dIJ2=rIJ_.squaredNorm();
				if(math::constant::ZERO<dIJ2 && dIJ2<=rc2){
					neigh_[i].push_back(Neighbor());
					neigh_[i].back().r()=rIJ_;
					neigh_[i].back().dr()=std::sqrt(dIJ2);
					neigh_[i].back().type()=struc.type(j);
					neigh_[i].back().index()=j;
					//if(neigh_[i].back().index()>=0) std::cout<<"neigh index "<<neigh_[i].back().index()<<"\n";
				}
			}
		}
	}
}

//**********************************************************************************************
// serialization
//**********************************************************************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const Neighbor& obj){
		if(STRUC_PRINT_FUNC>0) std::cout<<"nbytes(const Neighbor&)\n";
		int size=0;
		size+=sizeof(double)*3;//r_
		size+=sizeof(double);//dr_
		size+=sizeof(int);//type_
		size+=sizeof(int);//index_
		return size;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const Neighbor& obj, char* arr){
		if(STRUC_PRINT_FUNC>0) std::cout<<"pack(const Neighbor&,char*):\n";
		int pos=0;
		pos+=pack(obj.r(),arr+pos);
		std::memcpy(arr+pos,&obj.dr(),sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&obj.type(),sizeof(int)); pos+=sizeof(int);
		std::memcpy(arr+pos,&obj.index(),sizeof(int)); pos+=sizeof(int);
		return pos;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(Neighbor& obj, const char* arr){
		if(STRUC_PRINT_FUNC>0) std::cout<<"unpack(Neighbor&,const char*):\n";
		int pos=0;
		pos+=unpack(obj.r(),arr+pos);
		std::memcpy(&obj.dr(),arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.type(),arr+pos,sizeof(int)); pos+=sizeof(int);
		std::memcpy(&obj.index(),arr+pos,sizeof(int)); pos+=sizeof(int);
		return pos;
	}
	
}
