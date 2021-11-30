#ifndef NEIGHBOR_HPP
#define NEIGHBOR_HPP

//c++ libraries
#include <iosfwd>
#include <string>
// ann - struc
#include "src/struc/structure.hpp"

//**********************************************************************************************
//Neighbor
//**********************************************************************************************

class Neighbor{
private:
	Eigen::Vector3d r_;//distance vector pointing from neighbor to central atom
	double dr_;//norm of r_;
	int type_;
	int index_;
public:
	Neighbor():dr_(0.0),type_(-1),index_(-1),r_(Eigen::Vector3d::Zero()){};
	~Neighbor(){};
	
	Eigen::Vector3d& r(){return r_;}
	const Eigen::Vector3d& r()const{return r_;}
	double& dr(){return dr_;}
	const double& dr()const{return dr_;}
	int& type(){return type_;}
	const int& type()const{return type_;}
	int& index(){return index_;}
	const int& index()const{return index_;}
	
	void clear();
};

//**********************************************************************************************
// NeigborList
//**********************************************************************************************

class NeighborList{
private:
	double rc_;
	std::vector<std::vector<Neighbor> > neigh_;
public:
	//==== constructors/destructors ====
	NeighborList(){defaults();}
	NeighborList(const Structure& struc, double rc){defaults();build(struc,rc);}
	~NeighborList(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const NeighborList& obj);
	
	//==== access ====
	double& rc(){return rc_;}
	const double& rc()const{return rc_;}
	int size(int i)const{return neigh_[i].size();}
	Neighbor& neigh(int i, int j){return neigh_[i][j];}
	const Neighbor& neigh(int i, int j)const{return neigh_[i][j];}
	
	//==== member functions ====
	void read(const char* str);
	void build(const Structure& struc, double rc);
	void build(const Structure& struc){build(struc,rc_);}
	void clear();
	void defaults(){clear();}
};

//**********************************************************************************************
// serialization
//**********************************************************************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const Neighbor& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const Neighbor& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(Neighbor& obj, const char* arr);
	
}

#endif