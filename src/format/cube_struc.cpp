//c libraries
#include <cstring>
#include <cmath>
//c++ libraries
#include <iostream>
//Eigen
#include <Eigen/Dense>
//local
#include "string.hpp"
#include "units.hpp"
#include "cube.hpp"

namespace CUBE{
		
void read(const char* file, const AtomType& atomT, Structure& struc, Grid& grid){
	if(CUBE_PRINT_FUNC>0) std::cout<<"read(const char*,const AtomType&,Structure&,Grid&):\n";
	//==== local variables ====
	//file i/o
		FILE* reader=NULL;
		char* input=new char[string::M];
	//structure
		Eigen::Vector3d origin;
		int natoms=0;
		int np=0;
		Eigen::Vector3i gridsize=Eigen::Vector3i::Zero();
		Eigen::Matrix3d lv,lvv;//lattice vectors
	//units
		double s_len=0.0;
		if(units::consts::system()==units::System::AU) s_len=units::BOHRpANG;
		else if(units::consts::system()==units::System::METAL) s_len=1.0;
		else throw std::runtime_error("Invalid units.");
		
	//==== open file ====
	if(CUBE_PRINT_STATUS>0) std::cout<<"opening file\n";
	reader=fopen(file,"r");
	if(reader==NULL) throw std::runtime_error("Could not open cube file");
	
	//==== read grid ====
	if(CUBE_PRINT_STATUS>0) std::cout<<"reading grid\n";
	fgets(input,string::M,reader);//skip line
	fgets(input,string::M,reader);//skip line
	//read natoms, origin
	std::sscanf(fgets(input,string::M,reader),"%i %lf %lf %lf",&natoms,&origin[0],&origin[1],&origin[2]);
	if(CUBE_PRINT_DATA>0) std::cout<<"natoms = "<<natoms<<"\n";
	if(CUBE_PRINT_DATA>1) std::cout<<"origin = "<<origin.transpose()<<"\n";
	//read grid
	std::sscanf(fgets(input,string::M,reader),"%i %lf %lf %lf",&gridsize[0],&lvv(0,0),&lvv(1,0),&lvv(2,0));
	std::sscanf(fgets(input,string::M,reader),"%i %lf %lf %lf",&gridsize[1],&lvv(0,1),&lvv(1,1),&lvv(2,1));
	std::sscanf(fgets(input,string::M,reader),"%i %lf %lf %lf",&gridsize[2],&lvv(0,2),&lvv(1,2),&lvv(2,2));
	if(CUBE_PRINT_DATA>1) std::cout<<"n = "<<gridsize.transpose()<<"\n";
	if(CUBE_PRINT_DATA>1) std::cout<<"lvv = \n"<<lvv<<"\n";
	//set units
	if(gridsize[0]<0){
		//lengths are in Angstroms
		if(units::consts::system()==units::System::AU) s_len=1.0/units::BOHR;
		else if(units::consts::system()==units::System::METAL) s_len=1.0;
		else throw std::runtime_error("Invalid units.");
	} else {
		//lengths are in Bohr
		if(units::consts::system()==units::System::AU) s_len=1.0;
		else if(units::consts::system()==units::System::METAL) s_len=units::BOHR;
		else throw std::runtime_error("Invalid units.");
	}
	lvv*=s_len;
	gridsize[0]=std::fabs(gridsize[0]);
	gridsize[1]=std::fabs(gridsize[1]);
	gridsize[2]=std::fabs(gridsize[2]);
	np=gridsize[0]*gridsize[1]*gridsize[2];
	//set lattice vector
	lv.col(0)=lvv.col(0)*gridsize[0];
	lv.col(1)=lvv.col(1)*gridsize[1];
	lv.col(2)=lvv.col(2)*gridsize[2];
	if(CUBE_PRINT_DATA>0) std::cout<<"s_len = "<<s_len<<"\n";
	if(CUBE_PRINT_DATA>1) std::cout<<"lv = \n"<<lv<<"\n";
	
	//=== read atoms ===
	if(CUBE_PRINT_STATUS>0) std::cout<<"reading atoms\n";
	//resize structure
	struc.resize(natoms,atomT);
	//read atoms
	for(int i=0; i<natoms; ++i){
		double an,mass;
		Eigen::Vector3d p;
		std::sscanf(fgets(input,string::M,reader),"%i %lf %lf %lf %lf",&an,&p[0],&p[1],&p[2],&mass);
		if(atomT.posn) struc.posn(i)=p*s_len;
		if(atomT.mass) struc.mass(i)=mass;
		if(atomT.an) struc.an(i)=an;
	}
	
	//==== read grid ====
	if(CUBE_PRINT_STATUS>0) std::cout<<"reading grid\n";
	const int nlines=np/6;
	const int rem=np-nlines*6;
	grid.resize(gridsize);
	grid.voxel()=lvv;
	int c=0;
	Eigen::VectorXd dat=Eigen::VectorXd::Zero(7);
	for(int i=0; i<nlines; ++i){
		std::sscanf(fgets(input,string::M,reader),"%lf %lf %lf %lf %lf %lf",
			&dat[0],&dat[1],&dat[2],&dat[3],&dat[4],&dat[5]
		);
		grid.data(c++)=dat[0];
		grid.data(c++)=dat[1];
		grid.data(c++)=dat[2];
		grid.data(c++)=dat[3];
		grid.data(c++)=dat[4];
		grid.data(c++)=dat[5];
	}
	fgets(input,string::M,reader);
	if(rem>0){
		grid.data(c++)=std::atof(std::strtok(input,string::WS));
		for(int i=1; i<rem; ++i){
			grid.data(c++)=std::atof(std::strtok(NULL,string::WS));
		}
	}
	
}

}