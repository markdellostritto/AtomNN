// c
#include <ctime>
#include <cstdio>
// str
#include "src/str/string.hpp"
#include "src/str/print.hpp"
// chem
#include "src/chem/units.hpp"
// struc
#include "src/struc/structure.hpp"
#include "src/struc/neighbor.hpp"
// torch
#include "src/torch/pot_ldamp_long.hpp"

Structure make_struc(){
	const double a0=5.3972587586;
	AtomType atomT; 
	atomT.name=true; atomT.an=true; atomT.index=true; atomT.type=true;
	atomT.charge=false; atomT.posn=true; atomT.symm=false; atomT.c6=false;
	Structure struc;
	const int natoms=4;
	struc.resize(natoms,atomT);
	Eigen::Matrix3d lv=Eigen::Matrix3d::Identity()*a0;
	struc.init(lv);
	struc.posn(0)<<0.0,0.0,0.0;
	struc.posn(1)<<0.0,0.5,0.5;
	struc.posn(2)<<0.5,0.0,0.5;
	struc.posn(3)<<0.5,0.5,0.0;
	for(int i=0; i<natoms; ++i){
		struc.posn(i)*=a0;
		struc.name(i)="Ar";
		struc.type(i)=0;
		struc.index(i)=i;
	}
	Structure ssuper;
	Eigen::Vector3i nlat;
	nlat<<3,3,3;
	Structure::super(struc,ssuper,nlat);
	return ssuper;
}

Structure make_NaCl_large(){
	const double a0=16.8599987030;
	AtomType atomT; 
	atomT.name=true; atomT.an=true; atomT.index=true; atomT.type=true;
	atomT.charge=true; atomT.posn=true; atomT.symm=false; atomT.c6
	Structure struc;
	const int natoms=216;
	struc.resize(natoms,atomT);
	Eigen::Matrix3d lv=Eigen::Matrix3d::Identity()*a0;
	struc.init(lv);
	struc.posn(0)<<0,0,0;
	struc.posn(1)<<0,0,0.333333343;
	struc.posn(2)<<0,0,0.666666687;
	struc.posn(3)<<0,0.333333343,0;
	struc.posn(4)<<0,0.333333343,0.333333343;
	struc.posn(5)<<0,0.333333343,0.666666687;
	struc.posn(6)<<0,0.666666687,0;
	struc.posn(7)<<0,0.666666687,0.333333343;
	struc.posn(8)<<0,0.666666687,0.666666687;
	struc.posn(9)<<0.333333343,0,0;
	struc.posn(10)<<0.333333343,0,0.333333343;
	struc.posn(11)<<0.333333343,0,0.666666687;
	struc.posn(12)<<0.333333343,0.333333343,0;
	struc.posn(13)<<0.333333343,0.333333343,0.333333343;
	struc.posn(14)<<0.333333343,0.333333343,0.666666687;
	struc.posn(15)<<0.333333343,0.666666687,0;
	struc.posn(16)<<0.333333343,0.666666687,0.333333343;
	struc.posn(17)<<0.333333343,0.666666687,0.666666687;
	struc.posn(18)<<0.666666687,0,0;
	struc.posn(19)<<0.666666687,0,0.333333343;
	struc.posn(20)<<0.666666687,0,0.666666687;
	struc.posn(21)<<0.666666687,0.333333343,0;
	struc.posn(22)<<0.666666687,0.333333343,0.333333343;
	struc.posn(23)<<0.666666687,0.333333343,0.666666687;
	struc.posn(24)<<0.666666687,0.666666687,0;
	struc.posn(25)<<0.666666687,0.666666687,0.333333343;
	struc.posn(26)<<0.666666687,0.666666687,0.666666687;
	struc.posn(27)<<0,0.166666672,0.166666672;
	struc.posn(28)<<0,0.166666672,0.5;
	struc.posn(29)<<0,0.166666672,0.833333313;
	struc.posn(30)<<0,0.5,0.166666672;
	struc.posn(31)<<0,0.5,0.5;
	struc.posn(32)<<0,0.5,0.833333313;
	struc.posn(33)<<0,0.833333313,0.166666672;
	struc.posn(34)<<0,0.833333313,0.5;
	struc.posn(35)<<0,0.833333313,0.833333313;
	struc.posn(36)<<0.333333343,0.166666672,0.166666672;
	struc.posn(37)<<0.333333343,0.166666672,0.5;
	struc.posn(38)<<0.333333343,0.166666672,0.833333313;
	struc.posn(39)<<0.333333343,0.5,0.166666672;
	struc.posn(40)<<0.333333343,0.5,0.5;
	struc.posn(41)<<0.333333343,0.5,0.833333313;
	struc.posn(42)<<0.333333343,0.833333313,0.166666672;
	struc.posn(43)<<0.333333343,0.833333313,0.5;
	struc.posn(44)<<0.333333343,0.833333313,0.833333313;
	struc.posn(45)<<0.666666687,0.166666672,0.166666672;
	struc.posn(46)<<0.666666687,0.166666672,0.5;
	struc.posn(47)<<0.666666687,0.166666672,0.833333313;
	struc.posn(48)<<0.666666687,0.5,0.166666672;
	struc.posn(49)<<0.666666687,0.5,0.5;
	struc.posn(50)<<0.666666687,0.5,0.833333313;
	struc.posn(51)<<0.666666687,0.833333313,0.166666672;
	struc.posn(52)<<0.666666687,0.833333313,0.5;
	struc.posn(53)<<0.666666687,0.833333313,0.833333313;
	struc.posn(54)<<0.166666672,0,0.166666672;
	struc.posn(55)<<0.166666672,0,0.5;
	struc.posn(56)<<0.166666672,0,0.833333313;
	struc.posn(57)<<0.166666672,0.333333343,0.166666672;
	struc.posn(58)<<0.166666672,0.333333343,0.5;
	struc.posn(59)<<0.166666672,0.333333343,0.833333313;
	struc.posn(60)<<0.166666672,0.666666687,0.166666672;
	struc.posn(61)<<0.166666672,0.666666687,0.5;
	struc.posn(62)<<0.166666672,0.666666687,0.833333313;
	struc.posn(63)<<0.5,0,0.166666672;
	struc.posn(64)<<0.5,0,0.5;
	struc.posn(65)<<0.5,0,0.833333313;
	struc.posn(66)<<0.5,0.333333343,0.166666672;
	struc.posn(67)<<0.5,0.333333343,0.5;
	struc.posn(68)<<0.5,0.333333343,0.833333313;
	struc.posn(69)<<0.5,0.666666687,0.166666672;
	struc.posn(70)<<0.5,0.666666687,0.5;
	struc.posn(71)<<0.5,0.666666687,0.833333313;
	struc.posn(72)<<0.833333313,0,0.166666672;
	struc.posn(73)<<0.833333313,0,0.5;
	struc.posn(74)<<0.833333313,0,0.833333313;
	struc.posn(75)<<0.833333313,0.333333343,0.166666672;
	struc.posn(76)<<0.833333313,0.333333343,0.5;
	struc.posn(77)<<0.833333313,0.333333343,0.833333313;
	struc.posn(78)<<0.833333313,0.666666687,0.166666672;
	struc.posn(79)<<0.833333313,0.666666687,0.5;
	struc.posn(80)<<0.833333313,0.666666687,0.833333313;
	struc.posn(81)<<0.166666672,0.166666672,0;
	struc.posn(82)<<0.166666672,0.166666672,0.333333343;
	struc.posn(83)<<0.166666672,0.166666672,0.666666687;
	struc.posn(84)<<0.166666672,0.5,0;
	struc.posn(85)<<0.166666672,0.5,0.333333343;
	struc.posn(86)<<0.166666672,0.5,0.666666687;
	struc.posn(87)<<0.166666672,0.833333313,0;
	struc.posn(88)<<0.166666672,0.833333313,0.333333343;
	struc.posn(89)<<0.166666672,0.833333313,0.666666687;
	struc.posn(90)<<0.5,0.166666672,0;
	struc.posn(91)<<0.5,0.166666672,0.333333343;
	struc.posn(92)<<0.5,0.166666672,0.666666687;
	struc.posn(93)<<0.5,0.5,0;
	struc.posn(94)<<0.5,0.5,0.333333343;
	struc.posn(95)<<0.5,0.5,0.666666687;
	struc.posn(96)<<0.5,0.833333313,0;
	struc.posn(97)<<0.5,0.833333313,0.333333343;
	struc.posn(98)<<0.5,0.833333313,0.666666687;
	struc.posn(99)<<0.833333313,0.166666672,0;
	struc.posn(100)<<0.833333313,0.166666672,0.333333343;
	struc.posn(101)<<0.833333313,0.166666672,0.666666687;
	struc.posn(102)<<0.833333313,0.5,0;
	struc.posn(103)<<0.833333313,0.5,0.333333343;
	struc.posn(104)<<0.833333313,0.5,0.666666687;
	struc.posn(105)<<0.833333313,0.833333313,0;
	struc.posn(106)<<0.833333313,0.833333313,0.333333343;
	struc.posn(107)<<0.833333313,0.833333313,0.666666687;
	struc.posn(108)<<0.166666672,0.166666672,0.166666672;
	struc.posn(109)<<0.166666672,0.166666672,0.5;
	struc.posn(110)<<0.166666672,0.166666672,0.833333313;
	struc.posn(111)<<0.166666672,0.5,0.166666672;
	struc.posn(112)<<0.166666672,0.5,0.5;
	struc.posn(113)<<0.166666672,0.5,0.833333313;
	struc.posn(114)<<0.166666672,0.833333313,0.166666672;
	struc.posn(115)<<0.166666672,0.833333313,0.5;
	struc.posn(116)<<0.166666672,0.833333313,0.833333313;
	struc.posn(117)<<0.5,0.166666672,0.166666672;
	struc.posn(118)<<0.5,0.166666672,0.5;
	struc.posn(119)<<0.5,0.166666672,0.833333313;
	struc.posn(120)<<0.5,0.5,0.166666672;
	struc.posn(121)<<0.5,0.5,0.5;
	struc.posn(122)<<0.5,0.5,0.833333313;
	struc.posn(123)<<0.5,0.833333313,0.166666672;
	struc.posn(124)<<0.5,0.833333313,0.5;
	struc.posn(125)<<0.5,0.833333313,0.833333313;
	struc.posn(126)<<0.833333313,0.166666672,0.166666672;
	struc.posn(127)<<0.833333313,0.166666672,0.5;
	struc.posn(128)<<0.833333313,0.166666672,0.833333313;
	struc.posn(129)<<0.833333313,0.5,0.166666672;
	struc.posn(130)<<0.833333313,0.5,0.5;
	struc.posn(131)<<0.833333313,0.5,0.833333313;
	struc.posn(132)<<0.833333313,0.833333313,0.166666672;
	struc.posn(133)<<0.833333313,0.833333313,0.5;
	struc.posn(134)<<0.833333313,0.833333313,0.833333313;
	struc.posn(135)<<0.166666672,0,0;
	struc.posn(136)<<0.166666672,0,0.333333343;
	struc.posn(137)<<0.166666672,0,0.666666687;
	struc.posn(138)<<0.166666672,0.333333343,0;
	struc.posn(139)<<0.166666672,0.333333343,0.333333343;
	struc.posn(140)<<0.166666672,0.333333343,0.666666687;
	struc.posn(141)<<0.166666672,0.666666687,0;
	struc.posn(142)<<0.166666672,0.666666687,0.333333343;
	struc.posn(143)<<0.166666672,0.666666687,0.666666687;
	struc.posn(144)<<0.5,0,0;
	struc.posn(145)<<0.5,0,0.333333343;
	struc.posn(146)<<0.5,0,0.666666687;
	struc.posn(147)<<0.5,0.333333343,0;
	struc.posn(148)<<0.5,0.333333343,0.333333343;
	struc.posn(149)<<0.5,0.333333343,0.666666687;
	struc.posn(150)<<0.5,0.666666687,0;
	struc.posn(151)<<0.5,0.666666687,0.333333343;
	struc.posn(152)<<0.5,0.666666687,0.666666687;
	struc.posn(153)<<0.833333313,0,0;
	struc.posn(154)<<0.833333313,0,0.333333343;
	struc.posn(155)<<0.833333313,0,0.666666687;
	struc.posn(156)<<0.833333313,0.333333343,0;
	struc.posn(157)<<0.833333313,0.333333343,0.333333343;
	struc.posn(158)<<0.833333313,0.333333343,0.666666687;
	struc.posn(159)<<0.833333313,0.666666687,0;
	struc.posn(160)<<0.833333313,0.666666687,0.333333343;
	struc.posn(161)<<0.833333313,0.666666687,0.666666687;
	struc.posn(162)<<0,0.166666672,0;
	struc.posn(163)<<0,0.166666672,0.333333343;
	struc.posn(164)<<0,0.166666672,0.666666687;
	struc.posn(165)<<0,0.5,0;
	struc.posn(166)<<0,0.5,0.333333343;
	struc.posn(167)<<0,0.5,0.666666687;
	struc.posn(168)<<0,0.833333313,0;
	struc.posn(169)<<0,0.833333313,0.333333343;
	struc.posn(170)<<0,0.833333313,0.666666687;
	struc.posn(171)<<0.333333343,0.166666672,0;
	struc.posn(172)<<0.333333343,0.166666672,0.333333343;
	struc.posn(173)<<0.333333343,0.166666672,0.666666687;
	struc.posn(174)<<0.333333343,0.5,0;
	struc.posn(175)<<0.333333343,0.5,0.333333343;
	struc.posn(176)<<0.333333343,0.5,0.666666687;
	struc.posn(177)<<0.333333343,0.833333313,0;
	struc.posn(178)<<0.333333343,0.833333313,0.333333343;
	struc.posn(179)<<0.333333343,0.833333313,0.666666687;
	struc.posn(180)<<0.666666687,0.166666672,0;
	struc.posn(181)<<0.666666687,0.166666672,0.333333343;
	struc.posn(182)<<0.666666687,0.166666672,0.666666687;
	struc.posn(183)<<0.666666687,0.5,0;
	struc.posn(184)<<0.666666687,0.5,0.333333343;
	struc.posn(185)<<0.666666687,0.5,0.666666687;
	struc.posn(186)<<0.666666687,0.833333313,0;
	struc.posn(187)<<0.666666687,0.833333313,0.333333343;
	struc.posn(188)<<0.666666687,0.833333313,0.666666687;
	struc.posn(189)<<0,0,0.166666672;
	struc.posn(190)<<0,0,0.5;
	struc.posn(191)<<0,0,0.833333313;
	struc.posn(192)<<0,0.333333343,0.166666672;
	struc.posn(193)<<0,0.333333343,0.5;
	struc.posn(194)<<0,0.333333343,0.833333313;
	struc.posn(195)<<0,0.666666687,0.166666672;
	struc.posn(196)<<0,0.666666687,0.5;
	struc.posn(197)<<0,0.666666687,0.833333313;
	struc.posn(198)<<0.333333343,0,0.166666672;
	struc.posn(199)<<0.333333343,0,0.5;
	struc.posn(200)<<0.333333343,0,0.833333313;
	struc.posn(201)<<0.333333343,0.333333343,0.166666672;
	struc.posn(202)<<0.333333343,0.333333343,0.5;
	struc.posn(203)<<0.333333343,0.333333343,0.833333313;
	struc.posn(204)<<0.333333343,0.666666687,0.166666672;
	struc.posn(205)<<0.333333343,0.666666687,0.5;
	struc.posn(206)<<0.333333343,0.666666687,0.833333313;
	struc.posn(207)<<0.666666687,0,0.166666672;
	struc.posn(208)<<0.666666687,0,0.5;
	struc.posn(209)<<0.666666687,0,0.833333313;
	struc.posn(210)<<0.666666687,0.333333343,0.166666672;
	struc.posn(211)<<0.666666687,0.333333343,0.5;
	struc.posn(212)<<0.666666687,0.333333343,0.833333313;
	struc.posn(213)<<0.666666687,0.666666687,0.166666672;
	struc.posn(214)<<0.666666687,0.666666687,0.5;
	struc.posn(215)<<0.666666687,0.666666687,0.833333313;
	for(int i=0; i<natoms; ++i){
		struc.posn(i)*=a0;
	}
	for(int i=0; i<natoms/2; ++i){
		struc.name(i)="Na";
	}
	for(int i=natoms/2; i<natoms; ++i){
		struc.name(i)="Cl";
	}
	for(int i=0; i<natoms/2; ++i){
		struc.type(i)=0;
	}
	for(int i=natoms/2; i<natoms; ++i){
		struc.type(i)=1;
	}
	for(int i=0; i<natoms/2; ++i){
		struc.charge(i)=1;
	}
	for(int i=natoms/2; i<natoms; ++i){
		struc.charge(i)=-1;
	}
	for(int i=0; i<natoms/2; ++i){
		struc.c6(i)=0.54204;
	}
	for(int i=natoms/2; i<natoms; ++i){
		struc.c6(i)=0.31551;
	}
	return struc;
}

void test_mem(){
	//rand
	std::srand(std::time(NULL));
	//constants
	const int ntypes=3;
	const double rc=6;
	const double prec=1e-12;
	//potential - initialization
	ptnl::PotLDampLong pot1;
	pot1.rc()=rc;
	pot1.prec()=prec;
	pot1.resize(ntypes);
	for(int i=0; i<ntypes; ++i){
		for(int j=i+1; j<ntypes; ++j){
			pot1.c6()(i,j)=(1.0*rand())/RAND_MAX*5;
			pot1.rvdw()(i,j)=(1.0*rand())/RAND_MAX*5;
		}
	}
	pot1.init();
	//pack/unpack
	char* arr=new char[serialize::nbytes(pot1)];
	serialize::pack(pot1,arr);
	ptnl::PotLDampLong pot2;
	serialize::unpack(pot2,arr);
	delete[] arr;
	//compare
	std::cout<<"err - serialize = "<<pot1-pot2<<"\n";
}

void test_unit_J(){
	//constants
	const double prec=1e-8;
	const double rcut=6.0;
	//generate NaCl crystal
	units::System unitsys=units::System::METAL;
	units::consts::init(unitsys);
	Structure struc=make_NaCl_large();
	const int natoms=struc.nAtoms();
	//ewald
	KSpace::London london;
	london.prec()=prec;
	london.rcut()=rcut;
	london.init(struc);
	//initialize the london object
	ptnl::PotLDampLong pot;
	pot.rc()=rcut;
	pot.prec()=prec;
	pot.coul()=coul;
	NeighborList nlist(struc,rcut);
	//compute the coulomb interaction matrix
	Eigen::MatrixXd J;
	pot.J(struc,nlist,J);
	//compute the energy
	Eigen::VectorXd q(natoms);
	for(int i=0; i<natoms; ++i) q[i]=struc.charge(i);
	const double energyJ=0.5*q.dot(J*q);
	const double energyC=pot.energy(struc,nlist);
	//print the total energy
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str)<<"\n";
	std::cout<<"TEST - UNIT - J\n";
	std::cout<<"energy/atom (ewald) = "<<energyC/struc.nAtoms()<<"\n";
	std::cout<<"energy/atom (mat)   = "<<energyJ/struc.nAtoms()<<"\n";
	std::cout<<"error (percent)     = "<<std::fabs((energyJ-energyC)/energyC*100.0)<<"\n";
	std::cout<<print::buf(str)<<"\n";
	delete[] str;
}

int main(int argc, char* argv[]){
	
	char* str=new char[print::len_buf];

	std::cout<<print::buf(str)<<"\n";
	std::cout<<print::title("POT_LONDON_LONG",str)<<"\n";
	test_mem();
	std::cout<<print::title("POT_LONDON_LONG",str)<<"\n";
	std::cout<<print::buf(str)<<"\n";
	
	return 0;
}