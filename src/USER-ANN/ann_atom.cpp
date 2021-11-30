// c libraries
#include <cstdlib>
#include <cstring>
#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
#include <cmath>
#elif (defined __ICC || defined __INTEL_COMPILER)
#include <mathimf.h> //intel math library
#endif
// c++ libraries
#include <iostream>
// ann - str
#include "ann_string.h"
// ann - atom
#include "ann_atom.h"
	
void AtomANN::clear(){
	if(ATOM_PRINT_FUNC>0) std::cout<<"AtomANN::clear():\n";
	name_=std::string("NULL");
	id_=string::hash(name_);
	mass_=0;
	energy_=0;
	charge_=0;
}

AtomANN& AtomANN::read(const char* str, AtomANN& atom){
	if(ATOM_PRINT_FUNC>0) std::cout<<"AtomANN::read(const char*,AtomANN&):\n";
	/*char* name=new char[10];
	std::sscanf(str,"%s %lf %lf %lf",name,&atom.mass(),&atom.energy(),&atom.charge());
	atom.name()=name;
	atom.id()=string::hash(atom.name());
	delete[] name;*/
	std::vector<std::string> strlist;
	const int nstr=string::split(str,string::WS,strlist);
	if(nstr!=4) throw std::invalid_argument("ERROR in AtomANN::read(const char*,AtomANN&): invalid atom format.");
	atom.name()=strlist[0];
	atom.mass()=std::atof(strlist[1].c_str());
	atom.energy()=std::atof(strlist[2].c_str());
	atom.charge()=std::atof(strlist[3].c_str());
	atom.id()=string::hash(atom.name());
	return atom;
}

std::ostream& operator<<(std::ostream& out, const AtomANN& atom){
	return out<<atom.name_<<" "<<atom.mass_<<" "<<atom.energy_<<" "<<atom.charge_;
}

void AtomANN::print(FILE* out, const AtomANN& atom){
	fprintf(out,"%s %f %f %f\n",atom.name().c_str(),atom.mass(),atom.energy(),atom.charge());
}

bool operator==(const AtomANN& atom1, const AtomANN& atom2){
	return (
		atom1.id()==atom2.id() &&
		fabs(atom1.mass()-atom2.mass())<1e-6 &&
		fabs(atom1.energy()-atom2.energy())<1e-6 &&
		fabs(atom1.charge()-atom2.charge())<1e-6
	);
}

namespace serialize{

//**********************************************
// byte measures
//**********************************************

template <> int nbytes(const AtomANN& obj){
	if(ATOM_PRINT_FUNC>0) std::cout<<"nbytes(const AtomANN&):\n";
	int size=0;
	size+=sizeof(double);//mass
	size+=sizeof(double);//energy
	size+=sizeof(double);//charge
	size+=sizeof(int);//id
	size+=nbytes(obj.name());//name
	return size;
}

//**********************************************
// packing
//**********************************************

template <> int pack(const AtomANN& obj, char* arr){
	if(ATOM_PRINT_FUNC>0) std::cout<<"pack(const AtomANN&,char*):\n";
	int pos=0;
	std::memcpy(arr+pos,&obj.mass(),sizeof(double)); pos+=sizeof(double);//mass
	std::memcpy(arr+pos,&obj.energy(),sizeof(double)); pos+=sizeof(double);//energy
	std::memcpy(arr+pos,&obj.charge(),sizeof(double)); pos+=sizeof(double);//charge
	std::memcpy(arr+pos,&obj.id(),sizeof(int)); pos+=sizeof(int);//id
	pos+=pack(obj.name(),arr+pos);//name
	return pos;
}

//**********************************************
// unpacking
//**********************************************

template <> int unpack(AtomANN& obj, const char* arr){
	if(ATOM_PRINT_FUNC>0) std::cout<<"unpack(AtomANN&,const char*):\n";
	int pos=0;
	std::memcpy(&obj.mass(),arr+pos,sizeof(double)); pos+=sizeof(double);//mass
	std::memcpy(&obj.energy(),arr+pos,sizeof(double)); pos+=sizeof(double);//energy
	std::memcpy(&obj.charge(),arr+pos,sizeof(double)); pos+=sizeof(double);//charge
	std::memcpy(&obj.id(),arr+pos,sizeof(int)); pos+=sizeof(int);//id
	pos+=unpack(obj.name(),arr+pos);//name
	return pos;
}
	
}
