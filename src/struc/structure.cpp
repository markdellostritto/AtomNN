//c++ libraries
#include <iostream>
//c libraries
#include <stdexcept>
// ann - strings
#include "src/str/string.hpp"
// ann - chemistry
#include "src/chem/ptable.hpp"
// ann - eigen
#include "src/math/eigen.hpp"
#include "src/math/const.hpp"
// ann - print
#include "src/str/print.hpp"
// ann - structure
#include "src/struc/structure.hpp"

//**********************************************************************************************
//AtomData
//**********************************************************************************************

//==== operators ====

std::ostream& operator<<(std::ostream& out, const AtomData& obj){
	out<<"natoms = "<<obj.nAtoms_<<"\n";
	out<<"type   = "<<obj.atomType_;
	return out;
}

//==== member functions ====

void AtomData::clear(){
	if(STRUC_PRINT_FUNC>0) std::cout<<"AtomData::clear():\n";
	//basic properties
	name_.clear();
	an_.clear();
	type_.clear();
	index_.clear();
	//serial properties
	mass_.clear();
	charge_.clear();
	radius_.clear();
	chi_.clear();
	eta_.clear();
	c6_.clear();
	js_.clear();
	//vector properties
	posn_.clear();
	vel_.clear();
	force_.clear();
	spin_.clear();
	//nnp
	symm_.clear();
}

//==== resizing ====

void AtomData::resize(int nAtoms, const AtomType& atomT){
	if(STRUC_PRINT_FUNC>0) std::cout<<"AtomData::resize(int,const AtomType&):\n";
	//check arguments
	if(nAtoms<=0) throw std::runtime_error("AtomData::resize(int,const AtomType&): invalid number of atoms");
	//set atom info
	atomType_=atomT;
	nAtoms_=nAtoms;
	//basic properties
	if(atomT.name)  name_.resize(nAtoms);
	if(atomT.an)    an_.resize(nAtoms,0);
	if(atomT.type)  type_.resize(nAtoms,-1);
	if(atomT.index) index_.resize(nAtoms,-1);
	//serial properties
	if(atomT.mass)   mass_.resize(nAtoms,0.0);
	if(atomT.charge) charge_.resize(nAtoms,0.0);
	if(atomT.radius) radius_.resize(nAtoms,0.0);
	if(atomT.chi)    chi_.resize(nAtoms,0.0);
	if(atomT.eta)    eta_.resize(nAtoms,0.0);
	if(atomT.c6)     c6_.resize(nAtoms,0.0);
	if(atomT.js)     js_.resize(nAtoms,0.0);
	//vector properties
	if(atomT.posn)  posn_.resize(nAtoms,Eigen::Vector3d::Zero());
	if(atomT.vel)   vel_.resize(nAtoms,Eigen::Vector3d::Zero());
	if(atomT.force) force_.resize(nAtoms,Eigen::Vector3d::Zero());
	if(atomT.spin)  spin_.resize(nAtoms,Eigen::Vector3d::Zero());
	//nnp
	if(atomT.symm)	symm_.resize(nAtoms);
	
	if(atomT.index) for(int i=0; i<nAtoms; ++i) index_[i]=i;
}

//**********************************************************************************************
//Structure
//**********************************************************************************************

//==== operators ====

std::ostream& operator<<(std::ostream& out, const Structure& struc){
	char* str=new char[print::len_buf];
	out<<print::buf(str)<<"\n";
	out<<print::title("STRUCTURE",str)<<"\n";
	out<<static_cast<const AtomData&>(struc)<<"\n";
	out<<static_cast<const Cell&>(struc)<<"\n";
	out<<static_cast<const State&>(struc)<<"\n";
	out<<print::buf(str);
	delete[] str;
	return out;
}

//==== member functions ====

void Structure::clear(){
	if(STRUC_PRINT_FUNC>0) std::cout<<"Structure::clear():\n";
	AtomData::clear();
	Cell::clear();
	State::clear();
}

//==== static functions ====

void Structure::write_binary(const Structure& struc, const char* file){
	if(STRUC_PRINT_FUNC>0) std::cout<<"Structure::write_binary(const char*):\n";
	//local variables
	char* arr=NULL;
	FILE* writer=NULL;
	bool error=false;
	int nWrite=-1;
	try{
		//open file
		writer=fopen(file,"wb");
		if(writer==NULL) throw std::runtime_error(std::string("write_binary(Structure&,const char*): Could not open file: ")+std::string(file));
		//allocate buffer
		const int nBytes=serialize::nbytes(struc);
		arr=new char[nBytes];
		if(arr==NULL) throw std::runtime_error("write_binary(Structure&,const char*): Could not allocate memory.");
		//write to buffer
		serialize::pack(struc,arr);
		//write to file
		nWrite=fwrite(&nBytes,sizeof(int),1,writer);
		if(nWrite!=1) throw std::runtime_error("write_binary(Structure&,const char*): Write error.");
		nWrite=fwrite(arr,sizeof(char),nBytes,writer);
		if(nWrite!=nBytes) throw std::runtime_error("write_binary(Structure&,const char*): Write error.");
		//close the file, free memory
		delete[] arr; arr=NULL;
		fclose(writer); writer=NULL;
	}catch(std::exception& e){
		std::cout<<"ERROR in write_binary(Structure& struc,const char*):\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	//free local variables
	if(arr!=NULL) delete[] arr;
	if(writer!=NULL) fclose(writer);
	if(error) throw std::runtime_error("Failed to write");
}

void Structure::read_binary(Structure& struc, const char* file){
	if(STRUC_PRINT_FUNC>0) std::cout<<"Structure::read_binary(const char*):\n";
	//local variables
	char* arr=NULL;
	FILE* reader=NULL;
	bool error=false;
	int nRead=-1;
	try{
		//open file
		reader=fopen(file,"rb");
		if(reader==NULL) throw std::runtime_error(std::string("read_binary(Structure&,const char*): Could not open file: ")+std::string(file));
		//find size
		int nBytes=0;
		nRead=fread(&nBytes,sizeof(int),1,reader);
		if(nRead!=1) throw std::runtime_error("read_binary(Structure&,const char*): Read error.");
		//allocate buffer
		arr=new char[nBytes];
		if(arr==NULL) throw std::runtime_error("read_binary(Structure&,const char*): Could not allocate memory.");
		//read from file
		nRead=fread(arr,sizeof(char),nBytes,reader);
		if(nRead!=nBytes) throw std::runtime_error("read_binary(Structure&,const char*): Read error.");
		//read from buffer
		serialize::unpack(struc,arr);
		//close the file, free memory
		delete[] arr; arr=NULL;
		fclose(reader); reader=NULL;
	}catch(std::exception& e){
		std::cout<<"ERROR in read_binary(Structure& struc,const char*):\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	//free local variables
	if(arr!=NULL) delete[] arr;
	if(reader!=NULL) fclose(reader);
	if(error) throw std::runtime_error("Failed to read");
}

Structure& Structure::super(const Structure& struc, Structure& superc, const Eigen::Vector3i nlat){
	if(nlat[0]<=0 || nlat[1]<=0 || nlat[2]<=0) throw std::invalid_argument("Invalid lattice.");
	const int np=nlat.prod();
	const int nAtomsT=struc.nAtoms()*np;
	superc.resize(nAtomsT,struc.atomType());
	//set the atomic properties
	int c=0;
	const AtomType& atomT=struc.atomType();
	for(int i=0; i<nlat[0]; ++i){
		for(int j=0; j<nlat[1]; ++j){
			for(int k=0; k<nlat[2]; ++k){
				const Eigen::Vector3d R=i*struc.R().col(0)+j*struc.R().col(1)+k*struc.R().col(2);
				for(int n=0; n<struc.nAtoms(); ++n){
					//set map
					Eigen::Vector3i index; index<<i,j,k;
					//basic properties
					if(atomT.name)		superc.name(c)=struc.name(n);
					if(atomT.an)		superc.an(c)=struc.an(n);
					if(atomT.type)		superc.type(c)=struc.type(n);
					if(atomT.index)	superc.index(c)=struc.index(n);
					//serial properties
					if(atomT.mass)		superc.mass(c)=struc.mass(n);
					if(atomT.charge)	superc.charge(c)=struc.charge(n);
					if(atomT.radius)	superc.radius(c)=struc.radius(n);
					if(atomT.chi)		superc.chi(c)=struc.chi(n);
					if(atomT.eta)		superc.eta(c)=struc.eta(n);
					if(atomT.c6)		superc.c6(c)=struc.c6(n);
					if(atomT.js)		superc.js(c)=struc.js(n);
					//vector properties
					if(atomT.posn)		superc.posn(c)=struc.posn(n)+R;
					if(atomT.vel) 		superc.vel(c)=struc.vel(n);
					if(atomT.force) 	superc.force(c)=struc.force(n);
					if(atomT.spin) 	superc.spin(c)=struc.spin(n);
					//nnp
					if(atomT.symm) 	superc.symm(c)=struc.symm(n);
					//increment
					c++;
				}
			}
		}
	}
	Eigen::MatrixXd Rnew=struc.R();
	Rnew.col(0)*=nlat[0];
	Rnew.col(1)*=nlat[1];
	Rnew.col(2)*=nlat[2];
	static_cast<Cell&>(superc).init(Rnew);
	return superc;
}

//**********************************************************************************************
//AtomSpecies
//**********************************************************************************************

//==== operators ====

std::ostream& operator<<(std::ostream& out, const AtomSpecies& obj){
	const int ns=obj.species_.size();
	for(int i=0; i<ns; ++i) out<<obj.species_[i]<<" "<<obj.nAtoms_[i]<<" ";
	return out;
}

//==== member functions ====

int AtomSpecies::nTot()const{
	int ntot=0;
	for(int i=0; i<nAtoms_.size(); ++i) ntot+=nAtoms_[i];
	return ntot;
}

void AtomSpecies::defaults(){
	if(STRUC_PRINT_FUNC>0) std::cout<<"AtomSpecies::defaults():\n";
	species_.clear();
	nAtoms_.clear();
	offsets_.clear();
}

void AtomSpecies::resize(const std::vector<std::string>& names, const std::vector<int>& nAtoms){
	if(STRUC_PRINT_FUNC>0) std::cout<<"AtomSpecies::resize(const std::vector<int>&,const std::vector<std::string>&):\n";
	if(nAtoms.size()!=names.size()) throw std::invalid_argument("Array size mismatch.");
	nSpecies_=names.size();
	species_=names;
	nAtoms_=nAtoms;
	offsets_.resize(nSpecies_,0);
	for(int i=1; i<nSpecies_; ++i) offsets_[i]=offsets_[i-1]+nAtoms_[i-1];
}

void AtomSpecies::resize(const Structure& struc){
	if(!struc.atomType().name) throw std::runtime_error("AtomSpecies::resize(const Structure&): cannot initialize without atom names");
	std::vector<std::string> names;
	std::vector<int> nAtoms;
	for(int i=0; i<struc.nAtoms(); ++i){
		int index=-1;
		for(int j=0; j<names.size(); ++j){
			if(names[j]==struc.name(i)){index=j;break;}
		}
		if(index<0){
			names.push_back(struc.name(i));
			nAtoms.push_back(1);
		} else ++nAtoms[index];
	}
	nSpecies_=names.size();
	species_=names;
	nAtoms_=nAtoms;
	offsets_.resize(nSpecies_,0);
	for(int i=1; i<nSpecies_; ++i) offsets_[i]=offsets_[i-1]+nAtoms_[i-1];
}

//==== static functions ====

int AtomSpecies::index_species(const std::string& str, const std::vector<std::string>& names){
	if(STRUC_PRINT_FUNC>0) std::cout<<"AtomSpecies::index_species(const std::string&, const std::vector<std::string>&):\n";
	for(int i=0; i<names.size(); ++i) if(str==names[i]) return i;
	return -1;
}

int AtomSpecies::index_species(const char* str, const std::vector<std::string>& names){
	if(STRUC_PRINT_FUNC>0) std::cout<<"AtomSpecies::index_species(const char*,const std::vector<std::string>&):\n";
	for(int i=0; i<names.size(); ++i) if(std::strcmp(str,names[i].c_str())==0) return i;
	return -1;
}

std::vector<int>& AtomSpecies::read_atoms(const AtomSpecies& as, const char* str, std::vector<int>& ids){
	//local function variables
	int nAtoms=0;
	std::vector<int> atomIndices;
	std::vector<std::string> atomNames;
	std::vector<int> nameIndices;
	
	//load the number of atoms, atom indices, and atom names
	nAtoms=AtomSpecies::read_natoms(str);
	AtomSpecies::read_indices(str,atomIndices);
	AtomSpecies::read_names(str,atomNames);
	nameIndices.resize(atomNames.size());
	for(int n=0; n<nameIndices.size(); ++n){
		nameIndices[n]=as.index_species(atomNames[n]);
		if(nameIndices[n]<0) throw std::invalid_argument("ERROR: Invalid atom name in atom string.");
	}
	
	//set the atom ids
	ids.resize(nAtoms,0);
	for(int n=0; n<nAtoms; ++n) ids[n]=as.index(nameIndices[n],atomIndices[n]);
	
	return ids;
}

int AtomSpecies::read_natoms(const char* str){
	const char* func_name="AtomSpecies::read_natoms(const char*)";
	if(STRUC_PRINT_FUNC>0) std::cout<<func_name<<":\n";
	//local function variables
	char* strtemp=new char[string::M];
	char* substr=new char[string::M];
	char* temp=new char[string::M];
	std::vector<std::string> substrs;
	int nStrs=0,nAtoms=0;
	bool error=false;
	
	try{
		//copy the string
		std::strcpy(strtemp,str);
		//find the number of substrings
		nStrs=string::substrN(strtemp,",");
		substrs.resize(nStrs);
		substrs[0]=std::string(std::strcpy(temp,std::strtok(strtemp,",")));
		for(int i=1; i<nStrs; ++i){
			substrs[i]=std::string(std::strcpy(temp,std::strtok(NULL,",")));
		}
		
		//parse the line by commas
		for(int i=0; i<nStrs; ++i){
			std::strcpy(substr,substrs[i].c_str());
			//find out if this substring has only one atom, or a set of atoms
			if(std::strpbrk(substr,":")==NULL) ++nAtoms; //single atom
			else {
				//set of atoms
				int beg, end;
				//find the beginning index
				std::strcpy(temp,std::strtok(substr,":"));
				if(std::strpbrk(temp,string::DIGITS)==NULL) throw std::invalid_argument("Invalid atom specification: no index.");
				else beg=std::atoi(std::strpbrk(temp,string::DIGITS))-1;
				//find the ending index
				std::strcpy(temp,std::strtok(NULL,":"));
				if(std::strpbrk(temp,string::DIGITS)==NULL) throw std::invalid_argument("Invalid atom specification: no index.");
				else end=std::atoi(std::strpbrk(temp,string::DIGITS))-1;
				//check the indices
				if(beg<0 || end<0 || end<beg) throw std::invalid_argument("Invalid atomic indices.");
				else nAtoms+=end-beg+1;
			}
		}
	}catch(std::exception& e){
		std::cout<<"ERROR in "<<func_name<<":\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	delete[] strtemp;
	delete[] substr;
	delete[] temp;
	
	if(error) throw std::invalid_argument("Invalid atom string.");
	else return nAtoms;
}
	
std::vector<int>& AtomSpecies::read_indices(const char* str, std::vector<int>& indices){
	const char* func_name="AtomSpecies::read_indices(const char*,std::vector<int>&)";
	if(STRUC_PRINT_FUNC>0) std::cout<<func_name<<":\n";
	//local function variables
	char* strtemp=new char[string::M];
	char* substr=new char[string::M];
	char* temp=new char[string::M];
	std::vector<std::string> substrs;
	int nStrs=0;
	bool error=false;
	
	try{
		//clear the vector
		indices.clear();
		//copy the string
		std::strcpy(strtemp,str);
		//find the number of substrings
		nStrs=string::substrN(strtemp,",");
		substrs.resize(nStrs);
		substrs[0]=std::string(std::strcpy(temp,std::strtok(strtemp,",")));
		for(int i=1; i<nStrs; ++i){
			substrs[i]=std::string(std::strcpy(temp,std::strtok(NULL,",")));
		}
		
		//parse the line by commas
		for(int i=0; i<nStrs; ++i){
			std::strcpy(substr,substrs[i].c_str());
			//find out if this substring has only one atom, or a set of atoms
			if(std::strpbrk(substr,":")==NULL){
				//single atom
				if(STRUC_PRINT_STATUS>0) std::cout<<"Single Atom\n";
				if(std::strpbrk(substr,string::DIGITS)==NULL) throw std::invalid_argument("Invalid atom specification: no index.");
				else indices.push_back(std::atoi(std::strpbrk(substr,string::DIGITS))-1);
			} else {
				//set of atoms
				if(STRUC_PRINT_STATUS>0) std::cout<<"Set of AtomData\n";
				int beg, end;
				//find the beginning index
				std::strcpy(temp,std::strtok(substr,":"));
				if(std::strpbrk(temp,string::DIGITS)==NULL) throw std::invalid_argument("Invalid atom specification: no index.");
				else beg=std::atoi(std::strpbrk(temp,string::DIGITS))-1;
				//find the ending index
				std::strcpy(temp,std::strtok(NULL,":"));
				if(std::strpbrk(temp,string::DIGITS)==NULL) throw std::invalid_argument("Invalid atom specification: no index.");
				else end=std::atoi(std::strpbrk(temp,string::DIGITS))-1;
				//check the indices
				if(beg<0 || end<0 || end<beg) throw std::invalid_argument("Invalid atomic indices.");
				for(int j=0; j<end-beg+1; j++) indices.push_back(beg+j);
			}
		}
	}catch(std::exception& e){
		std::cout<<"ERROR in "<<func_name<<":\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	delete[] strtemp;
	delete[] substr;
	delete[] temp;
	
	if(error){
		indices.clear();
		throw std::invalid_argument("Invalid atom string.");
	} else return indices;
}

std::vector<std::string>& AtomSpecies::read_names(const char* str, std::vector<std::string>& names){
	const char* func_name="AtomSpecies::read_names(const char*,std::vector<std::string>&)";
	if(STRUC_PRINT_FUNC>0) std::cout<<func_name<<":\n";
	//local function variables
	char* strtemp=new char[string::M];
	char* substr=new char[string::M];
	char* temp=new char[string::M];
	char* atomName=new char[string::M];
	std::vector<std::string> substrs;
	int nStrs=0;
	bool error=false;
	
	try{
		//clear the vector
		names.clear();
		//copy the string
		std::strcpy(strtemp,str);
		//find the number of substrings
		nStrs=string::substrN(strtemp,",");
		substrs.resize(nStrs);
		substrs[0]=std::string(std::strcpy(temp,std::strtok(strtemp,",")));
		for(int i=1; i<nStrs; ++i){
			substrs[i]=std::string(std::strcpy(temp,std::strtok(NULL,",")));
		}
		
		//parse the line by commas
		for(int i=0; i<nStrs; ++i){
			std::strcpy(substr,substrs[i].c_str());
			//find out if this substring has only one atom, or a set of atoms
			if(std::strpbrk(substr,":")==NULL){
				//single atom
				if(std::strpbrk(substr,string::DIGITS)==NULL) throw std::invalid_argument("Invalid atom specification: no index.");
				else names.push_back(std::string(string::trim_right(substr,string::DIGITS)));
			} else {
				//set of atoms
				int beg, end;
				//find the beginning index
				std::strcpy(temp,std::strtok(substr,":"));
				if(std::strpbrk(temp,string::DIGITS)==NULL) throw std::invalid_argument("Invalid atom specification: no index.");
				else beg=std::atoi(std::strpbrk(temp,string::DIGITS))-1;
				//find the ending index
				std::strcpy(temp,std::strtok(NULL,":"));
				if(std::strpbrk(temp,string::DIGITS)==NULL) throw std::invalid_argument("Invalid atom specification: no index.");
				else end=std::atoi(std::strpbrk(temp,string::DIGITS))-1;
				//check the indices
				if(beg<0 || end<0 || end<beg) throw std::invalid_argument("Invalid atomic indices");
				else {
					std::strcpy(atomName,string::trim_right(temp,string::DIGITS));
					for(int j=0; j<end-beg+1; ++j) names.push_back(std::string(atomName));
				}
			}
		}
	}catch(std::exception& e){
		std::cout<<"ERROR in "<<func_name<<":\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	delete[] strtemp;
	delete[] substr;
	delete[] temp;
	delete[] atomName;
	
	if(error){
		names.clear();
		throw std::invalid_argument("Invalid atom string.");
	} else return names;
}

void AtomSpecies::set_species(const AtomSpecies& as, Structure& struc){
	//find number of atoms, species
	int nAtomsT=0;
	const int nSpecies=as.species().size();
	for(int i=0; i<as.nAtoms().size(); ++i) nAtomsT+=as.nAtoms(i);
	if(nAtomsT!=struc.nAtoms()) throw std::runtime_error("AtomSpecies::set_species(const AtomSpecies&,Structure&): Mismatch in number of atoms.");
	//set species data
	if(struc.atomType().name){
		for(int i=0; i<nSpecies; ++i){
			for(int j=0; j<as.nAtoms(i); ++j){
				struc.name(as.index(i,j))=as.species(i);
			}
		}
	}
	if(struc.atomType().type){
		for(int i=0; i<nSpecies; ++i){
			for(int j=0; j<as.nAtoms(i); ++j){
				struc.type(as.index(i,j))=i;
			}
		}
	}
	if(struc.atomType().index){
		for(int i=0; i<nSpecies; ++i){
			for(int j=0; j<as.nAtoms(i); ++j){
				struc.index(as.index(i,j))=j;
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
	
	template <> int nbytes(const AtomData& obj){
		if(STRUC_PRINT_FUNC>0) std::cout<<"nbytes(const AtomData&)\n";
		int size=0;
		//atom type
		size+=nbytes(obj.atomType());
		//number of atoms
		size+=sizeof(obj.nAtoms());
		//basic properties
		if(obj.atomType().name)   size+=nbytes(obj.name());
		if(obj.atomType().an)     size+=nbytes(obj.an());
		if(obj.atomType().type)   size+=nbytes(obj.type());
		if(obj.atomType().index)  size+=nbytes(obj.index());
		//serial properties
		if(obj.atomType().mass)   size+=nbytes(obj.mass());
		if(obj.atomType().charge) size+=nbytes(obj.charge());
		if(obj.atomType().radius) size+=nbytes(obj.radius());
		if(obj.atomType().chi)    size+=nbytes(obj.chi());
		if(obj.atomType().eta)    size+=nbytes(obj.eta());
		if(obj.atomType().c6)     size+=nbytes(obj.c6());
		if(obj.atomType().js)     size+=nbytes(obj.js());
		//vector properties
		if(obj.atomType().posn)   size+=nbytes(obj.posn());
		if(obj.atomType().vel)    size+=nbytes(obj.vel());
		if(obj.atomType().force)  size+=nbytes(obj.force());
		if(obj.atomType().spin)   size+=nbytes(obj.spin());
		//nnp
		if(obj.atomType().symm)   size+=nbytes(obj.symm());
		//return
		return size;
	}
	template <> int nbytes(const Structure& obj){
		if(STRUC_PRINT_FUNC>0) std::cout<<"nbytes(const Structure&)\n";
		int size=0;
		size+=nbytes(static_cast<const Cell&>(obj));
		size+=nbytes(static_cast<const State&>(obj));
		size+=nbytes(static_cast<const AtomData&>(obj));
		return size;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const AtomData& obj, char* arr){
		if(STRUC_PRINT_FUNC>0) std::cout<<"pack(const AtomData&,char*):\n";
		int pos=0;
		//atom type
		pos+=pack(obj.atomType(),arr+pos);
		//natoms
		std::memcpy(arr+pos,&obj.nAtoms(),sizeof(int)); pos+=sizeof(int);
		//basic properties
		if(obj.atomType().name)   pos+=pack(obj.name(),arr+pos);
		if(obj.atomType().an)     pos+=pack(obj.an(),arr+pos);
		if(obj.atomType().type)   pos+=pack(obj.type(),arr+pos);
		if(obj.atomType().index)  pos+=pack(obj.index(),arr+pos);
		//serial properties
		if(obj.atomType().mass)   pos+=pack(obj.mass(),arr+pos);
		if(obj.atomType().charge) pos+=pack(obj.charge(),arr+pos);
		if(obj.atomType().radius) pos+=pack(obj.radius(),arr+pos);
		if(obj.atomType().chi)    pos+=pack(obj.chi(),arr+pos);
		if(obj.atomType().eta)    pos+=pack(obj.eta(),arr+pos);
		if(obj.atomType().c6)     pos+=pack(obj.c6(),arr+pos);
		if(obj.atomType().js)     pos+=pack(obj.js(),arr+pos);
		//vector properties
		if(obj.atomType().posn)   pos+=pack(obj.posn(),arr+pos);
		if(obj.atomType().vel)    pos+=pack(obj.vel(),arr+pos);
		if(obj.atomType().force)  pos+=pack(obj.force(),arr+pos);
		if(obj.atomType().spin)   pos+=pack(obj.spin(),arr+pos);
		//nnp
		if(obj.atomType().symm)   pos+=pack(obj.symm(),arr+pos);
		//return
		return pos;
	}
	template <> int pack(const Structure& obj, char* arr){
		if(STRUC_PRINT_FUNC>0) std::cout<<"pack(const Structure&,char*):\n";
		int pos=0;
		pos+=pack(static_cast<const Cell&>(obj),arr+pos);
		pos+=pack(static_cast<const State&>(obj),arr+pos);
		pos+=pack(static_cast<const AtomData&>(obj),arr+pos);
		return pos;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(AtomData& obj, const char* arr){
		if(STRUC_PRINT_FUNC>0) std::cout<<"unpack(AtomData&,const char*):\n";
		int pos=0;
		//atom type
		AtomType atomT;
		pos+=unpack(atomT,arr+pos);
		//natoms
		int nAtoms=0;
		std::memcpy(&nAtoms,arr+pos,sizeof(int)); pos+=sizeof(int);
		//resize
		obj.resize(nAtoms,atomT);
		//basic properties
		if(obj.atomType().name)  pos+=unpack(obj.name(),arr+pos);
		if(obj.atomType().an)    pos+=unpack(obj.an(),arr+pos);
		if(obj.atomType().type)  pos+=unpack(obj.type(),arr+pos);
		if(obj.atomType().index) pos+=unpack(obj.index(),arr+pos);
		//serial properties
		if(obj.atomType().mass)   pos+=unpack(obj.mass(),arr+pos);
		if(obj.atomType().charge) pos+=unpack(obj.charge(),arr+pos);
		if(obj.atomType().radius) pos+=unpack(obj.radius(),arr+pos);
		if(obj.atomType().chi)    pos+=unpack(obj.chi(),arr+pos);
		if(obj.atomType().eta)    pos+=unpack(obj.eta(),arr+pos);
		if(obj.atomType().c6)     pos+=unpack(obj.c6(),arr+pos);
		if(obj.atomType().js)     pos+=unpack(obj.js(),arr+pos);
		//vector properties
		if(obj.atomType().posn)   pos+=unpack(obj.posn(),arr+pos);
		if(obj.atomType().vel)    pos+=unpack(obj.vel(),arr+pos);
		if(obj.atomType().force)  pos+=unpack(obj.force(),arr+pos);
		if(obj.atomType().spin)   pos+=unpack(obj.spin(),arr+pos);
		//nnp
		if(obj.atomType().symm) pos+=unpack(obj.symm(),arr+pos);
		//return
		return pos;
	}
	template <> int unpack(Structure& obj, const char* arr){
		if(STRUC_PRINT_FUNC>0) std::cout<<"unpack(Structure&,const char*):\n";
		int pos=0;
		pos+=unpack(static_cast<Cell&>(obj),arr+pos);
		pos+=unpack(static_cast<State&>(obj),arr+pos);
		pos+=unpack(static_cast<AtomData&>(obj),arr+pos);
		return pos;
	}
	
}
