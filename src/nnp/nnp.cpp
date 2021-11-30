// c libraries
#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
#include <cmath>
#elif (defined __ICC || defined __INTEL_COMPILER)
#include <mathimf.h> //intel math library
#else
#include <cmath>
#endif
// c++ libraries
#include <iostream>
// ann - structure
#include "src/struc/structure.hpp"
#include "src/struc/cell_list.hpp"
// ann - math
#include "src/math/const.hpp"
// ann - str
#include "src/str/print.hpp"
// ann - nnp
#include "src/nnp/nnp.hpp"

//************************************************************
// NEURAL NETWORK HAMILTONIAN
//************************************************************

//==== operators ====

/**
* print neural network hamiltonian
* @param out - output stream
* @param nnh - neural network hamiltonian
*/
std::ostream& operator<<(std::ostream& out, const NNH& nnh){
	char* str=new char[print::len_buf];
	out<<print::buf(str)<<"\n";
	out<<print::title("NN - HAMILTONIAN",str)<<"\n";
	//hamiltonian
	out<<"ATOM     = "<<nnh.atom_<<"\n";
	//species
	out<<"NSPECIES = "<<nnh.nspecies_<<"\n";
	//potential parameters
	out<<"N_INPUT  = "; std::cout<<nnh.nInput_<<" "; std::cout<<"\n";
	out<<"N_INPUTR = "; std::cout<<nnh.nInputR_<<" "; std::cout<<"\n";
	out<<"N_INPUTA = "; std::cout<<nnh.nInputA_<<" "; std::cout<<"\n";
	out<<nnh.nn_<<"\n";
	out<<print::title("NN - HAMILTONIAN",str)<<"\n";
	out<<print::buf(str);
	delete[] str;
	return out;
}

//==== member functions ====

/**
* set NNH defaults
*/
void NNH::defaults(){
	if(NNP_PRINT_FUNC>0) std::cout<<"NNH::defaults()\n";
	//hamiltonian
		nspecies_=0;
		atom_.clear();
		nn_.clear();
	//basis for pair/triple interactions
		basisR_.clear();
		basisA_.clear();
	//network configuration
		nInput_=0;
		nInputR_=0;
		nInputA_=0;
		offsetR_.clear();
		offsetA_.clear();
}

/**
* resize the number of species
* @param nspecies - the total number of species
*/
void NNH::resize(int nspecies){
	if(NNP_PRINT_FUNC>0) std::cout<<"NNH::resize(int)\n";
	if(nspecies<0) throw std::invalid_argument("NNH::resize(int): invalid number of species.");
	nspecies_=nspecies;
	if(nspecies_>0){
		basisR_.resize(nspecies_);
		basisA_.resize(nspecies_);
		offsetR_.resize(nspecies_);
		offsetA_.resize(nspecies_);
	}
}

/**
* Initialize the number of inputs and offsets associated with the basis functions.
* Must be done after the basis has been defined, otherwise the values will make no sense.
* Different from resizing: resizing sets the number of species, this sets the number of inputs
* associated with the basis associated with each species.
*/
void NNH::init_input(){
	if(NNP_PRINT_FUNC>0) std::cout<<"NNH::init_input()\n";
	//radial inputs
	nInputR_=0;
	for(int i=0; i<nspecies_; ++i){
		nInputR_+=basisR_[i].nfR();
	}
	//radial offsets
	for(int i=0; i<nspecies_; ++i){
		if(i==0) offsetR_[i]=0;
		else offsetR_[i]=offsetR_[i-1]+basisR_[i-1].nfR();
	}
	//angular inputs
	nInputA_=0;
	for(int i=0; i<nspecies_; ++i){
		for(int j=i; j<nspecies_; ++j){
			nInputA_+=basisA_(j,i).nfA();
		}
	}
	//angular offsets
	for(int i=1; i<basisA_.size(); ++i){
		if(i==0) offsetA_[i]=0;
		else offsetA_[i]=offsetA_[i-1]+basisA_[i-1].nfA();
	}
	//total number of inputs
	nInput_=nInputR_+nInputA_;
}

/**
* compute energy of atom with symmetry function "symm"
* @param symm - the symmetry function
*/
double NNH::energy(const Eigen::VectorXd& symm){
	if(NNP_PRINT_FUNC>0) std::cout<<"NNH::energy(const Eigen::VectorXd&)\n";
	return nn_.execute(symm)[0]+atom_.energy();
}

//************************************************************
// NNP - Neural Network Potential
//************************************************************

//==== operators ====

/**
* print the nnp to screen
* @param out - output stream
* @param nnp - the neural network potential
*/
std::ostream& operator<<(std::ostream& out, const NNP& nnp){
	char* str=new char[print::len_buf];
	out<<print::buf(str)<<"\n";
	out<<print::title("NN - POT",str)<<"\n";
	out<<"R_CUT    = "<<nnp.rc_<<"\n";
	out<<"NSPECIES = "<<nnp.nspecies_<<"\n";
	for(int i=0; i<nnp.nspecies_; ++i) std::cout<<nnp.nnh_[i]<<"\n";
	out<<print::title("NN - POT",str)<<"\n";
	out<<print::buf(str);
	delete[] str;
	return out;
}

//==== member functions ====

/**
* set defaults for the neural network potential
*/
void NNP::defaults(){
	if(NNP_PRINT_FUNC>0) std::cout<<"NNP::defaults():\n";
	//species
		nspecies_=0;
		map_.clear();
		nnh_.clear();
	//cutoff
		rc_=0;
}

//==== resizing ====

/**
* resize the number of species and each NNH
* @param species - the species of the neural network potential
*/
void NNP::resize(const std::vector<Atom>& species){
	if(NNP_PRINT_FUNC>0) std::cout<<"NNP::resize(const std::vector<Atom>&)\n";
	if(species.size()<0) throw std::invalid_argument("NNP::resize(const std::vector<Atom>&): invalid number of species");
	nspecies_=species.size();
	nnh_.resize(nspecies_);
	for(int i=0; i<nspecies_; ++i){
		nnh_[i].resize(nspecies_);
		nnh_[i].atom()=species[i];
		map_.add(string::hash(species[i].name()),i);
	}
}

//==== static functions ====

//read/write basis

/**
* Read the basis for a given species from file.
* @param file - the name of the file from which the object will be read
* @param nnp - the neural network potential to be written
* @param atomName - the species for which we will read the basis
*/
void NNP::read_basis(const char* file, NNP& nnp, const char* atomName){
	if(NNP_PRINT_FUNC>0) std::cout<<"NNP::read(const char*,NNP&,const char*):\n";
	FILE* reader=NULL;
	reader=fopen(file,"r");
	if(reader!=NULL){
		NNP::read_basis(reader,nnp,atomName);
		fclose(reader);
		reader=NULL;
	} else throw std::runtime_error(std::string("NNP::read(const char*,NNP&): Could not open nnp file: \"")+std::string(file)+std::string("\""));
}

/**
* Read the basis for a given species from file.
* @param reader - the file pointer from which the object will be read
* @param nnp - the neural network potential to be written
* @param atomName - the species for which we will read the basis
*/
void NNP::read_basis(FILE* reader, NNP& nnp, const char* atomName){
	if(NNP_PRINT_FUNC>0) std::cout<<"NNP::read_basis(FILE*,NNP&,const char*):\n";
	//==== local function variables ====
	std::vector<std::string> strlist;
	char* input=new char[string::M];
	//==== get atom species ====
	const int atomIndex=nnp.index(atomName);
	//==== global cutoff ====
	string::split(fgets(input,string::M,reader),string::WS,strlist);
	const double rc=std::atof(strlist.at(1).c_str());
	if(rc!=nnp.rc()) throw std::invalid_argument("NNP::read_basis(FILE*,NNP&,const char*): invalid cutoff.");
	//==== number of species ====
	string::split(fgets(input,string::M,reader),string::WS,strlist);
	const int nspecies=std::atoi(strlist.at(1).c_str());
	if(nspecies!=nnp.nspecies()) throw std::invalid_argument("NNP::read_basis(FILE*,NNP&,const char*): invalid number of species.");
	//==== central species ====
	string::split(fgets(input,string::M,reader),string::WS,strlist);
	const int II=nnp.index(strlist.at(1));
	//==== check indices ====
	if(atomIndex!=II) throw std::invalid_argument("NNP::read_basis(FILE*,NNP&,const char*): invalid central species.\n");
	//==== basis - radial ====
	for(int j=0; j<nspecies; ++j){
		//read species
		string::split(fgets(input,string::M,reader),string::WS,strlist);
		const int JJ=nnp.index(strlist.at(1));
		//read basis
		BasisR::read(reader,nnp.nnh(II).basisR(JJ));
	}
	//==== basis - angular ====
	for(int j=0; j<nspecies; ++j){
		for(int k=j; k<nspecies; ++k){
			//read species
			string::split(fgets(input,string::M,reader),string::WS,strlist);
			const int JJ=nnp.index(strlist.at(1));
			const int KK=nnp.index(strlist.at(2));
			//read basis
			BasisA::read(reader,nnp.nnh(II).basisA(JJ,KK));
		}
	}
	//==== initialize the inputs ====
	nnp.nnh(II).init_input();
	//==== clear local variables ====
	delete[] input;
}

//read/write nnp

/**
* Write the neural network to file
* @param file - the name of the file to which the object will be written
* @param nnp - the neural network potential to be written
*/
void NNP::write(const char* file, const NNP& nnp){
	if(NNP_PRINT_FUNC>0) std::cout<<"NNP::write(const char*,const NNP&):\n";
	FILE* writer=NULL;
	writer=fopen(file,"w");
	if(writer!=NULL){
		NNP::write(writer,nnp);
		fclose(writer);
		writer=NULL;
	} else throw std::runtime_error(std::string("NNP::write(const char*,const NNP&): Could not write to nnh file: \"")+std::string(file)+std::string("\""));
}

/**
* Read the neural network from file
* @param file - the name of the file fro
* @param nnp - stores the neural network potential to be read
*/
void NNP::read(const char* file, NNP& nnp){
	if(NNP_PRINT_FUNC>0) std::cout<<"NNP::read(const char*,NNP&):\n";
	FILE* reader=NULL;
	reader=fopen(file,"r");
	if(reader!=NULL){
		NNP::read(reader,nnp);
		fclose(reader);
		reader=NULL;
	} else throw std::runtime_error(std::string("NNP::read(const char*,NNP&): Could not open nnp file: \"")+std::string(file)+std::string("\""));
}

/**
* Write the neural network to file
* @param writer - the file pointer used to write the object to file
* @param nnp - the neural network potential to be written
*/
void NNP::write(FILE* writer, const NNP& nnp){
	if(NNP_PRINT_FUNC>0) std::cout<<"NNP::write(FILE*,const NNP&):\n";
	//==== header ====
	fprintf(writer,"ann\n");
	//==== species ====
	fprintf(writer, "nspecies %i\n",nnp.nspecies());
	for(int n=0; n<nnp.nspecies(); ++n){
		const Atom& atom=nnp.nnh(n).atom();
		fprintf(writer,"%s %f %f %f\n",atom.name().c_str(),atom.mass(),atom.energy(),atom.charge());
	}
	//==== cutoff ====
	fprintf(writer,"rc %f\n",nnp.rc());
	//==== basis ====
	for(int i=0; i<nnp.nspecies(); ++i){
		//write central species
		fprintf(writer,"basis %s\n",nnp.nnh(i).atom().name().c_str());
		//write basis - radial
		for(int j=0; j<nnp.nspecies(); ++j){
			//write species
			fprintf(writer,"basis_radial %s\n",nnp.nnh(j).atom().name().c_str());
			//write basis
			BasisR::write(writer,nnp.nnh(i).basisR(j));
		}
		//write basis - angular
		for(int j=0; j<nnp.nspecies(); ++j){
			for(int k=j; k<nnp.nspecies(); ++k){
				//write species
				fprintf(writer,"basis_angular %s %s\n",nnp.nnh(j).atom().name().c_str(),nnp.nnh(k).atom().name().c_str());
				//write basis
				BasisA::write(writer,nnp.nnh(i).basisA(j,k));
			}
		}
	}
	//==== neural network ====
	for(int n=0; n<nnp.nspecies(); ++n){
		//write central species
		fprintf(writer,"nn %s\n",nnp.nnh(n).atom().name().c_str());
		//write the network
		NN::ANN::write(writer,nnp.nnh(n).nn());
	}
}

/**
* Read the neural network from file
* @param reader - the file pointer used to read the object from file
* @param nnp - stores the neural network potential to be read
*/
void NNP::read(FILE* reader, NNP& nnp){
	if(NNP_PRINT_FUNC>0) std::cout<<"NNP::read(FILE*,NNP&):\n";
	//==== local function variables ====
	std::vector<std::string> strlist;
	char* input=new char[string::M];
	//==== header ====
	fgets(input,string::M,reader);
	//==== number of species ====
	string::split(fgets(input,string::M,reader),string::WS,strlist);
	const int nspecies=std::atoi(strlist.at(1).c_str());
	if(nspecies<=0) throw std::invalid_argument("NNP::read(FILE*,NNP&): invalid number of species.");
	//==== species ====
	std::vector<Atom> species(nspecies);
	for(int n=0; n<nspecies; ++n){
		Atom::read(fgets(input,string::M,reader),species[n]);
	}
	//==== resize ====
	nnp.resize(species);
	//==== global cutoff ====
	string::split(fgets(input,string::M,reader),string::WS,strlist);
	const double rc=std::atof(strlist.at(1).c_str());
	if(rc<=0) throw std::invalid_argument("NNP::read(FILE*,NNP&): invalid cutoff.");
	else nnp.rc()=rc;
	//==== basis ====
	for(int i=0; i<nspecies; ++i){
		//read central species
		string::split(fgets(input,string::M,reader),string::WS,strlist);
		const int II=nnp.index(strlist.at(1));
		//read basis - radial
		for(int j=0; j<nspecies; ++j){
			//read species
			string::split(fgets(input,string::M,reader),string::WS,strlist);
			const int JJ=nnp.index(strlist.at(1));
			//read basis
			BasisR::read(reader,nnp.nnh(II).basisR(JJ));
		}
		//read basis - angular
		for(int j=0; j<nspecies; ++j){
			for(int k=j; k<nspecies; ++k){
				//read species
				string::split(fgets(input,string::M,reader),string::WS,strlist);
				const int JJ=nnp.index(strlist.at(1));
				const int KK=nnp.index(strlist.at(2));
				//read basis
				BasisA::read(reader,nnp.nnh(II).basisA(JJ,KK));
			}
		}
	}
	//==== initialize inputs ====
	for(int i=0; i<nspecies; ++i){
		nnp.nnh(i).init_input();
	}
	//==== neural network ====
	for(int n=0; n<nspecies; ++n){
		//read species
		string::split(fgets(input,string::M,reader),string::WS,strlist);
		const int II=nnp.index(strlist.at(1));
		//read network
		NN::ANN::read(reader,nnp.nnh(II).nn());
		//resize gradient object
		nnp.nnh(II).dOutDVal().resize(nnp.nnh(II).nn());
	}
	//==== clear local variables ====
	delete[] input;
}

//calculation

/**
* resize the symmetry function vectors to store the inputs
* @param nnp - the neural network potential
* @param struc - the structure which we will compute
*/
void NNP::init(const NNP& nnp, Structure& struc){
	if(NNP_PRINT_FUNC>0) std::cout<<"NNP::init(const NNP&,Structure&):\n";
	for(int n=0; n<struc.nAtoms(); ++n){
		struc.symm(n).resize(nnp.nnh(nnp.index(struc.name(n))).nInput());
	}
}

/**
* Compute the symmetry functions for a given structure.
* @param nnp - the neural network potential
* @param struc - the structure which we will compute
* @param nlist - neighbor list for each atom (includes periodic images)
* The symmetry functions are computed by looping over all nearest-neighbor pairs
* and all unique nearest-neighbor triples.  Thus, the neighbor list must be set 
* for the structure, and the type must correspond to the index of the atomic species
* in the NNP.
*/
void NNP::symm(NNP& nnp, Structure& struc, const NeighborList& nlist){
	if(NNP_PRINT_FUNC>0) std::cout<<"NNP::symm(const NNP&,Structure&,const NeighborList&):\n";
	for(int i=0; i<struc.nAtoms(); ++i){
		//reset the inputs
		struc.symm(i).setZero();
		//get the index of species i
		const int II=struc.type(i);
		//loop over all neighbors
		for(int j=0; j<nlist.size(i); ++j){
			//get the index of species j
			const int JJ=nlist.neigh(i,j).type();
			//get the distance from J to I
			const Eigen::Vector3d& rIJ=nlist.neigh(i,j).r();
			const double dIJ=nlist.neigh(i,j).dr();
			{
			//set the basis
			const int offsetR_=nnp.nnh(II).offsetR(JJ);
			BasisR& basisRij_=nnp.nnh(II).basisR(JJ);
			//compute the IJ contribution to all radial basis functions
			basisRij_.symm(dIJ);
			for(int nr=0; nr<basisRij_.nfR(); ++nr){
				struc.symm(i)[offsetR_+nr]+=basisRij_.symm()[nr];
			}
			}
			//loop over all unique triplets
			for(int k=j+1; k<nlist.size(i); ++k){
				//find the index of the species of atom k
				const int KK=nlist.neigh(i,k).type();
				//get the distance from K to I
				const Eigen::Vector3d& rIK=nlist.neigh(i,k).r();
				const double dIK=nlist.neigh(i,k).dr();
				//get the distance from J to K
				const double dJK=(nlist.neigh(i,k).r()-nlist.neigh(i,j).r()).norm();
				//compute the cosIJK angle and store the distances
				const double cosIJK=rIJ.dot(rIK)/(dIJ*dIK);
				const double d[3]={dIJ,dIK,dJK};
				//set the basis
				const int offsetA_=nnp.nnh(II).nInputR()+nnp.nnh(II).offsetA(JJ,KK);
				BasisA& basisAijk_=nnp.nnh(II).basisA(JJ,KK);
				//compute the IJ,IK,JK contribution to all angular basis functions
				basisAijk_.symm(cosIJK,d);
				for(int na=0; na<basisAijk_.nfA(); ++na){
					struc.symm(i)[offsetA_+na]+=basisAijk_.symm()[na];
				}
			}
		}
	}
}

/**
* execute all atomic networks and return energy
* @param nnp - the neural network potential
* @param struc - the structure for which we will compute the energy
* @return total energy
* it is assumed that the symmetry functions have been computed
*/
double NNP::energy(NNP& nnp, Structure& struc){
	if(NNP_PRINT_FUNC>0) std::cout<<"NNP::energy(const NNP&,const Structure&):\n";
	double energy=0;
	//loop over atoms
	for(int n=0; n<struc.nAtoms(); ++n){
		//set the index
		const int ii=nnp.index(struc.name(n));
		//compute the energy
		//std::cout<<"energy "<<struc.name(n)<<" "<<nnp.nnh(ii).nn().execute(struc.symm(n))[0]<<" "<<nnp.nnh(ii).atom().energy()<<"\n";
		energy+=nnp.nnh(ii).energy(struc.symm(n));
	}
	struc.energy()=energy;
	return struc.energy();
}

/**
* Compute the forces on the atoms for a given structure
* @param nnp - the neural network potential
* @param struc - the structure which we will compute
* @param nlist - neighbor list for each atom (includes periodic images)
* The forces are computed by looping over all nearest-neighbor pairs
* and all unique nearest-neighbor triples.  Thus, the neighbor list must be set 
* for the structure, and the type must correspond to the index of the atomic species
* in the NNP.  In addition, the index for each neighbor must be set to -1 if it is
* a periodic image or within [0,natoms] in which case the Newton's third law force
* pair must be added to the neighbor atom.
*/
void NNP::force(NNP& nnp, Structure& struc, const NeighborList& nlist){
	if(NNP_PRINT_FUNC>0) std::cout<<"NNP::force(Structure&,const NeighborList&):\n";
	//reset the force
	for(int i=0; i<struc.nAtoms(); ++i) struc.force(i).setZero();
	//compute the forces
	for(int i=0; i<struc.nAtoms(); ++i){
		//get the index of species i
		const int II=struc.type(i);
		//execute the appropriate network
		nnp.nnh(II).nn().execute(struc.symm(i));
		//calculate the network gradient
		nnp.nnh(II).dOutDVal().grad(nnp.nnh(II).nn());
		//set the gradient (n.b. dodi - do/di - deriv. of out w.r.t. in)
		const Eigen::VectorXd& dEdG=nnp.nnh(II).dOutDVal().dodi().row(0);
		//loop over all neighbors
		for(int j=0; j<nlist.size(i); ++j){
			//get the indices of the jth neighbor
			const int JJ=nlist.neigh(i,j).type();
			const int jj=nlist.neigh(i,j).index();
			//get the distance from J to I
			const Eigen::Vector3d& rIJ=nlist.neigh(i,j).r();
			const double dIJ=nlist.neigh(i,j).dr();
			const double dIJi=1.0/dIJ;
			//compute the IJ contribution to the radial force
			{
			const int offsetR_=nnp.nnh(II).offsetR(JJ);
			const double amp=nnp.nnh(II).basisR(JJ).force(dIJ,dEdG.data()+offsetR_)*dIJi;
			struc.force(i).noalias()+=amp*rIJ;
			if(jj>=0) struc.force(jj).noalias()-=amp*rIJ;
			}
			//loop over all unique triplets
			for(int k=j+1; k<nlist.size(i); ++k){
				//find the index of the species of atom k
				const int KK=nlist.neigh(i,k).type();
				const int kk=nlist.neigh(i,k).index();
				//get the distance from K to I
				const Eigen::Vector3d& rIK=nlist.neigh(i,k).r();
				const double dIK=nlist.neigh(i,k).dr();
				const double dIKi=1.0/dIK;
				//get the distance from J to K
				const Eigen::Vector3d rJK=(nlist.neigh(i,k).r()-nlist.neigh(i,j).r());
				const double dJK=rJK.norm();
				const double dJKi=1.0/dJK;
				//set the basis
				const int offsetA_=nnp.nnh(II).nInputR()+nnp.nnh(II).offsetA(JJ,KK);
				BasisA& basisAijk_=nnp.nnh(II).basisA(JJ,KK);
				//compute the IJ,IK,JK contribution to the angular force
				double phi=0; double eta[3]={0,0,0};
				const double cosIJK=rIJ.dot(rIK)*dIJi*dIKi;
				const double d[3]={dIJ,dIK,dJK};
				nnp.nnh(II).basisA(JJ,KK).force(phi,eta,cosIJK,d,dEdG.data()+offsetA_);
				struc.force(i).noalias()+=(phi*(dIKi-cosIJK*dIJi)+eta[0])*rIJ*dIJi;
				struc.force(i).noalias()+=(phi*(dIJi-cosIJK*dIKi)+eta[1])*rIK*dIKi;
				if(jj>=0){
					struc.force(jj).noalias()-=(-phi*cosIJK*dIJi+eta[0])*rIJ*dIJi+phi*dIJi*rIK*dIKi;
					struc.force(jj).noalias()-=eta[2]*rJK*dJKi;
				}
				if(kk>=0){
					struc.force(kk).noalias()-=(-phi*cosIJK*dIKi+eta[1])*rIK*dIKi+phi*dIKi*rIJ*dIJi;
					struc.force(kk).noalias()+=eta[2]*rJK*dJKi;
				}
			}
		}
	}
}

/**
* Compute the charges of the atoms for a given structure
* @param nnp - the neural network potential (trained for charges)
* @param struc - the structure which we will compute
* @param nlist - neighbor list for each atom (includes periodic images)
* it is assumed that the symmetry functions have been computed
*/
void NNP::charge(NNP& nnp, Structure& struc){
	if(NNP_PRINT_FUNC>0) std::cout<<"NNP::energy(const NNP&,const Structure&):\n";
	double energy=0;
	//loop over atoms
	for(int n=0; n<struc.nAtoms(); ++n){
		//set the index
		const int ii=nnp.index(struc.name(n));
		//compute the charge
		struc.charge(n)=nnp.nnh(ii).nn().execute(struc.symm(n))[0];
	}
}

//************************************************************
// serialization
//************************************************************

namespace serialize{
	
//**********************************************
// byte measures
//**********************************************

template <> int nbytes(const NNH& obj){
	if(NNP_PRINT_FUNC>0) std::cout<<"nbytes(const NNH&):\n";
	int size=0;
	//hamiltonian
	size+=nbytes(obj.atom());
	size+=nbytes(obj.nn());
	//species
	size+=nbytes(obj.nspecies());//nspecies_
	//basis for pair/triple interactions
	for(int j=0; j<obj.nspecies(); ++j){
		size+=nbytes(obj.basisR(j));
	}
	for(int j=0; j<obj.nspecies(); ++j){
		for(int k=j; k<obj.nspecies(); ++k){
			size+=nbytes(obj.basisA(j,k));
		}
	}
	//return the size
	return size;
}
template <> int nbytes(const NNP& obj){
	if(NNP_PRINT_FUNC>0) std::cout<<"nbytes(const NNP&):\n";
	int size=0;
	//species
	size+=nbytes(obj.nspecies());
	size+=nbytes(obj.map());
	for(int i=0; i<obj.nspecies(); ++i){
		size+=nbytes(obj.nnh(i));
	}
	//cutoff
	size+=nbytes(obj.rc());
	//return the size
	return size;
}

//**********************************************
// packing
//**********************************************

template <> int pack(const NNH& obj, char* arr){
	if(NNP_PRINT_FUNC>0) std::cout<<"pack(const NNH&,char*):\n";
	int pos=0;
	//hamiltonian
	pos+=pack(obj.atom(),arr+pos);
	pos+=pack(obj.nn(),arr+pos);
	//species
	pos+=pack(obj.nspecies(),arr+pos);
	//basis for pair/triple interactions
	for(int j=0; j<obj.nspecies(); ++j){
		pos+=pack(obj.basisR(j),arr+pos);
	}
	for(int j=0; j<obj.nspecies(); ++j){
		for(int k=j; k<obj.nspecies(); ++k){
			pos+=pack(obj.basisA(j,k),arr+pos);
		}
	}
	//return bytes written
	return pos;
}
template <> int pack(const NNP& obj, char* arr){
	if(NNP_PRINT_FUNC>0) std::cout<<"pack(const NNP&,char*):\n";
	int pos=0;
	//species
	pos+=pack(obj.nspecies(),arr+pos);
	pos+=pack(obj.map(),arr+pos);
	for(int i=0; i<obj.nspecies(); ++i){
		pos+=pack(obj.nnh(i),arr+pos);
	}
	//cutoff
	pos+=pack(obj.rc(),arr+pos);
	//return bytes written
	return pos;
}

//**********************************************
// unpacking
//**********************************************

template <> int unpack(NNH& obj, const char* arr){
	if(NNP_PRINT_FUNC>0) std::cout<<"unpack(NNH&,const char*):\n";
	int pos=0;
	//hamiltonian
	pos+=unpack(obj.atom(),arr+pos);
	pos+=unpack(obj.nn(),arr+pos);
	obj.dOutDVal().resize(obj.nn());
	//species
	int nspecies=0;
	pos+=unpack(nspecies,arr+pos);
	obj.resize(nspecies);
	//basis for pair/triple interactions
	for(int j=0; j<obj.nspecies(); ++j){
		pos+=unpack(obj.basisR(j),arr+pos);
	}
	for(int j=0; j<obj.nspecies(); ++j){
		for(int k=j; k<obj.nspecies(); ++k){
			pos+=unpack(obj.basisA(j,k),arr+pos);
		}
	}
	//intialize the inputs and offsets
	obj.init_input();
	//return bytes read
	return pos;
}
template <> int unpack(NNP& obj, const char* arr){
	if(NNP_PRINT_FUNC>0) std::cout<<"unpack(NNP&,const char*):\n";
	int pos=0;
	//species
	int nspecies=0;
	Map<int,int> map;
	pos+=unpack(nspecies,arr+pos);
	pos+=unpack(map,arr+pos);
	std::vector<NNH> nnh(nspecies);
	std::vector<Atom> species(nspecies);
	for(int i=0; i<nspecies; ++i){
		pos+=unpack(nnh[i],arr+pos);
		species[i]=nnh[i].atom();
	}
	obj.resize(species);
	for(int i=0; i<obj.nspecies(); ++i){
		obj.nnh(i)=nnh[i];
	}
	obj.map()=map;
	//cutoff
	pos+=unpack(obj.rc(),arr+pos);
	//return bytes read
	return pos;
}

}
