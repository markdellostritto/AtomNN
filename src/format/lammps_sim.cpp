// c libraries
#include <ctime>
// c++ libraries
#include <iostream>
// ann_v0 - text
#include "src/str/string.hpp"
// ann_v0 - chem
#include "src/chem/ptable.hpp"
#include "src/chem/units.hpp"
// ann_v0 - structure
#include "src/format/lammps_sim.hpp"

namespace LAMMPS{

//*****************************************************
//STYLE_ATOM struct
//*****************************************************

STYLE_ATOM::type STYLE_ATOM::read(const char* str){
	if(std::strcmp(str,"FULL")==0) return STYLE_ATOM::FULL;
	else if(std::strcmp(str,"BOND")==0) return STYLE_ATOM::BOND;
	else if(std::strcmp(str,"ATOMIC")==0) return STYLE_ATOM::ATOMIC;
	else if(std::strcmp(str,"CHARGE")==0) return STYLE_ATOM::CHARGE;
	else return STYLE_ATOM::UNKNOWN;
}

std::ostream& operator<<(std::ostream& out, STYLE_ATOM::type& t){
	if(t==STYLE_ATOM::FULL) out<<"FULL";
	else if(t==STYLE_ATOM::BOND) out<<"BOND";
	else if(t==STYLE_ATOM::ATOMIC) out<<"ATOMIC";
	else if(t==STYLE_ATOM::CHARGE) out<<"CHARGE";
	else out<<"UNKNOWN";
	return out;
}

//*****************************************************
//FORMAT_ATOM struct
//*****************************************************

std::ostream& operator<<(std::ostream& out, FORMAT_ATOM& f){
	out<<"index = "<<f.index<<"\n";
	out<<"mol   = "<<f.mol<<"\n";
	out<<"type  = "<<f.type<<"\n";
	out<<"x     = "<<f.x<<"\n";
	out<<"y     = "<<f.y<<"\n";
	out<<"z     = "<<f.z<<"\n";
	out<<"q     = "<<f.q<<"\n";
	out<<"fx    = "<<f.fx<<"\n";
	out<<"fy    = "<<f.fy<<"\n";
	out<<"fz    = "<<f.fz;
	return out;
}

//*****************************************************
//DUMP files
//*****************************************************

namespace DUMP{

void read(const char* file, const Interval& interval, const AtomType& atomT, Simulation& sim){
	static const char* funcName="read(const char*,const Interval&,AtomType&,Simulation&,Format&)";
	if(LAMMPS_PRINT_FUNC>0) std::cout<<NAMESPACE_GLOBAL<<"::"<<funcName<<":\n";
	//======== local function variables ========
	//==== file i/o ====
		FILE* reader=NULL;
		char* input=new char[string::M];
		char* temp=new char[string::M];
	//==== time info ====
		int ts=0;//number of timesteps
	//==== cell info ====
		Eigen::Matrix3d lv=Eigen::Matrix3d::Zero();
	//==== atom info ====
		int natoms=0;
		DATA_ATOM dataAtom;
		FORMAT_ATOM formatAtom;
	//==== timing ====
		clock_t start,stop;
		double time;
	//==== misc ====
		bool error=false;
	//==== units ====
		double s_len=1.0,s_energy=1.0;
		
	try{
		//==== start the timer ====
		start=std::clock();
		
		//==== open the file ====
		if(LAMMPS_PRINT_STATUS>0) std::cout<<"opening file\n";
		reader=fopen(file,"r");
		if(reader==NULL) throw std::runtime_error("I/O Error: Could not open file.");
		
		//==== find the number of timesteps ====
		if(LAMMPS_PRINT_STATUS>0) std::cout<<"reading number of timesteps\n";
		while(fgets(input,string::M,reader)!=NULL){
			if(std::strstr(input,"TIMESTEP")!=NULL) ++ts;
		}
		std::rewind(reader);
		
		//==== read in the atom format ====
		if(LAMMPS_PRINT_STATUS>0) std::cout<<"reading atom format\n";
		while(fgets(input,string::M,reader)!=NULL){
			if(std::strstr(input,"ITEM: ATOMS")!=NULL){
				int nTokens=string::substrN(input,string::WS)-2;
				std::strtok(input,string::WS);
				std::strtok(NULL,string::WS);
				for(int i=0; i<nTokens; ++i){
					std::strcpy(temp,std::strtok(NULL,string::WS));
					if(std::strcmp(temp,"id")==0) formatAtom.index=i;
					else if(std::strcmp(temp,"mol")==0) formatAtom.mol=i;
					else if(std::strcmp(temp,"type")==0) formatAtom.type=i;
					else if(std::strcmp(temp,"q")==0) formatAtom.q=i;
					else if(std::strcmp(temp,"mass")==0) formatAtom.m=i;
					else if(std::strcmp(temp,"x")==0) formatAtom.x=i;
					else if(std::strcmp(temp,"y")==0) formatAtom.y=i;
					else if(std::strcmp(temp,"z")==0) formatAtom.z=i;
					else if(std::strcmp(temp,"vx")==0) formatAtom.vx=i;
					else if(std::strcmp(temp,"vy")==0) formatAtom.vy=i;
					else if(std::strcmp(temp,"vz")==0) formatAtom.vz=i;
					else if(std::strcmp(temp,"fx")==0) formatAtom.fx=i;
					else if(std::strcmp(temp,"fy")==0) formatAtom.fy=i;
					else if(std::strcmp(temp,"fz")==0) formatAtom.fz=i;
				}
				break;
			}
		}
		if(LAMMPS_PRINT_DATA>0) std::cout<<"formatAtom = \n"<<formatAtom<<"\n";
		std::rewind(reader);
		
		//==== read in the number of atoms ====
		if(LAMMPS_PRINT_STATUS>0) std::cout<<"reading number of atoms\n";
		while(fgets(input,string::M,reader)!=NULL){
			if(std::strstr(input,"NUMBER OF ATOMS")!=NULL){
				natoms=std::atoi(fgets(input,string::M,reader));
				break;
			}
		}
		std::rewind(reader);
		
		//==== set the timesteps ====
		int beg=interval.beg-1;
		int end=interval.end-1;
		if(interval.end<0) end=ts+interval.end;
		if(beg>=ts) throw std::runtime_error("Invalid beginning timestep");
		if(end>=ts) throw std::runtime_error("Invalid ending timestep");
		const int tsint=end-beg+1;
		if(LAMMPS_PRINT_DATA>1){
			std::cout<<"interval  = "<<interval<<"\n";
			std::cout<<"ts        = "<<ts<<"\n";
			std::cout<<"(beg,end) = ("<<beg<<","<<end<<")\n";
		}
		
		//==== resize the simulation ====
		if(LAMMPS_PRINT_STATUS>0) std::cout<<"allocating memory\n";
		sim.resize(tsint/interval.stride,natoms,atomT);
		
		//==== read atoms ====
		if(LAMMPS_PRINT_STATUS>0) std::cout<<"reading atoms\n";
		int timestep=0;
		while(fgets(input,string::M,reader)!=NULL){
			if(std::strstr(input,"BOX")!=NULL){
				//check timestep
				if(timestep<beg){continue;}
				if(timestep%interval.stride!=0){continue;}
				if(LAMMPS_PRINT_DATA>1) std::cout<<"Cell: "<<timestep<<"\n";
				//local variables
				std::vector<std::string> strlist;
				lv.setZero();
				//x
				string::split(fgets(input,string::M,reader),string::WS,strlist);
				const double xlob=std::atof(strlist[0].c_str());//xlo
				const double xhib=std::atof(strlist[1].c_str());//xhi
				double xy=0;
				if(strlist.size()==3) xy=std::atof(strlist[2].c_str());//xy
				//y
				string::split(fgets(input,string::M,reader),string::WS,strlist);
				const double ylob=std::atof(strlist[0].c_str());//ylo
				const double yhib=std::atof(strlist[1].c_str());//yhi
				double xz=0;
				if(strlist.size()==3) xz=std::atof(strlist[2].c_str());//xz
				//z
				string::split(fgets(input,string::M,reader),string::WS,strlist);
				const double zlob=std::atof(strlist[0].c_str());//zlo
				const double zhib=std::atof(strlist[1].c_str());//zhi
				double yz=0;
				if(strlist.size()==3) yz=std::atof(strlist[2].c_str());//yz
				//set xlo,xhi,ylo,yhi,zlo,zhi
				const double xlo=xlob-std::min(0.0,std::min(xy,std::min(xz,xy+xz)));
				const double xhi=xhib-std::max(0.0,std::max(xy,std::max(xz,xy+xz)));
				const double ylo=ylob-std::min(0.0,yz);
				const double yhi=yhib-std::max(0.0,yz);
				const double zlo=zlob;
				const double zhi=zhib;
				//set lv
				lv(0,0)=xhi-xlo;
				lv(1,1)=yhi-ylo;
				lv(2,2)=zhi-zlo;
				lv(0,1)=xy;
				lv(0,2)=xz;
				lv(1,2)=yz;
				//set cell
				static_cast<Cell&>(sim.frame(timestep/interval.stride-beg)).init(lv*s_len);
			}
			if(std::strstr(input,"ITEM: ATOMS")!=NULL){
				if(LAMMPS_PRINT_DATA>1) std::cout<<"Atoms: "<<timestep<<"\n";
				if(timestep<beg){++timestep; continue;}
				if(timestep%interval.stride!=0){++timestep; continue;}
				const int lts=timestep/interval.stride-beg;//local time step
				std::vector<std::string> tokens;
				for(int i=0; i<natoms; ++i){
					//read in the next line
					fgets(input,string::M,reader);
					//split the line into tokens
					string::split(input,string::WS,tokens);
					//read in the data
					if(formatAtom.q>=0) dataAtom.q=std::atof(tokens[formatAtom.q].c_str());
					if(formatAtom.m>=0) dataAtom.q=std::atof(tokens[formatAtom.m].c_str());
					if(formatAtom.x>=0) dataAtom.posn[0]=std::atof(tokens[formatAtom.x].c_str());
					if(formatAtom.y>=0) dataAtom.posn[1]=std::atof(tokens[formatAtom.y].c_str());
					if(formatAtom.z>=0) dataAtom.posn[2]=std::atof(tokens[formatAtom.z].c_str());
					if(formatAtom.vx>=0) dataAtom.vel[0]=std::atof(tokens[formatAtom.vx].c_str());
					if(formatAtom.vy>=0) dataAtom.vel[1]=std::atof(tokens[formatAtom.vy].c_str());
					if(formatAtom.vz>=0) dataAtom.vel[2]=std::atof(tokens[formatAtom.vz].c_str());
					if(formatAtom.fx>=0) dataAtom.force[0]=std::atof(tokens[formatAtom.fx].c_str());
					if(formatAtom.fy>=0) dataAtom.force[1]=std::atof(tokens[formatAtom.fy].c_str());
					if(formatAtom.fz>=0) dataAtom.force[2]=std::atof(tokens[formatAtom.fz].c_str());
					if(formatAtom.index>=0) dataAtom.index=std::atoi(tokens[formatAtom.index].c_str())-1;
					if(formatAtom.type>=0) dataAtom.type=std::atoi(tokens[formatAtom.type].c_str());
					//set the simulation data
					if(atomT.type) sim.frame(lts).type(dataAtom.index)=dataAtom.type;
					if(atomT.index) sim.frame(lts).index(dataAtom.index)=dataAtom.index;
					if(atomT.posn) sim.frame(lts).posn(dataAtom.index)=dataAtom.posn*s_len;
					if(atomT.charge) sim.frame(lts).charge(dataAtom.index)=dataAtom.q;
					if(atomT.mass) sim.frame(lts).mass(dataAtom.index)=dataAtom.m;
					if(atomT.vel) sim.frame(lts).vel(dataAtom.index)=dataAtom.vel*s_len;
					if(atomT.force) sim.frame(lts).force(dataAtom.index)=dataAtom.force*s_energy/s_len;
					if(atomT.name) sim.frame(lts).name(dataAtom.index)=std::string("X")+std::to_string(dataAtom.type);
				}
				++timestep;
			}
			if(timestep>end) break;
		}
		
		//==== check if the cell is fixed ====
		sim.cell_fixed()=true;
		for(int t=1; t<sim.timesteps(); ++t){
			if(static_cast<Cell&>(sim.frame(t))!=static_cast<Cell&>(sim.frame(0))){sim.cell_fixed()=false;break;}
		}
		
		//==== move the atoms within the cell ====
		for(int t=0; t<sim.timesteps(); ++t){
			for(int n=0; n<sim.frame(t).nAtoms(); ++n){
				sim.frame(t).modv(sim.frame(t).posn(n),sim.frame(t).posn(n));
			}
		}
		
		//==== stop the timer ====
		stop=std::clock();
		
		//==== print the time ====
		time=((double)(stop-start))/CLOCKS_PER_SEC;
		std::cout<<"positions read in "<<time<<" seconds\n";
	}catch(std::exception& e){
		std::cout<<"ERROR in "<<NAMESPACE_GLOBAL<<"::"<<funcName<<":\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	//free local variables
	if(reader!=NULL) fclose(reader);
	delete[] input;
	delete[] temp;
	
	if(error) throw std::runtime_error("I/O Exception Occurred.");
}

void write(const char* file, const Interval& interval, const AtomType& atomT, Simulation& sim){
	static const char* funcName="write<AtomT>(const char*,const Interval&,const AtomType&,Simulation&)";
	if(LAMMPS_PRINT_FUNC>0) std::cout<<NAMESPACE_GLOBAL<<"::"<<funcName<<":\n";
	//local variables
	FILE* writer=NULL;
	bool error=false;
	
	try{
		//open the file
		writer=fopen(file,"w");
		if(writer==NULL) throw std::runtime_error("Unable to open file.");
		
		//set the beginning and ending timesteps
		int lbeg=interval.beg-1;
		int lend=interval.end-1;
		if(lend<0) lend=sim.timesteps()+interval.end;
		if(lbeg<0 || lend>sim.timesteps() || lbeg>lend) throw std::invalid_argument("Invalid beginning and ending timesteps.");
		if(LAMMPS_PRINT_DATA>0) std::cout<<"Interval = ("<<lbeg<<","<<lend<<")\n";
		
		for(int t=lbeg; t<=lend; ++t){
			fprintf(writer,"ITEM: TIMESTEP\n");
			fprintf(writer,"%i\n",t);
			fprintf(writer,"ITEM: NUMBER OF ATOMS\n");
			fprintf(writer,"%i\n",sim.frame(t).nAtoms());
			fprintf(writer,"ITEM: BOX BOUNDS pp pp pp\n");
			fprintf(writer,"%f %f\n",0.0,sim.frame(t).R()(0,0));
			fprintf(writer,"%f %f\n",0.0,sim.frame(t).R()(1,1));
			fprintf(writer,"%f %f\n",0.0,sim.frame(t).R()(2,2));
			fprintf(writer,"ITEM: ATOMS id type x y z\n");
			for(int n=0; n<sim.frame(t).nAtoms(); ++n){
				fprintf(writer,"%i %i %f %f %f\n",n+1,sim.frame(t).type(n),
					sim.frame(t).posn(n)[0],sim.frame(t).posn(n)[1],sim.frame(t).posn(n)[2]
				);
			}
		}
		
		fclose(writer);
		writer=NULL;
	}catch(std::exception& e){
		std::cout<<"ERROR in "<<NAMESPACE_GLOBAL<<"::"<<funcName<<":\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	if(error) throw std::runtime_error("I/O Exception Occurred.");
}
	
}

}
