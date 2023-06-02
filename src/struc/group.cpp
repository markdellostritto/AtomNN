//c++
#include <iostream>
#include <algorithm>
// structure
#include "src/struc/structure.hpp"
#include "src/struc/group.hpp"
// text
#include "src/str/string.hpp"

//***********************************************************************
// Group
//***********************************************************************

//==== Style ====

std::ostream& operator<<(std::ostream& out, const Group::Style::Type& type){
	switch(type){
		case Group::Style::ID: out<<"ID"; break;
		case Group::Style::TYPE: out<<"TYPE"; break;
		case Group::Style::NAME: out<<"NAME"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

const char* Group::Style::name(const Group::Style::Type& type){
	switch(type){
		case Group::Style::ID: return "ID";
		case Group::Style::TYPE: return "TYPE";
		case Group::Style::NAME: return "NAME";
		default: return "UNKNOWN";
	}
}

Group::Style::Type Group::Style::read(const char* str){
	if(std::strcmp(str,"ID")==0) return Group::Style::ID;
	else if(std::strcmp(str,"TYPE")==0) return Group::Style::TYPE;
	else if(std::strcmp(str,"NAME")==0) return Group::Style::NAME;
	else return Group::Style::UNKNOWN;
}

//==== operators ====

std::ostream& operator<<(std::ostream& out, const Group& group){
	return out<<group.label();
}

//==== member functions ====

void Group::clear(){
	label_="NULL";
	id_=string::hash("NULL");
	atoms_.clear();
}

void Group::init(const std::string& label, std::vector<int> atoms){
	label_=label;
	id_=string::hash(label_);
	atoms_=atoms;
	std::sort(atoms_.begin(),atoms_.begin()+atoms.size());
}

bool Group::contains(int atom){
	for(int i=0; i<atoms_.size(); ++i){
		if(atom<=atoms_[i]){
			if(atom==atoms_[i]) return true;
			else return false;
		}
	}
	return false;
}

int Group::find(int atom){
	int index=-1;
	for(int i=0; i<atoms_.size(); ++i){
		if(atom==atoms_[i]){
			index=i; break;
		}
	}
	return index;
}

//==== static functions ====

Group& Group::read(Token& token, Group& group){
	//group label id 1:100:1
	std::vector<int> atoms;
	//read label
	const std::string label=token.next();
	//read style
	Group::Style style=Group::Style::read(string::to_upper(token.next()).c_str());
	if(style==Group::Style::ID){
		//read atom lists
		while(!token.end()){
			//split atom string
			std::string atomstr=token.next();
			Token atok(atomstr.c_str(),":");
			//read atom limits
			int beg=-1,end=-1,stride=1;
			beg=std::atoi(atok.next().c_str())-1;
			if(!atok.end()) end=std::atoi(atok.next().c_str())-1;
			else end=beg;
			if(!atok.end()) stride=std::atoi(atok.next().c_str());
			//check limits
			if(beg<0) throw std::invalid_argument("Group::read(const Token&,Group&,const Structure): Invalid atom limits: beg");
			if(end<0) throw std::invalid_argument("Group::read(const Token&,Group&,const Structure): Invalid atom limits: end");
			if(end<beg) throw std::invalid_argument("Group::read(const Token&,Group&,const Structure): Invalid atom limits: order");
			if(stride<=0) throw std::invalid_argument("Group::read(const Token&,Group&,const Structure): Invalid atom limits: stride");
			//set atoms
			for(int i=beg; i<=end; i+=stride){
				atoms.push_back(i);
			}
		}
	} else {
		throw std::invalid_argument("Invalid group style.");
	}
	//init group
	group.init(label,atoms);
	//return group
	return group;
}

Group& Group::read(Token& token, Group& group, const Structure& struc){
	//group label id 1:100:1
	std::vector<int> atoms;
	//read label
	const std::string label=token.next();
	//read style
	Group::Style style=Group::Style::read(string::to_upper(token.next()).c_str());
	if(style==Group::Style::ID){
		//read atom lists
		while(!token.end()){
			//split atom string
			std::string atomstr=token.next();
			Token atok(atomstr.c_str(),":");
			//read atom limits
			int beg=-1,end=-1,stride=1;
			beg=std::atoi(atok.next().c_str())-1;
			if(!atok.end()) end=std::atoi(atok.next().c_str())-1;
			else end=beg;
			if(!atok.end()) stride=std::atoi(atok.next().c_str());
			//check limits
			if(beg<0) throw std::invalid_argument("Group::read(const Token&,Group&,const Structure): Invalid atom limits: beg");
			if(end<0) throw std::invalid_argument("Group::read(const Token&,Group&,const Structure): Invalid atom limits: end");
			if(end<beg) throw std::invalid_argument("Group::read(const Token&,Group&,const Structure): Invalid atom limits: order");
			if(stride<=0) throw std::invalid_argument("Group::read(const Token&,Group&,const Structure): Invalid atom limits: stride");
			if(end>=struc.nAtoms()) throw std::invalid_argument("Group::read(const Token&,Group&,const Structure): Invalid atom limits: end");
			//set atoms
			for(int i=beg; i<=end; i+=stride){
				atoms.push_back(i);
			}
		}
	} else if(style==Group::Style::TYPE){
		//read types
		std::vector<int> types;
		while(!token.end()){
			types.push_back(std::atoi(token.next().c_str())-1);
		}
		//find number of types
		int ntypes=-1;
		for(int i=0; i<struc.nAtoms(); ++i){
			if(struc.type(i)>ntypes) ntypes=struc.type(i);
		}
		ntypes++;
		//check types
		for(int i=0; i<types.size(); ++i){
			if(types[i]<=0 || types[i]>=ntypes) throw std::invalid_argument("Group::read(const Token&,Group&,const Structure): Invalid type");
		}
		//set atoms
		for(int i=0; i<struc.nAtoms(); ++i){
			for(int j=0; j<types.size(); ++j){
				if(struc.type(i)==types[j]){
					atoms.push_back(i);
					break;
				}
			}
		}
	} else if(style==Group::Style::NAME){
		//read names
		std::vector<std::string> names;
		while(!token.end()){
			names.push_back(token.next());
		}
		//set atoms
		for(int i=0; i<struc.nAtoms(); ++i){
			for(int j=0; j<names.size(); ++j){
				if(struc.name(i)==names[j]){
					atoms.push_back(i);
					break;
				}
			}
		}
	} else {
		throw std::invalid_argument("Invalid group style.");
	}
	//init group
	group.init(label,atoms);
	//return group
	return group;
}