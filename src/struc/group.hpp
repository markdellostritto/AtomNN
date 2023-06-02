#pragma once
#ifndef GROUP_HPP
#define GROUP_HPP

// structure
#include "src/struc/structure.hpp"
// str
#include "src/str/string.hpp"
#include "src/str/token.hpp"

class Group{
private:
	int id_;
	std::string label_;
	std::vector<int> atoms_;
public:
	struct Style{
	private:
		//prevent automatic conversion for other built-in types
		template<typename T> operator T() const;
	public:
		//enum
		enum Type{ID,TYPE,NAME,UNKNOWN};
		Type t_;
		//constructor
		Style(Type t):t_(t){}
		//operators
		operator Type()const{return t_;}
		//member functions
		static Type read(const char* str);
		static const char* name(const Type& type);
	};

	//==== constructors/destructors ====
	Group():label_("NULL"),id_(string::hash("NULL")){}
	~Group(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Group& group);
	
	//==== access ====
	int& id(){return id_;}
	const int& id()const{return id_;}
	std::string& label(){return label_;}
	const std::string& label()const{return label_;}
	std::vector<int>& atoms(){return atoms_;}
	const std::vector<int>& atoms()const{return atoms_;}
	int& atom(int i){return atoms_[i];}
	const int& atom(int i)const{return atoms_[i];}
	int size()const{return atoms_.size();}
	
	//==== member functions ====
	void clear();
	void init(const std::string& label, std::vector<int> atoms);
	bool contains(int atom);
	int find(int atom);
	
	//==== static functions ====
	static Group& read(Token& token, Group& group);
	static Group& read(Token& token, Group& group, const Structure& struc);
};

//==== operators ====

inline bool operator==(const Group& g1, const Group& g2){return g1.id()==g2.id();}
inline bool operator!=(const Group& g1, const Group& g2){return g1.id()!=g2.id();}

#endif