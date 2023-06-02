#pragma once
#ifndef MOLGRAPH_HPP
#define MOLGRAPH_HPP

#include "src/chem/molgraph.hpp"

namespace molgraph{

static Molecule& build(const Structure& struc, Molecule& molecule){
	//resize
	molecule.atoms().resize(struc.nAtoms());
	//set atoms
	for(int i=0; i<struc.nAtoms(); ++i){
		molecule.atom(i).name()=struc.name(i);
		molecule.type(i).name()=struc.type(i);
		molecule.an(i).name()=struc.an(i);
	}
	//set bonds
	Eigen::Vector3d rtmp;
	for(int i=0; i<struc.nAtoms(); ++i){
		for(int j=i+1; j<struc.nAtoms(); ++j){
			const double dr=struc.dist(struc.posn(i),struc.posn(j),rtmp);
			const double bl=ptable::radius_covalent(struc.an(i))+ptable::radius_covalent(struc.an(j));
			if(dr<bl){
				molecule.atom(i).bonds().push_back(j);
				molecule.atom(j).bonds().push_back(i);
			}
		}
	}
}
	
};