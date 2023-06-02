#pragma once
#ifndef LAMMPS_STRUC_HPP
#define LAMMPS_STRUC_HPP

// c++ libraries
#include <vector>
//format
#include "src/format/lammps.hpp"
// structure
#include "src/struc/structure.hpp"

namespace LAMMPS{

namespace DUMP{

void read(const char* file, const AtomType& atomT, Structure& struc){}
void write(const char* file, const AtomType& atomT, Structure& struc);
void write(FILE* writer, const AtomType& atomT, Structure& struc);

}

}

#endif
