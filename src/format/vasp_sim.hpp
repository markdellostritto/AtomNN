#pragma once
#ifndef VASP_SIM_HPP
#define VASP_SIM_HPP

// c++ libraries
#include <vector>
#include <string>
// ann - structure
#include "src/format/format.hpp"
#include "src/struc/sim.hpp"

#ifndef __cplusplus
	#error A C++ compiler is required
#endif

#ifndef VASP_PRINT_FUNC
#define VASP_PRINT_FUNC 0
#endif

#ifndef VASP_PRINT_STATUS
#define VASP_PRINT_STATUS 0
#endif

#ifndef VASP_PRINT_DATA
#define VASP_PRINT_DATA 0
#endif

namespace VASP{

//static variables
static const int HEADER_SIZE=7;//number of lines in the header before the atomic positions
static const char* NAMESPACE_GLOBAL="VASP";
	
namespace XDATCAR{

static const char* NAMESPACE_LOCAL="XDATCAR";
void read(const char* file, const Interval& interval, const AtomType& atomT, Simulation& sim);
void write(const char* file, const Interval& interval, const AtomType& atomT, const Simulation& sim);

}

}

#endif