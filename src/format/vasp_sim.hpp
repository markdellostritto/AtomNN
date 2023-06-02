#pragma once
#ifndef VASP_SIM_HPP
#define VASP_SIM_HPP

// c++ libraries
#include <vector>
#include <string>
// format
#include "src/format/vasp.hpp"
// structure
#include "src/format/format.hpp"
#include "src/struc/sim.hpp"

namespace VASP{

namespace XDATCAR{

static const char* NAMESPACE_LOCAL="XDATCAR";
void read(const char* file, const Interval& interval, const AtomType& atomT, Simulation& sim);
void write(const char* file, const Interval& interval, const AtomType& atomT, const Simulation& sim);

}

}

#endif