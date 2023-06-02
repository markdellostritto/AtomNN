#pragma once
#ifndef XYZ_SIM_HPP
#define XYZ_SIM_HPP

// eigen libraries
#include <Eigen/Dense>
//format
#include "src/format/xyz.hpp"
// structure
#include "src/struc/sim.hpp"

namespace XYZ{
	
//unwrapping

void unwrap(Structure& struc);

void read(const char* file, const Interval& interval, const AtomType& atomT, Simulation& sim);
void write(const char* file, const Interval& interval, const AtomType& atomT, const Simulation& sim);

}

#endif