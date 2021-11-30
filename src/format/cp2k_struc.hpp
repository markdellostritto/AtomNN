#pragma once
#ifndef CP2K_HPP
#define CP2K_HPP

// eigen libraries
#include <Eigen/Dense>
// DAME - structure
#include "src/struc/structure_fwd.hpp"

#ifndef CP2K_PRINT_FUNC
#define CP2K_PRINT_FUNC 0
#endif

#ifndef CP2K_PRINT_STATUS
#define CP2K_PRINT_STATUS 0
#endif

#ifndef CP2K_PRINT_DATA
#define CP2K_PRINT_DATA 0
#endif

#ifndef __cplusplus
	#error A C++ compiler is required
#endif

namespace CP2K{

//*****************************************************
//reading
//*****************************************************

void read(const char* file, const AtomType& atomT, Structure& struc);

}

#endif