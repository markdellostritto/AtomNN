#ifndef FILE_HPP
#define FILE_HPP

// ann - structure
#include "src/struc/structure_fwd.hpp"
// ann - file i/o
#include "src/format/format.hpp"

Structure& read_struc(const char* file, FILE_FORMAT::type format, const AtomType& atomT, Structure& struc);

#endif