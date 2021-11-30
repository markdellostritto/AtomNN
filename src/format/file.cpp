// c++ libraries
#include <iostream>
#include <exception>
// ann - structure
#include "src/struc/structure.hpp"
// ann - file i/o
#include "src/format/vasp_struc.hpp"
#include "src/format/qe_struc.hpp"
#include "src/format/xyz_struc.hpp"
#include "src/format/cp2k_struc.hpp"
#include "src/format/ame_struc.hpp"
#include "src/format/file.hpp"

Structure& read_struc(const char* file, FILE_FORMAT::type format, const AtomType& atomT, Structure& struc){
	switch(format){
		case FILE_FORMAT::VASP_XML:
			VASP::XML::read(file,0,atomT,struc);
		break;
		case FILE_FORMAT::QE:
			QE::OUT::read(file,atomT,struc);
		break;
		case FILE_FORMAT::XYZ:
			XYZ::read(file,atomT,struc);
		break;
		case FILE_FORMAT::CP2K:
			CP2K::read(file,atomT,struc);
		break;
		case FILE_FORMAT::AME:
			AME::read(file,atomT,struc);
		break;
		default:
			throw std::invalid_argument("ERROR in read(const char*,FILE_Format::type,const AtomType&,Structure&): invalid file format.");
		break;
	}
	return struc;
}