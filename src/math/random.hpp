#ifndef RANDOM_HPP
#define RANDOM_HPP

//c libraries
#if __cplusplus > 199711L
#include <cstdint>
#endif
//c++ libraries
#include <iosfwd>
//math - const
#include "src/math/const.hpp"

namespace rng{

namespace dist{
	
	//******************************************************
	// Distribution - Name
	//******************************************************

	class Name{
	public:
		enum Type{
			UNKNOWN=0,
			UNIFORM=1,
			EXP=2,
			NORMAL=3,
			RAYLEIGH=4,
			LOGISTIC=5,
			CAUCHY=6
		};
		//constructor
		Name():t_(Type::UNKNOWN){}
		Name(Type t):t_(t){}
		//operators
		operator Type()const{return t_;}
		//member functions
		static Name read(const char* str);
		static const char* name(const Name& name);
	private:
		Type t_;
		//prevent automatic conversion for other built-in types
		//template<typename T> operator T() const;
	};
	std::ostream& operator<<(std::ostream& out, const Name& name);
	
} //end namespace dist

} //end namespace rng

#endif