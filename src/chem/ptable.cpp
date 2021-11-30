// c libraries
#include <cstring>
#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
#include <cmath>
#elif (defined __ICC || defined __INTEL_COMPILER)
#include <mathimf.h> //intel math library
#else
#include <cmath>
#endif
// ann - ptable
#include "src/chem/ptable.hpp"

namespace ptable{
	
//*********************************************
//Function
//*********************************************

//************** NAME ***************
const char* name(int an){return NAME[an-1];}
//********** ATOMIC_NUMBER **********
int an(const char* name){
	for(int i=0; i<N_ELEMENTS; i++){
		if(std::strcmp(name,NAME[i])==0) return i+1;
	}
	return 0;
}
int an(double mass){
	double min=100;
	int an=0;
	for(int i=0; i<N_ELEMENTS; ++i){
		if(fabs(mass-MASS[i])<min){
			min=fabs(mass-MASS[i]);
			an=i+1;
		}
	}
	return an;
}
//************** MASS ***************
double mass(int an){return MASS[an-1];}
//************* RADIUS **************
double radius_atomic(int an){return RADIUS_ATOMIC[an-1];}
double radius_cov(int an){return RADIUS_COVALENT[an-1];}

}

