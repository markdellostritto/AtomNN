###############################################################################
# COMPILERS AND FLAGS
###############################################################################

CXX_THREAD=mpic++
CXX_SERIAL=g++ 
#CXX_FLAGS=-std=gnu++11 -w -O3 -march=native 
CXX_FLAGS=-std=gnu++11 -w -O3 -march=native -DEIGEN_NO_DEBUG
#CXX_FLAGS=-std=gnu++11 -w -O3 -march=native -ffast-math

