##################################################################################
# EXTERNAL LIBRARY PATHS
##################################################################################
EIGEN = /usr/local/include/eigen-3.4-rc1/ # eigen library
ANN   = /home/tuh39227/Documents/code/AtomNN # AtomNN
INC_LIST = $(EIGEN) $(ANN)
INC = $(foreach d, $(INC_LIST), -I$d)

##################################################################################
# GENERAL SETTINGS
##################################################################################
COMP=gnu

# Include global settings.
include Makefile.$(COMP)

##################################################################################
# OBJECT FILES
##################################################################################
neural = batch.o data.o nn.o nn_train.o
nnp = nnp.o cutoff.o
basisr = basis_radial.o symm_radial.o symm_radial_g1.o symm_radial_g2.o symm_radial_t1.o
basisa = basis_angular.o symm_angular.o symm_angular_g3.o symm_angular_g4.o
struc = structure.o cell.o cell_list.o atom.o neighbor.o thermo.o atom_type.o
format = file.o format.o qe_struc.o vasp_struc.o xyz_struc.o cp2k_struc.o ame_struc.o
math = func.o special.o accumulator.o eigen.o random.o
chem = units.o ptable.o ewald3D.o
str = string.o print.o 
util = compiler.o time.o
mem = serialize.o map.o
thread = parallel.o
entropy = entropy_rho.o rho_atomic.o
signal = fft.o window.o
objects = $(neural) $(basisr) $(basisa) $(struct) $(format) $(math) $(chem) $(str) $(util) $(thread)

##################################################################################
# OBJECT FILE MAKE RULES
##################################################################################

# machine learning
nn.o: ml/nn.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c ml/nn.cpp
batch.o: ml/batch.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c ml/batch.cpp
data.o: ml/data.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c ml/data.cpp
nn_train.o: ml/nn_train.cpp
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -c ml/nn_train.cpp
# nnp
nnp.o: nnp/nnp.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c nnp/nnp.cpp
atom.o: nnp/atom.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c nnp/atom.cpp
cutoff.o: nnp/cutoff.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c nnp/cutoff.cpp
# nnp - basis - radial
basis_radial.o: nnp/basis_radial.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c nnp/basis_radial.cpp
symm_radial.o: nnp/symm_radial.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c nnp/symm_radial.cpp
symm_radial_g1.o: nnp/symm_radial_g1.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c nnp/symm_radial_g1.cpp
symm_radial_g2.o: nnp/symm_radial_g2.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c nnp/symm_radial_g2.cpp
symm_radial_t1.o: nnp/symm_radial_t1.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c nnp/symm_radial_t1.cpp
# nnp - basis - angular
basis_angular.o: nnp/basis_angular.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c nnp/basis_angular.cpp
symm_angular.o: nnp/symm_angular.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c nnp/symm_angular.cpp
symm_angular_g3.o: nnp/symm_angular_g3.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c nnp/symm_angular_g3.cpp
symm_angular_g4.o: nnp/symm_angular_g4.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c nnp/symm_angular_g4.cpp
# struc
structure.o: struc/structure.cpp 
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c struc/structure.cpp
sim.o: struc/sim.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c struc/sim.cpp
cell.o: struc/cell.cpp 
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c struc/cell.cpp
cell_list.o: struc/cell_list.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c struc/cell_list.cpp
neighbor.o: struc/neighbor.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c struc/neighbor.cpp
atom_type.o: struc/atom_type.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c struc/atom_type.cpp
thermo.o: struc/thermo.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c struc/thermo.cpp
grid.o: struc/grid.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c struc/grid.cpp
# format
file.o: format/file.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c format/file.cpp
format.o: format/format.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c format/format.cpp
vasp_struc.o: format/vasp_struc.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c format/vasp_struc.cpp
vasp_sim.o: format/vasp_sim.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c format/vasp_sim.cpp
lammps_sim.o: format/lammps_sim.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c format/lammps_sim.cpp
qe_struc.o: format/qe_struc.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c format/qe_struc.cpp
xyz_struc.o: format/xyz_struc.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c format/xyz_struc.cpp
cp2k_struc.o: format/cp2k_struc.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c format/cp2k_struc.cpp
ame_struc.o: format/ame_struc.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c format/ame_struc.cpp
# math
func.o: math/func.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c math/func.cpp
special.o: math/special.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c math/special.cpp
accumulator.o: math/accumulator.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c math/accumulator.cpp
eigen.o: math/eigen.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c math/eigen.cpp
random.o: math/random.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c math/random.cpp
# optimization
optimize.o: opt/optimize.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c opt/optimize.cpp
# chem
units.o: chem/units.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c chem/units.cpp
ptable.o: chem/ptable.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c chem/ptable.cpp
ewald3D.o: chem/ewald3D.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c chem/ewald3D.cpp
# memory
serialize.o: mem/serialize.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c mem/serialize.cpp
map.o: mem/map.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c mem/map.cpp
# thread
parallel.o: thread/parallel.cpp
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -c thread/parallel.cpp
# string
string.o: str/string.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c str/string.cpp
print.o: str/print.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c str/print.cpp
input.o: str/input.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c str/input.cpp
# utility
compiler.o: util/compiler.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c util/compiler.cpp
time.o: util/time.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c util/time.cpp
# entropy
rho_atomic.o: entropy/rho_atomic.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c entropy/rho_atomic.cpp
# signal
fft.o: signal/fft.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c signal/fft.cpp -lfftw3
window.o: signal/window.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c signal/window.cpp

##################################################################################
# TARGETS
##################################################################################

clean: 
	rm *.o

#**** UNIT TESTS ****

test_unit_optimize: optimize.o string.o serialize.o eigen.o print.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_unit_optimize.exe opt/test_unit_optimize.cpp optimize.o string.o serialize.o eigen.o print.o
test_unit_nn: nn.o string.o serialize.o eigen.o print.o special.o random.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_unit_nn.exe ml/test_unit_nn.cpp nn.o string.o serialize.o eigen.o print.o special.o random.o
test_unit_nnp: nnp.o nn.o string.o eigen.o print.o special.o random.o atom.o cutoff.o $(basisr) $(basisa) $(mem)
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_unit_nnp.exe nnp/test_unit_nnp.cpp nnp.o nn.o string.o eigen.o print.o special.o random.o atom.o cutoff.o $(basisr) $(basisa) $(mem)
test_unit_struc: structure.o cell.o cell_list.o eigen.o serialize.o atom.o print.o atom_type.o string.o thermo.o units.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_unit_struc.exe struc/test_unit_struc.cpp structure.o cell.o cell_list.o eigen.o serialize.o atom.o print.o atom_type.o string.o thermo.o units.o
test_unit_eigen:
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_unit_eigen.exe math/test_unit_eigen.cpp

#**** NN OPTIMIZATION ****

nn_fit: $(neural) $(str) $(math) $(util) optimize.o serialize.o parallel.o
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o nn_fit.exe ml/nn_fit.cpp $(neural) $(str) $(math) $(util) optimize.o serialize.o parallel.o

#**** NNP OPTIMIZATION ****

nnp_train: $(nnp) batch.o data.o nn.o $(str) $(math) $(util) $(mem) $(thread) $(basisr) $(basisa) $(struct) $(format) $(chem) $(struc) optimize.o
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o nnp_train.exe nnp/nnp_train.cpp $(nnp) batch.o data.o nn.o $(str) $(math) $(util) $(mem) $(thread) $(basisr) $(basisa) $(struct) $(format) $(chem) $(struc) optimize.o
nnp_compute: $(nnp) nn.o $(str) $(math) $(util) $(mem) $(thread) $(basisr) $(basisa) $(struct) $(format) $(chem) $(struc)
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o nnp_compute.exe nnp/nnp_compute.cpp $(nnp) nn.o $(str) $(math) $(util) $(mem) $(thread) $(basisr) $(basisa) $(struct) $(format) $(chem) $(struc)
nnp_compute_sim: $(nnp) nn.o $(str) $(math) $(util) $(mem) $(thread) $(basisr) $(basisa) $(struct) $(format) $(chem) $(struc) sim.o vasp_sim.o
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o nnp_compute_sim.exe nnp/nnp_compute_sim.cpp $(nnp) nn.o $(str) $(math) $(util) $(mem) $(thread) $(basisr) $(basisa) $(struct) $(format) $(chem) $(struc) sim.o vasp_sim.o
intgrad: intgrad.cpp $(struct) $(chem) $(utility) $(neural) $(basisr) $(basisa) $(math) 
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o intgrad.exe intgrad.cpp $(struct) $(chem) $(utility) $(neural) $(basisr) $(basisa) $(math) 

#**** ENTROPY ****
entropy_struc: $(chem) $(struc) $(str) $(mem) eigen.o vasp_struc.o grid.o format.o rho_atomic.o fft.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o entropy_struc.exe entropy/entropy_struc.cpp $(chem) $(struc) $(str) $(mem) eigen.o vasp_struc.o grid.o format.o rho_atomic.o fft.o -lfftw3
entropy_old: $(chem) $(struc) $(str) $(mem) eigen.o vasp_struc.o grid.o format.o rho_atomic.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o entropy_old.exe entropy/entropy_old.cpp $(chem) $(struc) $(str) $(mem) eigen.o vasp_struc.o grid.o format.o rho_atomic.o
entropy_sim: $(chem) $(struc) $(str) $(mem) eigen.o vasp_sim.o lammps_sim.o grid.o format.o rho_atomic.o sim.o parallel.o fft.o
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o entropy_sim.exe entropy/entropy_sim.cpp $(chem) $(struc) $(str) $(mem) eigen.o vasp_sim.o lammps_sim.o grid.o format.o rho_atomic.o sim.o parallel.o fft.o -lfftw3

