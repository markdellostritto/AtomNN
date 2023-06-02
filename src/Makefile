##################################################################################
# EXTERNAL LIBRARY PATHS
##################################################################################
EIGEN = /usr/local/include/eigen-3.4-rc1/ # eigen library
ANN   = /home/tuh39227/Documents/code/ann_v9/ # AtomNN
#ANN:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
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
nnp = nnh.o nnp.o cutoff.o type.o
basis = basis.o basis_radial.o basis_angular.o
struc = structure.o cell.o cell_list.o neighbor.o state.o atom_type.o
format = format.o lammps.o
format_struc = file_struc.o qe_struc.o vasp_struc.o xyz_struc.o cp2k_struc.o ame_struc.o raw_struc.o
format_sim = file_sim.o vasp_sim.o lammps_sim.o xyz_sim.o cp2k_sim.o
math = special.o eigen.o random.o reduce.o corr.o
opt = loss.o stop.o algo.o objective.o decay.o
chem = units.o ptable.o 
str = string.o print.o token.o
util = compiler.o time.o
mem = serialize.o map.o
thread = comm.o dist.o
entropy = entropy_rho.o rho_atomic.o
signal = fft.o window.o
pot = pot.o pot_factory.o pot_coul_cut.o pot_coul_wolf.o pot_coul_dsf.o pot_coul_long.o pot_gauss_cut.o pot_gauss_dsf.o pot_gauss_long.o pot_lj_cut.o pot_lj_long.o pot_ldamp_cut.o pot_ldamp_long.o pot_spin_ex.o pot_nnpe.o pot_nnpeq.o pot_qeq_gl.o
kspace = kspace.o kspace_coul.o kspace_london.o
analyze = pair.o group.o compute.o shell.o hbond.o rdf.o distmat.o avg_posn.o
property = set_property.o set_property_type.o set_property_mass.o set_property_charge.o set_property_velocity.o set_property_temp.o set_property_factory.o
objects = $(neural) $(basis) $(struct) $(format) $(math) $(chem) $(str) $(util) $(thread)

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
pca.o: ml/pca.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c ml/pca.cpp
# nnp
nnh.o: nnp/nnh.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c nnp/nnh.cpp
nnp.o: nnp/nnp.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c nnp/nnp.cpp
type.o: nnp/type.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c nnp/type.cpp
cutoff.o: nnp/cutoff.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c nnp/cutoff.cpp
# nnp - basis
basis.o: nnp/basis.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c nnp/basis.cpp
basis_radial.o: nnp/basis_radial.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c nnp/basis_radial.cpp
basis_angular.o: nnp/basis_angular.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c nnp/basis_angular.cpp
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
verlet.o: struc/verlet.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c struc/verlet.cpp
pair.o: struc/pair.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c struc/pair.cpp
atom_type.o: struc/atom_type.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c struc/atom_type.cpp
state.o: struc/state.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c struc/state.cpp
grid.o: struc/grid.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c struc/grid.cpp
group.o: struc/group.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c struc/group.cpp
# format
file_struc.o: format/file_struc.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c format/file_struc.cpp
file_sim.o: format/file_sim.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c format/file_sim.cpp
format.o: format/format.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c format/format.cpp
vasp_struc.o: format/vasp_struc.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c format/vasp_struc.cpp
vasp_sim.o: format/vasp_sim.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c format/vasp_sim.cpp
lammps.o: format/lammps.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c format/lammps.cpp
lammps_sim.o: format/lammps_sim.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c format/lammps_sim.cpp
xyz_sim.o: format/xyz_sim.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c format/xyz_sim.cpp
cp2k_sim.o: format/cp2k_sim.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c format/cp2k_sim.cpp
qe_struc.o: format/qe_struc.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c format/qe_struc.cpp
xyz_struc.o: format/xyz_struc.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c format/xyz_struc.cpp
raw_struc.o: format/raw_struc.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c format/raw_struc.cpp
cp2k_struc.o: format/cp2k_struc.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c format/cp2k_struc.cpp
ame_struc.o: format/ame_struc.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c format/ame_struc.cpp
# math
special.o: math/special.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c math/special.cpp
accumulator.o: math/accumulator.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c math/accumulator.cpp
corr.o: math/corr.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c math/corr.cpp
reduce.o: math/reduce.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c math/reduce.cpp
eigen.o: math/eigen.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c math/eigen.cpp
random.o: math/random.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c math/random.cpp
hist.o: math/hist.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c math/hist.cpp
graph.o: math/graph.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c math/graph.cpp
# optimization
loss.o: opt/loss.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c opt/loss.cpp
stop.o: opt/stop.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c opt/stop.cpp
algo.o: opt/algo.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c opt/algo.cpp
decay.o: opt/decay.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c opt/decay.cpp
objective.o: opt/objective.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c opt/objective.cpp
optimize.o: opt/optimize.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c opt/optimize.cpp
# chem
units.o: chem/units.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c chem/units.cpp
ptable.o: chem/ptable.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c chem/ptable.cpp
molecule.o: chem/molecule.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c chem/molecule.cpp
alias.o: chem/alias.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c chem/alias.cpp
# memory
serialize.o: mem/serialize.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c mem/serialize.cpp
map.o: mem/map.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c mem/map.cpp
# thread
dist.o: thread/dist.cpp
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -c thread/dist.cpp
comm.o: thread/comm.cpp
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -c thread/comm.cpp
# string
string.o: str/string.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c str/string.cpp
print.o: str/print.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c str/print.cpp
token.o: str/token.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c str/token.cpp
parse.o: str/parse.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c str/parse.cpp
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
# analysis
compute.o: analysis/compute.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c analysis/compute.cpp
shell.o: analysis/shell.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c analysis/shell.cpp
hbond.o: analysis/hbond.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c analysis/hbond.cpp
rdf.o: analysis/rdf.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c analysis/rdf.cpp
distmat.o: analysis/distmat.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c analysis/distmat.cpp
avg_posn.o: analysis/avg_posn.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c analysis/avg_posn.cpp
# torch
job.o: torch/job.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/job.cpp
qeq.o: torch/qeq.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/qeq.cpp
qtpie.o: torch/qtpie.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/qtpie.cpp
set_property.o: torch/set_property.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/set_property.cpp
set_property_type.o: torch/set_property_type.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/set_property_type.cpp
set_property_mass.o: torch/set_property_mass.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/set_property_mass.cpp
set_property_charge.o: torch/set_property_charge.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/set_property_charge.cpp
set_property_velocity.o: torch/set_property_velocity.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/set_property_velocity.cpp
set_property_temp.o: torch/set_property_temp.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/set_property_temp.cpp
set_property_factory.o: torch/set_property_factory.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/set_property_factory.cpp
pot.o: torch/pot.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/pot.cpp
pot_factory.o: torch/pot_factory.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/pot_factory.cpp
pot_coul_cut.o: torch/pot_coul_cut.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/pot_coul_cut.cpp
pot_coul_wolf.o: torch/pot_coul_wolf.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/pot_coul_wolf.cpp
pot_coul_dsf.o: torch/pot_coul_dsf.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/pot_coul_dsf.cpp
pot_coul_long.o: torch/pot_coul_long.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/pot_coul_long.cpp
pot_gauss_cut.o: torch/pot_gauss_cut.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/pot_gauss_cut.cpp
pot_gauss_long.o: torch/pot_gauss_long.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/pot_gauss_long.cpp
pot_gauss_dsf.o: torch/pot_gauss_dsf.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/pot_gauss_dsf.cpp
pot_qeq_gl.o: torch/pot_qeq_gl.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/pot_qeq_gl.cpp
pot_lj_cut.o: torch/pot_lj_cut.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/pot_lj_cut.cpp 
pot_lj_long.o: torch/pot_lj_long.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/pot_lj_long.cpp
pot_ldamp_cut.o: torch/pot_ldamp_cut.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/pot_ldamp_cut.cpp
pot_ldamp_long.o: torch/pot_ldamp_long.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/pot_ldamp_long.cpp
pot_spin_ex.o: torch/pot_spin_ex.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/pot_spin_ex.cpp
pot_nnpe.o: torch/pot_nnpe.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/pot_nnpe.cpp
pot_nnpeq.o: torch/pot_nnpeq.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/pot_nnpeq.cpp
kspace.o: torch/kspace.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/kspace.cpp
kspace_coul.o: torch/kspace_coul.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/kspace_coul.cpp
kspace_london.o: torch/kspace_london.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/kspace_london.cpp
engine.o: torch/engine.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/engine.cpp
monte_carlo.o: torch/monte_carlo.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/monte_carlo.cpp
integrator.o: torch/integrator.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/integrator.cpp
dump.o: torch/dump.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/dump.cpp
stochastic.o: torch/stochastic.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/stochastic.cpp

##################################################################################
# TARGETS
##################################################################################

clean: 
	rm *.o

#==== UNIT TESTS ====

#** optimization**
test_opt: $(opt) $(str) serialize.o eigen.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_opt.exe opt/test_opt.cpp $(opt) $(str) serialize.o eigen.o
test_loss: $(opt) $(str) serialize.o eigen.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_loss.exe opt/test_loss.cpp $(opt) $(str) serialize.o eigen.o
#** math **
test_math_special: time.o special.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_math_special.exe math/test_math_special.cpp time.o special.o
test_fft_unit: fft.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_fft_unit.exe signal/test_fft_unit.cpp fft.o -lfftw3
test_reduce_unit: reduce.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_reduce_unit.exe math/test_reduce_unit.cpp reduce.o
#** machine learning **
test_nn_unit: nn.o string.o serialize.o eigen.o print.o special.o random.o token.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_nn_unit.exe ml/test_nn_unit.cpp nn.o string.o serialize.o eigen.o print.o special.o random.o token.o
test_nn_transfer: nn.o string.o serialize.o eigen.o print.o special.o random.o token.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_nn_transfer.exe ml/test_nn_transfer.cpp nn.o string.o serialize.o eigen.o print.o special.o random.o token.o
test_nn_grad: nn.o string.o serialize.o eigen.o print.o special.o random.o token.o reduce.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_nn_grad.exe ml/test_nn_grad.cpp nn.o string.o serialize.o eigen.o print.o special.o random.o token.o reduce.o
test_nn_cost: nn.o string.o serialize.o eigen.o print.o special.o random.o token.o reduce.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_nn_cost.exe ml/test_nn_cost.cpp nn.o string.o serialize.o eigen.o print.o special.o random.o token.o reduce.o
test_pca_gauss: pca.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_pca_gauss.exe ml/test_pca_gauss.cpp pca.o
#** torch **
test_kspace_coul: $(struc) $(str) $(math) $(mem) $(kspace) $(chem) pot_coul_long.o pot.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_kspace_coul.exe torch/test_kspace_coul.cpp $(struc) $(str) $(math) $(mem) $(kspace) $(chem) pot_coul_long.o pot.o
test_kspace_london: $(struc) $(str) $(chem) $(kspace) $(pot) $(thread) $(basis) $(nnp) $(math) nn.o $(mem)
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o test_kspace_london.exe torch/test_kspace_london.cpp $(struc) $(str) $(chem) $(kspace) $(pot) $(thread) $(basis) $(nnp) $(math) nn.o $(mem)
test_pot_coul_cut: $(struc) $(str) eigen.o serialize.o units.o pot_coul_cut.o pot.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_pot_coul_cut.exe torch/test_pot_coul_cut.cpp $(struc) $(str) eigen.o serialize.o units.o pot_coul_cut.o pot.o
test_pot_coul_long: $(struc) $(str) eigen.o serialize.o units.o kspace.o kspace_coul.o pot_coul_long.o pot_coul_cut.o pot.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_pot_coul_long.exe torch/test_pot_coul_long.cpp $(struc) $(str) eigen.o serialize.o units.o kspace.o kspace_coul.o pot_coul_long.o pot_coul_cut.o pot.o
test_pot_gauss_cut: $(struc) $(str) eigen.o serialize.o units.o pot_gauss_cut.o pot.o ptable.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_pot_gauss_cut.exe torch/test_pot_gauss_cut.cpp $(struc) $(str) eigen.o serialize.o units.o pot_gauss_cut.o pot.o ptable.o
test_pot_gauss_long: $(struc) $(str) eigen.o serialize.o units.o kspace.o kspace_coul.o pot_gauss_long.o pot_coul_long.o pot.o ptable.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_pot_gauss_long.exe torch/test_pot_gauss_long.cpp $(struc) $(str) eigen.o serialize.o units.o kspace.o kspace_coul.o pot_gauss_long.o pot_coul_long.o pot.o ptable.o
test_pot_gauss_dsf: $(struc) $(str) eigen.o serialize.o units.o pot_gauss_dsf.o pot_coul_dsf.o pot.o ptable.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_pot_gauss_dsf.exe torch/test_pot_gauss_dsf.cpp $(struc) $(str) eigen.o serialize.o units.o pot_gauss_dsf.o pot_coul_dsf.o pot.o ptable.o
test_pot_ldamp_long: $(struc) $(str) eigen.o serialize.o units.o kspace.o kspace_london.o pot_ldamp_long.o pot.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_pot_ldamp_long.exe torch/test_pot_ldamp_long.cpp $(struc) $(str) eigen.o serialize.o units.o kspace.o kspace_london.o pot_ldamp_long.o pot.o
#** nnp **
test_nnp_unit: $(basis) $(mem) $(struc) $(str) nnh.o nnp.o nn.o eigen.o special.o random.o cutoff.o type.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_nnp_unit.exe nnp/test_nnp_unit.cpp $(basis) $(mem) $(struc) $(str) nnh.o nnp.o nn.o eigen.o special.o random.o cutoff.o type.o
test_nnp_symm: $(basis) $(mem) $(struc) $(str) nnh.o nnp.o nn.o eigen.o special.o random.o cutoff.o type.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_nnp_symm.exe nnp/test_nnp_symm.cpp $(basis) $(mem) $(struc) $(str) nnh.o nnp.o nn.o eigen.o special.o random.o cutoff.o type.o
test_struc_unit: $(struc) $(str) eigen.o serialize.o units.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_struc_unit.exe struc/test_struc_unit.cpp $(struc) $(str) eigen.o serialize.o units.o
test_neigh_unit: $(struc) $(str) eigen.o serialize.o units.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_neigh_unit.exe struc/test_neigh_unit.cpp $(struc) $(str) eigen.o serialize.o units.o
test_cutoff: $(mem) $(str) $(math) basis.o cutoff.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_cutoff.exe nnp/test_cutoff.cpp $(mem) $(str) $(math) basis.o cutoff.o
test_basis_radial: $(mem) $(str) $(math) basis.o basis_radial.o cutoff.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_basis_radial.exe nnp/test_basis_radial.cpp $(mem) $(str) $(math) basis.o basis_radial.o cutoff.o
test_basis_angular: $(mem) $(str) $(math) basis.o basis_angular.o cutoff.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_basis_angular.exe nnp/test_basis_angular.cpp $(mem) $(str) $(math) basis.o basis_angular.o cutoff.o
test_omp_out:
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -fopenmp -o test_omp_out.exe omp/test_omp_out.cpp
test_omp_for: time.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -fopenmp -o test_omp_for.exe omp/test_omp_for.cpp time.o
test_omp_eigen: 
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -fopenmp -o test_omp_eigen.exe omp/test_omp_eigen.cpp 

#==== NN OPTIMIZATION ====

nn_fit: $(neural) $(str) $(math) $(util) $(thread) $(opt) serialize.o
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o nn_fit.exe ml/nn_fit.cpp $(neural) $(str) $(math) $(util) $(thread) $(opt) serialize.o 

#==== NNP OPTIMIZATION ====

symm: $(nnp) $(str) $(util) $(mem) $(basis) $(struct) $(format_struc) $(struc) $(chem) nn.o format.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o symm.exe nnp/symm.cpp $(nnp) $(str) $(util) $(mem) $(basis) $(struct) $(format_struc) $(struc) $(chem) $(math) nn.o format.o
nnpte: $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(pot) $(kspace) $(opt) batch.o nn.o format.o alias.o
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o nnpte.exe nnp/nnpte.cpp $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(kspace) $(opt) batch.o nn.o pot.o pot_ldamp_long.o pot_gauss_long.o pot_ldamp_cut.o format.o alias.o
nnptef: $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(pot) $(kspace) $(opt) batch.o nn.o format.o alias.o
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o nnptef.exe nnp/nnptef.cpp $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(kspace) $(opt) batch.o nn.o pot.o pot_ldamp_long.o pot_gauss_long.o pot_ldamp_cut.o format.o alias.o
nnpqa: $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(pot) $(kspace) $(opt) batch.o nn.o format.o alias.o
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o nnpqa.exe nnp/nnpqa.cpp $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(kspace) $(opt) batch.o nn.o pot.o pot_ldamp_long.o pot_gauss_long.o pot_ldamp_cut.o format.o alias.o
nnpqa2: $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(pot) $(kspace) $(opt) batch.o nn.o format.o alias.o
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o nnpqa2.exe nnp/nnpqa2.cpp $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(kspace) $(opt) batch.o nn.o pot.o pot_ldamp_long.o pot_gauss_long.o pot_ldamp_cut.o format.o alias.o
nnpqa3: $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(pot) $(kspace) $(opt) batch.o nn.o format.o alias.o parse.o
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o nnpqa3.exe nnp/nnpqa3.cpp $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(kspace) $(opt) batch.o nn.o pot.o pot_ldamp_long.o pot_gauss_long.o pot_ldamp_cut.o format.o alias.o parse.o
nnpqn3: $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(pot) $(kspace) $(opt) batch.o nn.o format.o alias.o parse.o
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o nnpqn3.exe nnp/nnpqn3.cpp $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(kspace) $(opt) batch.o nn.o pot.o pot_ldamp_long.o pot_gauss_long.o pot_ldamp_cut.o format.o alias.o parse.o
nnptk: $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(pot) $(kspace) $(opt) batch.o nn.o format.o
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o nnptk.exe nnp/nnptk.cpp $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(kspace) $(opt) batch.o nn.o pot.o pot_ldamp_long.o pot_gauss_long.o pot_ldamp_cut.o format.o
nnpteq: $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(pot) $(kspace) $(opt) $(pot) qeq.o batch.o nn.o format.o
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o nnpteq.exe nnp/nnpteq.cpp $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(kspace) $(opt) $(pot) qeq.o batch.o nn.o format.o
nnptex: $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(pot) $(kspace) $(opt) $(pot) qtpie.o batch.o nn.o format.o
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o nnptex.exe nnp/nnptex.cpp $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(kspace) $(opt) $(pot) qtpie.o batch.o nn.o format.o
nnptes: $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(pot) $(kspace) optimize.o batch.o nn.o 
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o nnptes.exe nnp/nnptes.cpp $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(pot) $(kspace) optimize.o batch.o nn.o
nnptev: $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(pot) $(kspace) optimize.o batch.o nn.o
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o nnptev.exe nnp/nnptev.cpp $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(pot) $(kspace) optimize.o batch.o nn.o
nnptez: $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(pot) $(kspace) optimize.o batch.o nn.o format.o
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o nnptez.exe nnp/nnptez.cpp $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(pot) $(kspace) optimize.o batch.o nn.o format.o
nnpce: $(nnp) nn.o $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) format.o
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o nnpce.exe nnp/nnpce.cpp $(nnp) nn.o $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) format.o
nnpces: $(nnp) nn.o $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_sim) $(chem) $(struc) format.o sim.o
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o nnpces.exe nnp/nnpces.cpp $(nnp) nn.o $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_sim) $(chem) $(struc) format.o sim.o 
nnpceq: $(nnp) nn.o $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(pot) $(kspace) format.o qeq.o
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o nnpceq.exe nnp/nnpceq.cpp $(nnp) nn.o $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(pot) $(kspace) format.o qeq.o
nnptef2: $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(pot) $(kspace) $(opt) batch.o nn.o format.o alias.o
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o nnptef2.exe nnp/nnptef2.cpp $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(kspace) $(opt) batch.o nn.o pot.o pot_ldamp_long.o pot_gauss_long.o pot_ldamp_cut.o format.o alias.o

#==== CHARGE EQULIBRATION ====

qeqm: qeq.o $(struc) $(format_struc) $(str) $(mem) $(util) $(chem) $(pot) $(kspace) $(nnp) $(basis) $(math) nn.o format.o 
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o qeqm.exe torch/qeqm.cpp qeq.o $(struc) $(format_struc) $(str) $(mem) $(util) $(chem) $(pot) $(kspace) $(nnp) $(basis) $(math) nn.o format.o 
qtpiem: qtpie.o $(struc) $(format_struc) $(str) $(mem) $(util) $(chem) $(pot) $(kspace) eigen.o format.o reduce.o type.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o qtpiem.exe torch/qtpiem.cpp qtpie.o $(struc) $(format_struc) $(str) $(mem) $(util) $(chem) $(pot) $(kspace) eigen.o format.o reduce.o type.o

#==== TORCH ====

torch: $(struc) $(format_struc) $(str) $(mem) $(util) $(chem) $(pot) $(math) $(kspace) $(basis) $(nnp) $(property) batch.o data.o nn.o monte_carlo.o integrator.o dump.o stochastic.o format.o engine.o qeq.o job.o group.o verlet.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o torch.exe torch/torch.cpp $(struc) $(format_struc) $(str) $(mem) $(util) $(chem) $(pot) $(math) $(kspace) $(basis) $(nnp) $(property) batch.o data.o nn.o monte_carlo.o integrator.o dump.o stochastic.o format.o engine.o qeq.o job.o group.o verlet.o
torch_train: $(struc) $(format_struc) $(str) $(mem) $(util) $(chem) $(pot) $(math) $(kspace) $(basis) $(nnp) batch.o data.o nn.o monte_carlo.o integrator.o dump.o stochastic.o format.o engine.o qeq.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o torch_train.exe torch/torch_train.cpp $(struc) $(format_struc) $(str) $(mem) $(util) $(chem) $(pot) $(math) $(kspace) $(basis) $(nnp) batch.o data.o nn.o monte_carlo.o integrator.o dump.o stochastic.o format.o engine.o qeq.o
phonon: $(struc) format.o vasp_struc.o vasp_sim.o lammps_sim.o $(str) $(mem) $(util) units.o ptable.o special.o eigen.o $(thread) sim.o
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o phonon.exe torch/phonon.cpp $(struc) format.o vasp_struc.o vasp_sim.o lammps_sim.o $(str) $(mem) $(util) units.o ptable.o special.o eigen.o $(thread) sim.o -lfftw3

#==== ANALYSIS ====

analysis: $(struc) sim.o format.o $(format_sim) $(str) $(mem) $(chem) $(math) $(analyze) hist.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o analysis.exe analysis/analysis.cpp $(struc) sim.o format.o $(format_sim) $(str) $(mem) $(chem) $(math) $(analyze) hist.o
atomsort: $(struc) $(format_struc) $(str) $(chem) $(mem) format.o parse.o eigen.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o atomsort.exe analysis/atomsort.cpp $(struc) $(format_struc) $(str) $(chem) $(mem) format.o parse.o eigen.o
vibrlx: $(struc) sim.o format.o $(format_sim) $(str) $(mem) $(chem) $(math) fft.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o vibrlx.exe analysis/vibrlx.cpp $(struc) sim.o format.o $(format_sim) $(str) $(mem) $(chem) $(math) fft.o -lfftw3
smod: $(struc) format.o $(format_struc) $(str) $(mem) $(chem) eigen.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o smod.exe analysis/smod.cpp $(struc) format.o $(format_struc) $(str) $(mem) $(chem) eigen.o
ffa: $(struc) format.o $(format_struc) $(str) $(mem) $(chem) $(math) molecule.o graph.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o ffa.exe chem/ffa.cpp $(struc) format.o $(format_struc) $(str) $(mem) $(chem) $(math) molecule.o graph.o
vdos: $(struc) sim.o format.o $(format_sim) $(str) $(mem) $(chem) $(math) $(signal) group.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o vdos.exe analysis/vdos.cpp $(struc) sim.o format.o $(format_sim) $(str) $(mem) $(chem) $(math) $(signal) group.o -lfftw3
vdos2: $(str) $(mem) $(chem) $(math) $(signal) group.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o vdos2.exe analysis/vdos2.cpp $(str) $(mem) $(chem) $(math) $(signal) group.o -lfftw3
angle: $(str) $(mem) $(chem) $(math) $(struc) group.o hist.o sim.o format.o $(format_sim) pair.o $(signal)
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o angle.exe analysis/angle.cpp $(str) $(mem) $(chem) $(math) $(struc) group.o hist.o sim.o format.o $(format_sim) pair.o $(signal) -lfftw3
rdf: $(str) $(mem) $(chem) $(math) $(struc) group.o hist.o sim.o format.o $(format_sim) pair.o $(signal)
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o rdf.exe analysis/rdf.cpp $(str) $(mem) $(chem) $(math) $(struc) group.o hist.o sim.o format.o $(format_sim) pair.o $(signal) -lfftw3
cluster: $(str) $(mem) $(chem) $(math) $(struc) group.o hist.o sim.o format.o $(format_sim) pair.o 
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o cluster.exe analysis/cluster.cpp $(str) $(mem) $(chem) $(math) $(struc) group.o hist.o sim.o format.o $(format_sim) pair.o

#==== TOOLS ====

convert_struc: $(struc) $(format_struc) $(str) $(mem) $(util) $(chem) eigen.o parse.o format.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o convert_struc.exe format/convert_struc.cpp $(struc) $(format_struc) $(str) $(mem) $(util) $(chem) eigen.o parse.o format.o
convert_sim: $(struc) $(format_sim) $(str) $(mem) $(util) $(chem) eigen.o parse.o sim.o format.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o convert_sim.exe format/convert_sim.cpp $(format_sim) $(struc) $(str) $(mem) $(util) $(chem) eigen.o parse.o sim.o format.o
nnp_compute: $(nnp) nn.o $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) format.o
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o nnp_compute.exe nnp/nnp_compute.cpp $(nnp) nn.o $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) format.o
nnp_compute_sim: $(nnp) nn.o $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_sim) $(chem) $(struc) sim.o vasp_sim.o format.o
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o nnp_compute_sim.exe nnp/nnp_compute_sim.cpp $(nnp) nn.o $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_sim) $(chem) $(struc) sim.o format.o
intgrad: $(struc) $(chem) $(util) $(neural) $(basis) $(math) $(str) $(format) $(nnp) $(mem) optimize.o
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o intgrad.exe nnp/intgrad.cpp $(struc) $(chem) $(util) $(neural) $(basis) $(math) $(str) $(format) $(nnp) $(mem) optimize.o
plot_loss: $(nnp) pca.o nn.o $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format) $(chem) $(struc) optimize.o
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o plot_loss.exe nnp/plot_loss.cpp $(nnp) pca.o nn.o $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format) $(chem) $(struc) optimize.o
make_basis: $(str) $(basis) $(math) cutoff.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o make_basis.exe nnp/make_basis.cpp $(str) $(basis) $(math) cutoff.o
make_eigen: $(str) $(struc) $(format_struc) $(chem) $(mem) eigen.o format.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o make_eigen.exe nnp/make_eigen.cpp $(str) $(struc) $(format_struc) $(chem) $(mem) eigen.o format.o
make_eigen2: $(str) $(struc) $(format_struc) $(chem) $(mem) eigen.o format.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o make_eigen2.exe nnp/make_eigen2.cpp $(str) $(struc) $(format_struc) $(chem) $(mem) eigen.o format.o

#==== QE ====
avg_dos: $(string)
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o avg_dos.exe qe/avg_dos.cpp $(str)
