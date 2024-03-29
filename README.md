# AtomicNN - Atomic Nueral Network

Author: Mark J. DelloStritto

A code to train neural network potentials using periodic atomic structures and ab-initio references energies.  
Also included is a an extension to LAMMPS (https://lammps.sandia.gov/) which implements neural network potentials 
for molecular dynamics simulations.

## CODE ORGANIZATION

**MACHINE LEARNING**
* nn.hpp               - neural network
* batch.hpp            - organizing samples into batches
* data.hpp             - organizing training/validation/testing data
* nn_train.hpp         - nn - training

**NEURAL NETWORK POTENTIAL**
* nnp.hpp       - neural network potential
* cutoff.hpp    - distance cutoff
* atom.hpp      - stores atom properties
* nnp_train.hpp - nnp - training

**BASIS - RADIAL**
* basis_radial.hpp    - radial basis (table of radial symmetry functions)
* symm_radial.hpp     - radial symmetry function header
* symm_radial_g1.hpp  - "G1" symmetry function [1]
* symm_radial_g2.hpp  - "G2" symmetry function [1]
* symm_radial_t1.hpp  - "T1" symmetry function (tanh)

**BASIS - ANGULAR**
* basis_angular.hpp   - angular basis (table of angular symmetry functions)
* symm_angular.hpp    - angular symmetry function header
* symm_angular_g3.hpp - "G3" symmetry function [1]
* symm_angular_g4.hpp - "G4" symmetry function [1]

**STRUCTURE**
* atom_type.hpp - type of atom data
* cell.hpp      - unit cell
* cell_list.hpp - divides unit cell into grid
* neighbor.hpp  - nearest neighbor list
* thermo.hpp    - thermodynamic data
* structure.hpp - atomic trajectories/properties
* sim.hpp       - time series of structures

**FORMAT**
* format.hpp - stores possible formats
* vasp_struc.hpp   - read/write VASP files
* qe_struc.hpp     - read/write QE files
* xyz_struc.hpp    - read/write XYZ files
* cp2k_struc.hpp   - read/write CP2K files
* ann_struc.hpp    - read/write ANN files
* vasp_sim.hpp     - read/write VASP simulations

**MATH**
* const.hpp        - mathematical constants
* special.hpp      - special functions
* func.hpp         - function evalulation
* cmp.hpp          - comparison functions
* accumulator.hpp  - statistical accumulator
* eigen.hpp        - Eigen utilities
* lmat.hpp         - lower triangular matrix
* random.hpp       - random number generation

**OPTIMIZATION**
* optimize.hpp     - optimization

**CHEMISTRY**
* units.hpp   - physical units
* ptable.hpp  - physical constants
* ewald3D.hpp - ewald sums

**STRING**
* print.hpp     - utilities for formatted output
* string.hpp    - string utilities

**MEMORY**
* map.hpp       - mapping two different objects to each other
* serialize.hpp - serialization of complex objects

**UTILITY**
* compiler.hpp  - utilities for printing compiler information
* typedef.hpp   - global typdefs
* time.hpp      - program timing

**MULTITHREADING**
* parallel.hpp - utilities for thread distribution

## INSTALLATION

This code requires the Eigen Matrix Library (http://eigen.tuxfamily.org)

The Eigen library is a header library, and thus does not need to be compiled. 
The location of the Eigen library must be specified in the Makefile.
The compiler can be specified with COMP and supports COMP=gnu and COMP=intel

**Makefile options:**
* test_unit_optimize - unit test - optimization
* test_unit_nn       - unit test - neural network
* test_unit_nnp      - unit test - neural network potential
* test_unit_struc    - unit test - structure object
* make nn_fit    - make training binary for functions
* make nnp_train - make training binary for NNPs
* make nnp_compute - make utility for computing energies/forces of separate structures with an existing potential
* make nnp_compute_sim - make utility for computing energies/forces of a simulation with an existing potential
* make intgrad - make utility for computing interated gradients
* make clean   - removes all object files

## EXECUTION

The nnp training code is executed taking a single argument, the parameter file, and
all output is sent to standard output.  The parameter file provides all data files
necessary for training, valiadation, and testing as well as all parameters defining
the neural network potential and the training process.  The binary can be run in
parallel using MPI, with the type of MPI specified during compilation.
	
## TRAINING - DATA

### data - files:

The training, validation, and testing data are read from data files, each of which
contains a list of files providing structures and associated energies. The structures
listed in each data file are read and combined to form the total data set in each
category. Each data file can be individually classified as training, validation, or testing.
Although one can use a single data file for each category, it is recommended to split 
the data into several data files based on different constructions, symmetries, chemistries, etc.
The use of separate data files facilitates the organization and division of data into
different categories for training and analysis.

### data - training:

The data used for training the neural network potential. This data is the ONLY data 
which determines the gradient for optimization. The training data should comprise 
the bulk of the reference data.  

### data - validation:

The data used for determining the end of the optimization of the neural network training. 
Although a stopping criterion is implemented, one should monitor the validation error 
closely and end optimization when the change the in validation error is negligible or 
when the validation error is less than a threshold value. The validation data should be a 
small (~10-20%) fraction of the training data and should reflect a target of the 
neural network potential.

### data - test:

Optional data for testing.  When provided in training mode, energies 
and forces are computed for this data at the end of optimization. 
When provided in test mode, no optimization is performed, and energies 
and forces are computed for this data.

## NEURAL NETWORK POTENTIAL - SPECIFICATION

The neural network potential for a set of species is stored in one file, as although some aspects
of the potential are solely associated with each species (e.g. the basis), the neural network Hamiltonian
is defined by the combination of all species.

A given neural network potential can be thought of as combination of three distinct elements:

* vacuum energy of atom (sets cohesive energy)
* radial and angular basis (set of symmetry functions)
* atomic neural network

The basis can be the same for all atoms, as the neural networks for each species are 
completely independent.  That is, since the energy is defined by a combination of the basis and the 
neural network, the basis can be the same for all elements.  Alternatively, the basis can be 
dieffernet for each species, reflecting different local environments.

One should specify a set of symmetry functions which best reflect the atomic forces.
Essentially, one should ensure that there are a large number of symmetry functions with large gradients
at the interatomic distances and angles where the gradient of the energy is greatest.
As the interatomic forces are computed as sums of the gradients of the symmetry functions,
the symmetry functions should also resemble the integral of the expected interatomic forces.

## NEURAL NETWORK POTENTIAL - TRAINING

The data can be organized in any number of data files. These data files are simply 
lists of file names which will be loaded into memory. Each data file can contain any number of structure files. 
The file format specifies the format of the data files.  The same format must be used for all files.

The units define the physical constants used to interpret the input variables and data.
The unit systems follow the same conventions as the LAMMPS unit systems. 
Currently, only AU and METAL units are implemented.

Each species in the systems requires a vacuum energy (default: zero), a neural network geometry,
and a file specifying the basis for the species.  The vacuum energy is used to bias the output of the neural
network for each species and thus should be closely related to the energy of the given species in isolation.
Determining the vacuum energy can be challening due to changes in spin or charge in the solid
state compared to vacuum and thus requires care.  The neural network geometry is the number of nodes
of each hidden layer of the network, not including the input or output layer, which are by definition
the number of symmetry functions and one, respectively.  It is suggested that each layer of the network
be assigned a number of nodes which is half of the number of nodes of the previous layer, rounded up
to the nearest integer.

The names of the output potential file and the restart must be provided.  The potential and restart
file will be written every "nwrite" steps with the optimization step appended to the end of the file
name.  At the end of a succesful optimization, the potential file and restart file will be written
without any indices appended.

One can restart the training of a neural network by specifying the appropriate flag.  The potential
and optimization parameters will be read from the restart file, but the symmetry functions will
still be read from the given list of structures.  Thus, upon restarting, the symmetry functions
must be recomputed, and one can change the number or composition of the training data upon restarting,
though this is not recommended.  In order to elide the recomputation of the symmetry functions, one
can write the symmetry functions to binary files and then read the binary format when restarting.
The binary files are written in the same folders as the original structure files with the same name
but with a ".dat" extension.  Finally, note that one can alter some optimization parameters when 
restarting, such as the step size "gamma" or the current decay count "step".  It is thus recommended that
one have two parameter files, and initialization parameter file and a restart parameter file, where in the
restart parameter file any optimization parameters you do not want to alter when restarting are
either removed or commented out.

The cutoff radius sets the distance at which the potential goes to zero. The batch size 
specifies the size of the random collection of items in the training data used for training. 
Pre-conditioning shifts and normalizes the inputs of the neural network using the average 
and standard deviation of the inputs.  These values are then incorporated into the neural network, 
included as a "scaling layer" which stores the average and standard deviation of each node
over the training data.

## OPTIMIZATION

There are three basic types of optimization:

* stochastic descent (rprop)
* gradient descent (sdg,sdm,nag,adagrad,adadelta,adam,nadam)
* conjugate-gradient (bfgs)

All gradient descent algorithms require a descent parameter (gamma).  Many also require 
a memory parameter (eta) which adds different forms of momentum to the descent by 
combining the current gradient with past gradients.  All gradient descent algorithms also
offer a decay parameter which decreases the descent parameter (gamma) over time.  The decay
parameter should be an floating point number which sets the time constant of the decay function.
Several different decay functions are available, including: exponential, inverse, power, and sqrt.
For exponential decay, at step n, gamma will be reduced by a factor of exp(-alpha n).
For inverse decay, at step n, gamma will be computed as gamma/(1+alpha n).
For power decay, at step n, gamma will be computed as gamma/(1+alpha n)^(p) for a given power p.
For sqrt decay, at step n, gamma will be computed as gamma/(1+alpha n)^(1/2).
When using a gradient descent method, it is recommended to use a relatively small but non-zero
decay parameter, typically on the order of the inverse number of steps taking before stopping (max_iter).

The SDM algorithm (Steepest Descent with Momentum) is an effective but slow method to train a neural network.
It is suggested one use a small gamma (around 1e-3) and a large momentum term (e.g. 0.9).  The advantage of
SDM is that it can yield very small errors, but requires a very long time to train and thus a very slow
decay of the optimization step.  

The ADAM algorithm is also effective, though it requires a good choice of gamma, and is 
one of the best gradient descent methods available. It is recommended that one use a small gamma (around 1e-3) 
and a small batch size (typically 32-128) , as the smaller 
batch size results in a stochastic gradient direction which tends to benefit 
training performance. For more difficult systems, one might consider using 
the NADAM algorithm, with is ADAM with a Nesterov-style momentum term which results in a more stable 
gradient in the optimization step.

All other algorithms work to some degree, but are not recommended.

When restarting, all optimization parameters (lambda, gamma, decay, etc.) are overwritten by the 
values provided in the parameter file. There are times when this is not desirable, i.e. when restarting 
the optimization when using a decay schedule for the optimization step (gamma). In this case, if one removes
or comments out a given parameter, then the optimization parameters are taken from the restart file.

## BASIS

A "basis" is defined as the collection of symmetry functions used as inputs for a given species.  A basis
is comprised of both radial and angular symmetry functions for all species in a given potential.
For example, if one is constructing a NNP for Ar, Kr, and Xe, each of these species will have its own basis.
The basis for Ar will include radial functions for neighboring Ar, Kr, and Xe atoms, as well as angular
functions for all unique combinations of pairs of Ar, Kr, and Xe atoms.

The basis is defined in a file which lists the central species at the top followed by a list of lists
of symmetry functions corresponding to each species or pair of species.

The choice of the word "basis" is intentional, as the basis in quantum chemistry serves much the same
purpose as a basis in NNPs.  In quantum mechanics, the Hamiltonian is an operator which thus has a 
concrete functional form only when paired with a given wavefunction.  The choice of basis thus defines
the possible energies of a given system and thus must be carefully choice to represent the possible states
a given molecule might occupy.  If one includes only s-orbitals for H, it will impossible to correctly
predict the polarizability.  In a similar way a NN is an operator which is well defined only
when paired with a given set of inputs.  The symmetry functions define the inputs and thus define the
possible energies a NN could produce.  If one includes only radial symmetry functions the impossibility
of uniquely assigning all possible atomic configurations to lists of interatomic distances will
make it impossible to generate an accurate NNP.

When specifying each set of symmetry functions, one must include the cutoff distance, the cutoff function,
the type of symmetry function, and the normalization scheme.  The normalization scheme can be either 
UNIT or VOL, corresponding to "unit" normalization or "volume" normalization, respectively.  For unit 
normalization, the inputs to the neural network are not normalized at all, making them suitable only for
activation functions whose gradients remain constant as the input goes to infinity.  For volume normalization,
the inputs are scaled by a constant related to the cutoff volume, guaranteeing they will lie within a 
small range, thereby making it possible to use activation functions whose gradients go to zero quickly
as the the inputs move away from zero.

## WRITING

Every "n_write" iterations the neural network potential is written to file, and a restart
file is written.  The optimization count and all optimization data is retained in the
restart file, allowing for consistent restarting from a completed run.  When the max
number of iterations is reached, the final potential and restart files are written,
allowing for rapid restart without moving or copying files. Note that the max iterations 
applies only to the current run, not to the total optimization count.  The restart file
is taken from the input file and given an extenion ".$i" where $i is the current optimization 
count, and the restart file is written without an extension when the max number of iterations is reached.
The nueral network potential is written every "nsave" ierations again with a ".$i" extension, 
with the final potential written at the end of the optimization (after max_iter is reached).

The error for the training and validation data is written to file very "n_print" timesteps.
The error is accumulated in the same file when restarting, appending the new iteration 
count and error to the end of the file.

There are several flags for writing the the basis set, energies, and forces to file at the end of optimization.
The energies and forces are guaranteed to be written in the same order as they are provided in the data files.

## PARALLELISM

The code used for training NNPs can be run on multiple processors using the MPI protocol.  
The use of multiple threads when training a NNP is difficult though, as although the NNP is trained on the
energies of structures, the computation involves individual atoms.  This then requires two levels of parellelism
in order to take full advantage of the available hardware.

The issue with parallelism is greatest when one is training on structure with many atoms or with structures which 
all have different numbers of atoms.  As a pathological example, imagine that the training data is composed of
100 structures, 99 of which have 1 atom per unit cell and one of which has 1000 atoms per unit cell.  In order to 
train on a given structure, one must compute the total energy of the structure, which in turn requires computing 
the energy of all atoms in the structure.  In this example, one could easily distribute the calculation to 100 
processors such that each processor computed the energy of a single structure.  However, the single structure with 
1000 atoms would dominate the computing time, leaving most processors idle most of the time.  On the other hand, 
even if all structures have the same number of atoms but a large number, this can significantly slow down optimization.
Generally, batches much larger than 32 are not beneficial, and so a naive parallel implementation where each processor 
computes the energy of a single structure would limit one to only a little more than 32 processors.

To overcome these issues, we have implemented a two-tiered multithreading strategy whereby a group of processors 
computes the energy of a single structure.  During initialization, the processors are split into NBATCH groups where 
NBATCH is the size of the batch, typically on the order of 32.  The training and validation data is then split as equally 
as possible between the NBATCH groups.  During optimization, each group of processors draws one random structure from 
its own set, and the atoms of that structure are then divided as equally as possible between all processors within the 
group.  The total energy of the structure is then summed over the local group and thus the contribution of the structure 
to the error and gradient is computed.  Finally, the error and gradients are summed over each of the NBATCH groups and 
the chosen algorithm is used to update the parameters of the NNPs.  Thus, for a batch size of NBATCH and a typical number 
of atoms NATOM for each structure, one can use a maximum of NBATCH x NATOM processors without losses in efficiency.
Note that using more processors in a local group than there are atoms in a structure does not impact the accuracy of 
the calculation, only the computational efficiency.  We illustrate the parallelization strategy in the figure below.

![Code Layout](/parallel.png)

## MOLECULAR DYNAMICS

An add-on package for LAMMPS is included so that the neural network potential may be used
for molecular dynamics simulations.  The potential reads in the same neural network potential
files output by the training code.
Installation of the package is documented in the README files in USER-ANN

## FILE FORMATS

A number of different file formats can be read to obtain atomic structures, forces, and energies.

**FORMATS**
* VASP xml - XML file produced by VASP which contains all relevant information, including positions, forces, and the energy.
* QE output - Quantum Espresso does not aggregate all relevant information in a single file, however, all relevant information 
is included in the standard output. Thus, if one pipes the output into a file, this can be read for training.
* CP2K output - A CP2K output file (only if the type of calculation is ENERGY)
* XYZ files - In addition to the standard XYZ format, one must also included the energy and lattice constants in the xyz file.
Thus, in the commment section, one must include a given comment, followed by the energy, followed by the x, y, and z lattice constants.
(Currently, only orthorhombic cells are implemented).
* ANN file - special format designed for this code
* BINARY file - a binary file storing the cell, energy, positions, charges (if specified), and symmetry functions of a given structure

## PARAMETER FILE REFERENCE

Unless specified otherwise, white space is used as a delimiter for all parameters.  The amount or type of white space 
(e.g. spaces or tabs) does not impact performance.  

Any characters to the right of a comment character "#" are ignored.

All strings are case insensitive with the exception of file names.  

Boolean values may be specified as true using "1", "true", or "t", and they may be specified as false using "0", "false", or "f".

### GENERAL PARAMETERS

* __seed__ (integer) - The seed for the random number generator. If negative, the seed is taken from the current time (default).
* __mode__ (string) - The execution mode, either "train" or "test".
* __format__ (string) - The format of the structure files (must all be the same format). Possible values:
	* __VASP_XML__ - VASP XML file
	* __QE__ - Quantum Espresso output 
	* __CP2K__ - CP2K output
	* __XYZ__ - modified XYZ format
	* __ANN__ - ANN format
	* __BINARY__ - binary file
* __units__ (string) - The unit system (follows LAMMPS conventions). Possible values:
	* __au__ - atomic units
	* __metal__ - metal units
* __charge__ (boolean) - Whether the atoms have a constant charge. If "true," then ewald energies are computed for each
structure and removed, allowing one to train on the remainder.

### PARAMETERS - DATA

* __data_train__ (string) - Data file containing files for training.
* __data_val__ (string) - Data file containing files for validation.
* __data_test__ (string) - Data file containing files for testing.

### PARAMETERS - ATOMS

Each __atom__ entry can be followed by a species name, a flag designating a property, and finally the associated data.  The possible flags and values follow below:

* __mass__ (float) - The mass of the atom in mass units, provided for convenience and clarity but most of the time not used.
* __energy__ (float) - The vacuum energy of the atom in energy units.
* __charge__ (float) - The charge of the atom in charge units (used only if "charge" is true).
* __basis__ (string) - The name of the file containing the basis for a given atom.
* __nh__ (integers) - Array of integers specifying the number of nodes per hidden layer from left (closest to input) to right (closest to output).

### READ/WRITE

* __file_ann__ (string) - The name of the file where the neural network potential will be stored.
* __file_restart__ (string) - The name of the file where the restart information will be stored.
* __restart__ (boolean) - Whether the optimization will be restarted from a restart file.
* __norm__ (boolean) - Whether to normalize the energies by the number of atoms when writing.

### OPTIMIZATION

* __algo__ - (string) The optimization algorithm.
	* __sgd__ - steepest gradient descent
	* __sdm__ - steepest descent with momentum
	* __nag__ - Nesterov accelerated gradient
	* __adagrad__ - ADAGRAD
	* __adadelta__ - ADADELTA
	* __rmsprop__ - RMSPROP
	* __adam__ - ADAM
	* __amsgrad__ - AMSGRAD
	* __bfgs__ - BFGS
	* __rprop__ - RPROP
* __opt_val__ - (string) Optimization value, determining how to stop optimization.
	* __ftol_abs__ - Stop when the absolute value of the objective function is below the tolerance.
	* __ftol_rel__ - Stop when the change in the value of the objective function is below the tolerance.
	* __xtol_rel__ - Stop when the change in the parameters is below the tolerance.
* __loss__ - (string) Defines the type of loss function.
	* __mse__ - mean squared error
	* __mae__ - mean absolute error
	* __huber__ - Huber loss function
* __decay__ - (string) Decay function.
	* __const__ - Constant decay; no change in the step.
	* __exp__ - Exponential decay.
	* __inv__ - Inverse function decay.
	* __sqrt__ - Inverse square root decay.
	* __pow__ - Inverse of a power law decay.
	* __step__ - Stepwise exponential decay.
* __tol__ - (float) Optimization tolerance, determining the stopping point, must be positive.
* __max_iter__ - (int) The maximum number of iterations in a given optimization run, must be greater than zero.
* __n_print__ - (int) Print the error to standard output every "n_print" steps.
* __n_write__ - (int) Write the restart file every "n_write" steps.
* __gamma__ - (float) Optimization step size (gradient descent), must be greater than zero (overrides restart)
* __eta__ - (float) Memory term for gradient descent methods which use momentum, must be between zero and one (overrides restart).
* __alpha__ - (int) Decay rate for the optimization step __gamma__ (overrides restart).
* __pow__ - (float) Sets the power for power law decay of the optimization step (overrides restart).
* __step__ - (int) Sets the step of the decay function, the optimization count is not affected (overrides restart).
* __labmda__ - (float) Parameter setting the influence of weight decay (overrides restart) (default: 0.0).
* __mix__ - (float) mixing parameter for Polyak-Ruppert averaging (0 - no mixing) (default: 0.0)
* __huberw__ - (float) width of the Huber loss function (default: 1.0)

### NN POTENTIAL

* __r_cut__ - (float) The cutoff radius of the potential in distance units, must be greater than zero.
* __transfer__ - (string) The name of the transfer function. Possible values:
	* __tanh__ - hyperbolic tanh function
	* __sigmoid__ - sigmoid function
	* __arctan__ - normalized ArcTan function
	* __isru__ - inverse square root unit
	* __softsign__ - softsign function
	* __softplus__ - softplus function (with ln(2) substracted)
	* __relu__ - rectified linear function
	* __elu__ - exponential linear function
	* __gelu__ - gaussian error linear function
	* __swish__ - SWISH activation function
	* __mish__ - MISH activation function
	* __tanhre__ - TANHRE activation function
	* __lin__ - linear function
* __dist__ - (string) Distribution used to generate the random initial values of the neural network.  Possible values:
	* __normal__ - Normal distribution (default).
	* __exp__ - Exponential distribution.
* __sigma__ - Width of the distribution used to generate the random initial values of the neural network.
* __init__ - (string) Initialization method used to normalize the random initial values of the neural network. Possible values:
	* __rand__ - No normalization, all values are simply drawn from a given random distribution.
	* __xavier__ - The random distribution for each layer is normalized by the inverse of the square root of the size of the previous layer.
	* __he__ - The random distribution for each layer is normalized by the inverse of twice the square root of the size of the previous layer.
	* __mean__ - Same as "he", but the denominator is the root of the average of the previous and next layers.
* __n_batch__ - (int) Number of structures in the batch, must be less than the total number of structures owned by a process
* __pre_cond__ - (bool) Whether to precondition the inputs by shifting to zero using the average and scaling the magnitude by the inverse of the standard deviation.
* __calc_force__ - (bool) Whether to compute the forces on the atoms at the end of optimization. An expensive calculation that is often unnecessary, though useful for testing purposes.

### OUTPUT

* __write_basis__  - Whether to write the basis as a function of distance/angle to file.
* __write_energy__ - Whether to write the training/validation/testing energies to file.
* __write_ewald__  - Whether to write the training/validation/testing ewald energies to file.
* __write_force__  - Whether to write the force on each atom to file.
* __write_input__  - Whether to write the inputs to file.
* __write_corr__   - Whether to write the input correlations to file.
* __write_symm__   - Whether to write the symmetry function values to binary files for restarting.

## REFERENCES

[1] Behler, J. Constructing High-Dimensional Neural Network Potentials: A Tutorial Review. Int. J. Quantum Chem. 2015, 115 (16), 1032–1050. https://doi.org/10.1002/qua.24890.
