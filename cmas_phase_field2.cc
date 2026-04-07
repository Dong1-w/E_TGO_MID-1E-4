/* -----------------------------------------------------------------------------
*
* CMAS Phase Field Corrosion Model (Parallel Version) - Revised
* 
* Changes Summary:
* 1. Concentration system expanded to 6 variables: c1, c2, c3, c4, n, N
*    - c1-c4: CMAS components (Diffuses in TC + TGO)
*    - n: Chemical corrosion (Only in TC, causes expansion)
*    - N: Mechanical degradation (Only in TGO, reduces strength, no expansion)
* 2. Elasticity: Eigenstrain only applied in TC layer based on 'n'.
* 3. Phase Field: Critical stress in TGO defined by 'N' (sigma * sqrt(1-N)).
*
* -----------------------------------------------------------------------------
*/

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/grid/grid_in.h>

#define FORCE_USE_OF_TRILINOS
namespace LA
{
#if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) && \
!(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
using namespace dealii::LinearAlgebraPETSc;
#  define USE_PETSC_LA
#elif defined(DEAL_II_WITH_TRILINOS)
using namespace dealii::LinearAlgebraTrilinos;
#else
#  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
} // namespace LA

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>

namespace CMASPhaseField
{
using namespace dealii;

// Domain parameters (from COMSOL model, scaled to mm)
namespace Domain
{
constexpr double x_min = 0.0;
constexpr double x_max = 8.1;     // mm
constexpr double y_min = 0.0;
constexpr double y_max = 1.25;     // mm

// Layer boundaries (y coordinates in mm)
constexpr double y_sub_top = 0.9;    // SUB layer top
constexpr double y_bc_top = 1.0;      // BC layer top
constexpr double y_bc_bottom = 0.9;      // BC layer bottom
constexpr double y_tgo_bottom = 1.0; // TGO layer bottom
constexpr double y_tgo_top = 1.04;    // TGO layer top
constexpr double y_tc_bottom = 1.04;  // TC layer bottom
constexpr double y_tc_top = 1.25;      // TC layer top

// CMAS application region at top boundary
constexpr double x_cmas_min = 3;    // mm
constexpr double x_cmas_max = 5;    // mm
constexpr double tgo_transition_width = 0.5; // mm, smoothing band width on each side of CMAS segment
}

// Material parameters
namespace Material
{
// TC layer
constexpr double E_TC = 40e9 * 1e-6;        // 40 GPa -> 40000 MPa
constexpr double nu_TC = 0.12;
constexpr double G_TC = 50.0 * 1e-3;        // 0.05 N/mm
constexpr double sigma_TC = 1000e6 * 1e-6;  // 1000 MPa

// TGO layer
constexpr double E_TGO_CORRODED = 1e-4;      // MPa (CMAS-corroded middle segment)
constexpr double E_TGO_INTACT = 40e9 * 1e-6; // 40 GPa (remaining TGO segment)
constexpr double nu_TGO = 0.12;
constexpr double G_TGO = 40 * 1e-3;       // Fracture-energy-like parameter (N/mm), not elastic shear modulus
constexpr double sigma_TGO = 40e6 * 1e-6;   // 40 MPa

// BC layer
constexpr double E_BC = 200e9 * 1e-6;       // 200 GPa
constexpr double nu_BC = 0.3;

// SUB layer
constexpr double E_SUB = 250e9 * 1e-6;      // 250 GPa
constexpr double nu_SUB = 0.3;

// CMAS properties
constexpr double rho_CMAS = 2630.0;  // g/mm³
constexpr double porosity = 0.16;

// Diffusion coefficients
constexpr double D_intact = 8.84e-13 * 1e6; // 8.84e-7 mm²/s 8.84e-13
constexpr double D_damaged = 4e-9 * 1e6;    // 4e-3 mm²/s

// Reaction rate constants
constexpr double k_reaction1 = 1.914e-8;  // TC layer reaction (for n)
constexpr double k_reaction2 = 2e-5;      // TGO degradation rate (xi for N)

// Phase field length scale
constexpr double length_scale = 2e-5 * 1e3; // 0.02 mm

// Viscous regularization parameter
constexpr double viscosity = 1500; 
}

namespace TimeStep
{
constexpr double total_time = 360.0 * 60.0;
constexpr double dt_initial = 0.01;
constexpr double dt_max = 20;
constexpr double dt_min = 1e-6;
constexpr int output_interval = 3;
}

// Numerical constants for Newton-Raphson solver
namespace NumericalConstants
{
constexpr double division_epsilon = 1e-10;  // Small value to avoid division by zero
constexpr double zero_threshold = 1e-15;    // Threshold for considering values as zero
constexpr double damage_threshold_tolerance = 1e-12;  // Tolerance for damage initiation check
constexpr double determinant_threshold = 1e-10;  // Threshold for checking if determinant J is near zero

// Newton-Raphson solver parameters for nonlinear elasticity
constexpr unsigned int max_newton_elasticity_iterations = 50;
constexpr double newton_elasticity_tolerance = 1e-6;
constexpr double newton_elasticity_linear_tolerance = 1e-8;
}

namespace MolarMass
{
constexpr double Fe2O3 = 160.0;
constexpr double MgO = 40.0;
constexpr double Al2O3 = 102.0;
constexpr double SiO2 = 60.0;
}

inline double lambda(double E, double nu)
{
return E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
}

inline double mu(double E, double nu)
{
return E / (2.0 * (1.0 + nu));
}

enum MaterialLayer { SUB, BC, TGO, TC };

MaterialLayer get_layer(const double y)
{
if (y <= Domain::y_sub_top) return SUB;
else if (y <= Domain::y_bc_top && y > Domain::y_sub_top) return BC;
else if (y <= Domain::y_tgo_top && y > Domain::y_tgo_bottom) return TGO;
else return TC;
}

inline double smoothstep(const double t_in)
{
const double t = std::max(0.0, std::min(1.0, t_in));
return t * t * (3.0 - 2.0 * t);
}

inline double get_tgo_E_base(const Point<2> &p)
{
if (!(p[1] > Domain::y_tgo_bottom && p[1] <= Domain::y_tgo_top))
return Material::E_TGO_INTACT;

const double transition_width = Domain::tgo_transition_width;
const double x_left_start = Domain::x_cmas_min - transition_width;
const double x_left_end = Domain::x_cmas_min;
const double x_right_start = Domain::x_cmas_max;
const double x_right_end = Domain::x_cmas_max + transition_width;

const double x = p[0];
if (x <= x_left_start || x >= x_right_end)
return Material::E_TGO_INTACT;

if (x >= x_left_end && x <= x_right_start)
return Material::E_TGO_CORRODED;

if (x > x_left_start && x < x_left_end)
{
const double t = smoothstep((x - x_left_start) / transition_width);
return Material::E_TGO_INTACT + (Material::E_TGO_CORRODED - Material::E_TGO_INTACT) * t;
}

const double t = smoothstep((x - x_right_start) / transition_width);
return Material::E_TGO_CORRODED + (Material::E_TGO_INTACT - Material::E_TGO_CORRODED) * t;
}

inline double get_layer_base_E(const Point<2> &p)
{
switch (get_layer(p[1]))
{
case TC:  return Material::E_TC;
case TGO: return get_tgo_E_base(p);
case BC:  return Material::E_BC;
case SUB: return Material::E_SUB;
}
return Material::E_TC;
}

// MODIFIED: Added parameter N for TGO degradation
double get_E(const Point<2> &p, double phi = 0.0, double n = 0.0, double N = 0.0)
{
MaterialLayer layer = get_layer(p[1]);
double E_base = get_layer_base_E(p);
double omega = 1.0;

switch(layer)
{
case TC:
{
double b0 = Material::length_scale;
double Kphi = 4.0 * E_base * Material::G_TC / 
(3.14159 * b0 * Material::sigma_TC * Material::sigma_TC);
double Nphi = (1.0 - phi) * (1.0 - phi);
double Dphi = Nphi + Kphi * phi * (1.0 - phi / 2.0);
omega = Nphi / (Dphi);
}
break;
case TGO:
{
double b0 = Material::length_scale;
// MODIFIED: TGO degradation depends on N, not n
// sigma_eff = sigma_TGO * sqrt(1 - N)
double degradation_factor = std::pow(1.0 - N, 1);
if (degradation_factor < 0.0) degradation_factor = 0.0;

double G_eff = Material::G_TGO * degradation_factor; // G scales with (1-N)? assuming similar to n
double sigma_eff = Material::sigma_TGO * std::sqrt(degradation_factor);

if (sigma_eff < 1e-6) sigma_eff = 1e-6;

double Kphi = 4.0 * E_base * G_eff / 
(3.14159 * b0 * sigma_eff * sigma_eff);
double Nphi = (1.0 - phi) * (1.0 - phi);
double Dphi = Nphi + Kphi * phi * (1.0 - phi / 2.0);
omega = Nphi / (Dphi );
}
break;
case BC:
break;
case SUB:
break;
}
return E_base * omega;
}

double get_nu(const Point<2> &p)
{
MaterialLayer layer = get_layer(p[1]);
switch(layer)
{
case TC:  return Material::nu_TC;
case TGO: return Material::nu_TGO;
case BC:  return Material::nu_BC;
case SUB: return Material::nu_SUB;
}
return Material::nu_TC;
}

double get_diffusivity(double phi)
{
double phi10 = std::pow(phi, 10);
return (1.0 - phi10) * Material::D_intact + phi10 * Material::D_damaged;
}

// MODIFIED: 6 components (c1, c2, c3, c4, n, N)
template <int dim>
class CMASBoundaryValues : public Function<dim>
{
public:
CMASBoundaryValues() : Function<dim>(6) {} 

virtual double value(const Point<dim> &p,
const unsigned int component = 0) const override
{
if (std::abs(p[1] - Domain::y_tc_top) < 1e-6 &&
p[0] >= Domain::x_cmas_min && p[0] <= Domain::x_cmas_max)
{
double total_mass = 35.0 * MolarMass::Fe2O3 + 10.0 * MolarMass::MgO + 
7.0 * MolarMass::Al2O3 + 48.0 * MolarMass::SiO2;
double c_total = Material::rho_CMAS * Material::porosity / total_mass;

switch (component)
{
case 0: return 35.0 * c_total; // c1
case 1: return 10.0 * c_total; // c2
case 2: return 14.0 * c_total; // c3
case 3: return 48.0 * c_total; // c4
case 4: return 0.0;            // n
case 5: return 0.0;            // N
}
}
return 0.0;
}
};

template <int dim>
class CMASProblem
{
public:
CMASProblem();
void run();

private:
void setup_mesh();
void setup_system();
void setup_constraints();
void setup_matrices();
void assemble_concentration_system();
void assemble_phase_field_system();
void assemble_phase_field_newton_system();  // Newton-Raphson Jacobian and residual
void assemble_elasticity_system();
void solve_concentration();
void solve_phase_field();
void solve_phase_field_newton();  // Newton-Raphson solver
void solve_elasticity();
void compute_principal_stress();
double compute_time_step_size();
double compute_max_tgo_stress();
void monitor_tc_top_midpoint_displacement();  // Monitor displacement at TC layer top midpoint
void output_results(const unsigned int step);
void update_history_field();

MPI_Comm mpi_communicator;
parallel::distributed::Triangulation<dim> triangulation;

// Concentration system (Size 6)
FESystem<dim> fe_concentration;
DoFHandler<dim> dof_handler_concentration;
IndexSet locally_owned_dofs_concentration;
IndexSet locally_relevant_dofs_concentration;
AffineConstraints<double> constraints_concentration;
LA::MPI::SparseMatrix system_matrix_concentration;
LA::MPI::Vector locally_relevant_solution_concentration;
LA::MPI::Vector completely_distributed_solution_concentration;
LA::MPI::Vector old_solution_concentration;
LA::MPI::Vector system_rhs_concentration;

// Phase field system
FE_Q<dim> fe_phase_field;
DoFHandler<dim> dof_handler_phase_field;
IndexSet locally_owned_dofs_phase_field;
IndexSet locally_relevant_dofs_phase_field;
AffineConstraints<double> constraints_phase_field;
LA::MPI::SparseMatrix system_matrix_phase_field;
LA::MPI::Vector locally_relevant_solution_phase_field;
LA::MPI::Vector completely_distributed_solution_phase_field;
LA::MPI::Vector old_solution_phase_field;
LA::MPI::Vector system_rhs_phase_field;
LA::MPI::Vector newton_update_phase_field;  // Newton-Raphson update vector

LA::MPI::Vector history_field;
std::vector<std::vector<double>> quadrature_point_history;

// Elasticity system
FESystem<dim> fe_elasticity;
DoFHandler<dim> dof_handler_elasticity;
IndexSet locally_owned_dofs_elasticity;
IndexSet locally_relevant_dofs_elasticity;
AffineConstraints<double> constraints_elasticity;
LA::MPI::SparseMatrix system_matrix_elasticity;
LA::MPI::Vector locally_relevant_solution_elasticity;
LA::MPI::Vector completely_distributed_solution_elasticity;
LA::MPI::Vector system_rhs_elasticity;

Vector<double> principal_stress_1;
Vector<double> principal_stress_2;
Vector<double> principal_stress_3;
Vector<double> stress_xx;
Vector<double> stress_yy;
Vector<double> stress_xy;
std::vector<std::pair<double, std::string>> times_and_names;

double time;
double time_step;
unsigned int timestep_number;

ConditionalOStream pcout;
TimerOutput computing_timer;
};

template <int dim>
CMASProblem<dim>::CMASProblem()
: mpi_communicator(MPI_COMM_WORLD)
, triangulation(mpi_communicator)
, fe_concentration(FE_Q<dim>(1), 6) // MODIFIED: Size 6 for [c1,c2,c3,c4,n,N]
, dof_handler_concentration(triangulation)
, fe_phase_field(1)
, dof_handler_phase_field(triangulation)
, fe_elasticity(FE_Q<dim>(1), dim)
, dof_handler_elasticity(triangulation)
, time(0.0)
, time_step(TimeStep::dt_initial)
, timestep_number(0)
, pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
, computing_timer(mpi_communicator, pcout, TimerOutput::never, TimerOutput::wall_times)
{
}

template <int dim>
void CMASProblem<dim>::setup_mesh()
{
TimerOutput::Scope ts(computing_timer, "setup_mesh");
    pcout << "Generating full-domain mesh..." << std::endl;

    const Point<dim> p1(Domain::x_min, Domain::y_min);
    const Point<dim> p2(Domain::x_max, Domain::y_max);
    constexpr unsigned int n_cells_x = 81;
    constexpr unsigned int n_cells_y = 125;
std::vector<unsigned int> repetitions(dim);
repetitions[0] = n_cells_x;
repetitions[1] = n_cells_y;
GridGenerator::subdivided_hyper_rectangle(triangulation, repetitions, p1, p2, true);

constexpr double boundary_id_tolerance = 1e-10;
for (const auto &cell : triangulation.active_cell_iterators())
for (const auto face_no : cell->face_indices())
if (cell->face(face_no)->at_boundary())
{
const Point<dim> c = cell->face(face_no)->center();
if (std::abs(c[0] - Domain::x_min) < boundary_id_tolerance)
cell->face(face_no)->set_boundary_id(3); // left
else if (std::abs(c[0] - Domain::x_max) < boundary_id_tolerance)
cell->face(face_no)->set_boundary_id(4); // right
          else if (std::abs(c[1] - Domain::y_max) < boundary_id_tolerance)
            cell->face(face_no)->set_boundary_id(1); // top CMAS
          else if (std::abs(c[1] - Domain::y_min) < boundary_id_tolerance)
            cell->face(face_no)->set_boundary_id(0); // bottom
}

pcout << "  Mesh generated. Active cells: " << triangulation.n_active_cells() << std::endl;
}

template <int dim>
void CMASProblem<dim>::setup_system()
{
// Phase 1 of 2: Distribute DOFs and extract index sets (needed for setup_constraints)
// Note: setup_matrices() must be called after setup_constraints() to complete the setup
TimerOutput::Scope ts(computing_timer, "setup_system");

dof_handler_concentration.distribute_dofs(fe_concentration);
locally_owned_dofs_concentration = dof_handler_concentration.locally_owned_dofs();
locally_relevant_dofs_concentration = DoFTools::extract_locally_relevant_dofs(dof_handler_concentration);

dof_handler_phase_field.distribute_dofs(fe_phase_field);
locally_owned_dofs_phase_field = dof_handler_phase_field.locally_owned_dofs();
locally_relevant_dofs_phase_field = DoFTools::extract_locally_relevant_dofs(dof_handler_phase_field);

dof_handler_elasticity.distribute_dofs(fe_elasticity);
locally_owned_dofs_elasticity = dof_handler_elasticity.locally_owned_dofs();
locally_relevant_dofs_elasticity = DoFTools::extract_locally_relevant_dofs(dof_handler_elasticity);

// Initialize stress vectors
principal_stress_1.reinit(triangulation.n_locally_owned_active_cells());
principal_stress_2.reinit(triangulation.n_locally_owned_active_cells());
principal_stress_3.reinit(triangulation.n_locally_owned_active_cells());
stress_xx.reinit(triangulation.n_locally_owned_active_cells());
stress_yy.reinit(triangulation.n_locally_owned_active_cells());
stress_xy.reinit(triangulation.n_locally_owned_active_cells());

{
const unsigned int n_q_points = QGauss<dim>(fe_phase_field.degree + 1).size();
quadrature_point_history.clear();
quadrature_point_history.resize(triangulation.n_locally_owned_active_cells(), std::vector<double>(n_q_points, 0.0));
}
}

template <int dim>
void CMASProblem<dim>::setup_matrices()
{
// Phase 2 of 2: Set up vectors, sparsity patterns, and matrices
// This must be called after setup_constraints() because the sparsity pattern
// needs to include entries for the constraint couplings
TimerOutput::Scope ts(computing_timer, "setup_matrices");

// Concentration system vectors and matrix
locally_relevant_solution_concentration.reinit(locally_owned_dofs_concentration, locally_relevant_dofs_concentration, mpi_communicator);
completely_distributed_solution_concentration.reinit(locally_owned_dofs_concentration, mpi_communicator);
old_solution_concentration.reinit(locally_owned_dofs_concentration, locally_relevant_dofs_concentration, mpi_communicator);
system_rhs_concentration.reinit(locally_owned_dofs_concentration, mpi_communicator);

DynamicSparsityPattern dsp_c(locally_relevant_dofs_concentration);
DoFTools::make_sparsity_pattern(dof_handler_concentration, dsp_c, constraints_concentration, false);
SparsityTools::distribute_sparsity_pattern(dsp_c, dof_handler_concentration.locally_owned_dofs(), mpi_communicator, locally_relevant_dofs_concentration);
system_matrix_concentration.reinit(locally_owned_dofs_concentration, locally_owned_dofs_concentration, dsp_c, mpi_communicator);

// Phase field system vectors and matrix
locally_relevant_solution_phase_field.reinit(locally_owned_dofs_phase_field, locally_relevant_dofs_phase_field, mpi_communicator);
completely_distributed_solution_phase_field.reinit(locally_owned_dofs_phase_field, mpi_communicator);
old_solution_phase_field.reinit(locally_owned_dofs_phase_field, locally_relevant_dofs_phase_field, mpi_communicator);
system_rhs_phase_field.reinit(locally_owned_dofs_phase_field, mpi_communicator);
newton_update_phase_field.reinit(locally_owned_dofs_phase_field, mpi_communicator);
history_field.reinit(locally_owned_dofs_phase_field, mpi_communicator);

DynamicSparsityPattern dsp_p(locally_relevant_dofs_phase_field);
DoFTools::make_sparsity_pattern(dof_handler_phase_field, dsp_p, constraints_phase_field, false);
SparsityTools::distribute_sparsity_pattern(dsp_p, dof_handler_phase_field.locally_owned_dofs(), mpi_communicator, locally_relevant_dofs_phase_field);
system_matrix_phase_field.reinit(locally_owned_dofs_phase_field, locally_owned_dofs_phase_field, dsp_p, mpi_communicator);

// Elasticity system vectors and matrix
locally_relevant_solution_elasticity.reinit(locally_owned_dofs_elasticity, locally_relevant_dofs_elasticity, mpi_communicator);
completely_distributed_solution_elasticity.reinit(locally_owned_dofs_elasticity, mpi_communicator);
system_rhs_elasticity.reinit(locally_owned_dofs_elasticity, mpi_communicator);

DynamicSparsityPattern dsp_e(locally_relevant_dofs_elasticity);
DoFTools::make_sparsity_pattern(dof_handler_elasticity, dsp_e, constraints_elasticity, false);
SparsityTools::distribute_sparsity_pattern(dsp_e, dof_handler_elasticity.locally_owned_dofs(), mpi_communicator, locally_relevant_dofs_elasticity);
system_matrix_elasticity.reinit(locally_owned_dofs_elasticity, locally_owned_dofs_elasticity, dsp_e, mpi_communicator);
}

template <int dim>
void CMASProblem<dim>::setup_constraints()
{
constraints_concentration.clear();
constraints_concentration.reinit(locally_relevant_dofs_concentration);
DoFTools::make_hanging_node_constraints(dof_handler_concentration, constraints_concentration);

// MODIFIED: Mask for 6 components
CMASBoundaryValues<dim> cmas_bc;
std::map<types::boundary_id, const Function<dim> *> boundary_functions;
boundary_functions[1] = &cmas_bc;

std::vector<bool> component_mask(6, false);
component_mask[0] = true; component_mask[1] = true; 
component_mask[2] = true; component_mask[3] = true;
// n (4) and N (5) are not Dirichlet constrained at boundary usually

VectorTools::interpolate_boundary_values(dof_handler_concentration, boundary_functions, constraints_concentration, ComponentMask(component_mask));
constraints_concentration.close();

constraints_phase_field.clear();
constraints_phase_field.reinit(locally_relevant_dofs_phase_field);
DoFTools::make_hanging_node_constraints(dof_handler_phase_field, constraints_phase_field);
constraints_phase_field.close();

constraints_elasticity.clear();
constraints_elasticity.reinit(locally_relevant_dofs_elasticity);
DoFTools::make_hanging_node_constraints(dof_handler_elasticity, constraints_elasticity);
VectorTools::interpolate_boundary_values(dof_handler_elasticity, 3, Functions::ZeroFunction<dim>(dim), constraints_elasticity);
VectorTools::interpolate_boundary_values(dof_handler_elasticity, 4, Functions::ZeroFunction<dim>(dim), constraints_elasticity);
constraints_elasticity.close();
}

template <int dim>
void CMASProblem<dim>::assemble_concentration_system()
{
TimerOutput::Scope ts(computing_timer, "assemble_concentration");
system_matrix_concentration = 0; 
system_rhs_concentration = 0;

QGauss<dim> quadrature(fe_concentration.degree + 1);
FEValues<dim> fe_values(fe_concentration, quadrature, update_values | update_gradients | update_quadrature_points | update_JxW_values);
FEValues<dim> fe_values_phase(fe_phase_field, quadrature, update_values | update_quadrature_points);

const unsigned int dofs_per_cell = fe_concentration.n_dofs_per_cell();
const unsigned int n_q_points = quadrature.size();

FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
Vector<double> cell_rhs(dofs_per_cell);
std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

std::vector<double> phi_values(n_q_points);
std::vector<Vector<double>> old_c_values(n_q_points, Vector<double>(6)); // MODIFIED: 6 components

for (const auto &cell : dof_handler_concentration.active_cell_iterators())
{
if (!cell->is_locally_owned()) continue;

cell_matrix = 0; 
cell_rhs = 0;

fe_values.reinit(cell);
const auto phase_cell = cell->as_dof_handler_iterator(dof_handler_phase_field);
fe_values_phase.reinit(phase_cell);

fe_values_phase.get_function_values(locally_relevant_solution_phase_field, phi_values);
fe_values.get_function_values(old_solution_concentration, old_c_values);

const Point<dim> cell_center = cell->center();
MaterialLayer layer = get_layer(cell_center[1]);

// MODIFIED: Logic for active regions based on variable type
bool can_diffuse = (layer == TC || layer == TGO); // c1-c4 diffuse in both
bool can_react_n = (layer == TC);                 // n reacts only in TC
bool can_react_N = (layer == TGO);                // N reacts only in TGO

for (unsigned int q = 0; q < n_q_points; ++q)
{
const double phi = phi_values[q];
const double D = get_diffusivity(phi);
const double JxW = fe_values.JxW(q);

double total_oxide_mass = old_c_values[q][0] * MolarMass::Fe2O3 + 
old_c_values[q][1] * MolarMass::MgO + 
old_c_values[q][2] * MolarMass::Al2O3 / 2.0 + 
old_c_values[q][3] * MolarMass::SiO2;
double n_old = old_c_values[q][4];
double N_old = old_c_values[q][5]; // TGO degradation

for (unsigned int i = 0; i < dofs_per_cell; ++i)
{
const unsigned int comp_i = fe_concentration.system_to_component_index(i).first;
const double shape_i = fe_values.shape_value(i, q);

double u_old_val = old_c_values[q][comp_i];
cell_rhs(i) += shape_i * u_old_val * JxW;

// MODIFIED: Reaction terms
// 1. Corrosion n (index 4) in TC
if (comp_i == 4 && can_react_n)
{
double reaction_rate = Material::k_reaction1 * (1.0 - n_old - 0.12) * total_oxide_mass;
cell_rhs(i) += shape_i * reaction_rate * time_step * JxW;
}
// 2. Degradation N (index 5) in TGO
// Equation: dN/dt = xi * (1 - N) * rho_CMAS
else if (comp_i == 5 && can_react_N)
{
double reaction_rate = Material::k_reaction2 * (1.0 - N_old) * total_oxide_mass;
cell_rhs(i) += shape_i * reaction_rate * time_step * JxW;
}

for (unsigned int j = 0; j < dofs_per_cell; ++j)
{
const unsigned int comp_j = fe_concentration.system_to_component_index(j).first;
if (comp_i == comp_j)
{
const double shape_j = fe_values.shape_value(j, q);

double mass_term = shape_i * shape_j;
double stiffness_term = 0.0;

// Diffusion only for c1-c4 in TC/TGO
if (comp_i < 4 && can_diffuse)
{
stiffness_term = time_step * D * fe_values.shape_grad(i, q) * fe_values.shape_grad(j, q);
}
// n and N do not diffuse, so stiffness_term remains 0 for them

cell_matrix(i, j) += (mass_term + stiffness_term) * JxW;
}
}
}
}
cell->get_dof_indices(local_dof_indices);
constraints_concentration.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, system_matrix_concentration, system_rhs_concentration);
}

system_matrix_concentration.compress(VectorOperation::add);
system_rhs_concentration.compress(VectorOperation::add);
}

template <int dim>
void CMASProblem<dim>::assemble_phase_field_system()
{
TimerOutput::Scope ts(computing_timer, "assemble_phase_field");
system_matrix_phase_field = 0; system_rhs_phase_field = 0;

QGauss<dim> quadrature(fe_phase_field.degree + 1);
FEValues<dim> fe_values(fe_phase_field, quadrature, update_values | update_gradients | update_quadrature_points | update_JxW_values);
FEValues<dim> fe_values_elastic(fe_elasticity, quadrature, update_gradients | update_quadrature_points);
FEValues<dim> fe_values_conc(fe_concentration, quadrature, update_values);

const unsigned int dofs_per_cell = fe_phase_field.n_dofs_per_cell();
const unsigned int n_q_points = quadrature.size();

FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
Vector<double> cell_rhs(dofs_per_cell);
std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
std::vector<double> phi_old(n_q_points);
std::vector<Vector<double>> conc_values(n_q_points, Vector<double>(6)); // Size 6
std::vector<std::vector<Tensor<1, dim>>> displacement_grads_all(n_q_points, std::vector<Tensor<1, dim>>(dim));

unsigned int cell_index = 0;

for (const auto &cell : dof_handler_phase_field.active_cell_iterators())
{
if (!cell->is_locally_owned()) continue;
cell_matrix = 0; cell_rhs = 0;
fe_values.reinit(cell);
const auto elastic_cell = cell->as_dof_handler_iterator(dof_handler_elasticity);
const auto conc_cell = cell->as_dof_handler_iterator(dof_handler_concentration);
fe_values_elastic.reinit(elastic_cell);
fe_values_conc.reinit(conc_cell);

fe_values.get_function_values(old_solution_phase_field, phi_old);
fe_values_conc.get_function_values(locally_relevant_solution_concentration, conc_values);

for (unsigned int d = 0; d < dim; ++d) {
FEValuesExtractors::Scalar displacement_component(d);
std::vector<Tensor<1, dim>> grad_u_component(n_q_points);
fe_values_elastic[displacement_component].get_function_gradients(locally_relevant_solution_elasticity, grad_u_component);
for (unsigned int q = 0; q < n_q_points; ++q) displacement_grads_all[q][d] = grad_u_component[q];
}

const Point<dim> cell_center = cell->center();
MaterialLayer layer = get_layer(cell_center[1]);
bool in_active_region = (layer == TC || layer == TGO);

if (cell_center[1] > Domain::y_tc_top - 0.05) in_active_region = false;

double E, nu, G_c, sigma_c;
// Initialize basic parameters
if (layer == TC) { E = Material::E_TC; nu = Material::nu_TC; G_c = Material::G_TC; sigma_c = Material::sigma_TC; in_active_region = false; }
else if (layer == TGO) { E = get_tgo_E_base(cell_center); nu = Material::nu_TGO; G_c = Material::G_TGO; sigma_c = Material::sigma_TGO; }
else if(layer == BC) {in_active_region = false;}
else { E = Material::E_BC; nu = Material::nu_BC; G_c = 1.0; sigma_c = 1e3; }

double b0 = Material::length_scale;
double ck = 2.0 * G_c * b0 / (3.14159 );
double ak = 2.0 * G_c / (3.14159 * b0);
double lam = lambda(E, nu);
double mu_val = mu(E, nu);

for (unsigned int q = 0; q < n_q_points; ++q)
{
const Point<dim> &x_q = fe_values.quadrature_point(q);
double n_conc = conc_values[q][4];
double N = conc_values[q][5]; // TGO degradation variable

// Expansion eigenstrain only in TC
double ext_strain = 0.0;
if (layer == TC) {
ext_strain = 0.176 * n_conc;
}

// Deformation gradient: F = I + grad_u (for large deformation)
Tensor<2, dim> F;
for (unsigned int i = 0; i < dim; ++i) {
for (unsigned int j = 0; j < dim; ++j) {
F[i][j] = (i == j ? 1.0 : 0.0) + displacement_grads_all[q][i][j];
}
}

// Right Cauchy-Green tensor: C = F^T * F
Tensor<2, dim> C;
for (unsigned int i = 0; i < dim; ++i) {
for (unsigned int j = 0; j < dim; ++j) {
C[i][j] = 0.0;
for (unsigned int k = 0; k < dim; ++k) {
C[i][j] += F[k][i] * F[k][j];
}
}
}

// Green-Lagrange strain: E_GL = 0.5 * (C - I)
SymmetricTensor<2, dim> E_GL;
for (unsigned int i = 0; i < dim; ++i) {
for (unsigned int j = i; j < dim; ++j) {
E_GL[i][j] = 0.5 * (C[i][j] - (i == j ? 1.0 : 0.0));
}
}

// Elastic Green-Lagrange strain
SymmetricTensor<2, dim> E_GL_elastic = E_GL;
for (unsigned int i = 0; i < dim; ++i) {
E_GL_elastic[i][i] -= ext_strain;
}

double tr_E_GL_2D = trace(E_GL_elastic);
double tr_E_GL_3D = tr_E_GL_2D - ext_strain;

// Second Piola-Kirchhoff stress
SymmetricTensor<2, dim> S;
for (unsigned int i = 0; i < dim; ++i) {
for (unsigned int j = i; j < dim; ++j) {
S[i][j] = lam * tr_E_GL_3D * (i == j ? 1.0 : 0.0) + 2.0 * mu_val * E_GL_elastic[i][j];
}
}

// Convert to Cauchy stress for principal stress calculation
double J = determinant(F);
if (std::abs(J) < NumericalConstants::determinant_threshold) J = 1.0;

SymmetricTensor<2, dim> cauchy_stress;
for (unsigned int i = 0; i < dim; ++i) {
for (unsigned int j = i; j < dim; ++j) {
double sigma_ij = 0.0;
for (unsigned int k = 0; k < dim; ++k) {
for (unsigned int l = 0; l < dim; ++l) {
sigma_ij += F[i][k] * S[k][l] * F[j][l];
}
}
cauchy_stress[i][j] = sigma_ij / J;
}
}

std::array<double, dim> principal_stresses = eigenvalues(cauchy_stress);
double sigma1 = *std::max_element(principal_stresses.begin(), principal_stresses.end());

// MODIFIED: TGO critical stress depends on N
double local_sigma_c = sigma_c;
if (layer == TGO) {
double factor = (1.0 - N);
if (factor < 0) factor = 0;
local_sigma_c = Material::sigma_TGO * std::sqrt(factor);
if (local_sigma_c < 1e-6) local_sigma_c = 1e-6;
}

double Y_bar = 0.5 * std::max(0.0, sigma1) * std::max(0.0, sigma1) / E;
double Y0 = 0.5 * local_sigma_c * local_sigma_c / E;  //sigma_c单位是Pa

// FIX for spurious phase field accumulation:
// Only allow damage evolution when stress exceeds threshold (Y_bar > Y0)
// or when damage has already initiated at this point (history > Y0).
double H_history = quadrature_point_history[cell_index][q];
bool damage_initiated = (H_history > Y0 + NumericalConstants::damage_threshold_tolerance) || (Y_bar > Y0);
double H = damage_initiated ? std::max(H_history, Y_bar) : Y0;

// Determine if phase field should evolve at this quadrature point
bool qpoint_active = in_active_region && damage_initiated;

double phi = phi_old[q];
// Note: Using local_sigma_c for Kphi calculation
double Kphi = 4.0 * E * G_c / (3.14159 * b0 * local_sigma_c * local_sigma_c);
double Nphi = (1.0 - phi) * (1.0 - phi);
double Dphi = Nphi + Kphi * phi * (1.0 - phi / 2.0);
double domega = (-(1.0 - phi) * 2.0 * Dphi - Nphi * (1.0 - phi) * (Kphi - 2.0)) / (Dphi * Dphi );
double fk = 2.0 * G_c / (3.14159 * b0) + domega * H;

for (unsigned int i = 0; i < dofs_per_cell; ++i) {
for (unsigned int j = 0; j < dofs_per_cell; ++j) {
if (qpoint_active)
cell_matrix(i, j) += (ck * fe_values.shape_grad(i, q) * fe_values.shape_grad(j, q) + ak * fe_values.shape_value(i, q) * fe_values.shape_value(j, q)) * fe_values.JxW(q);
else
cell_matrix(i, j) += fe_values.shape_value(i, q) * fe_values.shape_value(j, q) * fe_values.JxW(q);
}
if (qpoint_active) cell_rhs(i) += fe_values.shape_value(i, q) * fk * fe_values.JxW(q);
}
}
cell->get_dof_indices(local_dof_indices);
constraints_phase_field.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, system_matrix_phase_field, system_rhs_phase_field);
++cell_index;
}
system_matrix_phase_field.compress(VectorOperation::add);
system_rhs_phase_field.compress(VectorOperation::add);
}

// Newton-Raphson assembly for phase-field: assembles Jacobian and residual
// Residual R(phi) = 0 where we want to find phi
// The equation is: -div(c_k * grad(phi)) + a_k * phi - f_k(phi) = 0
// where f_k depends nonlinearly on phi through omega'(phi) * H
template <int dim>
void CMASProblem<dim>::assemble_phase_field_newton_system()
{
TimerOutput::Scope ts(computing_timer, "assemble_phase_field_newton");
system_matrix_phase_field = 0; 
system_rhs_phase_field = 0;

QGauss<dim> quadrature(fe_phase_field.degree + 1);
FEValues<dim> fe_values(fe_phase_field, quadrature, update_values | update_gradients | update_quadrature_points | update_JxW_values);
FEValues<dim> fe_values_elastic(fe_elasticity, quadrature, update_gradients | update_quadrature_points);
FEValues<dim> fe_values_conc(fe_concentration, quadrature, update_values);

const unsigned int dofs_per_cell = fe_phase_field.n_dofs_per_cell();
const unsigned int n_q_points = quadrature.size();

FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
Vector<double> cell_rhs(dofs_per_cell);
std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

// Current phi values and gradients for Newton
std::vector<double> phi_values(n_q_points);
std::vector<double> old_phi_values(n_q_points); 
std::vector<Tensor<1, dim>> phi_gradients(n_q_points);
std::vector<Vector<double>> conc_values(n_q_points, Vector<double>(6));
std::vector<std::vector<Tensor<1, dim>>> displacement_grads_all(n_q_points, std::vector<Tensor<1, dim>>(dim));

unsigned int cell_index = 0;

for (const auto &cell : dof_handler_phase_field.active_cell_iterators())
{
if (!cell->is_locally_owned()) continue;
cell_matrix = 0; 
cell_rhs = 0;
fe_values.reinit(cell);
const auto elastic_cell = cell->as_dof_handler_iterator(dof_handler_elasticity);
const auto conc_cell = cell->as_dof_handler_iterator(dof_handler_concentration);
fe_values_elastic.reinit(elastic_cell);
fe_values_conc.reinit(conc_cell);

// Get current iteration phi (from locally_relevant_solution_phase_field)
fe_values.get_function_values(locally_relevant_solution_phase_field, phi_values);
fe_values.get_function_values(old_solution_phase_field, old_phi_values);
fe_values.get_function_gradients(locally_relevant_solution_phase_field, phi_gradients);
fe_values_conc.get_function_values(locally_relevant_solution_concentration, conc_values);

for (unsigned int d = 0; d < dim; ++d) {
FEValuesExtractors::Scalar displacement_component(d);
std::vector<Tensor<1, dim>> grad_u_component(n_q_points);
fe_values_elastic[displacement_component].get_function_gradients(locally_relevant_solution_elasticity, grad_u_component);
for (unsigned int q = 0; q < n_q_points; ++q) 
displacement_grads_all[q][d] = grad_u_component[q];
}

const Point<dim> cell_center = cell->center();
MaterialLayer layer = get_layer(cell_center[1]);
bool in_active_region = (layer == TC || layer == TGO);

if (cell_center[1] > Domain::y_tc_top - 0.05) in_active_region = false;

double E, nu, G_c, sigma_c;
if (layer == TC) { 
E = Material::E_TC; nu = Material::nu_TC; G_c = Material::G_TC; sigma_c = Material::sigma_TC; 
in_active_region = false; 
}
else if (layer == TGO) { 
E = get_tgo_E_base(cell_center); nu = Material::nu_TGO; G_c = Material::G_TGO; sigma_c = Material::sigma_TGO; 
}
else if(layer == BC) {
in_active_region = false;
E = Material::E_BC; nu = Material::nu_BC; G_c = 1.0; sigma_c = 1e3;
}
else { 
E = Material::E_BC; nu = Material::nu_BC; G_c = 1.0; sigma_c = 1e3; 
}

double b0 = Material::length_scale;
double ck = 2.0 * G_c * b0 / 3.14159;
double ak = 2.0 * G_c / (3.14159 * b0);
double lam = lambda(E, nu);
double mu_val = mu(E, nu);

for (unsigned int q = 0; q < n_q_points; ++q)
{
double n_conc = conc_values[q][4];
double N = conc_values[q][5];

double ext_strain = 0.0;
if (layer == TC) {
ext_strain = 0.176 * n_conc;
}

// Deformation gradient: F = I + grad_u (for large deformation)
Tensor<2, dim> F;
for (unsigned int i = 0; i < dim; ++i) {
for (unsigned int j = 0; j < dim; ++j) {
F[i][j] = (i == j ? 1.0 : 0.0) + displacement_grads_all[q][i][j];
}
}

// Right Cauchy-Green tensor: C = F^T * F
Tensor<2, dim> C;
for (unsigned int i = 0; i < dim; ++i) {
for (unsigned int j = 0; j < dim; ++j) {
C[i][j] = 0.0;
for (unsigned int k = 0; k < dim; ++k) {
C[i][j] += F[k][i] * F[k][j];
}
}
}

// Green-Lagrange strain: E_GL = 0.5 * (C - I)
SymmetricTensor<2, dim> E_GL;
for (unsigned int i = 0; i < dim; ++i) {
for (unsigned int j = i; j < dim; ++j) {
E_GL[i][j] = 0.5 * (C[i][j] - (i == j ? 1.0 : 0.0));
}
}

// Elastic Green-Lagrange strain
SymmetricTensor<2, dim> E_GL_elastic = E_GL;
for (unsigned int i = 0; i < dim; ++i) {
E_GL_elastic[i][i] -= ext_strain;
}

double tr_E_GL_2D = trace(E_GL_elastic);
double tr_E_GL_3D = tr_E_GL_2D - ext_strain;

// Second Piola-Kirchhoff stress
SymmetricTensor<2, dim> S;
for (unsigned int i = 0; i < dim; ++i) {
for (unsigned int j = i; j < dim; ++j) {
S[i][j] = lam * tr_E_GL_3D * (i == j ? 1.0 : 0.0) + 2.0 * mu_val * E_GL_elastic[i][j];
}
}

// Convert to Cauchy stress for principal stress calculation
double J = determinant(F);
if (std::abs(J) < NumericalConstants::determinant_threshold) J = 1.0;

SymmetricTensor<2, dim> cauchy_stress;
for (unsigned int i = 0; i < dim; ++i) {
for (unsigned int j = i; j < dim; ++j) {
double sigma_ij = 0.0;
for (unsigned int k = 0; k < dim; ++k) {
for (unsigned int l = 0; l < dim; ++l) {
sigma_ij += F[i][k] * S[k][l] * F[j][l];
}
}
cauchy_stress[i][j] = sigma_ij / J;
}
}

std::array<double, dim> principal_stresses = eigenvalues(cauchy_stress);
double sigma1 = *std::max_element(principal_stresses.begin(), principal_stresses.end());

double local_sigma_c = sigma_c;
if (layer == TGO) {
double factor = (1.0 - N);
if (factor < 0) factor = 0;
local_sigma_c = Material::sigma_TGO * std::sqrt(factor);
if (local_sigma_c < 1e-6) local_sigma_c = 1e-6;
}

double Y_bar = 0.5 * std::max(0.0, sigma1) * std::max(0.0, sigma1) / E;
double Y0 = 0.5 * local_sigma_c * local_sigma_c / E;

// FIX for spurious phase field accumulation:
// Only allow damage evolution when stress exceeds the threshold (Y_bar > Y0)
// or when damage has already been initiated at this point (history > Y0).
// This prevents small numerical errors from causing unwanted phase field growth.
double H_history = quadrature_point_history[cell_index][q];
bool damage_initiated = (H_history > Y0 + NumericalConstants::damage_threshold_tolerance) || (Y_bar > Y0);
double H = damage_initiated ? std::max(H_history, Y_bar) : Y0;

// Determine if phase field should evolve at this quadrature point
bool qpoint_active = in_active_region && damage_initiated;

// Current phi value at this quadrature point
double phi = phi_values[q];
// Clamp phi to valid range for stability
phi = std::max(0.0, std::min(1.0, phi));

// Compute degradation function derivatives
double Kphi = 4.0 * E * G_c / (3.14159 * b0 * local_sigma_c * local_sigma_c);

// N(phi) = (1-phi)^2
double Nphi = (1.0 - phi) * (1.0 - phi);
// D(phi) = N(phi) + K * phi * (1 - phi/2)
double Dphi = Nphi + Kphi * phi * (1.0 - phi / 2.0);

// omega(phi) = N(phi) / D(phi)
// omega'(phi) = (N' * D - N * D') / D^2
// N'(phi) = -2(1-phi)
// D'(phi) = N'(phi) + K * (1 - phi) = -2(1-phi) + K(1-phi) = (K-2)(1-phi)
double Nphi_prime = -2.0 * (1.0 - phi);
double Dphi_prime = Nphi_prime + Kphi * (1.0 - phi);
// Second derivatives
double Nphi_pp = 2.0;
double Dphi_pp = Nphi_pp - Kphi;

double eps = NumericalConstants::division_epsilon;
double domega = (Nphi_prime * Dphi - Nphi * Dphi_prime) / (Dphi * Dphi + eps);

// Second derivative of omega for Jacobian
// omega' = (N'D - ND') / D^2
// omega'' = ((N''D + N'D' - N'D' - ND'')D^2 - (N'D - ND') * 2D*D') / D^4
//         = ((N''D - ND'')D^2 - 2D'*(N'D - ND')) / D^4
//         = (N'' - omega*D'')/D - 2D'*omega'/D
double omega_current = Nphi / (Dphi + eps);
double d2omega = (Nphi_pp - omega_current * Dphi_pp) / (Dphi + eps) - 2.0 * Dphi_prime * domega / (Dphi + eps);

// f_k(phi) = 2*G_c/(pi*b0) + omega'(phi)*H
double fk = ak + domega * H;  // Note: ak = 2*G_c/(pi*b0)

// df_k/dphi = omega''(phi) * H
double dfk_dphi = d2omega * H;

// Viscous regularization
double phi_old = old_phi_values[q];
double dt = time_step; 
double viscous_term = Material::viscosity * (phi - phi_old) / dt;
double viscous_term_derivative = Material::viscosity / dt;

const double JxW = fe_values.JxW(q);
const Tensor<1, dim> &grad_phi_q = phi_gradients[q];

for (unsigned int i = 0; i < dofs_per_cell; ++i) {
const double shape_i = fe_values.shape_value(i, q);
const Tensor<1, dim> &grad_shape_i = fe_values.shape_grad(i, q);

if (qpoint_active) {
// Residual R_i = c_k * (grad_phi, grad_v_i) + a_k * (phi, v_i) - (f_k, v_i)
double residual_contribution = ck * (grad_phi_q * grad_shape_i)
+ ak * phi * shape_i
- fk * shape_i
+ viscous_term * shape_i;
cell_rhs(i) -= residual_contribution * JxW;  // Note: -R for Newton

for (unsigned int j = 0; j < dofs_per_cell; ++j) {
const double shape_j = fe_values.shape_value(j, q);
const Tensor<1, dim> &grad_shape_j = fe_values.shape_grad(j, q);

// Jacobian J_ij = dR_i/dphi_j
// = c_k * (grad_v_j, grad_v_i) + a_k * (v_j, v_i) - df_k/dphi * v_j * v_i
double jacobian_contribution = ck * (grad_shape_j * grad_shape_i)
+ ak * shape_j * shape_i
- dfk_dphi * shape_j * shape_i
+ viscous_term_derivative * shape_j * shape_i;
cell_matrix(i, j) += jacobian_contribution * JxW;
}
}
else {
// Non-active region or damage not initiated: phi should stay at its current value (or 0)
// Just use identity-like system: phi = phi_old (no evolution)
cell_rhs(i) -= phi * shape_i * JxW;
for (unsigned int j = 0; j < dofs_per_cell; ++j) {
cell_matrix(i, j) += fe_values.shape_value(i, q) * fe_values.shape_value(j, q) * JxW;
}
}
}
}
cell->get_dof_indices(local_dof_indices);
constraints_phase_field.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, system_matrix_phase_field, system_rhs_phase_field);
++cell_index;
}
system_matrix_phase_field.compress(VectorOperation::add);
system_rhs_phase_field.compress(VectorOperation::add);
}


// Large deformation (finite strain) elasticity assembly
// Uses Green-Lagrange strain: E = 0.5(F^T F - I) where F = I + grad_u
// Uses Second Piola-Kirchhoff stress: S = lambda * tr(E) * I + 2 * mu * E
// Includes both material tangent and geometric stiffness
template <int dim>
void CMASProblem<dim>::assemble_elasticity_system()
{
TimerOutput::Scope ts(computing_timer, "assemble_elasticity");
system_matrix_elasticity = 0; system_rhs_elasticity = 0;

QGauss<dim> quadrature(fe_elasticity.degree + 1);
FEValues<dim> fe_values(fe_elasticity, quadrature, update_values | update_gradients | update_quadrature_points | update_JxW_values);
FEValues<dim> fe_values_phase(fe_phase_field, quadrature, update_values);
FEValues<dim> fe_values_conc(fe_concentration, quadrature, update_values);

const unsigned int dofs_per_cell = fe_elasticity.n_dofs_per_cell();
const unsigned int n_q_points = quadrature.size();

FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
Vector<double> cell_rhs(dofs_per_cell);
std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
std::vector<double> phi_values(n_q_points);
std::vector<Vector<double>> conc_values(n_q_points, Vector<double>(6));

// Storage for displacement gradients at quadrature points
std::vector<Tensor<2, dim>> grad_u_all(n_q_points);

for (const auto &cell : dof_handler_elasticity.active_cell_iterators())
{
if (!cell->is_locally_owned()) continue;
cell_matrix = 0; cell_rhs = 0;
fe_values.reinit(cell);
const auto phase_cell = cell->as_dof_handler_iterator(dof_handler_phase_field);
const auto conc_cell = cell->as_dof_handler_iterator(dof_handler_concentration);
fe_values_phase.reinit(phase_cell);
fe_values_conc.reinit(conc_cell);

fe_values_phase.get_function_values(locally_relevant_solution_phase_field, phi_values);
fe_values_conc.get_function_values(locally_relevant_solution_concentration, conc_values);

// Get current displacement gradients for nonlinear strain
for (unsigned int d = 0; d < dim; ++d) {
FEValuesExtractors::Scalar displacement_component(d);
std::vector<Tensor<1, dim>> grad_u_component(n_q_points);
fe_values[displacement_component].get_function_gradients(locally_relevant_solution_elasticity, grad_u_component);
for (unsigned int q = 0; q < n_q_points; ++q) {
for (unsigned int e = 0; e < dim; ++e) {
grad_u_all[q][d][e] = grad_u_component[q][e];
}
}
}

const Point<dim> cell_center = cell->center();
MaterialLayer layer = get_layer(cell_center[1]);

for (unsigned int q = 0; q < n_q_points; ++q)
{
const Point<dim> &x_q = fe_values.quadrature_point(q);
double phi = phi_values[q];
double n_conc = conc_values[q][4];
double N = conc_values[q][5];

double E_mod = get_E(x_q, phi, n_conc, N);
double nu = get_nu(x_q);
double lam = lambda(E_mod, nu);
double mu_val = mu(E_mod, nu);

// Eigenstrain from corrosion (only in TC layer)
double ext_strain = 0.0;
if (layer == TC) {
ext_strain = 0.176 * n_conc;
}

// Deformation gradient: F = I + grad_u
Tensor<2, dim> F;
for (unsigned int i = 0; i < dim; ++i) {
for (unsigned int j = 0; j < dim; ++j) {
F[i][j] = (i == j ? 1.0 : 0.0) + grad_u_all[q][i][j];
}
}

// Right Cauchy-Green tensor: C = F^T * F
Tensor<2, dim> C;
for (unsigned int i = 0; i < dim; ++i) {
for (unsigned int j = 0; j < dim; ++j) {
C[i][j] = 0.0;
for (unsigned int k = 0; k < dim; ++k) {
C[i][j] += F[k][i] * F[k][j];
}
}
}

// Green-Lagrange strain: E_GL = 0.5 * (C - I)
SymmetricTensor<2, dim> E_GL;
for (unsigned int i = 0; i < dim; ++i) {
for (unsigned int j = i; j < dim; ++j) {
E_GL[i][j] = 0.5 * (C[i][j] - (i == j ? 1.0 : 0.0));
}
}

// Elastic Green-Lagrange strain (subtract eigenstrain)
SymmetricTensor<2, dim> E_GL_elastic = E_GL;
for (unsigned int i = 0; i < dim; ++i) {
E_GL_elastic[i][i] -= ext_strain;
}

// Trace of elastic strain (2D, account for plane strain in 3rd direction)
double tr_E_GL_2D = trace(E_GL_elastic);
double tr_E_GL_3D = tr_E_GL_2D - ext_strain; // Account for out-of-plane eigenstrain

// Second Piola-Kirchhoff stress: S = lambda * tr(E) * I + 2 * mu * E
// (St. Venant-Kirchhoff material model)
SymmetricTensor<2, dim> S;
for (unsigned int i = 0; i < dim; ++i) {
for (unsigned int j = i; j < dim; ++j) {
S[i][j] = lam * tr_E_GL_3D * (i == j ? 1.0 : 0.0) + 2.0 * mu_val * E_GL_elastic[i][j];
}
}

const double JxW = fe_values.JxW(q);

// Assemble residual and tangent stiffness
for (unsigned int i = 0; i < dofs_per_cell; ++i) {
const unsigned int comp_i = fe_elasticity.system_to_component_index(i).first;
const Tensor<1, dim> &grad_Ni = fe_values.shape_grad(i, q);

// Compute F^T * grad_Ni for the weak form
Tensor<1, dim> FT_grad_Ni;
for (unsigned int k = 0; k < dim; ++k) {
FT_grad_Ni[k] = 0.0;
for (unsigned int m = 0; m < dim; ++m) {
FT_grad_Ni[k] += F[m][k] * (m == comp_i ? grad_Ni[m] : 0.0);
}
}

// Actually for vectorial shape functions, grad_Ni[comp_i] gives dN_i/dx_comp_i
// We need dN_i/dX_j where shape function i corresponds to component comp_i
// The contribution to variation of E_GL is:
// delta_E_GL[k][l] = 0.5 * (F[m][k] * grad_Ni_m * delta[comp_i][m] * delta[l] + ...)
// Simplified: for residual, R_i = sum over k,l of S[k][l] * d(E_GL[k][l])/du_i

// Residual contribution: -R_i = -integral S : delta_E dV
// delta_E[k][l] = 0.5 * (F^T[k][m] * dN_i/dX_l + F^T[l][m] * dN_i/dX_k) for component m = comp_i
double residual_contribution = 0.0;
for (unsigned int k = 0; k < dim; ++k) {
for (unsigned int l = 0; l < dim; ++l) {
// Variation of E_GL w.r.t. displacement component comp_i at node i
// dE_GL[k][l]/du_i^comp_i = 0.5 * (F[comp_i][k] * grad_Ni[l] + F[comp_i][l] * grad_Ni[k])
double dE_kl = 0.5 * (F[comp_i][k] * grad_Ni[l] + F[comp_i][l] * grad_Ni[k]);
residual_contribution += S[k][l] * dE_kl;
}
}


// 定义扰动区域：在鼓泡预期的中心位置 (x=4.0mm 附近)
// 定义扰动时间：仅在损伤初期施加，一旦变形开始就移除，或者一直保持微小值
bool is_center_region = (std::abs(cell_center[0] - 4.0) < 0.5); // 宽度 1mm 的中心区域
bool is_top_surface = (std::abs(cell_center[1] - Domain::y_tc_top) < 0.05); // 靠近上表面

// 扰动强度：不需要很大，只需要破坏对称性
// 比如 0.1 MPa 的牵引力 (相对于 40GPa 的模量微不足道，但足以诱导方向)
double perturbation_pressure = 0.0;

// 仅在中心上表面且时间较早时施加，或者当位移还很小时施加
if (is_center_region && is_top_surface && time < 100000.0) // 假设在前1000秒内施加
{
perturbation_pressure = 0.1; // 1 MPa 的向上牵引力
}
// 或者：始终施加一个极小的重  反向力
// perturbation_pressure = 0.1; 

// 将扰动力加到残差向量中 (RHS)
// 注意：elasticity 的 RHS 是 -Residual，所以外力 F 应该以 +F 的形式加进去
// 对应的弱形式项是： - integral( sigma : grad_v ) + integral( f * v ) = 0
// 所以 Residual = integral( sigma : grad_v ) - integral( f * v )
// 所以 RHS = -Residual = -integral( sigma : grad_v ) + integral( f * v )

if (std::abs(perturbation_pressure) > 1e-10)
{
// 遍历当前单元的自由度
for (unsigned int i = 0; i < dofs_per_cell; ++i)
{
const unsigned int comp_i = fe_elasticity.system_to_component_index(i).first;

// 仅在 Y 方向 (comp_i == 1) 施加力
if (comp_i == 1)
{
const double shape_i = fe_values.shape_value(i, q);
// 施加力项： f * v * JxW
// 这里简化为体积力形式施加在表层单元上，或者面积分
// 由于是在 cell 积分中，这相当于一个体积力 density
// 为了方便，直接加在 RHS 上
cell_rhs(i) += perturbation_pressure * shape_i * JxW; 
}
}
}
// RHS = -Residual (for Newton iteration we want -R so that K*du = -R)
cell_rhs(i) -= residual_contribution * JxW;

// Tangent stiffness matrix (material + geometric contributions)
for (unsigned int j = 0; j < dofs_per_cell; ++j) {
const unsigned int comp_j = fe_elasticity.system_to_component_index(j).first;
const Tensor<1, dim> &grad_Nj = fe_values.shape_grad(j, q);

double tangent_contribution = 0.0;

// 1. Material stiffness: d²Ψ/(dE dE) : (dE/du_i) : (dE/du_j)
// For St. Venant-Kirchhoff: C_ijkl = lambda * delta_ij * delta_kl + 2*mu * 0.5*(delta_ik*delta_jl + delta_il*delta_jk)
for (unsigned int k = 0; k < dim; ++k) {
for (unsigned int l = 0; l < dim; ++l) {
double dE_i_kl = 0.5 * (F[comp_i][k] * grad_Ni[l] + F[comp_i][l] * grad_Ni[k]);

for (unsigned int m = 0; m < dim; ++m) {
for (unsigned int n = 0; n < dim; ++n) {
double dE_j_mn = 0.5 * (F[comp_j][m] * grad_Nj[n] + F[comp_j][n] * grad_Nj[m]);

// Material tangent C_klmn
double C_klmn = lam * (k == l ? 1.0 : 0.0) * (m == n ? 1.0 : 0.0)
+ mu_val * ((k == m ? 1.0 : 0.0) * (l == n ? 1.0 : 0.0) 
+ (k == n ? 1.0 : 0.0) * (l == m ? 1.0 : 0.0));

tangent_contribution += dE_i_kl * C_klmn * dE_j_mn;
}
}
}
}

// 2. Geometric stiffness: S : d²E/(du_i du_j)
// d²E[k][l]/(du_i du_j) = 0.5 * delta[comp_i][comp_j] * (grad_Ni[k] * grad_Nj[l] + grad_Ni[l] * grad_Nj[k])
// when comp_i == comp_j
if (comp_i == comp_j) {
for (unsigned int k = 0; k < dim; ++k) {
for (unsigned int l = 0; l < dim; ++l) {
double d2E_kl = 0.5 * (grad_Ni[k] * grad_Nj[l] + grad_Ni[l] * grad_Nj[k]);
tangent_contribution += S[k][l] * d2E_kl;
}
}
}

cell_matrix(i, j) += tangent_contribution * JxW;
}
}
}
cell->get_dof_indices(local_dof_indices);
constraints_elasticity.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, system_matrix_elasticity, system_rhs_elasticity);
}
system_matrix_elasticity.compress(VectorOperation::add);
system_rhs_elasticity.compress(VectorOperation::add);
}

// ... (Solvers remain largely same)
template <int dim>
void CMASProblem<dim>::solve_concentration()
{
TimerOutput::Scope ts(computing_timer, "solve_concentration");
SolverControl solver_control(10000, 1e-12 * system_rhs_concentration.l2_norm());
SolverCG<LA::MPI::Vector> solver(solver_control);
LA::MPI::PreconditionAMG::AdditionalData data;
#ifdef USE_PETSC_LA
data.symmetric_operator = true;
#endif
LA::MPI::PreconditionAMG preconditioner;
preconditioner.initialize(system_matrix_concentration, data);
solver.solve(system_matrix_concentration, completely_distributed_solution_concentration, system_rhs_concentration, preconditioner);
constraints_concentration.distribute(completely_distributed_solution_concentration);
locally_relevant_solution_concentration = completely_distributed_solution_concentration;
pcout << "  Concentration: " << solver_control.last_step() << " CG iterations" << std::endl;
}

template <int dim>
void CMASProblem<dim>::solve_phase_field()
{
TimerOutput::Scope ts(computing_timer, "solve_phase_field");
SolverControl solver_control(10000, 1e-12 * system_rhs_phase_field.l2_norm());
SolverCG<LA::MPI::Vector> solver(solver_control);
LA::MPI::PreconditionAMG::AdditionalData data;
#ifdef USE_PETSC_LA
data.symmetric_operator = true;
#endif
LA::MPI::PreconditionAMG preconditioner;
preconditioner.initialize(system_matrix_phase_field, data);
solver.solve(system_matrix_phase_field, completely_distributed_solution_phase_field, system_rhs_phase_field, preconditioner);
constraints_phase_field.distribute(completely_distributed_solution_phase_field);
for (auto it = locally_owned_dofs_phase_field.begin(); it != locally_owned_dofs_phase_field.end(); ++it)
{
const types::global_dof_index i = *it;
double new_val = completely_distributed_solution_phase_field[i];
double old_val = old_solution_phase_field[i];
if (new_val < old_val) new_val = old_val;
new_val = std::max(0.0, std::min(1.0, new_val));
completely_distributed_solution_phase_field[i] = new_val;
}
locally_relevant_solution_phase_field = completely_distributed_solution_phase_field;
pcout << "  Phase field: " << solver_control.last_step() << " CG iterations" << std::endl;
}

// Newton-Raphson solver for phase-field equation using fixed-point iteration
// with relaxation and bound enforcement
template <int dim>
void CMASProblem<dim>::solve_phase_field_newton()
{
TimerOutput::Scope ts(computing_timer, "solve_phase_field_newton");

const unsigned int max_newton_iterations = 100;
const double newton_tol = 1e-6;
const double relaxation = 0.5;  // Under-relaxation factor for stability

// Initialize current solution from old solution
completely_distributed_solution_phase_field = old_solution_phase_field;
locally_relevant_solution_phase_field = completely_distributed_solution_phase_field;

unsigned int newton_iteration = 0;
double max_delta = 1.0;

while (newton_iteration < max_newton_iterations && max_delta > newton_tol)
{
// Store current solution for convergence check
LA::MPI::Vector prev_solution;
prev_solution.reinit(locally_owned_dofs_phase_field, mpi_communicator);
prev_solution = completely_distributed_solution_phase_field;

// Assemble the linearized system around current phi
assemble_phase_field_newton_system();

// Solve the linear system
newton_update_phase_field = 0;

double rhs_norm = system_rhs_phase_field.l2_norm();
if (rhs_norm < NumericalConstants::zero_threshold) {
pcout << "  Phase field Newton: RHS is near zero at iteration " << newton_iteration << std::endl;
break;
}

SolverControl solver_control(10000, NumericalConstants::division_epsilon * rhs_norm);
SolverCG<LA::MPI::Vector> solver(solver_control);
LA::MPI::PreconditionAMG::AdditionalData data;
#ifdef USE_PETSC_LA
data.symmetric_operator = false;
#endif
LA::MPI::PreconditionAMG preconditioner;
preconditioner.initialize(system_matrix_phase_field, data);

try {
solver.solve(system_matrix_phase_field, newton_update_phase_field, system_rhs_phase_field, preconditioner);
}
catch (std::exception &e) {
pcout << "  Phase field Newton: linear solver exception at iteration " << newton_iteration 
<< ": " << e.what() << std::endl;
break;
}

constraints_phase_field.distribute(newton_update_phase_field);

// Update solution with relaxation: phi_new = phi_old + relaxation * delta_phi
max_delta = 0.0;
for (auto it = locally_owned_dofs_phase_field.begin(); 
it != locally_owned_dofs_phase_field.end(); ++it)
{
const types::global_dof_index i = *it;
double delta = newton_update_phase_field[i];
double new_val = prev_solution[i] + relaxation * delta;

// Enforce bounds
new_val = std::max(0.0, std::min(1.0, new_val));
// Enforce irreversibility: phi can only increase from previous time step
double old_time_val = old_solution_phase_field[i];
if (new_val < old_time_val) new_val = old_time_val;

completely_distributed_solution_phase_field[i] = new_val;
max_delta = std::max(max_delta, std::abs(new_val - prev_solution[i]));
}

max_delta = Utilities::MPI::max(max_delta, mpi_communicator);
locally_relevant_solution_phase_field = completely_distributed_solution_phase_field;

++newton_iteration;
}

if (newton_iteration >= max_newton_iterations) {
pcout << "  Phase field Newton: did not converge in " << max_newton_iterations 
<< " iterations, max_delta = " << max_delta << std::endl;
} else {
pcout << "  Phase field Newton: converged in " << newton_iteration 
<< " iterations, max_delta = " << max_delta << std::endl;
}
}

// Newton-Raphson solver for nonlinear (large deformation) elasticity
template <int dim>
void CMASProblem<dim>::solve_elasticity()
{
TimerOutput::Scope ts(computing_timer, "solve_elasticity");

// Use named constants for Newton-Raphson parameters
const unsigned int max_newton_iterations = NumericalConstants::max_newton_elasticity_iterations;
const double newton_tol = NumericalConstants::newton_elasticity_tolerance;
const double linear_tol = NumericalConstants::newton_elasticity_linear_tolerance;

// Newton update vector
LA::MPI::Vector newton_update;
newton_update.reinit(locally_owned_dofs_elasticity, mpi_communicator);

unsigned int newton_iteration = 0;
double residual_norm = 1.0;
double initial_residual_norm = 1.0;

while (newton_iteration < max_newton_iterations)
{
// Assemble the tangent matrix and residual
assemble_elasticity_system();

residual_norm = system_rhs_elasticity.l2_norm();

if (newton_iteration == 0) {
initial_residual_norm = residual_norm;
if (initial_residual_norm < NumericalConstants::zero_threshold) {
pcout << "  Elasticity Newton: initial residual is zero, no update needed" << std::endl;
break;
}
}

double relative_residual = residual_norm / (initial_residual_norm + NumericalConstants::division_epsilon);

pcout << "    [Elasticity Newton iter " << newton_iteration << "] residual = " 
<< residual_norm << ", relative = " << relative_residual << std::endl;

if (relative_residual < newton_tol || residual_norm < NumericalConstants::zero_threshold) {
pcout << "  Elasticity Newton: converged in " << newton_iteration << " iterations" << std::endl;
break;
}

// Solve the linear system: K * delta_u = -R (RHS already has the sign)
newton_update = 0;

SolverControl solver_control(10000, linear_tol * residual_norm);
SolverCG<LA::MPI::Vector> solver(solver_control);
LA::MPI::PreconditionAMG::AdditionalData data;
#ifdef USE_PETSC_LA
// Note: symmetric_operator is set to false because the tangent stiffness matrix
// in large deformation includes geometric stiffness terms (S : d²E/du²) that
// make the matrix non-symmetric in general.
data.symmetric_operator = false;
#endif
LA::MPI::PreconditionAMG preconditioner;
preconditioner.initialize(system_matrix_elasticity, data);

try {
solver.solve(system_matrix_elasticity, newton_update, system_rhs_elasticity, preconditioner);
}
catch (std::exception &e) {
pcout << "  Elasticity Newton: linear solver exception at iteration " << newton_iteration 
<< ": " << e.what() << std::endl;
break;
}

constraints_elasticity.distribute(newton_update);

// Update solution: u = u + delta_u
completely_distributed_solution_elasticity.add(1.0, newton_update);
locally_relevant_solution_elasticity = completely_distributed_solution_elasticity;

++newton_iteration;
}

if (newton_iteration >= max_newton_iterations) {
pcout << "  WARNING: Elasticity Newton did not converge in " << max_newton_iterations 
<< " iterations, final residual = " << residual_norm << std::endl;
}
}

// Compute principal stresses using Green-Lagrange strain and Second Piola-Kirchhoff stress
template <int dim>
void CMASProblem<dim>::compute_principal_stress()
{
QGauss<dim> quadrature(fe_elasticity.degree + 1);
FEValues<dim> fe_values(fe_elasticity, quadrature, update_gradients | update_quadrature_points);
FEValues<dim> fe_values_phase(fe_phase_field, quadrature, update_values);
FEValues<dim> fe_values_conc(fe_concentration, quadrature, update_values);

const unsigned int n_q_points = quadrature.size();
std::vector<double> phi_values(n_q_points);
std::vector<Vector<double>> conc_values(n_q_points, Vector<double>(6));
unsigned int cell_index = 0;

for (const auto &cell : dof_handler_elasticity.active_cell_iterators())
{
if (!cell->is_locally_owned()) continue;
fe_values.reinit(cell);
const auto phase_cell = cell->as_dof_handler_iterator(dof_handler_phase_field);
const auto conc_cell = cell->as_dof_handler_iterator(dof_handler_concentration);
fe_values_phase.reinit(phase_cell);
fe_values_conc.reinit(conc_cell);
fe_values_phase.get_function_values(locally_relevant_solution_phase_field, phi_values);
fe_values_conc.get_function_values(locally_relevant_solution_concentration, conc_values);

double avg_sigma_xx = 0.0; double avg_sigma_yy = 0.0; double avg_sigma_xy = 0.0;
double max_principal_stress_1 = 0.0; double avg_principal_stress_2 = 0.0; double avg_principal_stress_3 = 0.0;

for (unsigned int q = 0; q < n_q_points; ++q)
{
const Point<dim> &x_q = fe_values.quadrature_point(q);
double phi = phi_values[q];
double n_conc = conc_values[q][4];
double N = conc_values[q][5];

double E_mod = get_E(x_q, phi, n_conc, N);
double nu = get_nu(x_q);
double lam = lambda(E_mod, nu);
double mu_val = mu(E_mod, nu);

MaterialLayer layer = get_layer(x_q[1]);
double ext_strain = 0.0;

if (layer == TC) {
ext_strain = 0.176 * n_conc;
}

// Get displacement gradient
Tensor<2, dim> grad_u;
for (unsigned int d = 0; d < dim; ++d) {
FEValuesExtractors::Scalar component(d);
std::vector<Tensor<1, dim>> grad_u_component(n_q_points);
fe_values[component].get_function_gradients(locally_relevant_solution_elasticity, grad_u_component);
for (unsigned int e = 0; e < dim; ++e) grad_u[d][e] = grad_u_component[q][e];
}

// Deformation gradient: F = I + grad_u
Tensor<2, dim> F;
for (unsigned int i = 0; i < dim; ++i) {
for (unsigned int j = 0; j < dim; ++j) {
F[i][j] = (i == j ? 1.0 : 0.0) + grad_u[i][j];
}
}

// Right Cauchy-Green tensor: C = F^T * F
Tensor<2, dim> C;
for (unsigned int i = 0; i < dim; ++i) {
for (unsigned int j = 0; j < dim; ++j) {
C[i][j] = 0.0;
for (unsigned int k = 0; k < dim; ++k) {
C[i][j] += F[k][i] * F[k][j];
}
}
}

// Green-Lagrange strain: E_GL = 0.5 * (C - I)
SymmetricTensor<2, dim> E_GL;
for (unsigned int i = 0; i < dim; ++i) {
for (unsigned int j = i; j < dim; ++j) {
E_GL[i][j] = 0.5 * (C[i][j] - (i == j ? 1.0 : 0.0));
}
}

// Elastic Green-Lagrange strain
SymmetricTensor<2, dim> E_GL_elastic = E_GL;
for (unsigned int i = 0; i < dim; ++i) {
E_GL_elastic[i][i] -= ext_strain;
}

double tr_E_GL_2D = trace(E_GL_elastic);
double tr_E_GL_3D = tr_E_GL_2D - ext_strain;

// Second Piola-Kirchhoff stress
SymmetricTensor<2, dim> S;
for (unsigned int i = 0; i < dim; ++i) {
for (unsigned int j = i; j < dim; ++j) {
S[i][j] = lam * tr_E_GL_3D * (i == j ? 1.0 : 0.0) + 2.0 * mu_val * E_GL_elastic[i][j];
}
}

// Convert to Cauchy stress for output: sigma = (1/J) * F * S * F^T
double J = determinant(F);
if (std::abs(J) < NumericalConstants::determinant_threshold) J = 1.0; // Prevent division by zero

SymmetricTensor<2, dim> cauchy_stress;
for (unsigned int i = 0; i < dim; ++i) {
for (unsigned int j = i; j < dim; ++j) {
double sigma_ij = 0.0;
for (unsigned int k = 0; k < dim; ++k) {
for (unsigned int l = 0; l < dim; ++l) {
sigma_ij += F[i][k] * S[k][l] * F[j][l];
}
}
cauchy_stress[i][j] = sigma_ij / J;
}
}

avg_sigma_xx += cauchy_stress[0][0]; 
avg_sigma_yy += cauchy_stress[1][1]; 
avg_sigma_xy += cauchy_stress[0][1];

std::array<double, dim> principal_stresses = eigenvalues(cauchy_stress);
double sigma1 = *std::max_element(principal_stresses.begin(), principal_stresses.end());
double sigma2 = *std::min_element(principal_stresses.begin(), principal_stresses.end());

double sigma3 = nu * (cauchy_stress[0][0] + cauchy_stress[1][1]) - E_mod * ext_strain;

max_principal_stress_1 = std::max(max_principal_stress_1, sigma1);
avg_principal_stress_2 += sigma2;
avg_principal_stress_3 += sigma3;
}
stress_xx[cell_index] = avg_sigma_xx / n_q_points;
stress_yy[cell_index] = avg_sigma_yy / n_q_points;
stress_xy[cell_index] = avg_sigma_xy / n_q_points;
principal_stress_1[cell_index] = max_principal_stress_1;
principal_stress_2[cell_index] = avg_principal_stress_2 / n_q_points;
principal_stress_3[cell_index] = avg_principal_stress_3 / n_q_points;
++cell_index;
}
}

template <int dim>
double CMASProblem<dim>::compute_time_step_size()
{
double local_max_change_conc = 0.0;
double local_max_abs_change_phi = 0.0; // 存储相场变量的最大绝对变化量

// 1. 计算浓度的变化（保持原有逻辑，基于绝对或相对变化，这里假设浓度仍然关注变化幅度）
for (auto it = locally_owned_dofs_concentration.begin();
it != locally_owned_dofs_concentration.end(); ++it)
{
const types::global_dof_index i = *it;
double old_val = old_solution_concentration[i];
double new_val = locally_relevant_solution_concentration[i];

if (std::abs(old_val) > 1e-4) 
{
double change = std::abs(new_val - old_val) / std::abs(old_val);
local_max_change_conc = std::max(local_max_change_conc, change);
}
}

// 2. 计算相场变量的绝对变化量
for (auto it = locally_owned_dofs_phase_field.begin();
it != locally_owned_dofs_phase_field.end(); ++it)
{
const types::global_dof_index i = *it;
double new_val = locally_relevant_solution_phase_field[i];
double old_val = old_solution_phase_field[i];
double abs_change = std::abs(new_val - old_val);
local_max_abs_change_phi = std::max(local_max_abs_change_phi, abs_change);
}

// MPI 规约获取全局最大值
double max_change_conc = Utilities::MPI::max(local_max_change_conc, mpi_communicator);
double max_abs_change_phi = Utilities::MPI::max(local_max_abs_change_phi, mpi_communicator);

double new_dt = time_step;

// 逻辑修改：如果相场绝对变化量超过 0.05，则步长减半
if (max_abs_change_phi > 0.05)
{
new_dt = std::max(time_step * 0.5, TimeStep::dt_min);
pcout << "    -> Adaptive DT: |Δphi| = " << max_abs_change_phi 
<< " > 0.05. Halving dt to " << new_dt << std::endl;
}
else
{
// 如果变化平缓，尝试增加步长（基于浓度变化控制）
if (max_change_conc < 0.01 && max_abs_change_phi < 0.01)
new_dt = std::min(time_step * 2.0, TimeStep::dt_max);
else if (max_change_conc < 0.20 && max_abs_change_phi < 0.03)
new_dt = std::min(time_step * 1.2, TimeStep::dt_max);
else if (max_change_conc < 0.50)
new_dt = time_step;
else
new_dt = std::max(time_step * 0.8, TimeStep::dt_min);
}

return new_dt;
}

template <int dim>
void CMASProblem<dim>::update_history_field()
{
old_solution_concentration = locally_relevant_solution_concentration;
old_solution_phase_field = locally_relevant_solution_phase_field;

QGauss<dim> quadrature(fe_phase_field.degree + 1);
FEValues<dim> fe_values_elastic(fe_elasticity, quadrature, update_gradients | update_quadrature_points);
FEValues<dim> fe_values_conc(fe_concentration, quadrature, update_values);

const unsigned int n_q_points = quadrature.size();
std::vector<Vector<double>> conc_values(n_q_points, Vector<double>(6));
std::vector<std::vector<Tensor<1, dim>>> displacement_grads_all(n_q_points, std::vector<Tensor<1, dim>>(dim));

unsigned int cell_index = 0;
for (const auto &cell : dof_handler_elasticity.active_cell_iterators())
{
if (!cell->is_locally_owned()) continue;
fe_values_elastic.reinit(cell);
const auto conc_cell = cell->as_dof_handler_iterator(dof_handler_concentration);
fe_values_conc.reinit(conc_cell);
fe_values_conc.get_function_values(locally_relevant_solution_concentration, conc_values);

for (unsigned int d = 0; d < dim; ++d) {
FEValuesExtractors::Scalar displacement_component(d);
std::vector<Tensor<1, dim>> grad_u_component(n_q_points);
fe_values_elastic[displacement_component].get_function_gradients(locally_relevant_solution_elasticity, grad_u_component);
for (unsigned int q = 0; q < n_q_points; ++q) displacement_grads_all[q][d] = grad_u_component[q];
}

const Point<dim> cell_center = cell->center();
MaterialLayer layer = get_layer(cell_center[1]);
bool in_active_region = (layer == TC || layer == TGO);

if (cell_center[1] > Domain::y_tc_top - 0.01) in_active_region = false;

double E, nu, sigma_c;
if (layer == TC) { E = Material::E_TC; nu = Material::nu_TC; sigma_c = Material::sigma_TC; }
else if (layer == TGO) { E = get_tgo_E_base(cell_center); nu = Material::nu_TGO; sigma_c = Material::sigma_TGO; }
else { E = Material::E_BC; nu = Material::nu_BC; sigma_c = 1e3; }
double lam = lambda(E, nu);
double mu_val = mu(E, nu);

for (unsigned int q = 0; q < n_q_points; ++q)
{
double n_conc = conc_values[q][4];
double N = conc_values[q][5]; // TGO degradation

// Expansion ONLY in TC
double ext_strain = 0.0;
if (layer == TC) ext_strain = 0.176 * n_conc;

// Deformation gradient: F = I + grad_u (for large deformation)
Tensor<2, dim> F;
for (unsigned int i = 0; i < dim; ++i) {
for (unsigned int j = 0; j < dim; ++j) {
F[i][j] = (i == j ? 1.0 : 0.0) + displacement_grads_all[q][i][j];
}
}

// Right Cauchy-Green tensor: C = F^T * F
Tensor<2, dim> C;
for (unsigned int i = 0; i < dim; ++i) {
for (unsigned int j = 0; j < dim; ++j) {
C[i][j] = 0.0;
for (unsigned int k = 0; k < dim; ++k) {
C[i][j] += F[k][i] * F[k][j];
}
}
}

// Green-Lagrange strain: E_GL = 0.5 * (C - I)
SymmetricTensor<2, dim> E_GL;
for (unsigned int i = 0; i < dim; ++i) {
for (unsigned int j = i; j < dim; ++j) {
E_GL[i][j] = 0.5 * (C[i][j] - (i == j ? 1.0 : 0.0));
}
}

// Elastic Green-Lagrange strain
SymmetricTensor<2, dim> E_GL_elastic = E_GL;
for (unsigned int i = 0; i < dim; ++i) {
E_GL_elastic[i][i] -= ext_strain;
}

double tr_E_GL_2D = trace(E_GL_elastic);
double tr_E_GL_3D = tr_E_GL_2D - ext_strain;

// Second Piola-Kirchhoff stress
SymmetricTensor<2, dim> S;
for (unsigned int i = 0; i < dim; ++i) {
for (unsigned int j = i; j < dim; ++j) {
S[i][j] = lam * tr_E_GL_3D * (i == j ? 1.0 : 0.0) + 2.0 * mu_val * E_GL_elastic[i][j];
}
}

// Convert to Cauchy stress for principal stress calculation
double J = determinant(F);
if (std::abs(J) < NumericalConstants::determinant_threshold) J = 1.0;

SymmetricTensor<2, dim> cauchy_stress;
for (unsigned int i = 0; i < dim; ++i) {
for (unsigned int j = i; j < dim; ++j) {
double sigma_ij = 0.0;
for (unsigned int k = 0; k < dim; ++k) {
for (unsigned int l = 0; l < dim; ++l) {
sigma_ij += F[i][k] * S[k][l] * F[j][l];
}
}
cauchy_stress[i][j] = sigma_ij / J;
}
}

std::array<double, dim> principal_stresses = eigenvalues(cauchy_stress);
double sigma1 = *std::max_element(principal_stresses.begin(), principal_stresses.end());

// History logic for TGO using N
double local_sigma_c = sigma_c;
if (layer == TGO) {
double factor = (1.0 - N);
if (factor < 0) factor = 0;
local_sigma_c = Material::sigma_TGO * std::sqrt(factor);
if (local_sigma_c < 1e-6) local_sigma_c = 1e-6;
}

double Y_bar = 0.5 * std::max(0.0, sigma1) * std::max(0.0, sigma1) / E;
double Y0 = 0.5 * local_sigma_c * local_sigma_c / E;

// FIX for spurious phase field accumulation:
// Only update history when stress exceeds threshold or damage already initiated
double H_history = quadrature_point_history[cell_index][q];
bool damage_initiated = (H_history > Y0 + NumericalConstants::damage_threshold_tolerance) || (Y_bar > Y0);

if (in_active_region && damage_initiated) {
quadrature_point_history[cell_index][q] = std::max(H_history, Y_bar);
}
}
++cell_index;
}
}

// Monitor displacement at TC layer top midpoint (x=4.05, y=1.25)
template <int dim>
void CMASProblem<dim>::monitor_tc_top_midpoint_displacement()
{
// TC layer top midpoint coordinates
const double target_x = (Domain::x_min + Domain::x_max) / 2.0;  // 4.05 mm
const double target_y = Domain::y_tc_top;                        // 1.25 mm

// Find the cell containing the target point and evaluate displacement
Point<dim> target_point;
target_point[0] = target_x;
target_point[1] = target_y;

double local_ux = 0.0;
double local_uy = 0.0;
bool found_locally = false;
double min_distance = 1e20;

// Simple approach: find the closest node/DOF to the target point
// For more accuracy, we could use point_value, but this requires more infrastructure
for (const auto &cell : dof_handler_elasticity.active_cell_iterators())
{
if (!cell->is_locally_owned()) continue;

// Check if this cell might contain our point
for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
{
const Point<dim> vertex = cell->vertex(v);
double dist = vertex.distance(target_point);

if (dist < min_distance)
{
min_distance = dist;

// Get the DOF indices for this vertex
std::vector<types::global_dof_index> local_dof_indices(fe_elasticity.n_dofs_per_cell());
cell->get_dof_indices(local_dof_indices);

// Find the DOFs corresponding to this vertex
for (unsigned int i = 0; i < fe_elasticity.n_dofs_per_cell(); ++i)
{
const unsigned int comp = fe_elasticity.system_to_component_index(i).first;
const unsigned int vertex_idx = fe_elasticity.system_to_component_index(i).second;

// Check if this DOF corresponds to our vertex
// For Q1 elements, vertex_idx matches the local vertex number
if (vertex_idx == v && locally_owned_dofs_elasticity.is_element(local_dof_indices[i]))
{
if (comp == 0)
local_ux = locally_relevant_solution_elasticity[local_dof_indices[i]];
else if (comp == 1)
local_uy = locally_relevant_solution_elasticity[local_dof_indices[i]];
}
}
found_locally = true;
}
}
}

// Gather results from all processes - find the one with minimum distance
// Note: Using direct MPI calls here because dealii Utilities::MPI doesn't have
// built-in support for MINLOC operation needed to find the process owning the
// closest node and broadcast its displacement values.
struct {
double distance;
double ux;
double uy;
int rank;
} local_result;

local_result.distance = found_locally ? min_distance : 1e20;
local_result.ux = local_ux;
local_result.uy = local_uy;
local_result.rank = Utilities::MPI::this_mpi_process(mpi_communicator);

// Use dealii's MPI wrapper for minimum distance
double global_min_distance = Utilities::MPI::min(local_result.distance, mpi_communicator);

// Find the winner rank (the process with minimum distance)
int winner_rank = -1;
if (std::abs(local_result.distance - global_min_distance) < NumericalConstants::division_epsilon) {
winner_rank = local_result.rank;
}

// Find the winning rank using dealii's MPI wrapper
int global_winner_rank = Utilities::MPI::max(winner_rank, mpi_communicator);

// Broadcast displacement from winner using direct MPI call
// (dealii doesn't have a broadcast utility for scalar values from arbitrary ranks)
double final_ux = local_ux;
double final_uy = local_uy;
MPI_Bcast(&final_ux, 1, MPI_DOUBLE, global_winner_rank, mpi_communicator);
MPI_Bcast(&final_uy, 1, MPI_DOUBLE, global_winner_rank, mpi_communicator);

// Output the result from rank 0
pcout << "  [MONITOR] TC layer top midpoint (x=" << target_x << ", y=" << target_y << "):" << std::endl;
pcout << "            u_x = " << final_ux << " mm, u_y = " << final_uy << " mm" << std::endl;
pcout << "            |u| = " << std::sqrt(final_ux*final_ux + final_uy*final_uy) << " mm" << std::endl;
}

template <int dim>
void CMASProblem<dim>::output_results(const unsigned int step)
{
TimerOutput::Scope ts(computing_timer, "output");
pcout << "  Writing output for step " << step << std::endl;

// MODIFIED: Added N to output names
std::vector<std::string> concentration_names = {"c1", "c2", "c3", "c4", "n", "N"};
std::vector<DataComponentInterpretation::DataComponentInterpretation> concentration_component_interpretation(6, DataComponentInterpretation::component_is_scalar);
std::vector<std::string> displacement_names(dim, "u");
std::vector<DataComponentInterpretation::DataComponentInterpretation> displacement_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);

DataOut<dim> data_out;
data_out.attach_dof_handler(dof_handler_phase_field);
data_out.add_data_vector(dof_handler_concentration, locally_relevant_solution_concentration, concentration_names, concentration_component_interpretation);
data_out.add_data_vector(dof_handler_elasticity, locally_relevant_solution_elasticity, displacement_names, displacement_component_interpretation);

// ... (rest of output function same)
Vector<double> subdomain(triangulation.n_active_cells());
unsigned int cell_idx = 0;
for (const auto &cell : triangulation.active_cell_iterators()) {
if (cell->is_locally_owned()) subdomain(cell_idx) = triangulation.locally_owned_subdomain();
else subdomain(cell_idx) = -1;
++cell_idx;
}
data_out.add_data_vector(subdomain, "subdomain", DataOut<dim>::type_cell_data);

Vector<double> sigma1_out(triangulation.n_active_cells());
Vector<double> sigma2_out(triangulation.n_active_cells());
Vector<double> sigma3_out(triangulation.n_active_cells());
Vector<double> sxx_out(triangulation.n_active_cells());
Vector<double> syy_out(triangulation.n_active_cells());
Vector<double> sxy_out(triangulation.n_active_cells());
Vector<double> E_modulus_out(triangulation.n_active_cells());

cell_idx = 0; unsigned int local_idx = 0;
for (const auto &cell : triangulation.active_cell_iterators()) {
if (cell->is_locally_owned()) {
sigma1_out(cell_idx) = principal_stress_1(local_idx);
sigma2_out(cell_idx) = principal_stress_2(local_idx);
sigma3_out(cell_idx) = principal_stress_3(local_idx);
sxx_out(cell_idx) = stress_xx(local_idx);
syy_out(cell_idx) = stress_yy(local_idx);
sxy_out(cell_idx) = stress_xy(local_idx);
const Point<dim> center = cell->center();
E_modulus_out(cell_idx) = get_layer_base_E(center);
++local_idx;
} else {
sigma1_out(cell_idx) = 0; sigma2_out(cell_idx) = 0; sigma3_out(cell_idx) = 0;
sxx_out(cell_idx) = 0; syy_out(cell_idx) = 0; sxy_out(cell_idx) = 0;
E_modulus_out(cell_idx) = 0;
}
++cell_idx;
}
data_out.add_data_vector(sigma1_out, "sigma1", DataOut<dim>::type_cell_data);
data_out.add_data_vector(sigma2_out, "sigma2", DataOut<dim>::type_cell_data);
data_out.add_data_vector(sigma3_out, "sigma3", DataOut<dim>::type_cell_data);
data_out.add_data_vector(sxx_out, "stress_xx", DataOut<dim>::type_cell_data);
data_out.add_data_vector(syy_out, "stress_yy", DataOut<dim>::type_cell_data);
data_out.add_data_vector(sxy_out, "stress_xy", DataOut<dim>::type_cell_data);
data_out.add_data_vector(E_modulus_out, "E_modulus", DataOut<dim>::type_cell_data);

data_out.build_patches();
data_out.write_vtu_with_pvtu_record("./", "solution", step, mpi_communicator, 2, 0);

// === 新增代码开始 ===
if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
{
// 记录当前物理时间和对应的文件名
// 注意：write_vtu_with_pvtu_record 生成的文件名格式通常是 "solution_步数.pvtu"
std::string filename = "solution_" + std::to_string(step) + ".pvtu";
times_and_names.push_back({time, filename});

// 每次输出都更新 pvd 文件
std::ofstream output("solution.pvd");
DataOutBase::write_pvd_record(output, times_and_names);
}
// === 新增代码结束 ===
pcout << "  Output written for step " << step << std::endl;

}
template <int dim>
double CMASProblem<dim>::compute_max_tgo_stress()
{
double local_max_tgo_sigma1 = -1e20; // 初始化为一个很小的数

// 遍历所有本地拥有的单元
unsigned int local_cell_index = 0; // 仅对本地拥有单元计数，对应 principal_stress_*
for (const auto &cell : triangulation.active_cell_iterators())
{
if (!cell->is_locally_owned())
continue;

// 获取单元中心点以判断是否属于 TGO 层
const Point<dim> cell_center = cell->center();
MaterialLayer layer = get_layer(cell_center[1]);

if (layer == TGO)
{
// 从之前计算好的 principal_stress_1 向量中获取该单元的应力
// principal_stress_1 存储的是每个单元内部积分点的最大主应力的最大值
double cell_sigma1 = principal_stress_1(local_cell_index);
if (cell_sigma1 > local_max_tgo_sigma1)
{
local_max_tgo_sigma1 = cell_sigma1;
}
}
++local_cell_index;
}

// 在所有 MPI 进程中取最大值
double global_max_tgo_sigma1 = Utilities::MPI::max(local_max_tgo_sigma1, mpi_communicator);

// 如果没有找到 TGO 单元（理论上不应发生），避免返回负无穷
if (global_max_tgo_sigma1 < -1e19) global_max_tgo_sigma1 = 0.0;

return global_max_tgo_sigma1;
}


template <int dim>
void CMASProblem<dim>::run()
{
    pcout << "CMAS codebase (full-domain concentration-displacement mode, Parallel Version)" << std::endl;
pcout << "=========================================================" << std::endl;
pcout << "Running with " << Utilities::MPI::n_mpi_processes(mpi_communicator) << " MPI rank(s)..." << std::endl;

setup_mesh();
setup_system();
setup_constraints();
setup_matrices();

completely_distributed_solution_concentration = 0; locally_relevant_solution_concentration = 0; old_solution_concentration = 0;
completely_distributed_solution_phase_field = 0; locally_relevant_solution_phase_field = 0; old_solution_phase_field = 0;
completely_distributed_solution_elasticity = 0; locally_relevant_solution_elasticity = 0;
history_field = 0;
newton_update_phase_field = 0;

compute_principal_stress();
output_results(0);

while (time < TimeStep::total_time)
{
time += time_step;
++timestep_number;
pcout << "\nTime step " << timestep_number << ", t = " << time / 60.0 << " min" << ", dt = " << time_step << " s" << std::endl;

// 1. 浓度场耦合较弱，通常不需要放入内部迭代，除非反应热非常强
pcout << "  Solving concentration equations..." << std::endl;
assemble_concentration_system();
solve_concentration();

pcout << "  Solving elasticity equations..." << std::endl;
assemble_elasticity_system();
solve_elasticity();

compute_principal_stress();

// Monitor TC layer top midpoint displacement for large deformation verification
monitor_tc_top_midpoint_displacement();

double new_dt = compute_time_step_size();

// update_history_field() 已经在循环中执行过了，这里不需要再次执行
// 只需要将当前步的解保存为 old_solution 供下一步使用
old_solution_concentration = locally_relevant_solution_concentration;

time_step = new_dt;

if (timestep_number % TimeStep::output_interval == 0) output_results(timestep_number);

double max_phi = locally_relevant_solution_phase_field.linfty_norm();
double max_sigma = principal_stress_1.linfty_norm();
max_phi = Utilities::MPI::max(max_phi, mpi_communicator);
max_sigma = Utilities::MPI::max(max_sigma, mpi_communicator);
pcout << "  Max phi = " << max_phi << ", Max sigma1 = " << max_sigma << " MPa" << std::endl;
}
}

} // namespace CMASPhaseField

int main(int argc, char *argv[])
{
try {
using namespace dealii;
using namespace CMASPhaseField;
Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
deallog.depth_console(0);
CMASProblem<2> cmas_problem;
cmas_problem.run();
}
catch (std::exception &exc) {
std::cerr << "Exception: " << exc.what() << std::endl;
return 1;
}
return 0;
}
