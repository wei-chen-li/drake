#include "drake/solvers/osqp_solver.h"

#include <optional>
#include <unordered_map>
#include <vector>

#include <osqp.h>

#include "drake/common/text_logging.h"
#include "drake/math/eigen_sparse_triplet.h"
#include "drake/solvers/aggregate_costs_constraints.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/specific_options.h"

// This function must appear in the global namespace -- the Serialize pattern
// uses ADL (argument-dependent lookup) and the namespace for the OSQPSettings
// struct is the global namespace. (We can't even use an anonymous namespace!)
static void Serialize(
    drake::solvers::internal::SpecificOptions* archive,
    // NOLINTNEXTLINE(runtime/references) to match Serialize concept.
    OSQPSettings& settings) {
  using drake::MakeNameValue;
  archive->Visit(MakeNameValue("rho", &settings.rho));
  archive->Visit(MakeNameValue("sigma", &settings.sigma));
  archive->Visit(MakeNameValue("max_iter", &settings.max_iter));
  archive->Visit(MakeNameValue("eps_abs", &settings.eps_abs));
  archive->Visit(MakeNameValue("eps_rel", &settings.eps_rel));
  archive->Visit(MakeNameValue("eps_prim_inf", &settings.eps_prim_inf));
  archive->Visit(MakeNameValue("eps_dual_inf", &settings.eps_dual_inf));
  archive->Visit(MakeNameValue("alpha", &settings.alpha));
  archive->Visit(MakeNameValue("delta", &settings.delta));
  archive->Visit(MakeNameValue("polish", &settings.polish));
  archive->Visit(MakeNameValue("polish_refine_iter",  // BR
                               &settings.polish_refine_iter));
  archive->Visit(MakeNameValue("verbose", &settings.verbose));
  archive->Visit(MakeNameValue("scaled_termination",  // BR
                               &settings.scaled_termination));
  archive->Visit(MakeNameValue("check_termination",  // BR
                               &settings.check_termination));
  archive->Visit(MakeNameValue("warm_start", &settings.warm_start));
  archive->Visit(MakeNameValue("scaling", &settings.scaling));
  archive->Visit(MakeNameValue("adaptive_rho", &settings.adaptive_rho));
  archive->Visit(MakeNameValue("adaptive_rho_interval",  // BR
                               &settings.adaptive_rho_interval));
  archive->Visit(MakeNameValue("adaptive_rho_tolerance",  // BR
                               &settings.adaptive_rho_tolerance));
  archive->Visit(MakeNameValue("adaptive_rho_fraction",  // BR
                               &settings.adaptive_rho_fraction));
  archive->Visit(MakeNameValue("time_limit", &settings.time_limit));
}

namespace drake {
namespace solvers {
namespace {
void ParseQuadraticCosts(const MathematicalProgram& prog,
                         Eigen::SparseMatrix<c_float>* P_upper,
                         std::vector<c_float>* q, double* constant_cost_term) {
  DRAKE_ASSERT(static_cast<int>(q->size()) == prog.num_vars());
  std::vector<Eigen::Triplet<c_float>> P_upper_triplets;
  internal::ParseQuadraticCosts(prog, &P_upper_triplets, q, constant_cost_term);
  // Scale the matrix P in the cost.
  // Note that the linear term is scaled in ParseLinearCosts().
  const auto& scale_map = prog.GetVariableScaling();
  if (!scale_map.empty()) {
    for (auto& triplet : P_upper_triplets) {
      // Column
      const auto column = scale_map.find(triplet.col());
      if (column != scale_map.end()) {
        triplet = Eigen::Triplet<double>(triplet.row(), triplet.col(),
                                         triplet.value() * (column->second));
      }
      // Row
      const auto row = scale_map.find(triplet.row());
      if (row != scale_map.end()) {
        triplet = Eigen::Triplet<double>(triplet.row(), triplet.col(),
                                         triplet.value() * (row->second));
      }
    }
  }

  P_upper->resize(prog.num_vars(), prog.num_vars());
  P_upper->setFromTriplets(P_upper_triplets.begin(), P_upper_triplets.end());
}

void ParseLinearCosts(const MathematicalProgram& prog, std::vector<c_float>* q,
                      double* constant_cost_term) {
  // Add the linear costs to the osqp cost.
  DRAKE_ASSERT(static_cast<int>(q->size()) == prog.num_vars());
  internal::ParseLinearCosts(prog, q, constant_cost_term);

  // Scale the vector q in the cost.
  const auto& scale_map = prog.GetVariableScaling();
  if (!scale_map.empty()) {
    for (const auto& [index, scale] : scale_map) {
      q->at(index) *= scale;
    }
  }
}

// OSQP defines its own infinity in osqp/include/glob_opts.h.
c_float ConvertInfinity(double val) {
  if (std::isinf(val)) {
    if (val > 0) {
      return OSQP_INFTY;
    }
    return -OSQP_INFTY;
  }
  return static_cast<c_float>(val);
}

// Will call this function to parse both LinearConstraint and
// LinearEqualityConstraint.
template <typename C>
void ParseLinearConstraints(
    const MathematicalProgram& prog,
    const std::vector<Binding<C>>& linear_constraints,
    std::vector<Eigen::Triplet<c_float>>* A_triplets, std::vector<c_float>* l,
    std::vector<c_float>* u, int* num_A_rows,
    std::unordered_map<Binding<Constraint>, int>* constraint_start_row) {
  // Loop over the linear constraints, stack them to get l, u and A.
  for (const auto& constraint : linear_constraints) {
    const std::vector<int> x_indices =
        prog.FindDecisionVariableIndices(constraint.variables());
    const std::vector<Eigen::Triplet<double>> Ai_triplets =
        math::SparseMatrixToTriplets(constraint.evaluator()->get_sparse_A());
    const Binding<Constraint> constraint_cast =
        internal::BindingDynamicCast<Constraint>(constraint);
    constraint_start_row->emplace(constraint_cast, *num_A_rows);
    // Append constraint.A to osqp A.
    for (const auto& Ai_triplet : Ai_triplets) {
      A_triplets->emplace_back(*num_A_rows + Ai_triplet.row(),
                               x_indices[Ai_triplet.col()],
                               static_cast<c_float>(Ai_triplet.value()));
    }
    const int num_Ai_rows = constraint.evaluator()->num_constraints();
    l->reserve(l->size() + num_Ai_rows);
    u->reserve(u->size() + num_Ai_rows);
    for (int i = 0; i < num_Ai_rows; ++i) {
      l->push_back(ConvertInfinity(constraint.evaluator()->lower_bound()(i)));
      u->push_back(ConvertInfinity(constraint.evaluator()->upper_bound()(i)));
    }
    *num_A_rows += num_Ai_rows;
  }
}

void ParseBoundingBoxConstraints(
    const MathematicalProgram& prog,
    std::vector<Eigen::Triplet<c_float>>* A_triplets, std::vector<c_float>* l,
    std::vector<c_float>* u, int* num_A_rows,
    std::unordered_map<Binding<Constraint>, int>* constraint_start_row) {
  // Loop over the linear constraints, stack them to get l, u and A.
  for (const auto& constraint : prog.bounding_box_constraints()) {
    const Binding<Constraint> constraint_cast =
        internal::BindingDynamicCast<Constraint>(constraint);
    constraint_start_row->emplace(constraint_cast, *num_A_rows);
    // Append constraint.A to osqp A.
    for (int i = 0; i < static_cast<int>(constraint.GetNumElements()); ++i) {
      A_triplets->emplace_back(
          *num_A_rows + i,
          prog.FindDecisionVariableIndex(constraint.variables()(i)),
          static_cast<c_float>(1));
    }
    const int num_Ai_rows = constraint.evaluator()->num_constraints();
    l->reserve(l->size() + num_Ai_rows);
    u->reserve(u->size() + num_Ai_rows);
    for (int i = 0; i < num_Ai_rows; ++i) {
      l->push_back(ConvertInfinity(constraint.evaluator()->lower_bound()(i)));
      u->push_back(ConvertInfinity(constraint.evaluator()->upper_bound()(i)));
    }
    *num_A_rows += num_Ai_rows;
  }
}

void ParseAllLinearConstraints(
    const MathematicalProgram& prog, Eigen::SparseMatrix<c_float>* A,
    std::vector<c_float>* l, std::vector<c_float>* u,
    std::unordered_map<Binding<Constraint>, int>* constraint_start_row) {
  std::vector<Eigen::Triplet<c_float>> A_triplets;
  l->clear();
  u->clear();
  int num_A_rows = 0;
  ParseLinearConstraints(prog, prog.linear_constraints(), &A_triplets, l, u,
                         &num_A_rows, constraint_start_row);
  ParseLinearConstraints(prog, prog.linear_equality_constraints(), &A_triplets,
                         l, u, &num_A_rows, constraint_start_row);
  ParseBoundingBoxConstraints(prog, &A_triplets, l, u, &num_A_rows,
                              constraint_start_row);

  // Scale the matrix A.
  // Note that we only scale the columns of A, because the constraint has the
  // form l <= Ax <= u where the scaling of x enters the columns of A instead of
  // rows of A.
  const auto& scale_map = prog.GetVariableScaling();
  if (!scale_map.empty()) {
    for (auto& triplet : A_triplets) {
      auto column = scale_map.find(triplet.col());
      if (column != scale_map.end()) {
        triplet = Eigen::Triplet<double>(triplet.row(), triplet.col(),
                                         triplet.value() * (column->second));
      }
    }
  }

  A->resize(num_A_rows, prog.num_vars());
  A->setFromTriplets(A_triplets.begin(), A_triplets.end());
}

// Convert an Eigen::SparseMatrix to csc_matrix, to be used by osqp.
// Make sure the input Eigen sparse matrix is compressed, by calling
// makeCompressed() function.
// The caller of this function is responsible for freeing the memory allocated
// here.
csc* EigenSparseToCSC(const Eigen::SparseMatrix<c_float>& mat) {
  // A csc matrix is in the compressed column major.
  c_float* values =
      static_cast<c_float*>(c_malloc(sizeof(c_float) * mat.nonZeros()));
  c_int* inner_indices =
      static_cast<c_int*>(c_malloc(sizeof(c_int) * mat.nonZeros()));
  c_int* outer_indices =
      static_cast<c_int*>(c_malloc(sizeof(c_int) * (mat.cols() + 1)));
  for (int i = 0; i < mat.nonZeros(); ++i) {
    values[i] = *(mat.valuePtr() + i);
    inner_indices[i] = static_cast<c_int>(*(mat.innerIndexPtr() + i));
  }
  for (int i = 0; i < mat.cols() + 1; ++i) {
    outer_indices[i] = static_cast<c_int>(*(mat.outerIndexPtr() + i));
  }
  return csc_matrix(mat.rows(), mat.cols(), mat.nonZeros(), values,
                    inner_indices, outer_indices);
}

template <typename C>
void SetDualSolution(
    const std::vector<Binding<C>>& constraints,
    const Eigen::VectorXd& all_dual_solution,
    const std::unordered_map<Binding<Constraint>, int>& constraint_start_row,
    MathematicalProgramResult* result) {
  for (const auto& constraint : constraints) {
    // OSQP uses the dual variable `y` as the negation of the shadow price, so
    // we need to negate `all_dual_solution` as Drake interprets dual solution
    // as the shadow price.
    const Binding<Constraint> constraint_cast =
        internal::BindingDynamicCast<Constraint>(constraint);
    result->set_dual_solution(
        constraint,
        -all_dual_solution.segment(constraint_start_row.at(constraint_cast),
                                   constraint.evaluator()->num_constraints()));
  }
}
}  // namespace

bool OsqpSolver::is_available() {
  return true;
}

void OsqpSolver::DoSolve2(const MathematicalProgram& prog,
                          const Eigen::VectorXd& initial_guess,
                          internal::SpecificOptions* options,
                          MathematicalProgramResult* result) const {
  OsqpSolverDetails& solver_details =
      result->SetSolverDetailsType<OsqpSolverDetails>();

  // OSQP solves a convex quadratic programming problem
  // min 0.5 xᵀPx + qᵀx
  // s.t l ≤ Ax ≤ u
  // OSQP is written in C, so this function will be in C style.

  // Get the cost for the QP.
  // Since OSQP 0.6.0 the P matrix is required to be upper triangular.
  Eigen::SparseMatrix<c_float> P_upper_sparse;
  std::vector<c_float> q(prog.num_vars(), 0);
  double constant_cost_term{0};

  ParseQuadraticCosts(prog, &P_upper_sparse, &q, &constant_cost_term);
  ParseLinearCosts(prog, &q, &constant_cost_term);

  // linear_constraint_start_row[binding] stores the starting row index in A
  // corresponding to the linear constraint `binding`.
  std::unordered_map<Binding<Constraint>, int> constraint_start_row;

  // Parse the linear constraints.
  Eigen::SparseMatrix<c_float> A_sparse;
  std::vector<c_float> l, u;
  ParseAllLinearConstraints(prog, &A_sparse, &l, &u, &constraint_start_row);

  // Now pass the constraint and cost to osqp data.
  OSQPData* data = nullptr;

  // Populate data.
  data = static_cast<OSQPData*>(c_malloc(sizeof(OSQPData)));

  data->n = prog.num_vars();
  data->m = A_sparse.rows();
  data->P = EigenSparseToCSC(P_upper_sparse);
  data->q = q.data();
  data->A = EigenSparseToCSC(A_sparse);
  data->l = l.data();
  data->u = u.data();

  // Create the settings, initialized to the upstream defaults.
  OSQPSettings* settings =
      static_cast<OSQPSettings*>(c_malloc(sizeof(OSQPSettings)));
  osqp_set_default_settings(settings);
  // Customize the defaults for Drake.
  // - Default polish to true, to get an accurate solution.
  // - Disable adaptive rho, for determinism.
  settings->polish = 1;
  settings->adaptive_rho_interval = ADAPTIVE_RHO_FIXED;
  // Apply the user's additional options (if any).
  options->Respell([](const auto& common, auto* respelled) {
    respelled->emplace("verbose", common.print_to_console ? 1 : 0);
    // OSQP does not support setting the number of threads so we ignore the
    // kMaxThreads option.
  });
  options->CopyToSerializableStruct(settings);

  // If any step fails, it will set the solution_result and skip other steps.
  std::optional<SolutionResult> solution_result;

  // Setup workspace.
  OSQPWorkspace* work = nullptr;
  if (!solution_result) {
    const c_int osqp_setup_err = osqp_setup(&work, data, settings);
    if (osqp_setup_err != 0) {
      solution_result = SolutionResult::kInvalidInput;
    }
  }

  if (!solution_result && initial_guess.array().isFinite().all()) {
    const c_int osqp_warm_err = osqp_warm_start_x(work, initial_guess.data());
    if (osqp_warm_err != 0) {
      solution_result = SolutionResult::kInvalidInput;
    }
  }

  // Solve problem.
  if (!solution_result) {
    DRAKE_THROW_UNLESS(work != nullptr);
    const c_int osqp_solve_err = osqp_solve(work);
    if (osqp_solve_err != 0) {
      solution_result = SolutionResult::kInvalidInput;
    }
  }

  // Extract results.
  if (!solution_result) {
    DRAKE_THROW_UNLESS(work->info != nullptr);

    solver_details.iter = work->info->iter;
    solver_details.status_val = work->info->status_val;
    solver_details.primal_res = work->info->pri_res;
    solver_details.dual_res = work->info->dua_res;
    solver_details.setup_time = work->info->setup_time;
    solver_details.solve_time = work->info->solve_time;
    solver_details.polish_time = work->info->polish_time;
    solver_details.run_time = work->info->run_time;
    solver_details.rho_updates = work->info->rho_updates;

    // We set the primal and dual variables as long as osqp_solve() is finished.
    const Eigen::Map<Eigen::Matrix<c_float, Eigen::Dynamic, 1>> osqp_sol(
        work->solution->x, prog.num_vars());

    // Scale solution back if `scale_map` is not empty.
    const auto& scale_map = prog.GetVariableScaling();
    if (!scale_map.empty()) {
      drake::VectorX<double> scaled_sol = osqp_sol.cast<double>();
      for (const auto& [index, scale] : scale_map) {
        scaled_sol(index) *= scale;
      }
      result->set_x_val(scaled_sol);
    } else {
      result->set_x_val(osqp_sol.cast<double>());
    }
    solver_details.y =
        Eigen::Map<Eigen::VectorXd>(work->solution->y, work->data->m);
    SetDualSolution(prog.linear_constraints(), solver_details.y,
                    constraint_start_row, result);
    SetDualSolution(prog.linear_equality_constraints(), solver_details.y,
                    constraint_start_row, result);
    SetDualSolution(prog.bounding_box_constraints(), solver_details.y,
                    constraint_start_row, result);

    switch (work->info->status_val) {
      case OSQP_SOLVED:
      case OSQP_SOLVED_INACCURATE: {
        result->set_optimal_cost(work->info->obj_val + constant_cost_term);
        solution_result = SolutionResult::kSolutionFound;
        break;
      }
      case OSQP_PRIMAL_INFEASIBLE:
      case OSQP_PRIMAL_INFEASIBLE_INACCURATE: {
        solution_result = SolutionResult::kInfeasibleConstraints;
        result->set_optimal_cost(MathematicalProgram::kGlobalInfeasibleCost);
        break;
      }
      case OSQP_DUAL_INFEASIBLE:
      case OSQP_DUAL_INFEASIBLE_INACCURATE: {
        solution_result = SolutionResult::kDualInfeasible;
        break;
      }
      case OSQP_MAX_ITER_REACHED: {
        solution_result = SolutionResult::kIterationLimit;
        break;
      }
      default: {
        solution_result = SolutionResult::kSolverSpecificError;
        break;
      }
    }
  }
  result->set_solution_result(solution_result.value());

  // Clean workspace.
  osqp_cleanup(work);
  c_free(data->P->x);
  c_free(data->P->i);
  c_free(data->P->p);
  c_free(data->P);
  c_free(data->A->x);
  c_free(data->A->i);
  c_free(data->A->p);
  c_free(data->A);
  c_free(data);
  c_free(settings);
}

}  // namespace solvers
}  // namespace drake
