#include "drake/solvers/mosek_solver.h"

#include <filesystem>

#include <gtest/gtest.h>

#include "drake/common/temp_directory.h"
#include "drake/common/test_utilities/expect_throws_message.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/mixed_integer_optimization_util.h"
#include "drake/solvers/test/exponential_cone_program_examples.h"
#include "drake/solvers/test/l2norm_cost_examples.h"
#include "drake/solvers/test/linear_program_examples.h"
#include "drake/solvers/test/quadratic_constrained_program_examples.h"
#include "drake/solvers/test/quadratic_program_examples.h"
#include "drake/solvers/test/second_order_cone_program_examples.h"
#include "drake/solvers/test/semidefinite_program_examples.h"
#include "drake/solvers/test/sos_examples.h"

namespace drake {
namespace solvers {
namespace test {
const double kInf = std::numeric_limits<double>::infinity();
TEST_P(LinearProgramTest, TestLP) {
  MosekSolver solver;
  prob()->RunProblem(&solver);
}

INSTANTIATE_TEST_SUITE_P(
    MosekTest, LinearProgramTest,
    ::testing::Combine(::testing::ValuesIn(linear_cost_form()),
                       ::testing::ValuesIn(linear_constraint_form()),
                       ::testing::ValuesIn(linear_problems())));

TEST_F(UnboundedLinearProgramTest0, Test) {
  MosekSolver solver;
  if (solver.available()) {
    const MathematicalProgram& const_prog = *prog_;
    MathematicalProgramResult result;
    solver.Solve(const_prog, {}, {}, &result);
    // Mosek can only detect dual infeasibility, not primal unboundedness.
    EXPECT_FALSE(result.is_success());
    EXPECT_EQ(result.get_solution_result(), SolutionResult::kDualInfeasible);
    const MosekSolverDetails& mosek_solver_details =
        result.get_solver_details<MosekSolver>();
    EXPECT_EQ(mosek_solver_details.rescode, 0);
    // This problem status is defined in
    // https://docs.mosek.com/11.0/capi/constants.html#mosek.prosta
    const int MSK_SOL_STA_DUAL_INFEAS_CER = 6;
    EXPECT_EQ(mosek_solver_details.solution_status,
              MSK_SOL_STA_DUAL_INFEAS_CER);
    const auto x_sol = result.GetSolution(x_);
    EXPECT_FALSE(std::isnan(x_sol(0)));
    EXPECT_FALSE(std::isnan(x_sol(1)));
  }
}

TEST_F(UnboundedLinearProgramTest1, Test) {
  MosekSolver solver;
  if (solver.available()) {
    MathematicalProgramResult result;
    solver.Solve(*prog_, {}, {}, &result);
    // Mosek can only detect dual infeasibility, not primal unboundedness.
    EXPECT_EQ(result.get_solution_result(), SolutionResult::kDualInfeasible);
  }
}

TEST_F(DuplicatedVariableLinearProgramTest1, Test) {
  MosekSolver solver;
  if (solver.available()) {
    CheckSolution(solver);
  }
}

TEST_P(QuadraticProgramTest, TestQP) {
  MosekSolver solver;
  prob()->RunProblem(&solver);
}

INSTANTIATE_TEST_SUITE_P(
    MosekTest, QuadraticProgramTest,
    ::testing::Combine(::testing::ValuesIn(quadratic_cost_form()),
                       ::testing::ValuesIn(linear_constraint_form()),
                       ::testing::ValuesIn(quadratic_problems())));

GTEST_TEST(QPtest, TestUnitBallExample) {
  MosekSolver solver;
  if (solver.available()) {
    TestQPonUnitBallExample(solver);
  }
}

GTEST_TEST(QPtest, TestQuadraticCostVariableOrder) {
  MosekSolver solver;
  if (solver.available()) {
    TestQuadraticCostVariableOrder(solver);
  }
}

TEST_P(TestEllipsoidsSeparation, TestSOCP) {
  MosekSolver mosek_solver;
  if (mosek_solver.available()) {
    SolveAndCheckSolution(mosek_solver);
  }
}

INSTANTIATE_TEST_SUITE_P(
    MosekTest, TestEllipsoidsSeparation,
    ::testing::ValuesIn(GetEllipsoidsSeparationProblems()));

GTEST_TEST(TestDuplicatedVariableQuadraticProgram, Test) {
  MosekSolver solver;
  if (solver.available()) {
    TestDuplicatedVariableQuadraticProgram(solver);
  }
}

TEST_P(TestQPasSOCP, TestSOCP) {
  MosekSolver mosek_solver;
  if (mosek_solver.available()) {
    SolveAndCheckSolution(mosek_solver);
  }
}

INSTANTIATE_TEST_SUITE_P(MosekTest, TestQPasSOCP,
                         ::testing::ValuesIn(GetQPasSOCPProblems()));

TEST_P(TestFindSpringEquilibrium, TestSOCP) {
  MosekSolver mosek_solver;
  if (mosek_solver.available()) {
    SolveAndCheckSolution(mosek_solver);
  }
}

INSTANTIATE_TEST_SUITE_P(
    MosekTest, TestFindSpringEquilibrium,
    ::testing::ValuesIn(GetFindSpringEquilibriumProblems()));

GTEST_TEST(TestSOCP, MaximizeGeometricMeanTrivialProblem1) {
  MaximizeGeometricMeanTrivialProblem1 prob;
  MosekSolver solver;
  if (solver.available()) {
    const auto result = solver.Solve(prob.prog(), {}, {});
    prob.CheckSolution(result, 1E-7);
  }
}

GTEST_TEST(TestSOCP, MaximizeGeometricMeanTrivialProblem2) {
  MaximizeGeometricMeanTrivialProblem2 prob;
  MosekSolver solver;
  if (solver.available()) {
    const auto result = solver.Solve(prob.prog(), {}, {});
    prob.CheckSolution(result, 1E-7);
  }
}

GTEST_TEST(TestSOCP, SmallestEllipsoidCoveringProblem) {
  MosekSolver solver;
  // Mosek 10 returns a solution that is accurate up to 1.3E-5 for this specific
  // problem. Might need to change the tolerance when we upgrade Mosek.
  SolveAndCheckSmallestEllipsoidCoveringProblems(solver, {}, 1.3E-5);
}

GTEST_TEST(TestSOCP, TestSocpDuplicatedVariable1) {
  MosekSolver solver;
  TestSocpDuplicatedVariable1(solver, std::nullopt, 1E-6);
}

GTEST_TEST(TestSOCP, TestSocpDuplicatedVariable2) {
  MosekSolver solver;
  TestSocpDuplicatedVariable2(solver, std::nullopt, 1E-6);
}

GTEST_TEST(TestSOCP, TestSocpDuplicatedVariable3) {
  MosekSolver solver;
  TestSocpDuplicatedVariable3(solver, std::nullopt, 1E-5);
}

GTEST_TEST(TestL2NormCost, ShortestDistanceToThreePoints) {
  MosekSolver solver;
  ShortestDistanceToThreePoints tester{};
  tester.CheckSolution(solver, std::nullopt, 1E-4);
}

GTEST_TEST(TestL2NormCost, ShortestDistanceFromCylinderToPoint) {
  MosekSolver solver;
  ShortestDistanceFromCylinderToPoint tester{};
  tester.CheckSolution(solver);
}

GTEST_TEST(TestL2NormCost, ShortestDistanceFromPlaneToTwoPoints) {
  MosekSolver solver;
  ShortestDistanceFromPlaneToTwoPoints tester{};
  tester.CheckSolution(solver, std::nullopt, 5E-4);
}

GTEST_TEST(TestSemidefiniteProgram, TrivialSDP) {
  MosekSolver mosek_solver;
  if (mosek_solver.available()) {
    TestTrivialSDP(mosek_solver, 1E-8);
  }
}

GTEST_TEST(TestSemidefiniteProgram, CommonLyapunov) {
  MosekSolver mosek_solver;
  if (mosek_solver.available()) {
    FindCommonLyapunov(mosek_solver, {}, 1E-8);
  }
}

GTEST_TEST(TestSemidefiniteProgram, OuterEllipsoid) {
  MosekSolver mosek_solver;
  if (mosek_solver.available()) {
    FindOuterEllipsoid(mosek_solver, {}, 1E-6);
  }
}

GTEST_TEST(TestSemidefiniteProgram, EigenvalueProblem) {
  MosekSolver mosek_solver;
  if (mosek_solver.available()) {
    SolveEigenvalueProblem(mosek_solver, {}, 1E-7, /*check_dual=*/true);
  }
}

GTEST_TEST(TestSemidefiniteProgram, SolveSDPwithSecondOrderConeExample1) {
  MosekSolver mosek_solver;
  if (mosek_solver.available()) {
    SolveSDPwithSecondOrderConeExample1(mosek_solver, 1E-7);
  }
}

GTEST_TEST(TestSemidefiniteProgram, SolveSDPwithSecondOrderConeExample2) {
  MosekSolver mosek_solver;
  if (mosek_solver.available()) {
    SolveSDPwithSecondOrderConeExample2(mosek_solver, 1E-7);
  }
}

GTEST_TEST(TestSemidefiniteProgram, SolveSDPwithOverlappingVariables) {
  MosekSolver mosek_solver;
  if (mosek_solver.available()) {
    SolveSDPwithOverlappingVariables(mosek_solver, 1E-7);
  }
}

GTEST_TEST(TestExponentialConeProgram, ExponentialConeTrivialExample) {
  MosekSolver solver;
  if (solver.available()) {
    ExponentialConeTrivialExample(solver, 1E-5, true);
  }
}

GTEST_TEST(TestExponentialConeProgram, MinimizeKLDivengence) {
  MosekSolver solver;
  if (solver.available()) {
    MinimizeKLDivergence(solver, 2E-5);
  }
}

GTEST_TEST(TestExponentialConeProgram, MinimalEllipsoidConveringPoints) {
  MosekSolver solver;
  if (solver.available()) {
    MinimalEllipsoidCoveringPoints(solver, 1E-6);
  }
}

GTEST_TEST(TestExponentialConeProgram, MatrixLogDeterminantLower) {
  MosekSolver mosek_solver;
  if (mosek_solver.available()) {
    MatrixLogDeterminantLower(mosek_solver, 1E-6);
  }
}

GTEST_TEST(MosekTest, TestLogging) {
  // Test if we can print the logging info to a log file.
  MathematicalProgram prog;
  const auto x = prog.NewContinuousVariables<2>();
  prog.AddLinearConstraint(x(0) + x(1) == 1);

  const std::string log_file = temp_directory() + "/mosek_logging.log";
  EXPECT_FALSE(std::filesystem::exists({log_file}));
  MosekSolver solver;
  MathematicalProgramResult result;
  solver.Solve(prog, {}, {}, &result);
  // By default, no logging file.
  EXPECT_FALSE(std::filesystem::exists({log_file}));
  // Print to console. We can only test this doesn't cause any runtime error. We
  // can't test if the logging message is actually printed to the console.
  SolverOptions solver_options;
  solver_options.SetOption(CommonSolverOption::kPrintToConsole, 1);
  solver.Solve(prog, {}, solver_options, &result);
  solver_options.SetOption(CommonSolverOption::kPrintToConsole, 0);
  // Output the logging to the console
  solver_options.SetOption(CommonSolverOption::kPrintFileName, log_file);
  solver.Solve(prog, {}, solver_options, &result);
  EXPECT_TRUE(std::filesystem::exists({log_file}));
  // Now set both print to console and the log file. This will cause an error.
  solver_options.SetOption(CommonSolverOption::kPrintToConsole, 1);
  DRAKE_EXPECT_THROWS_MESSAGE(
      solver.Solve(prog, {}, solver_options, &result),
      ".*cannot print to both the console and a file.*");
}

GTEST_TEST(MosekTest, SolverOptionsTest) {
  // We test that passing solver options changes the behavior of
  // MosekSolver::Solve().
  MathematicalProgram prog;
  auto x = prog.NewContinuousVariables<2>();
  prog.AddLinearConstraint(100 * x(0) + 100 * x(1) <= 1);
  prog.AddConstraint(x(0) >= 0);
  prog.AddConstraint(x(1) >= 0);
  prog.AddLinearCost(1E5 * x(0) + x(1));

  MosekSolver mosek_solver;
  SolverOptions solver_options;
  MathematicalProgramResult result;

  // Set a string option, to at least make sure nothing crashes. Unfortunately,
  // there is no MOSEK string option that affects the output or logging, so we
  // cannot actually test that the option is propagated correctly.
  solver_options.SetOption(MosekSolver::id(), "MSK_SPAR_BAS_SOL_FILE_NAME",
                           "/tmp/mosek.bas");

  // Solve with 1e3 => failed.
  solver_options.SetOption(MosekSolver::id(), "MSK_DPAR_DATA_TOL_C_HUGE", 1E3);
  mosek_solver.Solve(prog, {}, solver_options, &result);
  EXPECT_FALSE(result.is_success());
  // This response code is defined in
  // https://docs.mosek.com/11.0/capi/response-codes.html#mosek.rescode
  const int MSK_RES_ERR_HUGE_C{1375};
  EXPECT_EQ(result.get_solver_details<MosekSolver>().rescode,
            MSK_RES_ERR_HUGE_C);

  // Solve with 1e6 => success.
  solver_options.SetOption(MosekSolver::id(), "MSK_DPAR_DATA_TOL_C_HUGE", 1E6);
  mosek_solver.Solve(prog, {}, solver_options, &result);
  EXPECT_TRUE(result.is_success());
}

GTEST_TEST(MosekSolver, SolverOptionsErrorTest) {
  // Set a non-existing option. Mosek should report error.
  MathematicalProgram prog;
  auto x = prog.NewContinuousVariables<2>();
  prog.AddLinearConstraint(x(0) + x(1) >= 0);
  MathematicalProgramResult result;
  MosekSolver mosek_solver;

  // Test `int`.
  {
    SolverOptions solver_options;
    solver_options.SetOption(MosekSolver::id(), "no_such_option", 42);
    DRAKE_EXPECT_THROWS_MESSAGE(
        mosek_solver.Solve(prog, {}, solver_options, &result),
        ".*cannot set Mosek option \'no_such_option\' to value \'42\'.*");
  }

  // Test `double`.
  {
    SolverOptions solver_options;
    solver_options.SetOption(MosekSolver::id(), "no_such_option", 0.5);
    DRAKE_EXPECT_THROWS_MESSAGE(
        mosek_solver.Solve(prog, {}, solver_options, &result),
        ".*cannot set Mosek option \'no_such_option\' to value \'0.5\'.*");
  }

  // Test `string`.
  {
    SolverOptions solver_options;
    solver_options.SetOption(MosekSolver::id(), "no_such_option", "foo");
    DRAKE_EXPECT_THROWS_MESSAGE(
        mosek_solver.Solve(prog, {}, solver_options, &result),
        ".*cannot set Mosek option \'no_such_option\' to value \'foo\'.*");
  }
}

GTEST_TEST(MosekTest, Write) {
  // Write model to a file.
  MathematicalProgram prog;
  auto x = prog.NewContinuousVariables<2>();
  prog.AddLinearEqualityConstraint(x(0) + x(1) == 1);
  prog.AddQuadraticCost(x(0) * x(0) + x(1) * x(1), true /* is_convex */);
  MosekSolver mosek_solver;
  SolverOptions solver_options;
  const std::string file = temp_directory() + "mosek.mps";
  EXPECT_FALSE(std::filesystem::exists(file));
  solver_options.SetOption(MosekSolver::id(), "writedata", file);
  mosek_solver.Solve(prog, {}, solver_options);
  EXPECT_TRUE(std::filesystem::exists(file));
  std::filesystem::remove(file);
  // Set "writedata" to "". Now expect no model file.
  solver_options.SetOption(MosekSolver::id(), "writedata", "");
  mosek_solver.Solve(prog, {}, solver_options);
  EXPECT_FALSE(std::filesystem::exists(file));
}

GTEST_TEST(MosekSolver, TestInitialGuess) {
  // Mosek allows to set initial guess for integer/binary variables.
  // Solve the following mixed-integer problem
  // Find a point C on one of the line segment A1A2, A2A3, A3A4, A4A1 such that
  // the distance from the point C to the point D = (0, 0) is minimized, where
  // A1 = (-1, 0), A2 = (0, 1), A3 = (2, 0), A4 = (1, -0.5)
  MathematicalProgram prog;
  auto lambda = prog.NewContinuousVariables<5>();
  auto y = prog.NewBinaryVariables<4>();
  AddSos2Constraint(&prog, lambda.cast<symbolic::Expression>(),
                    y.cast<symbolic::Expression>());
  Eigen::Matrix<double, 2, 5> pts_A;
  pts_A.col(0) << -1, 0;
  pts_A.col(1) << 0, 1;
  pts_A.col(2) << 2, 0;
  pts_A.col(3) << 1, -0.5;
  pts_A.col(4) = pts_A.col(0);
  // point C in the documentation above.
  auto pt_C = prog.NewContinuousVariables<2>();
  prog.AddLinearEqualityConstraint(pt_C == pts_A * lambda);
  prog.AddQuadraticCost(pt_C(0) * pt_C(0) + pt_C(1) * pt_C(1));

  MosekSolver solver;
  SolverOptions solver_options;
  // Allow only one solution (the one corresponding to the initial guess).
  solver_options.SetOption(solver.id(), "MSK_IPAR_MIO_MAX_NUM_SOLUTIONS", 1);
  // By setting y = (0, 1, 0, 0), pt_C = (0, 1), lambda = (0, 1, 0, 0, 0) point
  // C is on the line segment A2A3. This is a valid integral solution with
  // distance to origin = 1.
  prog.SetInitialGuess(pt_C, Eigen::Vector2d(0, 1));
  prog.SetInitialGuess(
      lambda, (Eigen::Matrix<double, 5, 1>() << 0, 1, 0, 0, 0).finished());
  prog.SetInitialGuess(y, Eigen::Vector4d(0, 1, 0, 0));
  MathematicalProgramResult result;
  solver.Solve(prog, prog.initial_guess(), solver_options, &result);
  const double tol = 1E-8;
  EXPECT_TRUE(result.is_success());
  EXPECT_NEAR(result.get_optimal_cost(), 1, tol);

  // By setting MSK_IPAR_MIO_CONSTRUCT_SOL=1, Mosek will first try to construct
  // the continuous solution given the initial binary variable solutions. In
  // this case it searches the point on line segment A2A3 that is closest
  // to the origin, which is (0.4, 0.8).
  solver_options.SetOption(solver.id(), "MSK_IPAR_MIO_CONSTRUCT_SOL", 1);
  solver.Solve(prog, prog.initial_guess(), solver_options, &result);
  EXPECT_TRUE(result.is_success());
  EXPECT_NEAR(result.get_optimal_cost(), 0.8, tol);

  // By setting y = (0, 0, 0, 1), point C is on the line segment A4A1. The
  // minimal squared distance is  1.0 / 17
  prog.SetInitialGuess(y, Eigen::Vector4d(0, 0, 0, 1));
  solver.Solve(prog, prog.initial_guess(), solver_options, &result);
  EXPECT_TRUE(result.is_success());
  EXPECT_NEAR(result.get_optimal_cost(), 1.0 / 17, tol);
}

GTEST_TEST(MosekTest, UnivariateQuarticSos) {
  UnivariateQuarticSos dut;
  MosekSolver solver;
  if (solver.available()) {
    const auto result = solver.Solve(dut.prog());
    dut.CheckResult(result, 1E-10);
  }
}

GTEST_TEST(MosekTest, BivariateQuarticSos) {
  BivariateQuarticSos dut;
  MosekSolver solver;
  if (solver.available()) {
    const auto result = solver.Solve(dut.prog());
    dut.CheckResult(result, 1E-10);
  }
}

GTEST_TEST(MosekTest, SimpleSos1) {
  SimpleSos1 dut;
  MosekSolver solver;
  if (solver.available()) {
    const auto result = solver.Solve(dut.prog());
    dut.CheckResult(result, 1E-8);
  }
}

GTEST_TEST(MosekTest, MotzkinPolynomial) {
  MotzkinPolynomial dut;
  MosekSolver solver;
  if (solver.available()) {
    const auto result = solver.Solve(dut.prog());
    dut.CheckResult(result, 1E-8);
  }
}

GTEST_TEST(MosekTest, UnivariateNonnegative1) {
  UnivariateNonnegative1 dut;
  MosekSolver solver;
  if (solver.available()) {
    const auto result = solver.Solve(dut.prog());
    dut.CheckResult(result, 1E-9);
  }
}

GTEST_TEST(MosekTest, MinimalDistanceFromSphereProblem) {
  for (bool with_linear_cost : {true, false}) {
    MinimalDistanceFromSphereProblem<3> dut1(Eigen::Vector3d(0, 0, 0),
                                             Eigen::Vector3d(0, 0, 0), 1,
                                             with_linear_cost);
    MinimalDistanceFromSphereProblem<3> dut2(Eigen::Vector3d(0, 0, 1),
                                             Eigen::Vector3d(0, 0, 0), 1,
                                             with_linear_cost);
    MinimalDistanceFromSphereProblem<3> dut3(Eigen::Vector3d(0, 1, 1),
                                             Eigen::Vector3d(0, 0, 0), 1,
                                             with_linear_cost);
    MinimalDistanceFromSphereProblem<3> dut4(Eigen::Vector3d(0, 1, 1),
                                             Eigen::Vector3d(-1, -1, 0), 1,
                                             with_linear_cost);
    MosekSolver solver;
    if (solver.available()) {
      const double tol = 1E-4;
      dut1.SolveAndCheckSolution(solver, tol);
      dut2.SolveAndCheckSolution(solver, tol);
      dut3.SolveAndCheckSolution(solver, tol);
      dut4.SolveAndCheckSolution(solver, tol);
    }
  }
}

GTEST_TEST(MosekTest, SolveSDPwithQuadraticCosts) {
  MosekSolver solver;
  if (solver.available()) {
    SolveSDPwithQuadraticCosts(solver, 1E-8);
  }
}

GTEST_TEST(MosekTest, TestTrivial1x1SDP) {
  MosekSolver solver;
  if (solver.available()) {
    TestTrivial1x1SDP(solver, 1E-8, /*check_dual=*/true, /*dual_tol=*/1E-8);
  }
}

GTEST_TEST(MosekTest, TestTrivial2x2SDP) {
  MosekSolver solver;
  if (solver.available()) {
    TestTrivial2x2SDP(solver, 1E-5, /*check_dual=*/true, /*dual_tol=*/1E-8);
  }
}

GTEST_TEST(MosekTest, Test1x1with3x3SDP) {
  MosekSolver solver;
  if (solver.available()) {
    Test1x1with3x3SDP(solver, 1E-4, /*check_dual=*/true, /*dual_tol=*/1E-7);
  }
}

GTEST_TEST(MosekTest, Test2x2with3x3SDP) {
  MosekSolver solver;
  if (solver.available()) {
    Test2x2with3x3SDP(solver, 1E-3, /*check_dual=*/true, /*dual_tol=*/1E-3);
  }
}

GTEST_TEST(MosekTest, TestTrivial1x1LMI) {
  MosekSolver solver;
  if (solver.available()) {
    TestTrivial1x1LMI(solver, 1E-7, /*check_dual=*/true, /*dual_tol=*/1E-7);
  }
}

GTEST_TEST(MosekTest, Test2X2LMI) {
  MosekSolver solver;
  if (solver.available()) {
    Test2x2LMI(solver, 1E-7, /*check_dual=*/true, /*dual_tol=*/1E-7);
  }
}

GTEST_TEST(MosekTest, TestHankel) {
  MosekSolver solver;
  if (solver.available()) {
    TestHankel(solver, 1E-5, true, 1E-6);
  }
}

GTEST_TEST(MosekTest, LPDualSolution1) {
  MosekSolver solver;
  if (solver.available()) {
    TestLPDualSolution1(solver, 1E-8);
  }
}

GTEST_TEST(MosekTest, LPDualSolution2) {
  MosekSolver solver;
  if (solver.available()) {
    TestLPDualSolution2(solver, 1E-8);
  }
}

GTEST_TEST(MosekTest, LPDualSolution2Scaled) {
  MosekSolver solver;
  if (solver.available()) {
    TestLPDualSolution2Scaled(solver, 1E-8);
  }
}

GTEST_TEST(MosekTest, LPDualSolution3) {
  MosekSolver solver;
  if (solver.available()) {
    TestLPDualSolution3(solver, 1E-8);
  }
}

GTEST_TEST(MosekTest, LPDualSolution4) {
  MosekSolver solver;
  if (solver.available()) {
    TestLPDualSolution4(solver, 1E-8);
  }
}

GTEST_TEST(MosekTest, LPDualSolution5) {
  MosekSolver solver;
  TestLPDualSolution5(solver, 1E-8);
}

GTEST_TEST(MosekTest, QPDualSolution1) {
  MosekSolver solver;
  if (solver.available()) {
    SolverOptions solver_options;
    // The default tolerance is 1E-8, and one entry of the optimal solution is
    // 0. This means the error on the primal and dual solution is in the order
    // of 1E-4, that is too large.
    solver_options.SetOption(solver.id(), "MSK_DPAR_INTPNT_QO_TOL_REL_GAP",
                             1e-12);
    TestQPDualSolution1(solver, solver_options, 6E-6);
  }
}

GTEST_TEST(MosekTest, QPDualSolution2) {
  MosekSolver solver;
  if (solver.available()) {
    TestQPDualSolution2(solver);
  }
}

GTEST_TEST(MosekTest, QPDualSolution3) {
  MosekSolver solver;
  if (solver.available()) {
    TestQPDualSolution3(solver, 1E-6, 3E-4);
  }
}

GTEST_TEST(MosekTest, EqualityConstrainedQPDualSolution1) {
  MosekSolver solver;
  if (solver.available()) {
    TestEqualityConstrainedQPDualSolution1(solver);
  }
}

GTEST_TEST(MosekTest, EqualityConstrainedQPDualSolution2) {
  MosekSolver solver;
  if (solver.available()) {
    TestEqualityConstrainedQPDualSolution2(solver);
  }
}

GTEST_TEST(MosekSolver, SocpDualSolution1) {
  MosekSolver solver;
  if (solver.available()) {
    SolverOptions solver_options{};
    TestSocpDualSolution1(solver, solver_options, 1E-7);
  }
}

GTEST_TEST(MosekSolver, SocpDualSolution2) {
  MosekSolver solver;
  if (solver.available()) {
    SolverOptions solver_options{};
    TestSocpDualSolution2(solver, solver_options, 1E-6);
  }
}

GTEST_TEST(MosekTest, SDPDualSolution1) {
  MosekSolver solver;
  if (solver.available()) {
    TestSDPDualSolution1(solver, 1E-4);
  }
}

GTEST_TEST(MosekTest, TestEllipsoid1) {
  // Test quadratically constrained program.
  MosekSolver solver;
  if (solver.available()) {
    TestEllipsoid1(solver, std::nullopt, 1E-6);
  }
}

GTEST_TEST(MosekTest, TestEllipsoid2) {
  // Test quadratically constrained program.
  MosekSolver solver;
  if (solver.available()) {
    TestEllipsoid2(solver, std::nullopt, 1E-5);
  }
}

GTEST_TEST(MosekTest, TestNonconvexQP) {
  MosekSolver solver;
  if (solver.available()) {
    TestNonconvexQP(solver, true);
  }
}

template <typename C>
void CheckDualSolutionNotNan(const MathematicalProgramResult& result,
                             const Binding<C>& constraint) {
  const auto dual_sol = result.GetDualSolution(constraint);
  for (int i = 0; i < dual_sol.rows(); ++i) {
    EXPECT_FALSE(std::isnan(dual_sol(i)));
  }
}
GTEST_TEST(MosekTest, InfeasibleLinearProgramTest) {
  // Solve an infeasible LP, make sure the infeasible solution is returned.
  MathematicalProgram prog;
  auto x = prog.NewContinuousVariables<2>();
  auto constraint1 = prog.AddLinearConstraint(x(0) + x(1) >= 3);
  auto constraint2 = prog.AddBoundingBoxConstraint(0, 1, x);
  prog.AddLinearCost(x(0) + 2 * x(1));
  MosekSolver solver;
  if (solver.available()) {
    const auto result = solver.Solve(prog);
    ASSERT_FALSE(result.is_success());
    EXPECT_EQ(result.get_optimal_cost(),
              MathematicalProgram::kGlobalInfeasibleCost);
    // Check that the primal solutions are not NAN.
    for (int i = 0; i < x.rows(); ++i) {
      EXPECT_FALSE(std::isnan(result.GetSolution(x(i))));
    }
    // Check that the dual solutions are not NAN.
    CheckDualSolutionNotNan(result, constraint1);
    CheckDualSolutionNotNan(result, constraint2);
    // Check that the optimal cost is not NAN.
    EXPECT_FALSE(std::isnan(result.get_optimal_cost()));
  }
}

GTEST_TEST(MosekTest, InfeasibleSemidefiniteProgramTest) {
  // Solve an infeasible SDP, make sure the infeasible solution is returned.
  MathematicalProgram prog;
  auto X = prog.NewSymmetricContinuousVariables<3>();
  auto constraint1 = prog.AddPositiveSemidefiniteConstraint(X);
  auto constraint2 =
      prog.AddLinearConstraint(X(0, 0) + X(1, 1) + X(2, 2) <= -1);
  prog.AddLinearCost(X(1, 2));
  MosekSolver solver;
  if (solver.available()) {
    const auto result = solver.Solve(prog);
    ASSERT_FALSE(result.is_success());
    EXPECT_EQ(result.get_optimal_cost(),
              MathematicalProgram::kGlobalInfeasibleCost);
    // Check that the primal solutions are not NAN.
    const auto X_sol = result.GetSolution(X);
    for (int i = 0; i < X.rows(); ++i) {
      for (int j = 0; j < X.cols(); ++j) {
        EXPECT_FALSE(std::isnan(X_sol(i, j)));
      }
    }
    // Check that the dual solutions are not NAN.
    CheckDualSolutionNotNan(result, constraint1);
    CheckDualSolutionNotNan(result, constraint2);

    // Check that the optimal cost is not NAN.
    EXPECT_FALSE(std::isnan(result.get_optimal_cost()));
  }
}

GTEST_TEST(MosekTest, LPNoBasisSelection) {
  // We solve an LP using interior point method (IPM), but don't do basis
  // identification (which cleans the solution) after IPM finishes. Hence the
  // basis solution is not available and Mosek can only acquire the IPM
  // solution.
  MathematicalProgram prog;
  auto x = prog.NewContinuousVariables<2>();
  prog.AddBoundingBoxConstraint(0, kInf, x);
  prog.AddLinearConstraint(x(0) + x(1) <= 1);
  prog.AddLinearCost(-x(0) - 2 * x(1));

  SolverOptions solver_options;
  solver_options.SetOption(MosekSolver::id(), "MSK_IPAR_INTPNT_BASIS", 0);
  MosekSolver solver;
  if (solver.available()) {
    auto result = solver.Solve(prog, std::nullopt, solver_options);
    EXPECT_TRUE(result.is_success());
    const auto x_sol = result.GetSolution(x);
    const double tol = 1E-6;
    EXPECT_TRUE(CompareMatrices(x_sol, Eigen::Vector2d(0, 1), tol));
  }
}
}  // namespace test
}  // namespace solvers
}  // namespace drake

int main(int argc, char** argv) {
  // Ensure that we have the MOSEK license for the entire duration of this test,
  // so that we do not have to release and re-acquire the license for every
  // test.
  auto mosek_license = drake::solvers::MosekSolver::AcquireLicense();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
