/** \file test_slae_solver_i.cpp
  Тесты для класса SlaeSolverI из slae_solver_i.h*/
#include "slae_solver_i.h"

#include "gtest/gtest.h"
#include "test_utils.h"

TEST(SlaeSolverITest, SolveSimpleSlaeWithCVMSolver) {
  SlaeSolverI *abstract_solver = new SlaeSolverCVM;
  std::vector<std::pair<int,int> > A_indexes = MakeSimpleSlaeAIndexes();
  std::vector<double> A_vals = MakeSimpleSlaeAValues();
  std::vector<double> b = MakeSimpleSlaeB();
  std::vector<double> etalon_x = MakeSimpleSlaeX();
  std::vector<double> x;
  x.reserve( b.size() );

  abstract_solver->Solve(A_indexes, A_vals, b, &x);
  EXPECT_TRUE( std::equal( x.begin(), x.end(), etalon_x.begin() ) );

  delete abstract_solver;
}