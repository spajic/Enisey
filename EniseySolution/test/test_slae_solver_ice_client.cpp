/** \file test_slae_solver_ice_client.cpp
  Тесты для класса SlaeSolverIceClient из slae_solver_ice_client.h*/
#include "slae_solver_ice_client.h"

#include "gtest/gtest.h"
#include "test_utils.h"

#include <vector>
#include "test_utils.h"

TEST(SlaeSolverIceClientTest, SolvesSimpleSlae) {
  SlaeSolverIceClient solver;
  std::vector<int> A_indexes = MakeSimpleSlaeAIndexes();
  std::vector<double> A_vals = MakeSimpleSlaeAValues();
  std::vector<double> b = MakeSimpleSlaeB();
  std::vector<double> etalon_x = MakeSimpleSlaeX();
  std::vector<double> x;
  x.reserve( b.size() );
  solver.Solve(A_indexes, A_vals, b, &x);
  EXPECT_TRUE( std::equal( x.begin(), x.end(), etalon_x.begin() ) );
}