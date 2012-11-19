/** \file test_saratov_etalon_loader.cpp
Тесты для класса загрузки эталона Саратова.*/
#include "gtest/gtest.h"
#include "test_utils.h"

#include "util_saratov_etalon_loader.h"
#include "slae_solver_cvm.h"

TEST(SaratovEtalonLoader, LoadMultipleSlae) {
// Проверка: заргуженное дублированное эталонное решение совпадает с решением,
// построенным решателем системы по дублированным исходным данным.
  std::vector<int> a_indices;
  std::vector<double> a;
  std::vector<double> b;
  std::vector<double> etalon;

  SaratovEtalonLoader loader;
  loader.LoadSaratovEtalonSlaeMultiple(&a_indices, &a, &b, &etalon, 2); 

  std::vector<double> x;
  SlaeSolverCVM solver;
  solver.Solve(a_indices, a, b, &x);

  ASSERT_EQ(etalon.size(), x.size());
  auto x_it = x.begin();
  for(auto it = etalon.begin(); it != etalon.end(); ++it) {
    ASSERT_NEAR( *it,  *x_it, 0.0000001);
    ++x_it;
  }
}