/** \file gas_transfer_system.cpp
Тесты для класса GasTransferSystem из GasTransferSystem.h*/
#include "gas_transfer_system.h"

#include "gtest/gtest.h"
#include "test_utils.h"

#include <string>
#include <iostream>
#include <list>

const std::string path_to_vesta_files = "C:\\Enisey\\data\\saratov_gorkiy\\";


class GasTransferSystemFromVestaTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    gts.LoadFromVestaFiles(path_to_vesta_files);
  }
  GasTransferSystem gts;
};

TEST_F(GasTransferSystemFromVestaTest, LoadsFromVestaAndWritesToGraphviz) {
  /** \todo Здесь стоит сделать автоматическую проверку того, что граф
  загружается и выгружается правильно. */  
  const std::string graphviz_filename = 
    "C:\\Enisey\\out\\GTSLoadsFromVestaAndWritesToGraphviz_test.dot";
  gts.WriteToGraphviz(graphviz_filename);
}
TEST_F(GasTransferSystemFromVestaTest, MakesInitialApprox) {
  gts.MakeInitialApprox();
  const std::string graphviz_filename = 
    "C:\\Enisey\\out\\GTSMakesInitialApprox.dot";
  // Проверяем корректность задания ограничений давлений.
  // bool ok = ChechPressureConstraintsForVertices(
  //  &graph,
  //  overall_p_min,
  //  overall_p_max);
  //EXPECT_EQ(ok, true);
  /// \todo Добавить автоматические проверки на корректность.
  gts.WriteToGraphviz(graphviz_filename);
}
TEST_F(GasTransferSystemFromVestaTest, CountsAllEdges) {
  gts.MakeInitialApprox();
  gts.CountAllEdges();
  gts.MixVertices();
  const std::string graphviz_filename = 
    "C:\\Enisey\\out\\GTSCountsAllEdges.dot";
  /// \todo Добавить автоматические проверки на корректность.
  gts.WriteToGraphviz(graphviz_filename);
  float d = gts.CountDisbalance();
  std::cout << "Disb1 = " << d << std::endl;
  std::list<float> disbs;
  float g = 1.0 / 3.0;
  float d_prev = 1000000;
  gts.SetSlaeRowNumsForVertices();
  for(int n = 0; n < 35; ++n) {
    gts.CountNewIteration(g);
    float d = gts.CountDisbalance();
    if(d_prev < d) {
      g *= 0.9;
    }
    d_prev = d;
    int d_int = gts.GetIntDisbalance();
    std::cout << d << " " << d_int << std::endl;
    if(d_int == 0) {
      break;
    }
    disbs.push_back(d);
  }
  /*for(auto d = disbs.begin(); d != disbs.end(); ++d) {
    std::cout << *d << std::endl;
  }*/
  gts.WriteToGraphviz("C:\\Enisey\\out\\Result.dot");
}
