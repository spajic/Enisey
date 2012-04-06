/** \file gas_transfer_system.cpp
Тесты для класса GasTransferSystem из GasTransferSystem.h*/
#include "gas_transfer_system.h"

#include "gtest/gtest.h"
#include "test_utils.h"

#include <string>

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
  const std::string graphviz_filename = 
    "C:\\Enisey\\out\\GTSCountsAllEdges.dot";
  /// \todo Добавить автоматические проверки на корректность.
  gts.WriteToGraphviz(graphviz_filename);
  float d = gts.CountDisbalance();
}