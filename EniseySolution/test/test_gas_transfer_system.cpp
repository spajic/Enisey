/** \file gas_transfer_system.cpp
Тесты для класса GasTransferSystem из GasTransferSystem.h*/
#include "gas_transfer_system.h"

#include "gtest/gtest.h"
#include "test_utils.h"

#include <string>

TEST(GasTransferSystem, LoadsFromVestaAndWritesToGraphviz) {
  /** \todo Здесь стоит сделать автоматическую проверку того, что граф
  загружается и выгружается правильно. */
  const std::string path_to_vesta_files = "C:\\Enisey\\data\\saratov_gorkiy\\";
  const std::string graphviz_filename = 
    "C:\\Enisey\\out\\GTSLoadsFromVestaAndWritesToGraphviz_test.dot";
  GasTransferSystem gts;
  gts.LoadFromVestaFiles(path_to_vesta_files);
  gts.WriteToGraphviz(graphviz_filename);
}

