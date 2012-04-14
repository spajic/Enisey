/** \file test_gas_transfer_system_i.cpp
Типизированный абстрактный тест для всех классов, реализующих интефрейс
GasTransferSystemI.
Отличается от test_gas_transfer_system тем, что здесь используются только
методы абстрактного интерфейса, а там - методы конкретного класса 
GasTrnsferSystem. Основное - здесь, тот вариант - больше для отладки. В связи
с этим, эталонное решенине тоже может быть сформировано там.*/
#include "gas_transfer_system_i.h"

#include "gtest/gtest.h"
#include "test_utils.h"

#include <string>
#include <iostream>
#include <list>
#include <fstream>
#include <iomanip>
#include "slae_solver_cvm.h"
#include "slae_solver_ice_client.h"
#include "manager_edge_model_pipe_sequential.h"
#include "gas_transfer_system.h"
#include "gas_transfer_system_ice_client.h"

// Test-fixture class.
template <typename T>
class GasTransferSystemTypedTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    abstract_gts = new T;
  }
  virtual void TearDown() {
    delete abstract_gts;
  }
  GasTransferSystemI *abstract_gts;
  ManagerEdge *manager_edge_model_pipe_sequential_;
  SlaeSolverI *slae_solver_cvm_;
  std::vector<std::string> result_of_balancing;
  std::vector<double> abs_disbalances;
  std::vector<int> int_disbalances;
};

// Список типов, для которых будут выполняться тесты.
typedef ::testing::Types<
    GasTransferSystem,
    GasTransferSystemIceClient
> GasTransferSystemTypes;

TYPED_TEST_CASE(GasTransferSystemTypedTest, GasTransferSystemTypes);

TYPED_TEST(GasTransferSystemTypedTest, PerformsBalancingOfSaratovGorkiy) {
  abstract_gts->PeroformBalancing(
      FileAsVectorOfStrings(path_to_vesta_files + "MatrixConnections.dat"),
      FileAsVectorOfStrings(path_to_vesta_files + "InOutGRS.dat"),
      FileAsVectorOfStrings(path_to_vesta_files + "PipeLine.dat"),
      &result_of_balancing,
      &abs_disbalances,
      &int_disbalances
  ); 
  CompareGTSDisbalancesFactToEtalon(abs_disbalances, int_disbalances);
}
