/** \file gas_transfer_system.cpp
Тесты для класса GasTransferSystem из GasTransferSystem.h*/
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

const std::string path_to_vesta_files_i = "C:\\Enisey\\data\\saratov_gorkiy\\";

class GasTransferSystemITest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    gts = new GasTransferSystem;
    manager_edge_model_pipe_sequential_ = new ManagerEdgeModelPipeSequential;
    gts->set_manager_edge(manager_edge_model_pipe_sequential_);
    slae_solver_cvm_ = new SlaeSolverIceClient;
    gts->set_slae_solver(slae_solver_cvm_);
    gts->LoadFromVestaFiles(path_to_vesta_files_i);
  }
  virtual void TearDown() {
    delete manager_edge_model_pipe_sequential_;
    delete slae_solver_cvm_;
    delete gts;
  }
  GasTransferSystemI *gts;
  ManagerEdge *manager_edge_model_pipe_sequential_;
  SlaeSolverI *slae_solver_cvm_;
};

TEST_F(GasTransferSystemITest, LoadsFromVestaAndWritesToGraphviz) {
  /** \todo Здесь стоит сделать автоматическую проверку того, что граф
  загружается и выгружается правильно. */  
  const std::string graphviz_filename = 
    "C:\\Enisey\\out\\GTSLoadsFromVestaAndWritesToGraphviz_test.dot";
  gts->WriteToGraphviz(graphviz_filename);
}
TEST_F(GasTransferSystemITest, MakesInitialApprox) {
  gts->MakeInitialApprox();
  const std::string graphviz_filename = 
    "C:\\Enisey\\out\\GTSMakesInitialApprox.dot";
  // Проверяем корректность задания ограничений давлений.
  // bool ok = ChechPressureConstraintsForVertices(
  //  &graph,
  //  overall_p_min,
  //  overall_p_max);
  //EXPECT_EQ(ok, true);
  /// \todo Добавить автоматические проверки на корректность.
  gts->WriteToGraphviz(graphviz_filename);
}
TEST_F(GasTransferSystemITest, FindsBalanceForSaratovGorkiy) {
  gts->MakeInitialApprox();
  gts->CountAllEdges();
  gts->MixVertices();
  const std::string graphviz_filename = 
     "C:\\Enisey\\out\\GTSCountsAllEdges.dot";
  /// \todo Добавить автоматические проверки на корректность.

  std::list<double> int_disbs;
  std::list<double> abs_disbs;

  gts->WriteToGraphviz(graphviz_filename);
  double d = gts->CountDisbalance();
  abs_disbs.push_back(d);
  int_disbs.push_back( gts->GetIntDisbalance() );
  double g = 1.0 / 2.0;
  double d_prev = 1000000;
  gts->SetSlaeRowNumsForVertices();
  for(int n = 0; n < 35; ++n) {
    gts->CountNewIteration(g);
    d = gts->CountDisbalance();
    abs_disbs.push_back(d);
    if(d_prev < d) {
      g *= 0.9;
    }
    d_prev = d;
    int d_int = gts->GetIntDisbalance();
    int_disbs.push_back(d_int);
    if(d_int == 0) {
      break;
    }
  }
  gts->WriteToGraphviz("C:\\Enisey\\out\\Result.dot");

  // Файл, в который запишем эталонную последовательность дисбалансов для
  // дальнейшего сравнения. Эта функциональность включается только тогда,
  // когда нужно сформировать новый эталон.
  // Файл имеет формат
  // на строчке: номер итерации, дисбаланс, число узлов c дисбалансом.
  const std::string etalon_saratov_gorkiy_balance = 
    "C:\\Enisey\\out\\SaratovGorkiy\\etalon_balance_find.txt";
//#define NEW_ETALON
#ifdef NEW_ETALON
  std::ofstream etalon_f(etalon_saratov_gorkiy_balance); 
  etalon_f << std::setprecision(18) << std::fixed;
  auto i_d = int_disbs.begin();
  auto a_d = abs_disbs.begin();
  for(unsigned int iter_num = 0; iter_num < abs_disbs.size(); ++iter_num) {
    etalon_f << iter_num << " " << *a_d << " " << *i_d << std::endl;
    ++a_d;
    ++i_d;
  }
#endif  
  // Загружаем информацию об эталоне и сравниваем с фактом.
#ifndef NEW_ETALON
  std::ifstream etalon_f(etalon_saratov_gorkiy_balance);
  std::list<double> etalon_abs_disbs;
  std::list<double> etalon_int_disbs;
  int et_iter(0);
  double et_abs_d(0.0);
  double et_int_d(0.0);
  while(etalon_f >> et_iter >> et_abs_d >> et_int_d) {
    etalon_abs_disbs.push_back(et_abs_d);
    etalon_int_disbs.push_back(et_int_d);
  }
  ASSERT_EQ( abs_disbs.size(), etalon_abs_disbs.size() );
  bool abs_disbs_equal = std::equal(
      etalon_abs_disbs.begin(), etalon_abs_disbs.end(),
      abs_disbs.begin() );
  bool int_disbs_equal = std::equal(
      etalon_int_disbs.begin(), etalon_int_disbs.end(),
      int_disbs.begin() );
  EXPECT_TRUE(abs_disbs_equal);
  EXPECT_TRUE(int_disbs_equal);
#endif
}
