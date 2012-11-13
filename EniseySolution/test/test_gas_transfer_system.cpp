/** \file gas_transfer_system.cpp
Тесты для класса GasTransferSystem из GasTransferSystem.h*/
#include "gas_transfer_system.h"

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

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>


class GasTransferSystemFromVestaTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    manager_edge_model_pipe_sequential_ = new ManagerEdgeModelPipeSequential;
    gts.set_manager_edge(manager_edge_model_pipe_sequential_);
    slae_solver_cvm_ = new SlaeSolverIceClient;
    gts.set_slae_solver(slae_solver_cvm_);
    gts.LoadFromVestaFiles(path_to_vesta_files);
  }
  virtual void TearDown() {
    delete manager_edge_model_pipe_sequential_;
    delete slae_solver_cvm_;
  }
  GasTransferSystem gts;
  ManagerEdge *manager_edge_model_pipe_sequential_;
  SlaeSolverI *slae_solver_cvm_;
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
TEST_F(GasTransferSystemFromVestaTest, FindsBalanceForSaratovGorkiy) {
  // Create an empty property tree object
  using boost::property_tree::ptree;
  ptree pt;
  read_json("C:\\Enisey\\src\\config\\config.json", pt);
  std::string path = 
      pt.get<std::string>("Testing.ParallelManagers.Etalon.Paths.RootDir");   
  bool regenerate_etalon = 
      pt.get<bool>("Testing.ParallelManagers.Etalon.RegenerateEtalon");   
  gts.MakeInitialApprox();
  ManagerEdgeModelPipeSequential *manager = 
      gts.manager_model_pipe_sequential();
  if(regenerate_etalon) {
    manager->SavePassportsToFile(
        path +  
        pt.get<std::string>("Testing.ParallelManagers.Etalon.Paths.Passports") );
  }
  gts.CopyGasStateFromVerticesToEdges();
  if(regenerate_etalon) {
    manager->SaveWorkParamsToFile(
        path+  
        pt.get<std::string>("Testing.ParallelManagers.Etalon.Paths.WorkParams") );
  }
  gts.CountAllEdges();
  if(regenerate_etalon) {
    manager->SaveCalculatedParamsToFile(
        path +
        pt.get<std::string>("Testing.ParallelManagers.Etalon.Paths.CalculatedParams"));
  }
  gts.MixVertices();
  const std::string graphviz_filename = 
     "C:\\Enisey\\out\\GTSCountsAllEdges.dot";
  /// \todo Добавить автоматические проверки на корректность.

  std::vector<int> int_disbs;
  std::vector<double> abs_disbs;

  gts.WriteToGraphviz(graphviz_filename);
  double d = gts.CountDisbalance();
  abs_disbs.push_back(d);
  int_disbs.push_back( gts.GetIntDisbalance() );
  double g = 1.0 / 2.0;
  double d_prev = 1000000;
  gts.SetSlaeRowNumsForVertices();
  for(int n = 0; n < 35; ++n) {
    gts.CountNewIteration(g);
    d = gts.CountDisbalance();
    abs_disbs.push_back(d);
    if(d_prev < d) {
      g *= 0.9;
    }
    d_prev = d;
    int d_int = gts.GetIntDisbalance();
    int_disbs.push_back(d_int);
    if(d_int == 0) {
      break;
    }
  }
  gts.WriteToGraphviz("C:\\Enisey\\out\\Result.dot");

  // Файл, в который запишем эталонную последовательность дисбалансов для
  // дальнейшего сравнения. Эта функциональность включается только тогда,
  // когда нужно сформировать новый эталон.
  // Файл имеет формат
  // на строчке: номер итерации, дисбаланс, число узлов c дисбалансом.
//#define NEW_ETALON
#ifdef NEW_ETALON
  std::ofstream etalon_f(etalon_saratov_gorkiy_balance); 
  etalon_f << std::setprecision(30) << std::fixed;
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
  CompareGTSDisbalancesFactToEtalon(abs_disbs, int_disbs);
#endif
}
