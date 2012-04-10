/** \file gas_transfer_system.cpp
����� ��� ������ GasTransferSystem �� GasTransferSystem.h*/
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
  /** \todo ����� ����� ������� �������������� �������� ����, ��� ����
  ����������� � ����������� ���������. */  
  const std::string graphviz_filename = 
    "C:\\Enisey\\out\\GTSLoadsFromVestaAndWritesToGraphviz_test.dot";
  gts.WriteToGraphviz(graphviz_filename);
}
TEST_F(GasTransferSystemFromVestaTest, MakesInitialApprox) {
  gts.MakeInitialApprox();
  const std::string graphviz_filename = 
    "C:\\Enisey\\out\\GTSMakesInitialApprox.dot";
  // ��������� ������������ ������� ����������� ��������.
  // bool ok = ChechPressureConstraintsForVertices(
  //  &graph,
  //  overall_p_min,
  //  overall_p_max);
  //EXPECT_EQ(ok, true);
  /// \todo �������� �������������� �������� �� ������������.
  gts.WriteToGraphviz(graphviz_filename);
}
TEST_F(GasTransferSystemFromVestaTest, CountsAllEdges) {
  gts.MakeInitialApprox();
  gts.CountAllEdges();
  gts.MixVertices();
  const std::string graphviz_filename = 
    "C:\\Enisey\\out\\GTSCountsAllEdges.dot";
  /// \todo �������� �������������� �������� �� ������������.
  gts.WriteToGraphviz(graphviz_filename);
  double d = gts.CountDisbalance();
  std::cout << "Disb1 = " << d << std::endl;
  std::list<double> disbs;
  double g = 1.0 / 3.0;
  double d_prev = 1000000;
  gts.SetSlaeRowNumsForVertices();
  for(int n = 0; n < 35; ++n) {
    gts.CountNewIteration(g);
    double d = gts.CountDisbalance();
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
