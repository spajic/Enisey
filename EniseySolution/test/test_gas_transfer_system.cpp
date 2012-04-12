/** \file gas_transfer_system.cpp
����� ��� ������ GasTransferSystem �� GasTransferSystem.h*/
#include "gas_transfer_system.h"

#include "gtest/gtest.h"
#include "test_utils.h"

#include <string>
#include <iostream>
#include <list>
#include <fstream>
#include <iomanip>

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
TEST_F(GasTransferSystemFromVestaTest, FindsBalanceForSaratovGorkiy) {
  gts.MakeInitialApprox();
  gts.CountAllEdges();
  gts.MixVertices();
  const std::string graphviz_filename = 
     "C:\\Enisey\\out\\GTSCountsAllEdges.dot";
  /// \todo �������� �������������� �������� �� ������������.

  std::list<double> int_disbs;
  std::list<double> abs_disbs;

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

  // ����, � ������� ������� ��������� ������������������ ����������� ���
  // ����������� ���������. ��� ���������������� ���������� ������ �����,
  // ����� ����� ������������ ����� ������.
  // ���� ����� ������
  // �� �������: ����� ��������, ���������, ����� ����� c �����������.
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
  // ��������� ���������� �� ������� � ���������� � ������.
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
