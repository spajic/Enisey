/** \file test_parallel_manager_pipe_i.cpp
Типизированный абстрактный тест для всех классов, реализующих интефрейс
ParallelManagerPipeI.
*/

//#include "gas_transfer_system_i.h"

#include "gtest/gtest.h"
#include "test_utils.h"

#include "parallel_manager_pipe_i.h"
#include "parallel_manager_pipe_singlecore.h"
#include "parallel_manager_pipe_openmp.h"
#include "parallel_manager_pipe_cuda.cuh"
#include "parallel_manager_pipe_ice.h"



#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

using std::vector;
using boost::property_tree::ptree;

// Test-fixture class.
template <typename T>
class ParallelManagerPipeTypedTest : public ::testing::Test {
protected:
  virtual void SetUp() {
    manager = new T;
    read_json("C:\\Enisey\\src\\config\\config.json", pt);
    etalons_path = pt.get<std::string>("Testing.ParallelManagers.Etalon.Paths.RootDir");
  }
  virtual void TearDown() {
    delete manager;
  }
  void LoadEtalon() {
    LoadEtalonPassports();
    LoadEtalonWorkParams();
    LoadEtalonCalculatedParams();
  }
  void LoadEtalonPassports() {
    LoadSerializableVectorFromFile(
        etalons_path + 
            pt.get<std::string>(
                "Testing.ParallelManagers.Etalon.Paths.Passports"),
        &etalon_passports
    );
  }
  void LoadEtalonWorkParams() {
    LoadSerializableVectorFromFile(
        etalons_path + 
            pt.get<std::string>(
                "Testing.ParallelManagers.Etalon.Paths.WorkParams"),
        &etalon_work_params
      );
  }
  void LoadEtalonCalculatedParams() {
    LoadSerializableVectorFromFile(
        etalons_path + 
            pt.get<std::string>(
                "Testing.ParallelManagers.Etalon.Paths.CalculatedParams"),
        &etalon_calculated_params
      );
  }
  void CheckIfEtalonMatchesCalculatedResults() {
    ASSERT_EQ( etalon_calculated_params.size(), calculated_params.size() );
    auto calc = calculated_params.begin();
    double eps = 0.0000001;
    for(auto etalon = etalon_calculated_params.begin(); 
        etalon != etalon_calculated_params.end(); ++etalon) {
      EXPECT_NEAR(etalon->q(),        calc->q(),        eps);
      EXPECT_NEAR(etalon->dq_dp_in(), calc->dq_dp_in(), eps);
      EXPECT_NEAR(etalon->dq_dp_out(),calc->dq_dp_out(),eps);
      EXPECT_NEAR(etalon->t_out(),    calc->t_out(),    eps);
      ++calc;
    }    
  }
  ptree pt;
  std::string etalons_path;
  ParallelManagerPipeI *manager;  
  vector<PassportPipe> etalon_passports;
  vector<WorkParams> etalon_work_params;
  vector<CalculatedParams> etalon_calculated_params;
  vector<CalculatedParams> calculated_params;
};

// Список типов, для которых будут выполняться тесты.
typedef ::testing::Types<
    ParallelManagerPipeSingleCore,
    ParallelManagerPipeOpenMP,
    ParallelManagerPipeCUDA,
    ParallelManagerPipeIce
> ParallelManagerPipeTypes;

TYPED_TEST_CASE(ParallelManagerPipeTypedTest, ParallelManagerPipeTypes);

TYPED_TEST(ParallelManagerPipeTypedTest, TestCorrectnessByEtalon) {
  LoadEtalon();  
  manager->TakeUnderControl(etalon_passports);    
  manager->SetWorkParams(etalon_work_params);
  manager->CalculateAll();  
  manager->GetCalculatedParams(&calculated_params);  
  CheckIfEtalonMatchesCalculatedResults();  
}