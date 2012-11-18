/** \file util_ssaratov_etalon_loader.cpp
Класс загружает эталон труб Саратова из файла. */
#include "util_saratov_etalon_loader.h"

#include <vector>
#include <string>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include "passport_pipe.h"
#include "work_params.h"
#include "calculated_params.h"

#include "test_utils.h"

SaratovEtalonLoader::SaratovEtalonLoader() {
  read_json("C:\\Enisey\\src\\config\\config.json", pt_);
  etalons_path_ = pt_.get<std::string>(
      "Testing.ParallelManagers.Etalon.Paths.RootDir");
  passports_path_         = etalons_path_ + pt_.get<std::string>(
      "Testing.ParallelManagers.Etalon.Paths.Passports");
  work_params_path_       = etalons_path_ + pt_.get<std::string>(
      "Testing.ParallelManagers.Etalon.Paths.WorkParams");
  calculated_params_path_ = etalons_path_ + pt_.get<std::string>(
      "Testing.ParallelManagers.Etalon.Paths.CalculatedParams");
  LoadEtalonToInternalStorage();
  saratov_size_ = passports_.size();
}

void SaratovEtalonLoader::LoadEtalonToInternalStorage() {
  LoadEtalonPassports       (&passports_        );
  LoadEtalonWorkParams      (&work_params_      );
  LoadEtalonCalculatedParams(&calculated_params_);
}

template <class VecType>
void FillVecByVec(VecType const &v_from, VecType *v_to) {
  v_to->clear();
  v_to->resize( v_from.size() );
  std::copy( v_from.begin(), v_from.end(), v_to->begin() );
}

void SaratovEtalonLoader::LoadSaratovEtalon(
  std::vector<PassportPipe>     *passports,
  std::vector<WorkParams>       *work_params,
  std::vector<CalculatedParams> *calculated_params) {
    FillVecByVec(passports_         , passports         );
    FillVecByVec(work_params_       , work_params       );
    FillVecByVec(calculated_params_ , calculated_params );
}

template <class VecType>
void FillVecByMultipleVec(
    unsigned int multiplicity,
    VecType const &v_from, 
    VecType *v_to) {
  v_to->clear();
  v_to->resize( v_from.size() * multiplicity );
  int i = 0;
  auto it_after_last_copied_elem = v_to->begin();
  do {
    it_after_last_copied_elem = std::copy(
        v_from.begin(), v_from.end(), it_after_last_copied_elem);
    ++i;
  } while(i < multiplicity);
}

void SaratovEtalonLoader::LoadSaratovMultipleEtalon(
    unsigned int              multiplicity, 
    std::vector<PassportPipe> *passports  , 
    std::vector<WorkParams>   *work_params) {  
  FillVecByMultipleVec(multiplicity, passports_   , passports   );
  FillVecByMultipleVec(multiplicity, work_params_ , work_params );
}

void SaratovEtalonLoader::LoadEtalonPassports(
    std::vector<PassportPipe> *passports) {
  LoadSerializableVectorFromFile(passports_path_, passports);
}

void SaratovEtalonLoader::LoadEtalonWorkParams(
    std::vector<WorkParams> *work_params) {
  LoadSerializableVectorFromFile(work_params_path_, work_params); 
}

void SaratovEtalonLoader::LoadEtalonCalculatedParams(
    std::vector<CalculatedParams> *calculated_params) {
  LoadSerializableVectorFromFile(calculated_params_path_, calculated_params);
}
