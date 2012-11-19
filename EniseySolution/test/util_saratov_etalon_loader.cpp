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
#include "util_sparse_indices_multiplicator.h"

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
  slae_path_ = "C:/Enisey/out/Testing/SLAE/";
  LoadEtalonToInternalStorage();
  saratov_size_ = passports_.size();
}

void SaratovEtalonLoader::LoadEtalonToInternalStorage() {
  LoadEtalonPassports       (&passports_        );
  LoadEtalonWorkParams      (&work_params_      );
  LoadEtalonCalculatedParams(&calculated_params_);
  LoadSlaeToInternalStorage();
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

void SaratovEtalonLoader::LoadSaratovEtalonSlaeMultiple(
    std::vector<int>    *a_indices, 
    std::vector<double> *a_values, 
    std::vector<double> *b, 
    std::vector<double> *x, 
    int                  multiplicity) {
  a_indices->clear();
  a_indices->reserve(a_indices_.size() * multiplicity);
  MultipicateSparceIndices(b_.size(), a_indices_, a_indices, multiplicity);

  a_values->clear();
  a_values->reserve(a_values_.size() * multiplicity);
  FillVecByMultipleVec(multiplicity, a_values_, a_values);

  b->clear();
  b->reserve(b_.size() * multiplicity);
  FillVecByMultipleVec(multiplicity, b_, b);

  x->clear();
  x->reserve(x_.size() * multiplicity);
  FillVecByMultipleVec(multiplicity, x_, x);
}

void SaratovEtalonLoader::LoadSaratovEtalonSlae(
    std::vector<int>    *a_indices, 
    std::vector<double> *a_values, 
    std::vector<double> *b, 
    std::vector<double> *x) {
  FillVecByVec(a_indices_, a_indices);
  FillVecByVec(a_values_, a_values);
  FillVecByVec(b_, b);
  FillVecByVec(x_, x);
}

void SaratovEtalonLoader::LoadSlaeToInternalStorage() {
  LoadSerializableVectorFromTextFile(
      slae_path_ + "Saratov_A_indices.txt", &a_indices_);
  LoadSerializableVectorFromTextFile(
      slae_path_ + "Saratov_A_vals.txt"   , &a_values_);
  LoadSerializableVectorFromTextFile(
      slae_path_ + "Saratov_b.txt"        , &b_);
  LoadSerializableVectorFromTextFile(
      slae_path_ + "Saratov_x.txt"        , &x_);
}

void SaratovEtalonLoader::LoadEtalonPassports(
    std::vector<PassportPipe> *passports) {
  LoadSerializableVectorFromXmlFile(passports_path_, passports);
}

void SaratovEtalonLoader::LoadEtalonWorkParams(
    std::vector<WorkParams> *work_params) {
  LoadSerializableVectorFromXmlFile(work_params_path_, work_params); 
}

void SaratovEtalonLoader::LoadEtalonCalculatedParams(
    std::vector<CalculatedParams> *calculated_params) {
  LoadSerializableVectorFromXmlFile(calculated_params_path_, calculated_params);
}
