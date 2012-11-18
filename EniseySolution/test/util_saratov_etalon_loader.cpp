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
}

void SaratovEtalonLoader::LoadSaratovEtalon(
  std::vector<PassportPipe>     *passports,
  std::vector<WorkParams>       *work_params,
  std::vector<CalculatedParams> *calculated_params) {
    LoadEtalonPassports       (passports        );
    LoadEtalonWorkParams      (work_params      );
    LoadEtalonCalculatedParams(calculated_params);
}
void SaratovEtalonLoader::LoadEtalonPassports(
    std::vector<PassportPipe> *passports) {
  LoadSerializableVectorFromFile(
    etalons_path_ + pt_.get<std::string>(
        "Testing.ParallelManagers.Etalon.Paths.Passports"),
    passports);
}

void SaratovEtalonLoader::LoadEtalonWorkParams(
    std::vector<WorkParams> *work_params) {
  LoadSerializableVectorFromFile(
      etalons_path_ + pt_.get<std::string>(
          "Testing.ParallelManagers.Etalon.Paths.WorkParams"),
      work_params); 
}

void SaratovEtalonLoader::LoadEtalonCalculatedParams(
    std::vector<CalculatedParams> *calculated_params) {
  LoadSerializableVectorFromFile(
      etalons_path_ + pt_.get<std::string>(
          "Testing.ParallelManagers.Etalon.Paths.CalculatedParams"),
    calculated_params);
}
