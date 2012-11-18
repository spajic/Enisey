/** \file util_ssaratov_etalon_loader.h
Класс загружает эталон труб Саратова из файла. */
#pragma once
#include <vector>
#include <string>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

//Forward-declarations:
struct PassportPipe;
class WorkParams;
class CalculatedParams;

class SaratovEtalonLoader {
public:
  SaratovEtalonLoader();
  void LoadSaratovEtalon(
    std::vector<PassportPipe>     *passports,
    std::vector<WorkParams>       *work_params,
    std::vector<CalculatedParams> *calculated_params);
private:
  boost::property_tree::ptree pt_;
  std::string etalons_path_;

  void LoadEtalonPassports  (std::vector<PassportPipe> *passports);
  void LoadEtalonWorkParams (std::vector<WorkParams>   *work_params);
  void LoadEtalonCalculatedParams(
      std::vector<CalculatedParams> *calculated_params);
};