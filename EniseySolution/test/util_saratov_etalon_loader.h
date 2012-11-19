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
  void LoadSaratovMultipleEtalon(      
      unsigned int              multiplicity,    
      std::vector<PassportPipe> *passports  ,
      std::vector<WorkParams>   *work_params);
  void LoadSaratovEtalonSlae(
      std::vector<int>    *a_indices,
      std::vector<double> *a_values,
      std::vector<double> *b,
      std::vector<double> *x);
  void LoadSaratovEtalonSlaeMultiple( 
      std::vector<int>    *a_indices,
      std::vector<double> *a_values,
      std::vector<double> *b,
      std::vector<double> *x,
      int                  multiplicity);
    
private:
  boost::property_tree::ptree pt_;
  std::string etalons_path_;
  std::string passports_path_;
  std::string work_params_path_;
  std::string calculated_params_path_;
  std::string slae_path_;

  std::vector<PassportPipe>     passports_;
  std::vector<WorkParams>       work_params_;
  std::vector<CalculatedParams> calculated_params_;

  std::vector<int>    a_indices_;
  std::vector<double> a_values_;
  std::vector<double> b_;
  std::vector<double> x_;

  int saratov_size_;

  void LoadEtalonToInternalStorage();
  void LoadEtalonPassports  (std::vector<PassportPipe> *passports);
  void LoadEtalonWorkParams (std::vector<WorkParams>   *work_params);
  void LoadEtalonCalculatedParams(
      std::vector<CalculatedParams> *calculated_params);
  void LoadSlaeToInternalStorage();
};