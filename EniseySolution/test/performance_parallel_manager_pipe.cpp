/** \file performance_parallel_manager_pipe.cpp
Замер производительности классов ParallelManagerPipe.
Нужно замерить все комбинации следующих аспектов:
Реализация менеджера: SingleCore, OpenMP, CUDA, ICE;
Имплементация сервера ICE: SingleCore, OpenMP, CUDA;
Расположение Сервера ICE: локально, в AWS;
Тестовый пример: Саратов, 10x, 100x, 1000x Саратов;
Использование: однократный расчёт, итеративная смена рабочих параметров;
Количество итераций при итеративном расчёте: 10, 100.
При работе тест должен логгировать тайминги основных событий.
ToDo:
0. Проверить корректность работы ParallelManagerPipeI при итеративном вызове
   SetWorkParams.
1. Написать тест для объекта типа ParallelManagerSingleCore.
    - Расчёт 100x Саратова [10 раз и усреднить] 
    - Расчёт 100х Саратова по 10 итераций [10 раз и усреднить]
    [100, или 1000000 - подобрать так, чтобы делалось заметное время, но 
    довольно быстро - чтобы было куда ускоряться] 
    [Параметры должны задаваться через конфигурационный файл]
2. Сделать тест обобщённым.
3. Использовать тест в рамках Запускающего теста с различными реализацями
   ParallelManagerPipeI - OpenMP, CUDA, ICE с различными локальными реализациями.
4. Запустить сервер ICE в облаке AWS, настроить доступ к нему и затестить с ним -
   тоже с различными реализациями.
*/

#include "gtest/gtest.h"
#include "test_utils.h"

#include <stdexcept>

#include <vector>
#include <string>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <boost/foreach.hpp>

#include <log4cplus/logger.h>
#include <log4cplus/loggingmacros.h>
#include <log4cplus/configurator.h>
#include <log4cplus/fileappender.h> 
#include <log4cplus/consoleappender.h>
#include <log4cplus/layout.h>

#include <iomanip>

#include <ctime>

#include "util_saratov_etalon_loader.h"

#include "parallel_manager_pipe_i.h"
#include "parallel_manager_pipe_singlecore.h"
#include "parallel_manager_pipe_openmp.h"
#include "parallel_manager_pipe_cuda.cuh"
#include "parallel_manager_pipe_ice.h"


using std::shared_ptr;
using std::make_shared;
using std::string;
using std::vector;

using boost::property_tree::ptree;

using namespace log4cplus;

typedef shared_ptr<ParallelManagerPipeI>    ParallelManagerPipeIPtr;
typedef shared_ptr<ParallelManagerPipeIce>  ParallelManagerPipeIcePtr;

ParallelManagerPipeIPtr CreateManager(
    string manager_name, 
    string ice_endpoint) {
  if(ice_endpoint == "NONE") { // Локальный расчёт
    if(manager_name == "SingleCore") {
      return make_shared<ParallelManagerPipeSingleCore>( );
    }
    if(manager_name == "OpenMP") {
      return make_shared<ParallelManagerPipeOpenMP>( );
    }
    if(manager_name == "CUDA") {
       return make_shared<ParallelManagerPipeCUDA>( );
    }
  }
  else {
    ParallelManagerPipeIcePtr manager_ice = 
        make_shared<ParallelManagerPipeIce>( ice_endpoint );    
    manager_ice->SetParallelManagerType(manager_name);
    return manager_ice;
  }     
  throw std::invalid_argument(
      "Can't create ParalelManager of type " + manager_name +
      "on endpoint " + ice_endpoint);
}

void TestManager (    
    string                      manager_name,
    string                      ice_endpoint,    
    unsigned int                multiplicity,
    vector<PassportPipe>  const &passports,
    vector<WorkParams>    const &work_params,
    vector<CalculatedParams>    *calculated_params,
    unsigned int                repeats) {
  clock_t begin       = 0;
  clock_t end         = 0;
  double elapsed_secs = 0;

  double take_under_control_time    = 0;
  double set_work_params_time       = 0;
  double calculate_all_time         = 0;
  double get_calculated_params_time = 0;
  
  log4cplus::Logger log = log4cplus::Logger::getInstance(
    LOG4CPLUS_TEXT("ParallelManagerPerformance"));

  LOG4CPLUS_INFO(log, 
      manager_name.c_str()  << "; ice_impl: "       << manager_name.c_str() << 
      "; Endpoint: "        << ice_endpoint.c_str() << 
      "; Multipilcity = "   << multiplicity         << 
      "; Repeats = "        << repeats);

  for(int i = 0; i < repeats; ++i) {    
    ParallelManagerPipeIPtr manager(CreateManager(manager_name, ice_endpoint));
    
    begin = clock();
      manager->TakeUnderControl    (passports);
    end = clock();    
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    take_under_control_time += elapsed_secs;

    begin = clock();
      manager->SetWorkParams       (work_params);
    end = clock();  
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    set_work_params_time += elapsed_secs;

    begin = clock();
      manager->CalculateAll();  
    end = clock();  
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    calculate_all_time += elapsed_secs;

    begin = clock();
      manager->GetCalculatedParams (calculated_params);
    end = clock();  
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    get_calculated_params_time += elapsed_secs;    
  }
  LOG4CPLUS_INFO(log, "TakeUnderControl    - " << 
    take_under_control_time / repeats << "s");
  LOG4CPLUS_INFO(log, "SetWorkParams       - " << 
    set_work_params_time / repeats << "s");
  LOG4CPLUS_INFO(log, "CalculateAll        - " << 
    calculate_all_time / repeats << "s");
  LOG4CPLUS_INFO(log, "GetCalculatedParams - " << 
    get_calculated_params_time / repeats);  
  LOG4CPLUS_INFO(log, "Total Time - " <<
    (take_under_control_time + set_work_params_time + 
    calculate_all_time + get_calculated_params_time) / repeats << "s" << std::endl);
}

TEST(ParallelManagerPerformance, Perf) {  
  boost::property_tree::ptree pt;
  read_json("C:\\Enisey\\src\\config\\config.json", pt);

  bool run_performance_tests = 
      pt.get<bool>("Performance.ParallelManagers.StartPerfTests");
  if(!run_performance_tests) return;

  string  log_file    = pt.get<string>("Performance.ParallelManagers.LogFile");
  tstring log_file_t  = tstring( log_file.begin(), log_file.end() );
  SharedAppenderPtr myAppender( new FileAppender(log_file_t) );
  myAppender->setName(LOG4CPLUS_TEXT("First"));
  log4cplus::SharedAppenderPtr cAppender(new log4cplus::ConsoleAppender());
  cAppender->setName(LOG4CPLUS_TEXT("Second"));
  std::auto_ptr<log4cplus::Layout> myLayout = 
    std::auto_ptr<log4cplus::Layout>(new log4cplus::TTCCLayout());
  std::auto_ptr<log4cplus::Layout> myLayout2 = 
    std::auto_ptr<log4cplus::Layout>(new log4cplus::TTCCLayout());
  myAppender->setLayout(myLayout);
  cAppender->setLayout(myLayout2);
  log4cplus::Logger log = log4cplus::Logger::getInstance(
    LOG4CPLUS_TEXT("ParallelManagerPerformance"));
  log.addAppender(myAppender);
  log.addAppender(cAppender);
  log.setLogLevel(log4cplus::DEBUG_LOG_LEVEL);

  auto test_configs = pt.get_child("Performance.ParallelManagers.Tests");

  BOOST_FOREACH(
  ptree::value_type &test, test_configs) {
    std::string ice_endpoint = test.second.get<string>("IceEndpoint");

    BOOST_FOREACH(
    ptree::value_type &type, test.second.get_child("TypesAndRepeats") ) {
      std::string manager_name = type.second.get<string>("Type");

      BOOST_FOREACH(
      ptree::value_type &repeat, type.second.get_child("Repeats") ) {
        int multiplicity  = repeat.second.get<int>("Multiplicity");
        int repeats       = repeat.second.get<int>("Repeats");

        std::vector<PassportPipe>     passports;
        std::vector<WorkParams>       work_params;
        std::vector<CalculatedParams> calculated_params;
        SaratovEtalonLoader loader;
        loader.LoadSaratovMultipleEtalon(multiplicity,&passports,&work_params);

        TestManager (
            manager_name, 
            ice_endpoint,
            multiplicity,
            passports,
            work_params,
            &calculated_params,
            repeats);
      }
    }
  }
} 
