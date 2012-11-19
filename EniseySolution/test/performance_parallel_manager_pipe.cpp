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

#include <vector>
#include <string>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

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

std::auto_ptr<ParallelManagerPipeI> CreateManager(
    std::string name, 
    std::string ice_impl,
    std::string endpoint) {
  if(name == "SingleCore") {
    return std::auto_ptr<ParallelManagerPipeI>(
        new ParallelManagerPipeSingleCore);
  }
  if(name == "OpenMP") {
    return std::auto_ptr<ParallelManagerPipeI>(
        new ParallelManagerPipeOpenMP);
  }
  if(name == "CUDA") {
    return std::auto_ptr<ParallelManagerPipeI> (
        new ParallelManagerPipeCUDA);
  }
  if(name == "ICE") {
    std::auto_ptr<ParallelManagerPipeIce> manager_ice(
        new ParallelManagerPipeIce(endpoint));    
    manager_ice->SetParallelManagerType(ice_impl);
    return manager_ice;
  }
}

void TestManager (    
    std::string manager_name,
    std::string ice_impl,
    std::string endpoint,
    unsigned int multiplicity,
    std::vector<PassportPipe> const &passports,
    std::vector<WorkParams> const &work_params,
    std::vector<CalculatedParams> *calculated_params,
    unsigned int repeats) {
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
      manager_name.c_str() << "; ice_impl: " << ice_impl.c_str() << 
      "; Endpoint: " << endpoint.c_str()  << 
      "; Multipilcity = " << multiplicity << 
      "; Repeats = " << repeats);

  for(int i = 0; i < repeats; ++i) {    
    std::auto_ptr<ParallelManagerPipeI> 
        manager( CreateManager(manager_name, ice_impl, endpoint) );
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

  bool run_performance_tests = pt.get<bool>("Performance.StartPerfTests");
  if(!run_performance_tests) return;
  //log4cplus::BasicConfigurator config;
  //config.configure();
  log4cplus::SharedAppenderPtr myAppender(
    new log4cplus::FileAppender(
    LOG4CPLUS_TEXT("c:/Enisey/out/log/myLogFile.log")));
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

  
  unsigned int multiplicity = pt.get<unsigned int> (
      "Performance.ParallelManagers.Multiplicity");
  unsigned int repeats = pt.get<unsigned int> (
      "Performance.ParallelManagers.Repeats");
  
  std::vector<PassportPipe>     passports;
  std::vector<WorkParams>       work_params;
  std::vector<CalculatedParams> calculated_params;
  SaratovEtalonLoader loader;
  loader.LoadSaratovMultipleEtalon(
      multiplicity,
      &passports,
      &work_params      
  );

  std::string endpoint = pt.get<std::string> (
      "Performance.ParallelManagers.Endpoint");

  /*TestManager (
      "ICE", "SingleCore", endpoint,
      multiplicity,
      passports,
      work_params,
      &calculated_params,
      repeats,
      &log);
  
  TestManager (
      "SingleCore", "None", "None",
      multiplicity,
      passports,
      work_params,
      &calculated_params,
      repeats,
      &log);*/
  
  TestManager (
      "OpenMP", "None", "None",
      multiplicity,
      passports,
      work_params,
      &calculated_params,
      repeats);
  TestManager (
      "CUDA", "None", "None",
      multiplicity,
      passports,
      work_params,
      &calculated_params,
      repeats);
  TestManager (
      "ICE", "OpenMP", endpoint,
      multiplicity,
      passports,
      work_params,
      &calculated_params,
      repeats);
  TestManager (
      "ICE", "CUDA", endpoint,
      multiplicity,
      passports,
      work_params,
      &calculated_params,
      repeats);
} 