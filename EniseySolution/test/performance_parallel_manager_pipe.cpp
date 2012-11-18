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
#include <log4cplus/layout.h>

#include <iomanip>

#include <ctime>

#include "util_saratov_etalon_loader.h"

template <class ManagerClass>
void TestManager (    
    std::string manager_name,
    ManagerClass *dummy_manager,
    unsigned int multiplicity,
    std::vector<PassportPipe> const &passports,
    std::vector<WorkParams> const &work_params,
    std::vector<CalculatedParams> *calculated_params,
    unsigned int repeats,
    log4cplus::Logger *log) {
  clock_t begin       = 0;
  clock_t end         = 0;
  double elapsed_secs = 0;

  double take_under_control_time    = 0;
  double set_work_params_time       = 0;
  double calculate_all_time         = 0;
  double get_calculated_params_time = 0;
  
  LOG4CPLUS_INFO(*log, 
    manager_name.c_str() << "; Multipilcity = " << multiplicity << 
    "; Repeats = " << repeats);
  for(int i = 0; i < repeats; ++i) {    
    ManagerClass manager;
    begin = clock();
      manager.TakeUnderControl    (passports);
    end = clock();    
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    take_under_control_time += elapsed_secs;

    begin = clock();
      manager.SetWorkParams       (work_params);
    end = clock();  
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    set_work_params_time += elapsed_secs;

    begin = clock();
      manager.CalculateAll();  
    end = clock();  
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    calculate_all_time += elapsed_secs;

    begin = clock();
      manager.GetCalculatedParams (calculated_params);
    end = clock();  
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    get_calculated_params_time += elapsed_secs;    
  }
  LOG4CPLUS_INFO(*log, "TakeUnderControl    - " << 
    take_under_control_time / repeats << "s");
  LOG4CPLUS_INFO(*log, "SetWorkParams       - " << 
    set_work_params_time / repeats << "s");
  LOG4CPLUS_INFO(*log, "CalculateAll        - " << 
    calculate_all_time / repeats << "s");
  LOG4CPLUS_INFO(*log, "GetCalculatedParams - " << 
    get_calculated_params_time / repeats << "s" << std::endl);  
}

TEST(ParallelManagerPerformance, Perf) {  
  //log4cplus::BasicConfigurator config;
  //config.configure();
  log4cplus::SharedAppenderPtr myAppender(
    new log4cplus::FileAppender(
    LOG4CPLUS_TEXT("c:/Enisey/out/log/myLogFile.log")));
  //myAppender->setName(LOG4CPLUS_TEXT("myAppenderName"));  
  std::auto_ptr<log4cplus::Layout> myLayout = 
    std::auto_ptr<log4cplus::Layout>(new log4cplus::TTCCLayout());
  log4cplus::Logger log = log4cplus::Logger::getInstance(
    LOG4CPLUS_TEXT("ParallelManagerPerformance"));
  myAppender->setLayout(myLayout);
  log.addAppender(myAppender);
  log.setLogLevel(log4cplus::DEBUG_LOG_LEVEL);

  boost::property_tree::ptree pt;
  read_json("C:\\Enisey\\src\\config\\config.json", pt);
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
  
  ParallelManagerPipeSingleCore manager_single_core;
  TestManager (
      "ParallelManagerSingleCore",
      &manager_single_core,
      multiplicity,
      passports,
      work_params,
      &calculated_params,
      repeats,
      &log);

  ParallelManagerPipeOpenMP manager_openMP;
  TestManager (
    "ParallelManagerOpenMP",
    &manager_openMP,
    multiplicity,
    passports,
    work_params,
    &calculated_params,
    repeats,
    &log);

  ParallelManagerPipeCUDA manager_CUDA;
  TestManager (
    "ParallelManagerCUDA",
    &manager_CUDA,
    multiplicity,
    passports,
    work_params,
    &calculated_params,
    repeats,
    &log);
} 