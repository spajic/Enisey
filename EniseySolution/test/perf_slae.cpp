/** \file perf_slae.cpp
Замер производительности классов SlaeSolverI.*/
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

#include "slae_solver_cvm.h"

std::auto_ptr<SlaeSolverI> slae_factory(std::string slae_type) {
  if(slae_type == "CVM") {
    return std::auto_ptr<SlaeSolverI>(new SlaeSolverCVM);
  }
}

void test_slae(
    std::string slae_type,
    unsigned int repeats) {  
  log4cplus::Logger log = log4cplus::Logger::getInstance(
      LOG4CPLUS_TEXT("SlaePerf"));  
  std::vector<int>    A_indices;
  std::vector<double> A_values;
  std::vector<double> b;
  std::vector<double> x_etalon;
  std::vector<double> x_calc;
  SaratovEtalonLoader loader;
  loader.LoadSaratovEtalonSlae(
    &A_indices,
    &A_values,
    &b,
    &x_etalon);

  std::auto_ptr<SlaeSolverI> slae = slae_factory(slae_type);
  LOG4CPLUS_INFO(log, 
      "SlaeType: " << slae_type.c_str() <<
    "; Repeats: "  << repeats);

  clock_t begin               = 0;
  clock_t end                 = 0;
  double  total_elapsed_secs  = 0;

  for(int i = 0; i < repeats; ++i) {
    begin = clock();    
      slae->Solve(A_indices, A_values, b, &x_calc);
    end = clock();    
    total_elapsed_secs += double(end - begin) / CLOCKS_PER_SEC;
  }
  LOG4CPLUS_INFO(log, 
      "Solve - " << total_elapsed_secs / repeats << "s" << std::endl);
    
}

TEST(SlaePerformance, Perf) {
  boost::property_tree::ptree pt;
  read_json("C:\\Enisey\\src\\config\\config.json", pt);

  bool run_performance_tests = pt.get<bool>("Performance.StartPerfTests");
  if(!run_performance_tests) return;

  log4cplus::SharedAppenderPtr myAppender(
      new log4cplus::FileAppender(
      LOG4CPLUS_TEXT("c:/Enisey/out/log/SLAE.log")));
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
    LOG4CPLUS_TEXT("SlaePerf"));
  log.addAppender(myAppender);
  log.addAppender(cAppender);
  log.setLogLevel(log4cplus::DEBUG_LOG_LEVEL);
  
  test_slae("CVM", 10);
}