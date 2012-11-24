/** \file perf_slae.cpp
Замер производительности классов SlaeSolverI.*/
#include "gtest/gtest.h"
#include "test_utils.h"

#include <vector>
#include <string>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>

#define foreach BOOST_FOREACH

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
#include "slae_solver_ice_client.h"
#include "slae_solver_cusp.cuh"

using namespace log4cplus;

using std::string;
using std::vector;

typedef std::shared_ptr<SlaeSolverI> SlaeSolverIPtr;
typedef std::shared_ptr<SlaeSolverIceClient> SlaeSolverIceClientPtr;

SlaeSolverIPtr slae_factory(string slae_type, string endpoint) {
  if(endpoint == "NONE") { // Локальный расчёт.
    if(slae_type == "CVM") {
      return std::make_shared<SlaeSolverCVM>();
    }
    if(slae_type == "CUSP") {
      return std::make_shared<SlaeSolverCusp>();
    }
  }  
  else {
    SlaeSolverIceClientPtr solver = std::make_shared<SlaeSolverIceClient>();
    solver->SetSolverType(slae_type);    
    return solver;
  }
  throw std::invalid_argument(
      "Can't produce slae with type = " + slae_type + 
      " and endpoint = " + endpoint
  );
}

void test_slae(
    std::string slae_type,
    std::string ice_endpoint,
    unsigned int repeats,
    unsigned int multiplicity) {  
  log4cplus::Logger log = log4cplus::Logger::getInstance(
      LOG4CPLUS_TEXT("SlaePerf") );  
  std::vector<int>    A_indices;
  std::vector<double> A_values;
  std::vector<double> b;
  std::vector<double> x_etalon;
  std::vector<double> x_calc;
  SaratovEtalonLoader loader;
  loader.LoadSaratovEtalonSlaeMultiple( 
      &A_indices,
      &A_values,
      &b,
      &x_etalon,
      multiplicity);

  SlaeSolverIPtr slae = slae_factory(slae_type, ice_endpoint);
  LOG4CPLUS_INFO(log, 
      "SlaeType: "   << slae_type.c_str() <<
      "; endpoint: " << ice_endpoint.c_str() <<
      "; Size: "     << b.size() <<
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

  bool run_performance_tests = pt.get<bool>("Performance.SLAE.StartPerfTests");
  if(!run_performance_tests) return;

  string  log_file = pt.get<string>("Performance.SLAE.LogFile"); 
  tstring log_file_t( log_file.begin(), log_file.end() ); 
  
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
    LOG4CPLUS_TEXT("SlaePerf"));
  log.addAppender(myAppender);
  log.addAppender(cAppender);
  log.setLogLevel(log4cplus::DEBUG_LOG_LEVEL);
  
  auto test_configs = pt.get_child("Performance.SLAE.Tests");
  foreach(ptree::value_type &test, test_configs) {
    string ice_endpoint = test.second.get<string>("IceEndpoint");

    foreach(ptree::value_type &type, test.second.get_child("TypesAndRepeats")){
      string solver_type = type.second.get<string>("Type");  
      
      foreach(ptree::value_type &repeat, type.second.get_child("Repeats") ) {
        int multiplicity  = repeat.second.get<int>("Multiplicity");
        int repeats       = repeat.second.get<int>("Repeats");

        test_slae(solver_type, ice_endpoint, repeats, multiplicity);
      }
    }  
  }
}
