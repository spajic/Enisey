/** \file perf_gts.cpp
Замер производительности классов GasTransferSystemI.*/
#include "gtest/gtest.h"
#include "test_utils.h"

#include <vector>
#include <string>

#include <stdexcept>

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

#include "gas_transfer_system_i.h"
#include "gas_transfer_system.h"
#include "gas_transfer_system_ice_client.h"

#include "test_utils.h"

#define foreach BOOST_FOREACH

using namespace log4cplus;
using namespace std;

typedef shared_ptr<GasTransferSystemI>          GasTransferSystemIPtr;
typedef shared_ptr<GasTransferSystemIceClient>  GasTransferSystemIceClientPtr;

GasTransferSystemIPtr GtsFactory(
    string endpoint, 
    int number_of_iterations) {
  if(endpoint == "NONE") { // Локальный расчёт.
    return make_shared<GasTransferSystem>();
  }
  else {
    GasTransferSystemIceClientPtr gts( new GasTransferSystemIceClient(endpoint) );
    gts->SetNumberOfIterations(number_of_iterations);
    return gts;
  }
  throw invalid_argument("Can't create GTS for endpoint = " + endpoint);
}

void TestGts(    
    std::string endpoint,
    int num_of_gts_to_count,
    int repeats) {
  Logger log = Logger::getInstance(LOG4CPLUS_TEXT("GtsPerf"));  

  clock_t begin               = 0;
  clock_t end                 = 0;
  double  total_elapsed_secs  = 0;
  
  vector<string> matrix_connections_strings;
  vector<string> in_out_grs_strings;
  vector<string> pipe_line_strings;
  matrix_connections_strings = 
      FileAsVectorOfStrings(path_to_vesta_files + "MatrixConnections.dat");
  in_out_grs_strings = 
      FileAsVectorOfStrings(path_to_vesta_files + "InOutGRS.dat");
  pipe_line_strings =
      FileAsVectorOfStrings(path_to_vesta_files + "PipeLine.dat");

LOG4CPLUS_INFO(log,     
    "Endpoint: "            << endpoint.c_str()      <<
    "; NumOfGTSToCount: "   << num_of_gts_to_count   <<
    "; repetas: "           << repeats
);

std::vector<std::string>  result_file_strings ;
std::vector<double>       result_abs_disbs    ;
std::vector<int>          result_int_disbs    ;

  for(int i = 0; i < repeats; ++i) {
    LOG4CPLUS_INFO(log, "--Start batch # " << i );

    for(int j = 0; j < num_of_gts_to_count; ++j) {
      GasTransferSystemIPtr gts = GtsFactory(endpoint, num_of_gts_to_count);
      result_file_strings.clear();
      result_abs_disbs.clear();
      result_int_disbs.clear();
      begin = clock(); 
      LOG4CPLUS_INFO(log, "----Call gts.PeroformBalancing");
        gts->PeroformBalancing( 
            matrix_connections_strings,
            in_out_grs_strings,
            pipe_line_strings,
            &result_file_strings,
            &result_abs_disbs,
            &result_int_disbs
        );
      LOG4CPLUS_INFO(log, "----Return from gts.PeroformBalancing");
      end = clock();
      total_elapsed_secs += double(end - begin) / CLOCKS_PER_SEC;
      if(endpoint != "NONE") { // Считаем на сервере ICE
        break; // Повторения и так делаются на сервере.
      }
    }  
  } 
  CompareGTSDisbalancesFactToEtalon(result_abs_disbs, result_int_disbs);
  LOG4CPLUS_INFO(log, 
      "PerformBalancing - " << total_elapsed_secs/repeats << "s" << std::endl);  
}

TEST(GtsPerf, Test) {
  boost::property_tree::ptree pt;
  read_json("C:\\Enisey\\src\\config\\config.json", pt);
  bool run_performance_tests = pt.get<bool>("Performance.GTS.StartPerfTests");
  if(!run_performance_tests) return;

  string  log_file = pt.get<string>("Performance.GTS.LogFile"); 
  tstring log_file_t( log_file.begin(), log_file.end() ); 
  
  SharedAppenderPtr myAppender(new FileAppender(log_file_t) );
  myAppender->setName(LOG4CPLUS_TEXT("First"));
  SharedAppenderPtr cAppender(new log4cplus::ConsoleAppender());
  cAppender->setName(LOG4CPLUS_TEXT("Second"));
  std::auto_ptr<Layout> myLayout1 ( new TTCCLayout() );
  std::auto_ptr<Layout> myLayout2 ( new TTCCLayout() );  
  myAppender->setLayout(myLayout1);
  cAppender ->setLayout(myLayout2);
  Logger log = log4cplus::Logger::getInstance(
    LOG4CPLUS_TEXT("GtsPerf"));
  log.addAppender(myAppender);
  log.addAppender(cAppender);
  log.setLogLevel(log4cplus::DEBUG_LOG_LEVEL); 

  auto test_configs = pt.get_child("Performance.GTS.Tests");
  foreach(ptree::value_type &test, test_configs) {
    string ice_endpoint = test.second.get<string>("IceEndpoint");

    foreach(ptree::value_type &type, test.second.get_child("TypesAndRepeats")){    

      foreach(ptree::value_type &repeat, type.second.get_child("Repeats") ) {
        int multiplicity  = repeat.second.get<int>("Multiplicity");
        int repeats       = repeat.second.get<int>("Repeats");

        TestGts(ice_endpoint, multiplicity, repeats);
      }
    }  
  }  
}