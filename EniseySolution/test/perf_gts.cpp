/** \file perf_gts.cpp
Замер производительности классов GasTransferSystemI.*/
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

#include "gas_transfer_system_i.h"
#include "gas_transfer_system.h"
#include "gas_transfer_system_ice_client.h"

#include "test_utils.h"

std::auto_ptr<GasTransferSystemI> GtsFactory(
    std::string type,
    std::string endpoint) {
  if(type == "GTS") {
    return std::auto_ptr<GasTransferSystemI>(new GasTransferSystem);
  }
  if(type == "ICE") {
    return std::auto_ptr<GasTransferSystemI>(
        new GasTransferSystemIceClient(endpoint) 
    );
  }
}

void TestGts(
    std::string type,
    std::string endpoint,
    int num_of_gts_to_count,
    int repeats) {
  log4cplus::Logger log = 
      log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("GtsPerf"));  

  clock_t begin               = 0;
  clock_t end                 = 0;
  double  total_elapsed_secs  = 0;
  
  std::vector<std::string> matrix_connections_strings;
  std::vector<std::string> in_out_grs_strings;
  std::vector<std::string> pipe_line_strings;
  matrix_connections_strings = 
      FileAsVectorOfStrings(path_to_vesta_files + "MatrixConnections.dat");
  in_out_grs_strings = 
      FileAsVectorOfStrings(path_to_vesta_files + "InOutGRS.dat");
  pipe_line_strings =
      FileAsVectorOfStrings(path_to_vesta_files + "PipeLine.dat");

LOG4CPLUS_INFO(log, 
    "GtsType: "             << type.c_str()          <<
    "; endpoint: "          << endpoint.c_str()      <<
    "; NumOfGTSToCount: "   << num_of_gts_to_count   <<
    "; repetas: "           << repeats
);

std::vector<std::string>  result_file_strings ;
std::vector<double>       result_abs_disbs    ;
std::vector<int>          result_int_disbs    ;

  if(type == "ICE") {
    // На сервере и так делается расчёт 10 ГТС.
    num_of_gts_to_count = 1;
  }
  for(int i = 0; i < repeats; ++i) {
    LOG4CPLUS_INFO(log, "--Start batch # " << i );
    for(int j = 0; j < 10; ++j) {
      std::auto_ptr<GasTransferSystemI> gts = GtsFactory(type, endpoint);
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
      if(type == "ICE") {break;}// На сервере и так делается расчёт 10 ГТС.
      }  
    } 
  CompareGTSDisbalancesFactToEtalon(result_abs_disbs, result_int_disbs);
  LOG4CPLUS_INFO(log, 
      "PerformBalancing - " << total_elapsed_secs/repeats << "s" << std::endl);  
}

TEST(GtsPerf, Test) {
  boost::property_tree::ptree pt;
  read_json("C:\\Enisey\\src\\config\\config.json", pt);
  bool run_performance_tests = pt.get<bool>("Performance.StartPerfTests");
  if(!run_performance_tests) return;

  log4cplus::SharedAppenderPtr myAppender(
    new log4cplus::FileAppender(
    LOG4CPLUS_TEXT("c:/Enisey/out/log/GTS.log")));
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
    LOG4CPLUS_TEXT("GtsPerf"));
  log.addAppender(myAppender);
  log.addAppender(cAppender);
  log.setLogLevel(log4cplus::DEBUG_LOG_LEVEL);

  int num_gts_to_count = pt.get<int>("Performance.GTS.NumberOfGtsToCount");
  std::string endpoint = pt.get<std::string>("Performance.GTS.Endpoint");
  int repeats = pt.get<int>("Performance.GTS.Repeats");
  TestGts("GTS", "None"   , num_gts_to_count, repeats);
  TestGts("ICE", endpoint , num_gts_to_count, repeats);
}