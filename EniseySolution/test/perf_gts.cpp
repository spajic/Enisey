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
    int num_of_gts_to_count) {
  log4cplus::Logger log = 
      log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("GtsPerf"));

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
  TestGts("GTS", endpoint, num_gts_to_count);
}