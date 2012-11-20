/** file ice_server.cpp
Точка входа для запуска сервера ICE.*/
#include "Ice/Ice.h"
#include "slae_solver_ice.h"
#include "slae_solver_ice_cvm.h"
#include "gas_transfer_system_ice.h"
#include "gas_transfer_system_ice_usual.h"
#include "parallel_manager_ice_servant.h"

#include <log4cplus/logger.h>
#include <log4cplus/loggingmacros.h>
#include <log4cplus/configurator.h>
#include <log4cplus/fileappender.h> 
#include <log4cplus/consoleappender.h>
#include <log4cplus/layout.h>

#include <iomanip>

#include "gas_transfer_system.h"

using namespace log4cplus;

Logger GasTransferSystem::log_ = 
  Logger::getInstance(LOG4CPLUS_TEXT("IceServer.Gts"));

class Server : public Ice::Application {
public:
  virtual int run(int, char*[]);
private:
   Logger log;
   void RunSlaeSolverCvmServant(Ice::ObjectAdapterPtr *adapter);
   void RunParallelManagerServant(Ice::ObjectAdapterPtr *adapter);
   void RunSlaeSolverGasTransferSystemServant(Ice::ObjectAdapterPtr *adapter);
};

void Server::
    RunSlaeSolverCvmServant(Ice::ObjectAdapterPtr *adapter) {
  // Создаём servant'а, реализующего интерфейс SlaeSolverIce.
  Enisey::SlaeSolverIceCVMPtr slae_solver_ice_cvm = 
    new Enisey::SlaeSolverIceCVM;
  slae_solver_ice_cvm->ActivateSelfInAdapter(*adapter);
  // Создаём servant'а, реализующего интерфейс SlaeSolverIce.
LOG4CPLUS_INFO(log, "Run SlaeSolverServant");
}
void Server::
    RunSlaeSolverGasTransferSystemServant(Ice::ObjectAdapterPtr *adapter) {
  // Создаём servant'а, реализующего интерфейс GasTransferSystemIce.
  Enisey::GasTransferSystemIceUsualPtr gts_usual = 
    new Enisey::GasTransferSystemIceUsual;
  gts_usual->ActivateSelfInAdapter(*adapter);
LOG4CPLUS_INFO(log, "Run SlaeSolverGTSServant");
}
void Server:: 
    RunParallelManagerServant(Ice::ObjectAdapterPtr *adapter) {
  Enisey::ParallelManagerIceServantPtr parallel_manager_servant = 
    new Enisey::ParallelManagerIceServant;
  parallel_manager_servant->ActivateSelfInAdapter(*adapter);
LOG4CPLUS_INFO(log, "Run ParallelManagerServant");
}

int	Server::run(int, char*[]) {
  shutdownOnInterrupt(); // Установка реакции shutdown на сигнал Interrupt.
  // Настраиваем логгер.
  std::auto_ptr<Layout> console_layout =  std::auto_ptr<Layout>(new TTCCLayout());
  std::auto_ptr<Layout> file_layout    =  std::auto_ptr<Layout>(new TTCCLayout());
  SharedAppenderPtr console_appender(new ConsoleAppender);
  console_appender->setLayout(console_layout);
  SharedAppenderPtr file_appender(
      new FileAppender( LOG4CPLUS_TEXT("server.log")) );
  file_appender->setLayout(file_layout);
  log = Logger::getInstance( LOG4CPLUS_TEXT("IceServer") );
  log.addAppender(console_appender);
  log.addAppender(file_appender);
  log.setLogLevel(DEBUG_LOG_LEVEL);
  
  LOG4CPLUS_INFO(log, "HELLO from Logger!");
  // Создаём адаптер - точку входа, к которой обращаются клиенты.
  Ice::ObjectAdapterPtr adapter =
      communicator()->createObjectAdapterWithEndpoints(
      "EniseyServerAdapter", // Имя адаптера.
      "tcp -p 10000");    // Endpoint.
  RunSlaeSolverCvmServant(&adapter);
  RunSlaeSolverGasTransferSystemServant(&adapter);  
  RunParallelManagerServant(&adapter);
  adapter->activate(); // Начинаем слушать соединения от клиентов асинхронно.
  communicator()->waitForShutdown(); // Приостанавливаем данный поток.
  if(interrupted()) {
    std::cerr << appName() << ": received signal, shutting down" << std::endl;
  }
  return 0;
}

int main(int argc, char* argv[]) {
  Server s;
  return s.main(argc, argv);
}
