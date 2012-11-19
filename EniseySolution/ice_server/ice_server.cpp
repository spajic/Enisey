/** file ice_server.cpp
Точка входа для запуска сервера ICE.*/
#include "Ice/Ice.h"
#include "slae_solver_ice.h"
#include "slae_solver_ice_cvm.h"
#include "gas_transfer_system_ice.h"
#include "gas_transfer_system_ice_usual.h"
#include "parallel_manager_ice_servant.h"

class Server : public Ice::Application {
 public:
  virtual int run(int, char*[]);
};

void RunSlaeSolverCvmServant(Ice::ObjectAdapterPtr *adapter) {
  // Создаём servant'а, реализующего интерфейс SlaeSolverIce.
  Enisey::SlaeSolverIceCVMPtr slae_solver_ice_cvm = 
    new Enisey::SlaeSolverIceCVM;
  slae_solver_ice_cvm->ActivateSelfInAdapter(*adapter);
  // Создаём servant'а, реализующего интерфейс SlaeSolverIce.
  std::cout << "RunSlaeSolverCvmServant\n";
}
void RunSlaeSolverGasTransferSystemServant(Ice::ObjectAdapterPtr *adapter) {
  // Создаём servant'а, реализующего интерфейс GasTransferSystemIce.
  Enisey::GasTransferSystemIceUsualPtr gts_usual = 
    new Enisey::GasTransferSystemIceUsual;
  gts_usual->ActivateSelfInAdapter(*adapter);
  std::cout << "RunSlaeSolverGasTransferSystemServant\n";
}
void RunParallelManagerServant(Ice::ObjectAdapterPtr *adapter) {
  Enisey::ParallelManagerIceServantPtr parallel_manager_servant = 
    new Enisey::ParallelManagerIceServant;
  parallel_manager_servant->ActivateSelfInAdapter(*adapter);
  std::cout << "RunParallelManagerServant\n";
}

int	Server::run(int, char*[]) {
  shutdownOnInterrupt(); // Установка реакции shutdown на сигнал Interrupt.
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
