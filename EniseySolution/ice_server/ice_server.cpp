/** file ice_server.cpp
Точка входа для запуска сервера ICE.*/
#include "Ice/Ice.h"
#include "slae_solver_ice.h"
#include "slae_solver_ice_cvm.h"

class Server : public Ice::Application {
 public:
  virtual int run(int, char*[]);
};

int	Server::run(int, char*[]) {
  shutdownOnInterrupt(); // Установка реакции shutdown на сигнал Interrupt.
  // Создаём адаптер - точку входа, к которой обращаются клиенты.
  Ice::ObjectAdapterPtr adapter =
      communicator()->createObjectAdapterWithEndpoints(
      "EniseyServerAdapter", // Имя адаптера.
      "default -p 10000");    // Endpoint.
  // Создаём servant'а, реализующего интерфейс SlaeSolverIce.
  Enisey::SlaeSolverIceCVMPtr slae_solver_ice_cvm = 
      new Enisey::SlaeSolverIceCVM;
  slae_solver_ice_cvm->ActivateSelfInAdapter(adapter);
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
