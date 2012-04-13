/** file ice_server.cpp
����� ����� ��� ������� ������� ICE.*/
#include "Ice/Ice.h"
#include "slae_solver_ice.h"
#include "slae_solver_ice_cvm.h"

class Server : public Ice::Application {
 public:
  virtual int run(int, char*[]);
};

int	Server::run(int, char*[]) {
  shutdownOnInterrupt(); // ��������� ������� shutdown �� ������ Interrupt.
  // ������ ������� - ����� �����, � ������� ���������� �������.
  Ice::ObjectAdapterPtr adapter =
      communicator()->createObjectAdapterWithEndpoints(
      "EniseyServerAdapter", // ��� ��������.
      "default -p 10000");    // Endpoint.
  // ������ servant'�, ������������ ��������� SlaeSolverIce.
  Enisey::SlaeSolverIceCVMPtr slae_solver_ice_cvm = 
      new Enisey::SlaeSolverIceCVM;
  slae_solver_ice_cvm->ActivateSelfInAdapter(adapter);
  adapter->activate(); // �������� ������� ���������� �� �������� ����������.
  communicator()->waitForShutdown(); // ���������������� ������ �����.
  if(interrupted()) {
    std::cerr << appName() << ": received signal, shutting down" << std::endl;
  }
  return 0;
}

int main(int argc, char* argv[]) {
  Server s;
  return s.main(argc, argv);
}
