// ICE_server.cpp : Defines the entry point for the console application.
//

#include <finder_temperature_pseudo_criticalI.h>
#include <Ice/Ice.h>

class Server : public Ice::Application
{
 public:
  virtual int run(int, char*[]);
};
int	Server::run(int, char*[]) {
  shutdownOnInterrupt(); // Shut down cleanly on interrupt.
  Ice::ObjectAdapterPtr adapter = 
	  communicator()->createObjectAdapterWithEndpoints(
	      "SandboxServerAdapter", // ��� ��������.
	      "default -p 10000");    // Endpoint.
  Enisey::FinderTemperaturePseudoCriticalIPtr serv = 
	  new Enisey::FinderTemperaturePseudoCriticalI();
  serv->Activate(adapter); // ������������ servant � ASM ��������.
  adapter->activate(); // All objects are created, allow client requests now.  
  communicator()->waitForShutdown(); // Wait until we are done.
  if(interrupted())
  {
  	std::cerr << appName() << ": received signal, shutting down" << std::endl;
  }
  return 0;
}

int main(int argc, char* argv[]) {
  Server s;
  return s.main(argc, argv);
}
