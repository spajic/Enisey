/** \file test_slae_solver_ice_client.cpp
  Реализация SlaeSolverIceClient.*/
#include "slae_solver_ice_client.h"
#include "Ice/Ice.h"
#include "slae_solver_ice.h"
#include <vector>

void SlaeSolverIceClient::Solve(
    std::vector<int> const &A_indexes, 
    std::vector<double> const &A_values, 
    std::vector<double> const &B, 
    std::vector<double> *X) {
  Ice::CommunicatorPtr ic;
  try {
    ic = Ice::initialize();
    Ice::ObjectPrx base = ic->stringToProxy("SlaeSolverIceCVM:default -p 10000");
    Enisey::SlaeSolverIcePrx solver_proxy = 
        Enisey::SlaeSolverIcePrx::checkedCast(base);
    if(!solver_proxy) {
      std::cout << "Invalid proxy";
    } else {
      solver_proxy->Solve(A_indexes, A_values, B, *X);
    }
  } catch(const Ice::Exception &ex){  
    std::cerr << ex << std::endl;
  } catch(const char *msg) {
    std::cerr << msg << std::endl;
  }
  if(ic) {
    try {
      ic->destroy();
    } catch(const Ice::Exception &e) {
      std::cerr << e << std::endl;
    }
  }
}