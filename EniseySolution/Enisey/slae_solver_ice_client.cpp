/** \file test_slae_solver_ice_client.cpp
  Реализация SlaeSolverIceClient.*/
#include "slae_solver_ice_client.h"
#include "Ice/Ice.h"
#include "slae_solver_ice.h"
#include <vector>

SlaeSolverIceClient::SlaeSolverIceClient() {
  Prepare("SlaeSolverIce:default -p 10000");
}

SlaeSolverIceClient::SlaeSolverIceClient(std::string endpoint) {
  Prepare(endpoint);
}

void SlaeSolverIceClient::Prepare(std::string endpoint) {
  try {
    ic_ = Ice::initialize();
    Ice::ObjectPrx base = ic_->stringToProxy(endpoint);
    solver_proxy_ = Enisey::SlaeSolverIcePrx::checkedCast(base);
    if(!solver_proxy_) {
      std::cout << "Invalid proxy";
    };      
  } catch(const Ice::Exception &ex) {  
    std::cerr << ex << std::endl;
  }
}

SlaeSolverIceClient::~SlaeSolverIceClient() {
  if(ic_) {
    try {
      ic_->destroy();
    } catch(const Ice::Exception &e) {
      std::cerr << e << std::endl;
    }
  }
}

void SlaeSolverIceClient::SetSolverType(std::string const &solver_type) {
  solver_proxy_->SetSolverType(solver_type);
}

void SlaeSolverIceClient::Solve(
    std::vector<int> const &A_indexes, 
    std::vector<double> const &A_values, 
    std::vector<double> const &B, 
    std::vector<double> *X) {
  try {
    solver_proxy_->Solve(A_indexes, A_values, B, *X);  
    } catch(const char *msg) {
        std::cerr << msg << std::endl;
      }       
}
