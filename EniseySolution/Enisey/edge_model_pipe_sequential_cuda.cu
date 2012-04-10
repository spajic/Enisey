#include "edge_model_pipe_sequential_cuda.cuh"

#include "gas.h"

#include "manager_edge_model_pipe_sequential_cuda.cuh"

#include <string>

EdgeModelPipeSequentialCuda::EdgeModelPipeSequentialCuda()
{

}

EdgeModelPipeSequentialCuda::~EdgeModelPipeSequentialCuda()
{

}

EdgeModelPipeSequentialCuda::EdgeModelPipeSequentialCuda(int index, ManagerEdgeModelPipeSequentialCuda* manager) :
index_(index), manager_(manager)
{

}


void EdgeModelPipeSequentialCuda::set_gas_in(const Gas* gas)
{
	manager_->set_gas_in(gas, index_);
}

void EdgeModelPipeSequentialCuda::set_gas_out(const Gas* gas)
{
	manager_->set_gas_out(gas, index_);
}

std::string EdgeModelPipeSequentialCuda::GetName()
{
	return "EdgeModelPipeSequentialCuda";
}

double EdgeModelPipeSequentialCuda::q()
{
	return manager_->q(index_);
}