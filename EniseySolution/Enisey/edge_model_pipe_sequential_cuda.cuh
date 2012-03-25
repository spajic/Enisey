#pragma once
#include <string> 

#include "edge.h"

// forward-declaration
struct Gas;
class ManagerEdgeModelPipeSequentialCuda;

class EdgeModelPipeSequentialCuda: public Edge
{
public:
	EdgeModelPipeSequentialCuda();
	EdgeModelPipeSequentialCuda(int index, ManagerEdgeModelPipeSequentialCuda* manager);
	~EdgeModelPipeSequentialCuda();
	void set_gas_in(const Gas* gas);
	void set_gas_out(const Gas* gas);
	std::string GetName();
	float q();

private:
	int index_;
	ManagerEdgeModelPipeSequentialCuda* manager_;
};