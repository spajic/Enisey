#pragma once

#include <string>

// forward declarations
struct Gas;

class Edge
{
public:
  virtual std::string GetName() = 0;
  virtual void set_gas_in(const Gas* gas) = 0;
  virtual void set_gas_out(const Gas* gas) = 0;
  virtual float q() = 0;
};