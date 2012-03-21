#pragma once
#include "finder_interface.h"
#include <vector>
class FinderCuda : public FinderInterface {
 public:
  virtual void Find(std::vector<float> &result);
};