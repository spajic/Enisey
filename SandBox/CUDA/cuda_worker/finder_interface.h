#pragma once
#include <vector>

class FinderInterface {
 public:
  virtual void Find(std::vector<float> &result) = 0;
};
