#pragma once

#include <string>

struct Passport
{
  virtual std::string GetName() = 0;
};