#pragma once
#include <vector>

class FinderTemperaturePseudoCriticalWorkerInterface{
 public:
   virtual void Find(
       const std::vector<float>& DensityInStandartConditions,
       const std::vector<float>& Nitrogen,
       const std::vector<float>& Hydrocarbon,
       std::vector<float>& TemperaturePseudoCritical
   ) = 0;
};