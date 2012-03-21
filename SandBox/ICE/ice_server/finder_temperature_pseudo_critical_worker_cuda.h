#pragma once
#include "finder_temperature_pseudo_critical_worker_interface.h"
#include <vector>

class FinderPressurePseudoCriticalWorkerCuda : 
  public FinderTemperaturePseudoCriticalWorkerInterface {
 public:
    virtual void Find(
      const std::vector<float>& DensityInStandartConditions,
      const std::vector<float>& Nitrogen,
      const std::vector<float>& Hydrocarbon,
      std::vector<float>& TemperaturePseudoCritical
    );
};