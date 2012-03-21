#include "finder_temperature_pseudo_critical_worker_cuda.h"
#include "thrust/device_vector.h"
#include <vector>

void FinderPressurePseudoCriticalWorkerCuda::Find(
    const ::std::vector<float>& DensityInStandartConditions,
    const ::std::vector<float>& Nitrogen,
    const ::std::vector<float>& Hydrocarbon,
    ::std::vector<float>& TemperaturePseudoCritical) {
  // Просто делаем что-бы то ни было на CUDA.
  thrust::device_vector<float> v(3);
  v[0] = 99;
  // Просто возвращаем [99, 99, 99].
  TemperaturePseudoCritical.resize(3);
  TemperaturePseudoCritical[0] = 99;
  TemperaturePseudoCritical[1] = 99;
  TemperaturePseudoCritical[2] = 99;
  return;
}