#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <iostream>

int main() {
  thrust::host_vector<float> h_x(3);
  h_x[0] = 22;
  h_x[1] = 22;
  h_x[2] = 22;
  thrust::host_vector<float> h_y(3);
  h_y[0] = 44;
  h_y[1] = 44;
  h_y[2] = 44;

  thrust::device_vector<float> d_x(3);
  d_x = h_x;
  thrust::device_vector<float> d_y(3);
  d_y = h_y;
  thrust::device_vector<float> d_z(3);

  // z = x + y
  thrust::transform(d_x.begin(), d_x.end(), d_y.begin(), d_z.begin(), thrust::plus<float>());

  thrust::host_vector<float> h_z(3);
  h_z = d_z;
  std::cout << h_z[0] << ", " << h_z[1] <<  ", " << h_z[2] << std::endl;
  return 0;
}