/** \file slae_solver_cusp.cpp
  Реализация SlaeSolverCusp.*/
#include "slae_solver_cusp.cuh"
#include <vector>

#include <cusp/coo_matrix.h>
#include <cusp/monitor.h>
#include <cusp/precond/smoothed_aggregation.h>
#include <cusp/precond/ainv.h>
#include <cusp/krylov/bicgstab.h>
#include <cusp/krylov/cg.h>
#include <cusp/krylov/gmres.h>

  #include <cusp/csr_matrix.h>
  #include <cusp/krylov/bicg.h>
  #include <cusp/gallery/poisson.h>
#include <cusp/io/matrix_market.h>

template <typename Monitor>
void report_status(Monitor& monitor)
{
    if (monitor.converged())
    {
        std::cout << "  Solver converged to " << monitor.tolerance() << " tolerance";
        std::cout << " after " << monitor.iteration_count() << " iterations";
        std::cout << " (" << monitor.residual_norm() << " final residual)" << std::endl;
    }
    else
    {
        std::cout << "  Solver reached iteration limit " << monitor.iteration_limit() << " before converging";
        std::cout << " to " << monitor.tolerance() << " tolerance ";
        std::cout << " (" << monitor.residual_norm() << " final residual)" << std::endl;
    }
}

void SlaeSolverCusp::Solve(
    std::vector<int> const &A_indexes, 
    std::vector<double> const &A_values, 
    std::vector<double> const &B, 
    std::vector<double> *X) {

  int size = B.size();
  int non_zeros = A_indexes.size();
 
  cusp::coo_matrix<int, double, cusp::host_memory>   A_host(size, size, non_zeros);  
  for(int i = 0; i < A_indexes.size(); ++i) {
    int index = A_indexes[i];
    int row = index / size;
    int col = index - ( row*size );
    A_host.row_indices   [i] = row;
    A_host.column_indices[i] = col;
    A_host.values        [i] = A_values[i];
  }

 // cusp::io::write_matrix_market_file(A_host, "Saratov50.mtx");
  

  cusp::coo_matrix<int, double, cusp::device_memory> A_dev (size, size, non_zeros);
  A_dev = A_host;

  cusp::array1d<double, cusp::host_memory> x_host(size, 0);
  cusp::array1d<double, cusp::host_memory> b_host( B.begin(), B.end() );
 // cusp::io::write_matrix_market_file(b_host, "b_Saratov50.mtx");

//return;


  cusp::array1d<double, cusp::device_memory> x_dev(size, 0);
  cusp::array1d<double, cusp::device_memory> b_dev;
  b_dev = b_host;

  // set stopping criteria:
  //  iteration_limit    = 100
  //  relative_tolerance = 1e-6
  cusp::verbose_monitor<double> monitor(b_dev, 500, 1e-6);

  //cusp::precond::smoothed_aggregation<int, double, cusp::device_memory> M(A_dev);
  cusp::precond::diagonal<double, cusp::device_memory> M(A_dev);
  //cusp::precond::nonsym_bridson_ainv<double, cusp::device_memory> M(A_dev);
  //cusp::precond::scaled_bridson_ainv<float, cusp::device_memory> M(A_dev, .1);    
  cusp::krylov::bicgstab(A_dev, x_dev, b_dev, monitor, M);
  //cusp::krylov::cg(A_dev, x_dev, b_dev, monitor, M);
  //cusp::krylov::gmres(A_dev, x_dev, b_dev,50, monitor);

  x_host = x_dev;
  cusp::io::write_matrix_market_file(x_host, "x_Saratov50.mtx");

  X->resize(size);
  thrust::copy(x_host.begin(), x_host.end(), X->begin());

  // report status
  report_status(monitor);  

  // print hierarchy information
  //std::cout << "\nPreconditioner statistics" << std::endl;
  //M.print();      

}
