/** \file slae_solver_cusp.cpp
  Реализация SlaeSolverCusp.*/
#include "slae_solver_cusp.cuh"
#include <vector>

#include <cusp/coo_matrix.h>
#include <cusp/monitor.h>
#include <cusp/precond/smoothed_aggregation.h>
#include <cusp/krylov/bicgstab.h>
#include <cusp/krylov/cg.h>

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
  // Матрица (size x size) with non_zeros non-zeros.
  cusp::coo_matrix<int, float, cusp::host_memory> A(size, size, non_zeros);
  for(int i = 0; i < A_indexes.size(); ++i) {
    int index = A_indexes[i];
    int row = index / size;
    int col = index - ( row*size );
    A.row_indices   [i] = row;
    A.column_indices[i] = col;
    A.values        [i] = A_values[i];
  }
  cusp::coo_matrix<int, float, cusp::device_memory> A_dev(size, size, non_zeros);
  A_dev = A;

  // allocate storage for solution (x) and right hand side (b)
  cusp::array1d<float, cusp::host_memory> x_host( X->begin(), X->end() );
  cusp::array1d<float, cusp::host_memory> b_host( B .begin(), B .end() );
  cusp::array1d<float, cusp::device_memory> x_dev( x_host.begin(), x_host.end() );
  cusp::array1d<float, cusp::device_memory> b_dev( b_host.begin(), b_host.end() );
  
  // set stopping criteria:
  //  iteration_limit    = 100
  //  relative_tolerance = 1e-3
  //cusp::verbose_monitor<double> monitor(b_dev, 100, 1e-3);
  // setup preconditioner
  //cusp::precond::smoothed_aggregation<int, float, cusp::device_memory> M(A_dev);
  
  // solve the linear system A * x = b with the BiConjugate Gradient Stabilized method
  //cusp::krylov::bicgstab(A_dev, x_dev, b_dev, monitor);

  cusp::krylov::cg(A_dev, x_dev, b_dev);
  
  // report status
  //report_status(monitor);
  // print hierarchy information
  std::cout << "\nPreconditioner statistics" << std::endl;
//  M.print();

  // Копируем результат на host.
  x_host = x_dev;
  
  // Записываем результатв out-параметр метода.
  X->clear();
  X->resize(size);
  std::copy( x_host.begin(), x_host.end(), X->begin() );
}
