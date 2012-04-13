/** \file test_slae_solver_i.cpp
  Тесты для класса SlaeSolverI из slae_solver_i.h*/
#include "slae_solver_i.h"

#include "gtest/gtest.h"
#include "test_utils.h"

template <typename T>
class SlaeSolverTypedTest : public ::testing::Test {
 public:
  void SetUp() {
    abstract_solver = new T;
    A_indexes = MakeSimpleSlaeAIndexes();
    A_vals = MakeSimpleSlaeAValues();
    b = MakeSimpleSlaeB();
    etalon_x = MakeSimpleSlaeX();
    x.reserve( b.size() );
  }
  void TearDown() {
    delete abstract_solver;
  }
  /* Всё для составления и решения СЛАУ AX = B
  [2 0 1]   [1]   [5]
  [0 3 0] * [2] = [6]
  [1 2 3]   [3]   [14]
  Формат передачи параметров:
  A_indexes - индексы (строка*(длина) + столбец) ненулевых коэф-ов А.
  A_vals - значения этих коэф-ов в соответствующем порядке.
  B - вектор b.
  X - вектор найденного решения. 
  Так сделано для эффективности передачи разреженных матриц. 
  Хотя для них, конечно, есть свои форматы типа CSR.*/
  std::vector<int> MakeSimpleSlaeAIndexes() {
    std::vector<int> A_indexes;
    int len = 3; // Длина строки.
    A_indexes.push_back( 0 * len + 0 );  
    A_indexes.push_back( 0 * len + 2 );  
    A_indexes.push_back( 1 * len + 1 );  
    A_indexes.push_back( 2 * len + 0 );  
    A_indexes.push_back( 2 * len + 1 );  
    A_indexes.push_back( 2 * len + 2 );  
    return A_indexes;
  } 
  std::vector<double> MakeSimpleSlaeAValues() {
    std::vector<double> A_vals;
    A_vals.push_back( 2 );
    A_vals.push_back( 1 );
    A_vals.push_back( 3 );
    A_vals.push_back( 1 );
    A_vals.push_back( 2 );
    A_vals.push_back( 3 );
    return A_vals;
  }
  std::vector<double> MakeSimpleSlaeB() {
    std::vector<double> b;
    b.push_back(5); b.push_back(6); b.push_back(14);
    return b;
  }
  std::vector<double> MakeSimpleSlaeX() {
    std::vector<double> x;
    x.push_back(1); x.push_back(2); x.push_back(3);
    return x;
  }
  SlaeSolverI *abstract_solver;
  std::vector<int> A_indexes;
  std::vector<double> A_vals;
  std::vector<double> b;
  std::vector<double> etalon_x;
  std::vector<double> x;
};

// Список типов, для которых будут выполняться тесты.
typedef ::testing::Types<
    SlaeSolverCVM, 
    SlaeSolverIceClient
> SlaeSolverTypes;

TYPED_TEST_CASE(SlaeSolverTypedTest, SlaeSolverTypes);

TYPED_TEST(SlaeSolverTypedTest, SolvesSimpleSlae) {
  abstract_solver->Solve(A_indexes, A_vals, b, &x);
  EXPECT_TRUE( std::equal( x.begin(), x.end(), etalon_x.begin() ) );
}
