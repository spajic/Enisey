/** \file util_sparse_indices_multiplicator.cpp
Функция для "дулбирования" вектора индексов ненулевых элементов.
Пример:
A = [1 2], дублируем A -> [1 2 0 0]
    [3 4]                 [3 4 0 0]
                          [0 0 1 2]
                          [0 0 3 4]
Вектор индексов ненулевых элементов A: [0 1 2 3]
Вектор индексов ненулевых элементов 2А:[0 1 4 5 10 11 14 15]

Преобразование видно, если воспользоваться представление (row, col).
A -> (0,0), (0,1), (1,0), (1,1)

2A-> (0,0), (0,1), (1,0), (1,1),
     (2,2), (2,3), (3,2), (3,3).
То есть к каждому элементу A прибавляется sqrt(A.size).
*/
#include "util_sparse_indices_multiplicator.h"
#include "math.h"

struct RowCol {
  int row;
  int col;
};

int RowColToIndex(RowCol rc, int length) {
  return (length*rc.row) + rc.col;
}
RowCol IndexToRowCol(int index, int length) {
  RowCol rc;
  rc.row = index / length;
  rc.col = index - (rc.row*length);
  return rc;
}
void MultipicateSparceIndices(
    std::vector<int> const &from,
    std::vector<int> *to,
    int multiplicity) {
  int length = sqrt( static_cast<double>( from.size() ) );
  std::vector<RowCol> rc_vec_from;
  rc_vec_from.reserve(length);
  for(auto it = from.begin(); it != from.end(); ++it) {
    rc_vec_from.push_back( IndexToRowCol(*it, length) );
  }
  
  std::vector<RowCol> rc_vec_to;
  rc_vec_to.reserve( length * multiplicity );
  for(int i = 0; i < multiplicity; ++i) {
    for(auto it = rc_vec_from.begin(); it != rc_vec_from.end(); ++it) {
      RowCol rc = *it;
      rc.col += length * i;
      rc.row += length * i;
      rc_vec_to.push_back(rc);
    }    
  }

  to->clear();
  to->reserve( length * multiplicity );
  for(auto it = rc_vec_to.begin(); it != rc_vec_to.end(); ++it) {
    to->push_back( RowColToIndex(*it, length*multiplicity) );
  }
}

