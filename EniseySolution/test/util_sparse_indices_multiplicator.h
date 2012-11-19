/** \file util_sparse_indices_multiplicator.h
Функция для "дулбирования" вектора индексов ненулевых элементов.
*/
#pragma once

#include <vector>
void MultipicateSparceIndices(
    unsigned int length,
    std::vector<int> const &from,
    std::vector<int> *to,
    int multiplicity);
