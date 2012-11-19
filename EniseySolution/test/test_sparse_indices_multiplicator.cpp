/** \file test_sparse_indices_multiplicator.cpp
Тесты для util_sparse_indices_multiplicator.cpp*/
#include "gas_transfer_system.h"

#include "gtest/gtest.h"
#include "test_utils.h"

#include "util_sparse_indices_multiplicator.h"

TEST(SparseIndicesMultiplicator, DuplicateExample) {
  std::vector<int> A;
  A.push_back(0);
  A.push_back(1);
  A.push_back(2);
  A.push_back(3);

  std::vector<int> A_dub_etalon;
  A_dub_etalon.push_back(0);
  A_dub_etalon.push_back(1);
  A_dub_etalon.push_back(4);
  A_dub_etalon.push_back(5);
  A_dub_etalon.push_back(10);
  A_dub_etalon.push_back(11);
  A_dub_etalon.push_back(14);
  A_dub_etalon.push_back(15);

  std::vector<int> A_dub_res;
  MultipicateSparceIndices(A, &A_dub_res, 2);
  
  ASSERT_EQ( A_dub_etalon.size(), A_dub_res.size() );
  EXPECT_TRUE(
      std::equal(
          A_dub_etalon.begin(), A_dub_etalon.end(),
          A_dub_res.begin() 
      )
  );
}

TEST(SparseIndicesMultiplicator, TriplicateExample) {
  std::vector<int> A;
  A.push_back(0);
  A.push_back(1);
  A.push_back(2);
  A.push_back(3);

  std::vector<int> A_trip_etalon;
  A_trip_etalon.push_back(0);
  A_trip_etalon.push_back(1);
  A_trip_etalon.push_back(6);
  A_trip_etalon.push_back(7);
  A_trip_etalon.push_back(14);
  A_trip_etalon.push_back(15);
  A_trip_etalon.push_back(20);
  A_trip_etalon.push_back(21);
  A_trip_etalon.push_back(28);
  A_trip_etalon.push_back(29);
  A_trip_etalon.push_back(34);
  A_trip_etalon.push_back(35);

  std::vector<int> A_trip_res;
  MultipicateSparceIndices(A, &A_trip_res, 3);

  ASSERT_EQ( A_trip_etalon.size(), A_trip_res.size() );
  EXPECT_TRUE(
      std::equal(
          A_trip_etalon.begin(), A_trip_etalon.end(),
          A_trip_res.begin() 
      )
  );
}
