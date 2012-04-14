/** \file enisey_vesta.cpp
����� ����� � ��� ���������, ���������� ������ ������ Calc'a.*/
#include <string>
#include <iostream>
#include <fstream>
#include "gas_transfer_system_ice_client.h"

const std::string kVestaFolder = "C:\\Program Files (x86)\\Vesta\\Data\\";

std::vector<std::string> FileAsVectorOfStrings(std::string filename) {
  std::vector<std::string> res;
  std::ifstream f(filename);
  std::string line;
  while( getline(f, line) ) {
    res.push_back(line);
  }
  return res;
}

int main(int argc, char* argv[]) {
  /* ����� ������� ������ �� ������, �������������� ������.
  ����� ��������� �������������� � ������� ������ �� ���� �������
  GasTransferSystemI - ���������� ����� ��� ������� � Ice-���������.
  ����� �������� ���������� ������ � ���� � ������� ����� � �����.*/

  GasTransferSystemIceClient gts;
  std::vector<std::string> result_of_balancing;
  std::vector<double> abs_disbalances;
  std::vector<int> int_disbalances;
  gts.PeroformBalancing(
      FileAsVectorOfStrings(kVestaFolder + "MatrixConnections.dat"),
      FileAsVectorOfStrings(kVestaFolder + "InOutGRS.dat"),
      FileAsVectorOfStrings(kVestaFolder + "PipeLine.dat"),
      &result_of_balancing,
      &abs_disbalances,
      &int_disbalances
  );
  std::ofstream f(kVestaFolder + "Result.dat");
  for(auto s = result_of_balancing.begin(); s != result_of_balancing.end(); 
      ++s) {
    f << *s << std::endl;
  }
	return 0;
}
