#include "finder_interface.h"
#include "FinderCuda.h"
#include <vector>
#include <iostream>

int main(int argc, char* argv[]) {
	FinderInterface* abstr_interface = new FinderCuda();
	std::vector<float> res(3);
	abstr_interface->Find(res);
	std::cout << res[0] << "-" << res[1] << "-" << res[2] << std::endl;
	return 0;
}