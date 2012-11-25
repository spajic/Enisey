#ifndef _GAS_TRANSFER_SYSTEM_ICE_ICE
#define _GAS_TRANSFER_SYSTEM_ICE_ICE

#include <C:\Enisey\src\EniseySolution\ice_server\common_types.ice>

module Enisey {
  interface GasTransferSystemIce {
	idempotent void SetNumberOfIterations(int NumberOfIterations);
    idempotent void PerformBalancing(
	    StringSequence MatrixConnectionsFile,
		StringSequence InOutGRSFile,
		StringSequence PipeLinesFile,
		out StringSequence ResultFile,
		out DoubleSequence AbsDisbalances,
		out IntSequence IntDisbalances
	);
  }; // ����� interface GasTransferSystemIce.
}; // ����� module Enisey.

#endif