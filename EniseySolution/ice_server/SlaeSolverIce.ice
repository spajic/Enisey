#ifndef _SLAE_SOLVER_ICE_ICE
#define _SLAE_SOLVER_ICE_ICE

#include <C:\Enisey\src\EniseySolution\ice_server\CommonTypesIce.ice>

module Enisey {
  interface SlaeSolverIce {
	idempotent void SetSolverType(string SolverType);
    idempotent void Solve(
	    IntSequence AIndexes,
		DoubleSequence AValues,
		DoubleSequence B,
		out DoubleSequence X);
  };
};

#endif