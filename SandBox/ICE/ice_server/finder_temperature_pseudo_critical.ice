#ifndef _FINDER_TEMPERATURE_PSEUDO_CRITICAL_ICE
#define _FINDER_TEMPERATURE_PSEUDO_CRITICAL_ICE

#include <C:\Enisey\src\SandBox\ICE\ICE_server\common_types.ice>

module Enisey {
  interface FinderTemperaturePseudoCritical {
    idempotent void Find(
	    NumberSequence DensityInStandartConditions,
		NumberSequence Nitrogen,
		NumberSequence Hydrocarbon,
		out NumberSequence TemperaturePseudoCritical);
  };
};

#endif