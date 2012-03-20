#ifndef _FINDER_TEMPERATURE_PSEUDO_CRITICAL_ICE
#define _FINDER_TEMPERATURE_PSEUDO_CRITICAL_ICE

#include <common_types.ice>

module Enisey {
  interface FinderPressurePseudoCritical {
    idempotent void Find(
	    NumberSequence DensityInStandartConditions,
		NumberSequence Nitrogen,
		NumberSequence Hydrocarbon,
		out NumberSequence TemperaturePseudoCritical);
  };
};

#endif