#ifndef __finder_pressure_pseudo_criticalI_h__
#define __finder_pressure_pseudo_criticalI_h__

#include <finder_pressure_pseudo_critical.h>

namespace Enisey
{

class FinderPressurePseudoCriticalI : virtual public FinderPressurePseudoCritical {
 public:
  virtual void Find(
      const ::Enisey::NumberSequence& DensityInStandartConditions,
      const ::Enisey::NumberSequence& Nitrogen,
      const ::Enisey::NumberSequence& Hydrocarbon,
      ::Enisey::NumberSequence& PressurePseudoCritical,
	  const Ice::Current& current
  );
	// Добавление себя в ASM (ActiveServantsMap переданного адаптера.
	void Activate(const Ice::ObjectAdapterPtr& adapter);
};

typedef IceUtil::Handle<FinderPressurePseudoCriticalI> 
	FinderPressurePseudoCriticalIPtr;
}

#endif
