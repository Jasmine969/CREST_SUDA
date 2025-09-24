#include "lammpsplugin.h"

#include "version.h"

#include <cstring>

#include "bond_bpm_spring_anharmonic.h"

using namespace LAMMPS_NS;

static Bond *BondBPMSpringAnharmoniccreator(LAMMPS *lmp)
{
  return new BondBPMSpringAnharmonic(lmp);
}

extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc)
{
  lammpsplugin_t plugin;
  lammpsplugin_regfunc register_plugin = (lammpsplugin_regfunc) regfunc;

  plugin.version = LAMMPS_VERSION;
  plugin.style = "bond";
  plugin.name = "bpm/spring/anharmonic";
  plugin.info = "k1*(r-r0)+k3*(r-r0)^3";
  plugin.author = "Jasmine Feng";
  plugin.creator.v1 = (lammpsplugin_factory1 *) &BondBPMSpringAnharmoniccreator;
  plugin.handle = handle;
  (*register_plugin)(&plugin, lmp);
}