#include "lammpsplugin.h"

#include "version.h"

#include <cstring>

#include "angle_harmonic_concave.h"

using namespace LAMMPS_NS;

static Angle *AngleHarmonicConcavecreator(LAMMPS *lmp)
{
  return new AngleHarmonicConcave(lmp);
}

extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc)
{
  lammpsplugin_t plugin;
  lammpsplugin_regfunc register_plugin = (lammpsplugin_regfunc) regfunc;

  plugin.version = LAMMPS_VERSION;
  plugin.style = "angle";
  plugin.name = "harmonic/concave";
  plugin.info = "theta range is [0, 2*PI]";
  plugin.author = "Jasmine Feng";
  plugin.creator.v1 = (lammpsplugin_factory1 *) &AngleHarmonicConcavecreator;
  plugin.handle = handle;
  (*register_plugin)(&plugin, lmp);
}