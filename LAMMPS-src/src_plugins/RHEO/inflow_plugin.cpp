#include "lammpsplugin.h"

#include "version.h"

#include <cstring>

#include "fix_rheo_deposit.h"

using namespace LAMMPS_NS;

static Fix *FixRHEODepositcreator(LAMMPS *lmp, int argc, char **argv)
{
  return new FixRHEODeposit(lmp, argc, argv);
}

extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc)
{
  lammpsplugin_t plugin;
  lammpsplugin_regfunc register_plugin = (lammpsplugin_regfunc) regfunc;

  plugin.version = LAMMPS_VERSION;
  plugin.style = "fix";
  plugin.name = "rheo/deposit";
  plugin.info = "Inflow BC modified from fix deposit v1.0";
  plugin.author = "Jasmine Feng";
  plugin.handle = handle;
  plugin.creator.v2 = (lammpsplugin_factory2 *) &FixRHEODepositcreator;
  (*register_plugin)(&plugin, lmp);
}