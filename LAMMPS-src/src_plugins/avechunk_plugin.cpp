#include "lammpsplugin.h"

#include "version.h"

#include <cstring>

#include "fix_ave_chunk_preforce.h"

using namespace LAMMPS_NS;

static Fix *FixAveChunkPreforcecreator(LAMMPS *lmp, int argc, char **argv)
{
  return new FixAveChunkPreforce(lmp, argc, argv);
}

extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc)
{
  lammpsplugin_t plugin;
  lammpsplugin_regfunc register_plugin = (lammpsplugin_regfunc) regfunc;

  plugin.version = LAMMPS_VERSION;
  plugin.style = "fix";
  plugin.name = "ave/chunk/preforce";
  plugin.info = "fix ave/chunk during pre_force v1.0";
  plugin.author = "Jasmine Feng";
  plugin.handle = handle;
  plugin.creator.v2 = (lammpsplugin_factory2 *) &FixAveChunkPreforcecreator;
  (*register_plugin)(&plugin, lmp);
}