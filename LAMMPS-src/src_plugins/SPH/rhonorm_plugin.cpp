#include "lammpsplugin.h"

#include "version.h"

#include <cstring>

#include "pair_sph_rhosum_norm.h"
#include "pair_sph_rho_mls.h"

using namespace LAMMPS_NS;

static Pair *PairSPHRhoSumNormcreator(LAMMPS *lmp)
{
  return new PairSPHRhoSumNorm(lmp);
}

static Pair *PairSPHRhoMLScreator(LAMMPS *lmp)
{
  return new PairSPHRhoMLS(lmp);
}

extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc)
{
  lammpsplugin_t plugin;
  lammpsplugin_regfunc register_plugin = (lammpsplugin_regfunc) regfunc;

  plugin.version = LAMMPS_VERSION;
  plugin.style = "pair";
  plugin.name = "sph/rhosum/norm";
  plugin.info = "Density normalization pair style v1.0";
  plugin.author = "Jasmine Feng";
  plugin.creator.v1 = (lammpsplugin_factory1 *) &PairSPHRhoSumNormcreator;
  plugin.handle = handle;
  (*register_plugin)(&plugin, lmp);

  plugin.name = "sph/rho/mls";
  plugin.info = "Moving least square pair style v1.0";
  plugin.author = "Jasmine Feng";
  plugin.creator.v1 = (lammpsplugin_factory1 *) &PairSPHRhoMLScreator;
  (*register_plugin)(&plugin, lmp);
}