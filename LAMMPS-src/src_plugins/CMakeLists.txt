# -*- CMake -*- file for plugin examples.
# The is meant to be used as a template for plugins that are
# distributed independent from the LAMMPS package.
##########################################

cmake_minimum_required(VERSION 3.16)

# enforce out-of-source build
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source builds are not allowed. You must create and use a build directory. "
    "Please remove CMakeCache.txt and CMakeFiles first.")
endif()

project(plugins VERSION 1.0 LANGUAGES CXX)

# when this file is included as subdirectory in the LAMMPS build, many settings are directly imported
if(LAMMPS_DIR)
  set(LAMMPS_HEADER_DIR ${LAMMPS_SOURCE_DIR})
else()
  # NOTE: the next line should be commented out when used outside of the LAMMPS package
  get_filename_component(LAMMPS_SOURCE_DIR ${PROJECT_SOURCE_DIR}/../src ABSOLUTE)
  set(LAMMPS_HEADER_DIR ${LAMMPS_SOURCE_DIR} CACHE PATH "Location of LAMMPS headers")
  if(NOT LAMMPS_HEADER_DIR)
    message(FATAL_ERROR "Must set LAMMPS_HEADER_DIR")
  endif()
  # by default, install into $HOME/.local (not /usr/local),
  # so that no root access (and sudo) is needed
  if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "$ENV{HOME}/.local" CACHE PATH "Default install path" FORCE)
  endif()
  # ugly hacks for MSVC which by default always reports an old C++ standard in the __cplusplus macro
  # and prints lots of pointless warnings about "unsafe" functions
  if(MSVC)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
      add_compile_options(/Zc:__cplusplus)
      add_compile_options(/wd4244)
      add_compile_options(/wd4267)
      add_compile_options(/wd4250)
      add_compile_options(/EHsc)
    endif()
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
  endif()
endif()

# we *require* C++11 without extensions but prefer C++17.
# Kokkos requires at least C++17 (currently)
if(NOT CMAKE_CXX_STANDARD)
  if(cxx_std_17 IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    set(CMAKE_CXX_STANDARD 17)
  else()
    set(CMAKE_CXX_STANDARD 11)
  endif()
endif()
if(CMAKE_CXX_STANDARD LESS 11)
  message(FATAL_ERROR "C++ standard must be set to at least 11")
endif()
if(CMAKE_CXX_STANDARD LESS 17)
  message(WARNING "Selecting C++17 standard is preferred over C++${CMAKE_CXX_STANDARD}")
endif()
if(PKG_KOKKOS AND (CMAKE_CXX_STANDARD LESS 17))
  set(CMAKE_CXX_STANDARD 17)
endif()
# turn off C++17 check in lmptype.h
if(LAMMPS_CXX11)
  add_compile_definitions(LAMMPS_CXX11)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Need -restrict with Intel compilers
if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -restrict")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
include(CheckIncludeFileCXX)
if(NOT LAMMPS_DIR)
  include(LAMMPSInterfaceCXX)
endif()

##########################
# building the plugins

add_library(rhonorm_plugin MODULE SPH/rhonorm_plugin.cpp SPH/pair_sph_rhosum_norm.cpp SPH/pair_sph_rho_mls.cpp)
target_link_libraries(rhonorm_plugin PRIVATE lammps)

add_library(anharmonic_plugin MODULE BPM/anharmonic_plugin.cpp BPM/bond_bpm_spring_anharmonic.cpp)
target_link_libraries(anharmonic_plugin PRIVATE lammps)

add_library(angle_concave_plugin MODULE MOLECULE/angle_concave_plugin.cpp MOLECULE/angle_harmonic_concave.cpp)
target_link_libraries(angle_concave_plugin PRIVATE lammps)

add_library(inflow_plugin MODULE RHEO/inflow_plugin.cpp RHEO/fix_rheo_deposit.cpp)
target_link_libraries(inflow_plugin PRIVATE lammps)

add_library(setrho_plugin MODULE setrho_plugin.cpp fix_setrho.cpp)
target_link_libraries(setrho_plugin PRIVATE lammps)

add_library(avechunk_plugin MODULE avechunk_plugin.cpp fix_ave_chunk_preforce.cpp)
target_link_libraries(avechunk_plugin PRIVATE lammps)

set_target_properties(rhonorm_plugin anharmonic_plugin angle_concave_plugin inflow_plugin setrho_plugin avechunk_plugin
 PROPERTIES PREFIX "" SUFFIX ".so")

# MacOS seems to need this
if(CMAKE_SYSTEM_NAME STREQUAL Darwin)
  set_target_properties(rhonorm_plugin anharmonic_plugin angle_concave_plugin inflow_plugin setrho_plugin avechunk_plugin
    PROPERTIES LINK_FLAGS "-Wl,-undefined,dynamic_lookup")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
# tell CMake to export all symbols to a .dll on Windows with special case for MinGW cross-compilers
  set_target_properties(rhonorm_plugin anharmonic_plugin angle_concave_plugin inflow_plugin setrho_plugin avechunk_plugin
    PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
  if(CMAKE_CROSSCOMPILING)
    set_target_properties(rhonorm_plugin anharmonic_plugin angle_concave_plugin inflow_plugin setrho_plugin avechunk_plugin
      PROPERTIES LINK_FLAGS "-Wl,--export-all-symbols")
  endif()
else()
  set_target_properties(rhonorm_plugin anharmonic_plugin angle_concave_plugin inflow_plugin setrho_plugin avechunk_plugin PROPERTIES
    LINK_FLAGS "-rdynamic")
endif()

add_custom_target(plugins ALL ${CMAKE_COMMAND} -E echo "Building Plugins"
  DEPENDS rhonorm_plugin anharmonic_plugin angle_concave_plugin inflow_plugin setrho_plugin avechunk_plugin)
