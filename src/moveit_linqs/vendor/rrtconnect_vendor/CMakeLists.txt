cmake_minimum_required(VERSION 3.15)
project(rrtconnect_vendor)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_vendor_package REQUIRED)


find_package(absl_vendor REQUIRED)
find_package(genit_vendor REQUIRED)
find_package(eigenmath_vendor REQUIRED)

# I do not want to initialize submodules, because they are broken and we have vendor packages.
# https://cmake.org/cmake/help/latest/policy/CMP0097.html
cmake_policy(SET CMP0097 NEW)

ament_vendor(rrtconnect_vendor
  VCS_URL https://github.com/gyger/x-edr-rrtconnect.git
  VCS_VERSION 32d9858b8412b5e32b7ca01765c0ae4fcefef7e2
  SOURCE_SUBDIR rrtconnect
  PATCHES patches
)

ament_export_dependencies(
  absl_vendor absl 
  genit_vendor genit 
  eigenmath_vendor eigenmath
)

ament_package(CONFIG_EXTRAS "rrtconnect_vendor-extras.cmake")