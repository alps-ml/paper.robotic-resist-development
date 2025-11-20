cmake_minimum_required(VERSION 3.10)
project(eigenmath_vendor)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_vendor_package REQUIRED)

find_package(absl_vendor REQUIRED)
find_package(genit_vendor REQUIRED)

ament_vendor(eigenmath_vendor
  VCS_URL https://github.com/theteamatx/x-edr-eigenmath.git
  # VCS_VERSION 1.0.0
  VCS_VERSION 73660d4a0280c0943c7cee0df210d835aee81c40
  SOURCE_SUBDIR eigenmath
#  CMAKE_ARGS
#    -DZSTD_BUILD_STATIC:BOOL=ON
  PATCHES patches
)

ament_export_dependencies(absl_vendor absl)
ament_package(CONFIG_EXTRAS "eigenmath_vendor-extras.cmake")