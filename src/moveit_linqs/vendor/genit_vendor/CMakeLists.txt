cmake_minimum_required(VERSION 3.10)
project(genit_vendor)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_vendor_package REQUIRED)

find_package(absl_vendor REQUIRED)

ament_vendor(genit_vendor
  VCS_URL https://github.com/theteamatx/x-edr-genit.git
  VCS_VERSION 542a05502bf8b0e94c7b756aead336dbcf2c7471
  SOURCE_SUBDIR genit
  PATCHES patches
)

ament_export_dependencies(absl_vendor absl)
ament_package(CONFIG_EXTRAS "genit_vendor-extras.cmake")