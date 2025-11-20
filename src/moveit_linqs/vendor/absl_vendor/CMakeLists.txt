cmake_minimum_required(VERSION 3.10)
project(absl_vendor)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_vendor_package REQUIRED)

ament_add_default_options()

find_package(absl 20250512.0.0 QUIET NO_CMAKE_PACKAGE_REGISTRY)

ament_vendor(absl_vendor
  SATISFIED ${absl_FOUND}
  VCS_URL https://github.com/abseil/abseil-cpp.git
  #VCS_VERSION 20250512.0
  VCS_VERSION bc257a88f7c1939f24e0379f14a3589e926c950c
  CMAKE_ARGS
    -DCMAKE_CXX_STANDARD=17
    -DCMAKE_CXX_STANDARD_REQUIRED:BOOL=ON
    -DBUILD_TESTING:BOOL=OFF
    -DBUILD_SHARED_LIBS:BOOL=${BUILD_SHARED_LIBS}
    -DABSL_ENABLE_INSTALL:BOOL=ON
)

ament_package(CONFIG_EXTRAS "absl_vendor-extras.cmake")