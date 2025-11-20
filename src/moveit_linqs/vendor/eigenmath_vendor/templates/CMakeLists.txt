# Create patchfile with
#diff -u --label a/eigenmath/CMakeLists.txt --label b/eigenmath/CMakeLists.txt /dev/null ./CMakeLists.txt > ../patches/00_eigenmath_cmakelists.patch

cmake_minimum_required(VERSION 3.16)
project(eigenmath        
        VERSION 1.0.0 
        LANGUAGES CXX)
include(GNUInstallDirs)

set(LIBRARY_TARGET_NAME eigenmath)

# Find required packages
find_package(Eigen3 REQUIRED)
find_package(absl REQUIRED)
find_package(genit REQUIRED)

# eigenmath library
add_library(${LIBRARY_TARGET_NAME} SHARED)
add_library(${PROJECT_NAME}::${LIBRARY_TARGET_NAME} ALIAS ${LIBRARY_TARGET_NAME})

target_sources(${LIBRARY_TARGET_NAME}
  PRIVATE
    covariance.cc
    quadratic_optimization.cc
    quadrature.cc
    quintic_spline.cc
    spline_factories.cc
    spline_fit.cc
    vector_utils.cc

    # Headers (public interface)
    constants.h
    covariance.h
    distribution.h
    halton_sequence.h
    interpolation.h
    line_search.h
    line_utils.h
    manifolds.h
    mean_and_covariance.h
    normal_distribution.h
    numerical_derivatives.h
    plane_conversions.h
    pose2.h
    pose3.h
    pose3_utils.h
    pose_and_covariance3_utils.h
    prime.h
    quadratic_optimization.h
    quadrature.h
    quintic_spline.h
    quintic_spline_segment.h
    rotation_utils.h
    scalar_utils.h
    similarity3.h
    simple_filters.h
    so2.h
    so2_interval.h
    so3.h
    spline_coefficients.h
    spline_factories.h
    spline_fit.h
    type_checks.h
    types.h
    utils.h
    vector_utils.h
)

target_include_directories(${LIBRARY_TARGET_NAME} PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )

target_compile_features(${LIBRARY_TARGET_NAME} PUBLIC cxx_std_17)
target_link_libraries(${LIBRARY_TARGET_NAME} 
    PUBLIC
        Eigen3::Eigen
        absl::base
        absl::flat_hash_set
        absl::log
        absl::status
        absl::statusor
        absl::strings
        absl::str_format
        absl::optional
        absl::span
        genit::iterators
)

# incremental_stats library (header-only)
add_library(incremental_stats INTERFACE)
add_library(${PROJECT_NAME}::incremental_stats ALIAS incremental_stats)
target_sources(incremental_stats INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/incremental_stats.h>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/incremental_stats.h>
)
target_include_directories(incremental_stats INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(incremental_stats INTERFACE
    absl::time
)


# conversions
# Would need protobuf library.
#add_library(conversions SHARED)
#target_sources(conversions
#  PRIVATE
#    conversions.h
#    conversions.cc
#)

#target_include_directories(
#    conversions
#    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>
#           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

#target_link_libraries(conversions
#  PUBLIC
#    absl::log
#    absl::status
#    absl::strings
#    eigenmath::eigenmath
#)

# matchers library (header-only)
add_library(matchers INTERFACE)
add_library(${PROJECT_NAME}::matchers ALIAS matchers)
target_sources(matchers INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/matchers.h>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/matchers.h>
)
target_include_directories(matchers INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(matchers
  INTERFACE
    Eigen3::Eigen
    eigenmath::eigenmath
    genit::iterators
)

# sampling library (header-only)
add_library(sampling INTERFACE)
add_library(${PROJECT_NAME}::sampling ALIAS sampling)
target_sources(sampling INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/sampling.h>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/sampling.h>
)

target_include_directories(sampling INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>
    $<INSTALL_INTERFACE:include>
)

# compare vectors (header-only)
add_library(compare_vectors INTERFACE)
add_library(${PROJECT_NAME}::compare_vectors ALIAS compare_vectors)
target_sources(compare_vectors INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/compare_vectors.h>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/compare_vectors.h>
)

target_include_directories(compare_vectors INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>
    $<INSTALL_INTERFACE:include>
)

# quasi_random_vector
add_library(quasi_random_vector SHARED)
add_library(${PROJECT_NAME}::quasi_random_vector ALIAS quasi_random_vector)
target_sources(quasi_random_vector
  PRIVATE
    quasi_random_vector.h
    quasi_random_vector.cc
)

target_include_directories(
    quasi_random_vector
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_link_libraries(quasi_random_vector
  PUBLIC
    eigenmath::eigenmath
)


install(
    TARGETS 
        ${LIBRARY_TARGET_NAME}
        incremental_stats
        matchers
        sampling
        compare_vectors
        quasi_random_vector
    EXPORT ${PROJECT_NAME}-config
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(
    EXPORT ${PROJECT_NAME}-config
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})

# Could perhaps be improved to not depend on me filing out the h files manually.
install(
    FILES 
        compare_vectors.h
        constants.h
        covariance.h
        distribution.h
        halton_sequence.h
        interpolation.h
        line_search.h
        line_utils.h
        manifolds.h
        matchers.h
        mean_and_covariance.h
        normal_distribution.h
        numerical_derivatives.h
        plane_conversions.h
        pose2.h
        pose3.h
        pose3_utils.h
        pose_and_covariance3_utils.h
        prime.h
        quadratic_optimization.h
        quadrature.h
        quasi_random_vector.h
        quintic_spline.h
        quintic_spline_segment.h
        rotation_utils.h
        sampling.h
        scalar_utils.h
        similarity3.h
        simple_filters.h
        so2.h
        so2_interval.h
        so3.h
        spline_coefficients.h
        spline_factories.h
        spline_fit.h
        type_checks.h
        types.h
        utils.h
        vector_utils.h
    DESTINATION 
        "${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}"
    )