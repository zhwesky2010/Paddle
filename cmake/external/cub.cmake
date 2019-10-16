if(NOT WITH_GPU)
  return()
endif()

include(ExternalProject)

SET(CUB_REPOSITORY "https://github.com/NVlabs/cub.git")
SET(CUB_TAG "v1.8.0")
set(CUB_SOURCES_DIR ${THIRD_PARTY_PATH}/cub)

cache_third_party(extern_cub
    REPOSITORY ${CUB_REPOSITORY}
    TAG        ${CUB_TAG}
    DIR        ${CUB_SOURCES_DIR})

set(CUB_INCLUDE_DIR ${SOURCE_DIR} CACHE PATH "cub include directory." FORCE)
INCLUDE_DIRECTORIES(${CUB_INCLUDE_DIR})

ExternalProject_Add(
  extern_cub
  ${EXTERNAL_PROJECT_LOG_ARGS}
  PREFIX          ${CUB_SOURCES_DIR}
  SOURCE_DIR      ${SOURCE_DIR}
  DOWNLOAD_COMMAND  ${DOWNLOAD_CMD}
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
  TEST_COMMAND      ""
)

if(${CMAKE_VERSION} VERSION_LESS "3.3.0")
  set(dummyfile ${CMAKE_CURRENT_BINARY_DIR}/cub_dummy.c)
  file(WRITE ${dummyfile} "const char *dummy = \"${dummyfile}\";")
  add_library(cub STATIC ${dummyfile})
else()
  add_library(cub INTERFACE)
endif()

add_dependencies(cub extern_cub)
