include(ExternalProject)

SET(DLPACK_REPOSITORY "https://github.com/dmlc/dlpack.git")
SET(DLPACK_TAG    "v0.2")
set(DLPACK_SOURCES_DIR ${THIRD_PARTY_PATH}/dlpack)

cache_third_party(extern_dlpack
  REPOSITORY ${DLPACK_REPOSITORY}
  TAG        ${DLPACK_TAG}
  DIR        ${DLPACK_SOURCES_DIR})

set(DLPACK_INCLUDE_DIR ${SOURCE_DIR}/include CACHE PATH "dlpack include directory." FORCE)
INCLUDE_DIRECTORIES(${DLPACK_INCLUDE_DIR})

ExternalProject_Add(
  extern_dlpack
  ${EXTERNAL_PROJECT_LOG_ARGS}
  PREFIX          ${DLPACK_SOURCES_DIR}
  SOURCE_DIR      ${SOURCE_DIR}
  DOWNLOAD_COMMAND  ${DOWNLOAD_CMD}
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
  TEST_COMMAND      ""
)

if(${CMAKE_VERSION} VERSION_LESS "3.3.0")
  set(dummyfile ${CMAKE_CURRENT_BINARY_DIR}/dlpack_dummy.c)
  file(WRITE ${dummyfile} "const char *dummy = \"${dummyfile}\";")
  add_library(dlpack STATIC ${dummyfile})
else()
  add_library(dlpack INTERFACE)
endif()

add_dependencies(dlpack extern_dlpack)
