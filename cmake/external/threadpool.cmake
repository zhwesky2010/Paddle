INCLUDE(ExternalProject)

SET(THREADPOOL_REPOSITORY "https://github.com/progschj/ThreadPool.git")
SET(THREADPOOL_TAG "9a42ec1329f259a5f4881a291db1dcb8f2ad9040")
SET(THREADPOOL_SOURCES_DIR ${THIRD_PARTY_PATH}/threadpool)

cache_third_party(extern_threadpool
  REPOSITORY ${THREADPOOL_REPOSITORY}
  TAG        ${THREADPOOL_TAG}
  DIR        ${THREADPOOL_SOURCES_DIR})

SET(THREADPOOL_INCLUDE_DIR ${SOURCE_DIR})
INCLUDE_DIRECTORIES(${THREADPOOL_INCLUDE_DIR})
  
message("extern_threadpool ${SOURCE_DIR} ${DOWNLOAD_CMD}")

ExternalProject_Add(
    extern_threadpool
    ${EXTERNAL_PROJECT_LOG_ARGS}
    PREFIX          ${THREADPOOL_SOURCES_DIR}
    SOURCE_DIR      ${SOURCE_DIR}
    DOWNLOAD_COMMAND  ${DOWNLOAD_CMD}
    UPDATE_COMMAND  ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND     ""
    INSTALL_COMMAND   ""
    TEST_COMMAND      ""
)

if (${CMAKE_VERSION} VERSION_LESS "3.3.0")
    set(dummyfile ${CMAKE_CURRENT_BINARY_DIR}/threadpool_dummy.c)
    file(WRITE ${dummyfile} "const char *dummy_threadpool = \"${dummyfile}\";")
    add_library(simple_threadpool STATIC ${dummyfile})
else()
    add_library(simple_threadpool INTERFACE)
endif()

add_dependencies(simple_threadpool extern_threadpool)
