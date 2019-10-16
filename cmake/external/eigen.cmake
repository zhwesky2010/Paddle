INCLUDE(ExternalProject)

SET(EIGEN_SOURCES_DIR ${THIRD_PARTY_PATH}/eigen3)


if(WIN32)
    set(EIGEN_REPOSITORY "https://github.com/wopeizl/eigen-git-mirror")
    set(EIGEN_TAG  "support_cuda9_win")
else()
    set(EIGEN_REPOSITORY "https://github.com/eigenteam/eigen-git-mirror")
    set(EIGEN_TAG  "917060c364181f33a735dc023818d5a54f60e54c")
endif()

if(WITH_AMD_GPU)
    set(EIGEN_REPOSITORY "https://github.com/sabreshao/hipeigen.git")
    set(EIGEN_TAG  "7cb2b6e5a4b4a1efe658abb215cd866c6fb2275e") 
endif(WITH_AMD_GPU)

cache_third_party(extern_eigen3
    REPOSITORY ${EIGEN_REPOSITORY}
    TAG        ${EIGEN_TAG}
    DIR        ${EIGEN_SOURCES_DIR})

SET(EIGEN_INCLUDE_DIR ${SOURCE_DIR} CACHE PATH "eigen include directory." FORCE)
INCLUDE_DIRECTORIES(${EIGEN_INCLUDE_DIR})

if(WITH_AMD_GPU)
    ExternalProject_Add(
        extern_eigen3
        ${EXTERNAL_PROJECT_LOG_ARGS}
        PREFIX          ${EIGEN_SOURCES_DIR}
        SOURCE_DIR      ${SOURCE_DIR}
        DOWNLOAD_COMMAND  ${DOWNLOAD_CMD}
        UPDATE_COMMAND  ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND     ""
        INSTALL_COMMAND   ""
        TEST_COMMAND      ""
    )
else()
    ExternalProject_Add(
        extern_eigen3
        ${EXTERNAL_PROJECT_LOG_ARGS}
        PREFIX          ${EIGEN_SOURCES_DIR}
        SOURCE_DIR      ${SOURCE_DIR}
        # eigen on cuda9.1 missing header of math_funtions.hpp
        # https://stackoverflow.com/questions/43113508/math-functions-hpp-not-found-when-using-cuda-with-eigen
        DOWNLOAD_COMMAND  ${DOWNLOAD_CMD}
        DOWNLOAD_NAME   "eigen"
        UPDATE_COMMAND  ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND     ""
        INSTALL_COMMAND   ""
        TEST_COMMAND      ""
    )
endif()

if (${CMAKE_VERSION} VERSION_LESS "3.3.0")
    set(dummyfile ${CMAKE_CURRENT_BINARY_DIR}/eigen3_dummy.c)
    file(WRITE ${dummyfile} "const char *dummy_eigen3 = \"${dummyfile}\";")
    add_library(eigen3 STATIC ${dummyfile})
else()
    add_library(eigen3 INTERFACE)
endif()

add_dependencies(eigen3 extern_eigen3)