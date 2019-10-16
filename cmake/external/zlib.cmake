# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

INCLUDE(ExternalProject)
SET(ZLIB_REPOSITORY "https://github.com/madler/zlib.git")
SET(ZLIB_TAG "v1.2.8")

SET(ZLIB_SOURCES_DIR ${THIRD_PARTY_PATH}/zlib)
SET(ZLIB_INSTALL_DIR ${THIRD_PARTY_PATH}/install/zlib)
SET(ZLIB_ROOT ${ZLIB_INSTALL_DIR} CACHE FILEPATH "zlib root directory." FORCE)
SET(ZLIB_INCLUDE_DIR "${ZLIB_INSTALL_DIR}/include" CACHE PATH "zlib include directory." FORCE)

macro(cache_third_party TARGET)
    SET(options "")
    SET(oneValueArgs REPOSITORY TAG DIR)
    SET(multiValueArgs "")
    cmake_parse_arguments(cache_third_party "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    STRING(LENGTH ${cache_third_party_TAG} tag_len)
    set(commitId_len 40)
    IF(WIN32 AND WITH_TP_CACHE)
       SET(CACHE_DIR ${CMAKE_SOURCE_DIR}/third_party_cache/)
       STRING(MD5 HASH1 ${cache_third_party_REPOSITORY})
       STRING(MD5 HASH2 ${cache_third_party_TAG})
       STRING(APPEND CACHE_DIR ${TARGET} "_${HASH1}" "_${HASH2}")
       SET(SOURCE_DIR ${CACHE_DIR})
       IF(IS_DIRECTORY ${CACHE_DIR})
          set(DOWNLOAD_CMD "")
          MESSAGE(STATUS "===========don't download ${TARGET}!=============")
       ELSE()
          IF(tag_len EQUAL commitId_len)
            set(DOWNLOAD_CMD git clone -b master ${cache_third_party_REPOSITORY} ${CACHE_DIR} 
                    && cd ${CACHE_DIR} && git reset --hard ${cache_third_party_TAG})
            set(CMD "git clone -b master ${cache_third_party_REPOSITORY} ${CACHE_DIR} --single-branch 
                    && cd ${CACHE_DIR} && git reset --hard ${cache_third_party_TAG}")
          ELSE()
            set(DOWNLOAD_CMD git clone -b ${cache_third_party_TAG} ${cache_third_party_REPOSITORY} ${CACHE_DIR} --depth 1 --single-branch)
            set(CMD "git clone -b ${cache_third_party_TAG} ${cache_third_party_REPOSITORY} ${CACHE_DIR} --depth 1 --single-branch")
          ENDIF()
          MESSAGE(STATUS "===========download to =====${CACHE_DIR}====${CMD}====")
       ENDIF()
    ELSE()
      set(SOURCE_DIR ${cache_third_party_DIR}/src/${TARGET})
      IF(tag_len EQUAL commitId_len)
        set(DOWNLOAD_CMD git clone -b master ${cache_third_party_REPOSITORY} ${SOURCE_DIR} 
              && cd ${SOURCE_DIR} && git reset --hard ${cache_third_party_TAG})
        set(CMD "git clone -b master ${cache_third_party_REPOSITORY} ${CACHE_DIR} --single-branch 
              && cd ${SOURCE_DIR} && git reset --hard ${cache_third_party_TAG}")
      ELSE()
        set(DOWNLOAD_CMD git clone -b ${cache_third_party_TAG} ${cache_third_party_REPOSITORY} ${SOURCE_DIR} --depth 1 --single-branch)
        set(CMD "git clone -b ${cache_third_party_TAG} ${cache_third_party_REPOSITORY} ${SOURCE_DIR} --depth 1 --single-branch")
      ENDIF()
      MESSAGE(STATUS "=========download to ======${SOURCE_DIR}=====${CMD}==")
    ENDIF()
endmacro(cache_third_party)

cache_third_party(extern_zlib
    REPOSITORY ${ZLIB_REPOSITORY}
    TAG        ${ZLIB_TAG}
    DIR        ${ZLIB_SOURCES_DIR})

message("extern_zlib ${SOURCE_DIR} ${DOWNLOAD_CMD}")

INCLUDE_DIRECTORIES(${ZLIB_INCLUDE_DIR}) # For zlib code to include its own headers.
INCLUDE_DIRECTORIES(${THIRD_PARTY_PATH}/install) # For Paddle code to include zlib.h.

ExternalProject_Add(
    extern_zlib
    ${EXTERNAL_PROJECT_LOG_ARGS}
    PREFIX          ${ZLIB_SOURCES_DIR}
    SOURCE_DIR      ${SOURCE_DIR}
    DOWNLOAD_COMMAND  ${DOWNLOAD_CMD}
    UPDATE_COMMAND  ""
    CMAKE_ARGS      -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                    -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
                    -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
                    -DCMAKE_INSTALL_PREFIX=${ZLIB_INSTALL_DIR}
                    -DBUILD_SHARED_LIBS=OFF
                    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                    -DCMAKE_MACOSX_RPATH=ON
                    -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
                    ${EXTERNAL_OPTIONAL_ARGS}
    CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${ZLIB_INSTALL_DIR}
                     -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
                     -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
)

IF(WIN32)
  SET(ZLIB_LIBRARIES "${ZLIB_INSTALL_DIR}/lib/zlibstatic.lib" CACHE FILEPATH "zlib library." FORCE)
ELSE(WIN32)
  SET(ZLIB_LIBRARIES "${ZLIB_INSTALL_DIR}/lib/libz.a" CACHE FILEPATH "zlib library." FORCE)
ENDIF(WIN32)

ADD_LIBRARY(zlib STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET zlib PROPERTY IMPORTED_LOCATION ${ZLIB_LIBRARIES})
ADD_DEPENDENCIES(zlib extern_zlib)
