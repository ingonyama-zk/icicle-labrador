#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Function to display usage information
show_help() {
  echo "Usage: $0 [-d DEVICE_TYPE] [-b BACKEND_INSTALL_DIR]"
  echo
  echo "Options:"
  echo "  -d DEVICE_TYPE            Specify the device type (default: CPU)"
  echo "  -b BACKEND_INSTALL_DIR    Specify the backend installation directory (default: empty)"
  echo "  -h                        Show this help message"
  exit 0
}

# Parse command line options
while getopts ":d:b:h" opt; do
  case ${opt} in
    d )
      DEVICE_TYPE=$OPTARG
      ;;
    b )
      ICICLE_BACKEND_INSTALL_DIR="$(realpath ${OPTARG})"
      ;;
    h )
      show_help
      ;;
    \? )
      echo "Invalid option: -$OPTARG" 1>&2
      show_help
      ;;
    : )
      echo "Invalid option: -$OPTARG requires an argument" 1>&2
      show_help
      ;;
  esac
done

# Set default values if not provided
: "${DEVICE_TYPE:=CPU}"
: "${ICICLE_BACKEND_INSTALL_DIR:=}"

DEVICE_TYPE_LOWERCASE=$(echo "$DEVICE_TYPE" | tr '[:upper:]' '[:lower:]')

# Create necessary directories
mkdir -p build/src
mkdir -p build/icicle

# Paths from root directory
ICILE_DIR=$(realpath "icicle/")
ICICLE_BACKEND_SOURCE_DIR="${ICILE_DIR}/backend/${DEVICE_TYPE_LOWERCASE}"

# Download and extract Icicle release for non-CPU devices if backend not specified
if [ "$DEVICE_TYPE" != "CPU" ] && [ -z "${ICICLE_BACKEND_INSTALL_DIR}" ]; then
  echo "Downloading Icicle release for ${DEVICE_TYPE}"
  mkdir -p build/icicle_release
  if [ -z "$(ls -A build/icicle_release)" ]; then
    wget -q https://github.com/ingonyama-zk/icicle/releases/download/v4.0.0/icicle_4_0_0-ubuntu22-cuda122.tar.gz -O build/icicle_release/icicle.tar.gz
    tar -xzf build/icicle_release/icicle.tar.gz -C build/icicle_release
  else
    echo "Icicle release already present in build/icicle_release, skipping download."
  fi
  export ICICLE_BACKEND_INSTALL_DIR=$(realpath "build/icicle_release")
  echo "Using downloaded Icicle release at ${ICICLE_BACKEND_INSTALL_DIR}"
  
  # Build only the src project when using pre-built release
  cmake -DCMAKE_BUILD_TYPE=Release -S "${ICILE_DIR}" -B build/icicle -DRING=babykoala
  cmake --build build/icicle -j
  cmake -DCMAKE_BUILD_TYPE=Release -S src -B build/src
  cmake --build build/src -j
# Build Icicle backend locally if source exists and no backend install dir specified
elif [ "$DEVICE_TYPE" != "CPU" ] && [ ! -d "${ICICLE_BACKEND_INSTALL_DIR}" ] && [ -d "${ICICLE_BACKEND_SOURCE_DIR}" ]; then
  echo "Building icicle and ${DEVICE_TYPE} backend"
  cmake -DCMAKE_BUILD_TYPE=Release -DRING=babykoala "-D${DEVICE_TYPE}_BACKEND"=local -S "${ICILE_DIR}" -B build/icicle
  export ICICLE_BACKEND_INSTALL_DIR=$(realpath "build/icicle/backend")
  
  # Build both icicle and src
  cmake --build build/icicle -j
  cmake -DCMAKE_BUILD_TYPE=Release -S src -B build/src
  cmake --build build/src -j
else
  echo "Building icicle without backend, ICICLE_BACKEND_INSTALL_DIR=${ICICLE_BACKEND_INSTALL_DIR}"
  export ICICLE_BACKEND_INSTALL_DIR="${ICICLE_BACKEND_INSTALL_DIR}"
  cmake -DCMAKE_BUILD_TYPE=Release -DRING=babykoala -S "${ICILE_DIR}" -B build/icicle
  
  # Build both icicle and src
  cmake --build build/icicle -j
  cmake -DCMAKE_BUILD_TYPE=Release -S src -B build/src
  cmake --build build/src -j
fi

./build/src/example "$DEVICE_TYPE"
# compute-sanitizer --tool memcheck --leak-check full ./build/src/example "$DEVICE_TYPE" > sanitizer_output.log 2>&1