#!/bin/bash
set -e

echo "Starting GTSAM Installation..."

# Step 1: Install Boost
sudo apt-get update
sudo apt-get install -y libboost-all-dev

# Step 2: Download GTSAM
cd /tmp
wget -qO gtsam.zip https://github.com/borglab/gtsam/archive/4.1.1.zip
unzip -qo gtsam.zip
cd gtsam-4.1.1

# Step 3: Build GTSAM
mkdir -p build && cd build
cmake -DGTSAM_BUILD_WITH_MARCH_NATIVE=ON -DGTSAM_USE_SYSTEM_EIGEN=ON ..
make -j$(nproc)
sudo make install

echo "GTSAM successfully installed!"
