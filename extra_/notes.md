cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_CUDA_MODULE=ON \
  -DBUILD_GUI=OFF \
  -DOpenCV_DIR=/home/pomo/DriveGXO/opencv/build \
  -DPYTHON_EXECUTABLE=/home/pomo/DriveGXO/venv/bin/python3 \
  ..
sudo apt install -y build-essential cmake
sudo apt install -y libeigen3-dev libgl1-mesa-dev libglu1-mesa-dev
sudo apt install -y libjsoncpp-dev liblzf-dev libpng-dev
sudo apt install -y libglfw3-dev libglew-dev
sudo apt install -y git liblapack-dev liblapacke-dev
sudo apt install -y libicu-dev libblas-dev libfmt-dev

git clone https://github.com/isl-org/Open3D.git

~/DriveGXO/venv/bin/cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_CUDA_MODULE=ON \
  -DBUILD_GUI=ON \
  -DOpenCV_DIR=~/DriveGXO/opencv/build \
  -DPYTHON_EXECUTABLE=~/DriveGXO/venv/bin/python3 \
  ..

 
ln -s  /home/pomo/.local/lib/python3.8/site-packages/torch/  ~/DriveGXO/venv/lib/python3.8/site-packages/torch
ln -s <path_to_torchvision> ~/DriveGXO/venv/lib/python3.8/site-packages/torchvision



wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null

sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main'

git show v0.13.0:CMakeLists.txt | grep cmake_minimum_required


wget https://github.com/Kitware/CMake/releases/download/v3.19.2/cmake-3.19.2-Linux-aarch64.sh

wget https://github.com/Kitware/CMake/releases/download/v3.29.2/cmake-3.29.2.tar.gz
./bootstrap --prefix=/usr/local
make -j$(nproc)
sudo make install




python3 - <<EOF
import open3d as o3d
print("CUDA available:", o3d.core.cuda.is_available())
EOF

python3 - <<EOF
import open3d as o3d
print(f"Open3D version: {o3d.__version__}")
print("CUDA:", o3d.core.cuda.is_available())
EOF

rostopic echo /livox/lidar --noarr


git clone https://github.com/Livox-SDK/livox_ros_driver2.git ws_livox/src/livox_ros_driver2

python3 - <<'PY'
import logging, rospy
print("root logger level after importing rospy =", logging.getLogger().level)
PY
