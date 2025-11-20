set shell := ["bash", "-c"]

export ROS2_INSTALL := "/opt/ros/jazzy/setup.bash"
export ROS2_WS      := env("ROS2_WS", justfile_directory())

source_ROS2    := "source ${ROS2_INSTALL}"
source_pyvenv  := "source ${ROS2_WS}/venv/bin/activate"
source_ROS2_WS := "source ${ROS2_WS}/install/setup.bash"

export RFS_CONFIG_CAMERA_CALIBRATION := "${ROS2_WS}/src/meca_perception/camera_settings_calibration_20250604.json"

source_all := source_ROS2 + " && " + source_pyvenv + " && " + source_ROS2_WS

#export RMW_IMPLEMENTATION := "rmw_cyclonedds_cpp"

# ROS environment for Robots for science (LINQS)
default:
  just --list

# Run any command with ROS workspace and Python environment sourced.
exec *ARG:
  {{ source_all }} && \
  {{ARG}}

# Run the MECA500 development demo.
[group('meca')]
meca-demo-develop:
  {{ source_all }} && \
  ros2 run meca_controller meca_demo_develop

[group('meca')]
launch-control:
  gnome-terminal --title="MECA Driver" -- bash -c "cd {{ ROS2_WS }} && just meca-driver; exec bash" & \
  sleep 1 && \
  gnome-terminal --title="Bringup All" -- bash -c "cd {{ ROS2_WS }} && just bringup-all; exec bash"

# Launch MECA500 driver for ROS2
[group('meca')]
meca-driver:
  {{ source_all }} && \
  ros2 run meca_controller meca_driver --ros-args -r __ns:=/robot1

# Bringup all nodes for development demo.
[group('meca')]
bringup:
  {{ source_all }} && \
  {{ justfile_directory() }}/scripts/bringup_all.sh

# Bringup all nodes for development demo, with RViz2 started.
[group('meca')]
bringup-rviz:
  {{ source_all }} && \
  {{ justfile_directory() }}/scripts/bringup_all.sh --rviz

# Kill all nodes started by bringup_all.sh.
[group('meca')]
bringdown:
  {{ source_all }} && \
  {{ justfile_directory() }}/scripts/kill_bringup_nodes.sh

  # Launch RViz2
[group('ros2')]
rviz2 *ARG:
  {{ source_ROS2 }} && \
  rviz2 {{ ARG }}

# Launch RViz2 with Meca Robot model
[group('ros2')]
rviz2-meca:
  {{ source_all }} && \
  ros2 launch mecademic_description meca.launch.py

# Launch MoveIt with Meca Robot model
[group('moveit')]
moveit-meca:
  {{ source_all }} && \
  ros2 launch meca_moveit_config moveit_rviz.launch.py

# Run ROS 2 commands
[group('ros2')]
ros2 *ARG:
  {{ source_all }} && \
  ros2 {{ ARG }}

# Build the ROS2 workspace
[group('build')]
build:
  {{ source_ROS2 }} && {{ source_pyvenv }} && \
  python -m colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

# Clean build using colcon. Choose packages (default) or workspace
[group('build')]
clean ARG='packages': 
  {{ source_ROS2 }} && {{ source_pyvenv }} && \
  python -m colcon clean {{ ARG }}

[group('build')]
colcon *ARG: 
  {{ source_ROS2 }} && {{ source_pyvenv }} && \
  python -m colcon {{ ARG }}

# Initialize the ROS Python venv, reusing python packages installed in the system.
[group('build')]
[confirm]
init_venv:
  {{ source_ROS2 }} && python -m venv {{ ROS2_WS + "/venv" }} --system-site-packages --clear && \
  {{ source_pyvenv }} && pip install -r requirements.txt && \
  touch {{ROS2_WS + "/venv/COLCON_IGNORE"}}
# See https://github.com/ros2/ros2/issues/1094

# Update ROS2 dependencies
[group('build')]
rosdep_update:
  {{ source_ROS2 }} && {{ source_pyvenv }} && \
  sudo apt update && \
  rosdep update && \
  rosdep install -r --from-paths src --ignore-src --rosdistro $ROS_DISTRO -y

# Installing ROS2 Jazzy on Ubuntu 24.04 only.
[group('build')]
[confirm]
install-ros2-jazzy:
  # Following the instructions from https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html
  sudo apt update
  sudo locale-gen en_US.UTF-8
  sudo apt install software-properties-common -y
  sudo add-apt-repository universe -y
  sudo apt update -y && sudo apt install curl -y
  sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
  export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}') && \
  curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo $VERSION_CODENAME)_all.deb"
  sudo dpkg -i /tmp/ros2-apt-source.deb
  #echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
  sudo apt update -y
  sudo apt install ros-jazzy-desktop ros-dev-tools -y
  sudo apt install python3-rosdep python3-vcstool python3-colcon-common-extensions python3-colcon-mixin python3-colcon-clean -y
  sudo apt install python3-venv python3-pip -y
  sudo rosdep init
  rosdep update

[group('build')]
[confirm]
uninstall-ros2-jazzy:
  sudo apt remove ros2-apt-source -y
  sudo apt update -y
  sudo apt autoremove -y
  sudo apt upgrade -y

[group('build')]
[confirm]
install-moveit:
  sudo apt install ros-jazzy-moveit -y

# Installing MoveIt for ROS2 Jazzy from Source on Ubuntu 24.04
[group('build')]
[confirm]
install-moveit-source:
  sudo apt install -y \
  build-essential \
  cmake \
  git \
  python3-colcon-common-extensions \
  python3-flake8 \
  python3-rosdep \
  python3-setuptools \
  python3-vcstool \
  wget
  sudo apt update
  sudo apt dist-upgrade
  rosdep update
  {{ source_ROS2 }} && {{ source_pyvenv }} && \
  git -C "src/moveit2" pull || git clone https://github.com/moveit/moveit2.git -b $ROS_DISTRO "src/moveit2" && \
  cd ./src && for repo in moveit2/moveit2.repos $(f="moveit2/moveit2_$ROS_DISTRO.repos"; test -r $f && echo $f); do vcs import < "$repo"; done && cd .. && \
  rosdep install -r --from-paths src --ignore-src --rosdistro $ROS_DISTRO -y && \
  sudo apt install ros-$ROS_DISTRO-rmw-cyclonedds-cpp

[group('build')]
[confirm]
clean-moveit-source:
  rm -R src/moveit2 src/moveit_msgs src/moveit_resources
