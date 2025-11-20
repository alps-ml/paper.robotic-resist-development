# ROS2 Workspace for MECA500 Robot Control for Nanofabrication

This repository contains the necessary tools and code to control a MECA500 robot 
developed in the [LINQS](https://linqs.stanford.edu/) lab, currently for nanofabrication tasks.
The project uses ROS2 Jazzy and provides an environment for both simulation and robot control through MoveIt.

At the moment we implement a resist development task. It identifies chiplets inside a chip-box, picks them up to move through developer and runs a nitrogen gun at the end controlled using a Serial-port controlled valve.

The hardware components that need to be 3D printed, can be found in `docs/assets/models`.

## Bill of Materials

| Component              | Description                                      | Product Number |
|------------------------|--------------------------------------------------|----------------|
| MECA500 Robot Arm      | Primary robotic arm for nanofabrication tasks    | Meca500        |
| End-effector - Gripper   | Mounts for Techni Tool Tweezer tips. Uses Dowel pins for alignment. | MEGP-25 LS        |
| Realsense   | Depth camera for object detection. | Realsense D405        |
| Tweezer Tips   | Techni-Pro Tweezer Tips. [Techni-Pro 758TW0304](https://www.techni-tool.com/category/Hand-And-Power-Tools/Hand-Tools/Tweezers/Replacement-Tip-Tweezers/758TW0304-758TW0304) | Techni-Pro 758TW0305      |
| Chip-box               | Holds chiplets for pickup and placing | Entegris H44-999-1415       |
| Nitrogen Valve           | Serial-port valve-controlled drying gun. |  [Beduan Solenoid Valve](https://www.amazon.com/Beduan-Normally-Closed-Electric-Solenoid/dp/B07N2LGFYS)   |
| Serial Relay       | USB Serial Port Relay Module. | [JESSINIE Relay Module](https://www.amazon.com/JESSINIE-Lightweight-Intelligent-Overcurrent-Protection/dp/B0BXDNNJ8K)    |
| Dowel Pins            | Used for angular alignment of tweezer tips. (M1.5 x 6mm)  | [Amazon Link](https://www.amazon.com/dp/B07ZF69XM1) |

## Setup Instructions

> [!IMPORTANT]  
> Please do not expect this project to just run for you. You can see it as inspiration, help with implementing some of the tasks. This is not a final library you can just install and run. 

The project also contains a few unfinished leads we were following, like recovering some old MoveIt controller or including an RRTC motion planning algorithm in moveit_linqs.

### Prerequisites
- Ubuntu 24.04
- [just](https://just.systems/) - A command runner (similar to make but simpler)

## Using the Workspace

All commands should be run from the root of the workspace directory. There's no need to modify your bashrc as the justfile handles environment sourcing.

### Running the Demo
The demo is simplest run by starting three different components and running these commands:

```bash
just meca-driver
just bringup
just meca-demo-develop 
```

### Available Commands

#### ROS2 Basic Commands
- **Run RViz2**:
  ```bash
  just rviz2
  ```

- **Run RViz2 with Meca Robot model**:
  ```bash
  just rviz2-meca
  ```

#### MoveIt Commands
- **Launch MoveIt with Meca Robot model**:
  ```bash
  just moveit-meca
  ```

#### General ROS2 Commands
- **Run any ROS2 command**:
  ```bash
  just ros2 <command>
  ```
  For example:
  ```bash
  just ros2 node list
  ```

### Installation

1. **Install ROS2 Jazzy**:
   ```bash
   just install-ros2-jazzy
   ```
   This will install:
   - ROS2 Jazzy desktop
   - Required development tools
   - Dependencies needed for the project

2. **Build the workspace**:
   ```bash
   just build_ws
   ```

3. **Update dependencies** (if needed):
   ```bash
   just rosdep_update
   ```

## Project Structure

The workspace is organized as follows:
- meca_controller: Controller nodes for the MECA500 robot
- meca_moveit_config: MoveIt configuration for motion planning
- mecademic_description: Robot description files (URDF, visualization)

## Running on Other Systems

If you need to run this on a non-Ubuntu system, you can use containerization:

### Using Toolbox on Other Linux Distributions

1. Install [toolbox](https://containertoolbx.org/)
2. Create an Ubuntu 24.04 container:
   ```bash
   toolbox create --distro ubuntu --release 24.04 ubuntu2404-ROS
   ```
3. Enter the toolbox:
   ```bash
   toolbox enter ubuntu2404-ROS
   ```
4. Inside the toolbox, clone this repository and follow the installation steps above

## Building

To rebuild the workspace after changes:
```bash
just build_ws
```