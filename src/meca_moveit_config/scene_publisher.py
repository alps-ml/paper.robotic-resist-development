#!/usr/bin/env python3
import os, yaml, rclpy, trimesh
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Vector3, Point
from moveit_msgs.msg import CollisionObject, AttachedCollisionObject
from shape_msgs.msg import SolidPrimitive, Mesh, MeshTriangle
from moveit_msgs.srv import ApplyPlanningScene
from ament_index_python.packages import get_package_share_directory

class ScenePublisher(Node):
    def __init__(self):
        super().__init__('scene_publisher')
        pkg = get_package_share_directory('meca_moveit_config')
        env_yaml = os.path.join(pkg, 'config', 'environment.yaml')
        data = yaml.safe_load(open(env_yaml, 'r'))

        # wait for the planning scene service
        client = self.create_client(ApplyPlanningScene, 'apply_planning_scene')
        client.wait_for_service()

        collision_objects = []
        attached_objects = []
        
        for obj in data['objects']:
            co = CollisionObject()
            co.id = obj['name']
            co.header.frame_id = 'meca_base_link'

            # handle primitive shapes
            if obj['type'] == 'box' or obj['type'] == 'cylinder':
                prim = SolidPrimitive()
                if obj['type'] == 'box':
                    prim.type = SolidPrimitive.BOX
                    prim.dimensions = [float(dim) for dim in obj['size']]      # [x,y,z]
                    # For box, z-position should be half the height to place bottom at z=0
                    z_offset = prim.dimensions[2] / 2.0
                else:
                    prim.type = SolidPrimitive.CYLINDER
                    prim.dimensions = [float(dim) for dim in obj['size']]      # [height, radius]
                    # For cylinder, z-position should be half the height to place bottom at z=0
                    z_offset = prim.dimensions[0] / 2.0
                co.primitives = [prim]

            # handle mesh
            elif obj['type'] == 'mesh':
                mesh = Mesh()
                mesh_file = os.path.join(pkg, 'meshes', obj['filename'])
                
                # Load mesh with trimesh
                tm = trimesh.load(mesh_file, force='mesh')
                
                # Get the mesh bounds to calculate z-offset
                bounds = tm.bounds
                z_offset = (bounds[1][2] - bounds[0][2]) / 2.0  # Half of the mesh height
                
                # Convert to triangles and vertices
                # triangles
                for tri in tm.faces:
                    mesh.triangles.append(MeshTriangle(vertex_indices=tri.tolist()))
                # vertices
                for v in tm.vertices:
                    point = Point()
                    point.x = float(v[0])
                    point.y = float(v[1])
                    point.z = float(v[2])
                    mesh.vertices.append(point)
                
                co.meshes = [mesh]

            else:
                self.get_logger().warn(f"Unknown object type: {obj['type']}, skipping")
                continue

            # pose
            ps = PoseStamped()
            ps.header.frame_id = 'meca_base_link'
            px, py, pz = [float(p) for p in obj['pose']['position']]
            # Add the z-offset to place the bottom of the object at z=0
            pz += z_offset
            ox, oy, oz, ow = [float(o) for o in obj['pose']['orientation']]
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = px, py, pz
            ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = ox, oy, oz, ow

            # assign poses
            if obj['type'] == 'mesh':
                co.mesh_poses = [ps.pose]
            else:
                co.primitive_poses = [ps.pose]

            co.operation = CollisionObject.ADD
            
            # If this is the breadboard, create an attached collision object
            if obj['name'] == 'breadboard':
                attached_object = AttachedCollisionObject()
                attached_object.object = co
                attached_object.link_name = "meca_base_link"
                attached_object.touch_links = ["meca_base_link"]
                attached_objects.append(attached_object)
            else:
                collision_objects.append(co)

        # send them in one planning‐scene diff
        req = ApplyPlanningScene.Request()
        req.scene.world.collision_objects = collision_objects
        req.scene.robot_state.attached_collision_objects = attached_objects
        req.scene.is_diff = True
        client.call_async(req)
        self.get_logger().info(f"Successfully added {len(collision_objects)} objects and {len(attached_objects)} attached objects")
        rclpy.shutdown()

def main():
    rclpy.init()
    ScenePublisher()
    # note: no rclpy.spin() since we shutdown immediately

if __name__ == '__main__':
    main()
