'''
import trimesh

# parameters
outer_radius = 0.1
inner_radius = 0.08
height       = 0.2
sections     = 64

# make solid and “drill” out the center
outer = trimesh.creation.cylinder(radius=outer_radius, 
                                   height=height,
                                   sections=sections)
inner = trimesh.creation.cylinder(radius=inner_radius, 
                                   height=height + 0.001,  # slightly taller for clean boolean
                                   sections=sections)

hollow = outer.difference(inner)

# save the result
hollow.export('/home/linqs/ros_ws/src/meca_moveit_config/meshes/beaker_test.stl')
'''

#!/usr/bin/env python3
import trimesh
import numpy as np
'''
# ── Parameters Nalgene beaker ──────────────────────────────────────────────────────────────
height     = 0.075    # total height in m
r_base     = 0.052/2    # inner radius at bottom
r_top      = 0.0575/2    # inner radius at top
wall_thickness = 0.002
sections   = 64      # resolution around
# ────────────────────────────────────────────────────────────────────────────
'''

# ── Parameters HDPE cups ──────────────────────────────────────────────────────────────
height     = 0.0563    # total height in m
r_base     = 0.0766/2    # inner radius at bottom
r_top      = 0.0939/2    # inner radius at top
wall_thickness = 0.0029
sections   = 64      # resolution around
# ────────────────────────────────────────────────────────────────────────────


# Create a straight cylinder of radius=r_base, height centered on z=0
mesh = trimesh.creation.cylinder(radius=r_top+wall_thickness,
                                 height=height,
                                 sections=sections)

# Stretch its XY coordinates linearly from r_base at z=-h/2 to r_top at z=+h/2
z = mesh.vertices[:, 2]
scale = np.interp(z, [-height/2, height/2], [(r_base+wall_thickness)/(r_top+wall_thickness), 1.0])
mesh.vertices[:, 0] *= scale
mesh.vertices[:, 1] *= scale

# shrink a copy for the inner surface
inner = mesh.copy()
inner_radius_factor = r_top / (r_top+wall_thickness)
inner.vertices[:, :2] *= inner_radius_factor

# Boolean difference: outer minus inner
hollow = mesh.difference(inner)

# Export both STL and DAE formats
hollow.export('/home/linqs/ros_ws/src/meca_moveit_config/meshes/cup_hollow_tapered.stl')
hollow.export('/home/linqs/ros_ws/src/meca_moveit_config/meshes/cup_hollow_tapered.dae')

print("\nSaved hollow tapered cup to:")
print("  - cup_hollow_tapered.stl")
print("  - cup_hollow_tapered.dae")


