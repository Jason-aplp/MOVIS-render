"""Blender script to render images of 3D models.

This script is used to render images of 3D models. It takes in a list of paths
to .glb files and renders images of each model. The images are from rotating the
object around the origin. The images are saved to the output directory.

Example usage:
    blender -b -P blender_script.py -- \
        --object_path my_object.glb \
        --output_dir ./views \
        --engine CYCLES \
        --scale 0.8 \
        --num_images 12 \
        --camera_dist 1.2

Here, input_model_paths.json is a json file containing a list of paths to .glb.
"""

import argparse
import json
import math
import os
import random
import sys
import time
import urllib.request
import uuid
from typing import Tuple
from mathutils import Vector, Matrix
import numpy as np

import bpy
from mathutils import Vector

parser = argparse.ArgumentParser()
parser.add_argument(
    "--object_path",
    type=str,
    required=True,
    help="Path to the object file",
)
parser.add_argument("--output_dir", type=str, default="c3dfs")
parser.add_argument(
    "--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"]
)
parser.add_argument("--scale", type=float, default=0.8)
parser.add_argument("--num_images", type=int, default=12)
parser.add_argument("--camera_dist", type=int, default=1.2)
    
argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)

print('===================', args.engine, '===================')

context = bpy.context
scene = context.scene
render = scene.render

cam = scene.objects["Camera"]
cam.location = (0, 1.2, 0)
cam.data.lens = 35
cam.data.sensor_width = 32

cam_constraint = cam.constraints.new(type="TRACK_TO")
cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
cam_constraint.up_axis = "UP_Y"

# setup lighting
bpy.ops.object.light_add(type="AREA")
light2 = bpy.data.lights["Area"]
light2.energy = 3000
bpy.data.objects["Area"].location[2] = 0.5
bpy.data.objects["Area"].scale[0] = 100
bpy.data.objects["Area"].scale[1] = 100
bpy.data.objects["Area"].scale[2] = 100

bpy.ops.object.light_add(type="AREA")
light3 = bpy.data.lights["Area.001"]
light3.energy = 3000
bpy.data.objects["Area.001"].location[2] = 0.5
bpy.data.objects["Area.001"].scale[0] = 100
bpy.data.objects["Area.001"].scale[1] = 100
bpy.data.objects["Area.001"].scale[2] = 100

bpy.ops.object.light_add(type="AREA")
light4 = bpy.data.lights["Area.002"]
light4.energy = 3000
bpy.data.objects["Area.002"].location[2] = 0.5
bpy.data.objects["Area.002"].scale[0] = 100
bpy.data.objects["Area.002"].scale[1] = 100
bpy.data.objects["Area.002"].scale[2] = 100

bpy.ops.object.light_add(type="AREA")
light5 = bpy.data.lights["Area.003"]
light5.energy = 3000
bpy.data.objects["Area.003"].location[2] = 0.5
bpy.data.objects["Area.003"].scale[0] = 100
bpy.data.objects["Area.003"].scale[1] = 100
bpy.data.objects["Area.003"].scale[2] = 100



render.engine = args.engine
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
render.resolution_x = 512
render.resolution_y = 512
render.resolution_percentage = 100

scene.cycles.device = "GPU"
scene.cycles.samples = 128
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transparent_max_bounces = 3
scene.cycles.transmission_bounces = 3
scene.cycles.filter_width = 0.01
scene.cycles.use_denoising = True
scene.render.film_transparent = True

bpy.context.preferences.addons["cycles"].preferences.get_devices()
# Set the device_type
bpy.context.preferences.addons[
    "cycles"
].preferences.compute_device_type = "CUDA" # or "OPENCL"

def sample_point_on_sphere(radius: float) -> Tuple[float, float, float]:
    theta = random.random() * 2 * math.pi
    phi = math.acos(2 * random.random() - 1)
    return (
        radius * math.sin(phi) * math.cos(theta),
        radius * math.sin(phi) * math.sin(theta),
        radius * math.cos(phi),
    )

def sample_spherical(radius=3.0, maxz=3.0, minz=0.):
    correct = False
    while not correct:
        vec = np.random.uniform(-1, 1, 3)
        vec[2] = np.abs(vec[2])
        vec = vec / np.linalg.norm(vec, axis=0) * radius
        if maxz > vec[2] > minz:
            correct = True
    return vec

def sample_spherical(radius_min=1.5, radius_max=2.0, maxz=1.6, minz=-0.75):
    correct = False
    while not correct:
        vec = np.random.uniform(-1, 1, 3)
#         vec[2] = np.abs(vec[2])
        radius = np.random.uniform(radius_min, radius_max, 1)
        vec = vec / np.linalg.norm(vec, axis=0) * radius[0]
        if maxz > vec[2] > minz:
            correct = True
    return vec

def randomize_camera(x, y, z, flg):
    
    azimuth = random.uniform(0., 360)
    if flg == 0:
        distance = random.uniform(1.3, 1.7)
    else:
        distance = random.uniform(1.8, 2.2)
    if flg == 0:
        elevation = random.uniform(2., 40.)
    else:
        elevation = random.uniform(10., 60.)
    return set_camera_location(elevation, azimuth, distance, x, y, z)

def randomize_lookat():
    elevation = random.uniform(0., 90.)
    azimuth = random.uniform(0., 360)
    distance = random.uniform(0.01, 0.2)
    return elevation, azimuth, distance

def set_camera_location(elevation, azimuth, distance, tx, ty, tz):
    # from https://blender.stackexchange.com/questions/18530/
    # x, y, z = sample_spherical(radius_min=1.2, radius_max=2.0, maxz=2.0, minz=0.0)
    x = distance * math.cos(math.radians(elevation)) * math.cos(math.radians(azimuth))
    y = distance * math.cos(math.radians(elevation)) * math.sin(math.radians(azimuth))
    z = distance * math.sin(math.radians(elevation))
    camera = bpy.data.objects["Camera"]
    camera.location = x, y, z

    print(camera.location)
    direction = - camera.location
    direction += Vector((tx, ty, tz))
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()
    return camera

def randomize_lighting() -> None:
    light2.energy = random.uniform(300, 600)
    bpy.data.objects["Area"].location[0] = random.uniform(-1., 1.)
    bpy.data.objects["Area"].location[1] = random.uniform(-1., 1.)
    bpy.data.objects["Area"].location[2] = random.uniform(0.5, 1.5)


def reset_lighting() -> None:
    light2.energy = 1000
    bpy.data.objects["Area"].location[0] = 0
    bpy.data.objects["Area"].location[1] = 0.6
    bpy.data.objects["Area"].location[2] = 0.6

    light3.energy = 1000
    bpy.data.objects["Area.001"].location[0] = 0
    bpy.data.objects["Area.001"].location[1] = -0.6
    bpy.data.objects["Area.001"].location[2] = 0.6

    light4.energy = 1000
    bpy.data.objects["Area.002"].location[0] = 0.6
    bpy.data.objects["Area.002"].location[1] = 0
    bpy.data.objects["Area.002"].location[2] = 0.6

    light5.energy = 1000
    bpy.data.objects["Area.003"].location[0] = -0.6
    bpy.data.objects["Area.003"].location[1] = 0
    bpy.data.objects["Area.003"].location[2] = 0.6

def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


# load the glb model
def load_object(object_path: str) -> None:
    """Loads a glb model into the scene."""
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")


def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj

# function from https://github.com/panmari/stanford-shapenet-renderer/blob/master/render_blender.py
#camera 2 world
#up vector is (0, 0, 1)
def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    # R_bcam2cv = Matrix(
    #     ((1, 0,  0),
    #     (0, 1, 0),
    #     (0, 0, 1)))

    # Transpose since the rotation is object rotation, 
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam @ location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam @ cam.location
    # Use location from matrix_world to account for constraints:     
    T_world2bcam = -1*R_world2bcam @ location

    # # Build the coordinate transform matrix from world to computer vision camera
    # R_world2cv = R_bcam2cv@R_world2bcam
    # T_world2cv = R_bcam2cv@T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2bcam[0][:] + (T_world2bcam[0],),
        R_world2bcam[1][:] + (T_world2bcam[1],),
        R_world2bcam[2][:] + (T_world2bcam[2],)
        ))
    return RT
#return True means z is the biggest
def normalize_scene():
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    print('scale:', scale)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    offset[2] = -bbox_min[2]
    print('offset:', offset)
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")


def render_mask(root_pth, cur_num):
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    tot_objects = [obj for obj in bpy.context.scene.objects]

    root_objects = []
    for i in tot_objects:
        if (i.parent is not None) and (i.parent.parent is None):
            root_objects.append(i)
    root_object_dict = {}
    for i in range(len(root_objects)):
        root_object_dict[root_objects[i]] = i + 1
    for i, obj in enumerate(mesh_objects):
        parent = obj
        while True:
            if parent in root_objects:
                obj.pass_index = root_object_dict[parent]
                break
            parent = parent.parent
    view_layer = bpy.context.scene.view_layers[0]

    view_layer.use_pass_object_index = True
    view_layer.use_pass_z = True
    view_layer.use_pass_normal = True
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree

    for node in tree.nodes:
        tree.nodes.remove(node)

    render_layers_node = tree.nodes.new(type='CompositorNodeRLayers')
    output_file_depth = tree.nodes.new('CompositorNodeOutputFile')
    output_file_node_masks = tree.nodes.new('CompositorNodeOutputFile')
    output_file_node_image = tree.nodes.new('CompositorNodeOutputFile')
    composite_node = tree.nodes.new(type='CompositorNodeComposite')

    output_file_node_masks.base_path = os.path.join(root_pth, f"mask_{cur_num:03d}") 
    output_file_node_image.base_path = root_pth 
    output_file_node_image.file_slots[0].path = f"{cur_num:03d}_#"
    output_file_depth.base_path = root_pth
    output_file_depth.format.file_format = "OPEN_EXR"
    output_file_depth.format.color_depth = '32' 
    output_file_depth.file_slots[0].path = f"depth_{cur_num:03d}_#"
    tree.links.new(render_layers_node.outputs['Depth'], output_file_depth.inputs[0])

    tree.links.new(render_layers_node.outputs['Image'], composite_node.inputs['Image'])
    tree.links.new(render_layers_node.outputs['Image'], output_file_node_image.inputs[0])  


    while len(output_file_node_masks.file_slots) < len(root_objects):
        output_file_node_masks.file_slots.new("")


    for idx in range(len(root_objects)):
        id_mask_node = tree.nodes.new(type='CompositorNodeIDMask')
        id_mask_node.index = idx + 1
        tree.links.new(render_layers_node.outputs['IndexOB'], id_mask_node.inputs['ID value'])
        tree.links.new(id_mask_node.outputs['Alpha'], output_file_node_masks.inputs[idx])
        output_file_node_masks.file_slots[idx].path = f"mask_{cur_num:03d}_{0}_{idx + 1}_#"

def render_instance(root_pth, cur_num):
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    tot_objects = [obj for obj in bpy.context.scene.objects]

    root_objects = []
    for i in tot_objects:
        if (i.parent is not None) and (i.parent.parent is None):
            root_objects.append(i)
    root_object_dict = {}
    for i in range(len(root_objects)):
        root_object_dict[root_objects[i]] = i + 1
    for i, obj in enumerate(mesh_objects):
        parent = obj
        while True:
            if parent in root_objects:
                obj.pass_index = root_object_dict[parent]
                break
            parent = parent.parent
    view_layer = bpy.context.scene.view_layers[0]
    view_layer.use_pass_object_index = True
    view_layer.use_pass_z = True
    view_layer.use_pass_normal = True
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree

    for node in tree.nodes:
        tree.nodes.remove(node)

    # create nodes
    render_layers_node = tree.nodes.new(type='CompositorNodeRLayers')
    output_file_node_masks = tree.nodes.new('CompositorNodeOutputFile')
    composite_node = tree.nodes.new(type='CompositorNodeComposite')

    output_file_node_masks.base_path = os.path.join(root_pth, f"mask_{cur_num:03d}")


    while len(output_file_node_masks.file_slots) < 1:
        output_file_node_masks.file_slots.new("")

    for obj in range(len(root_objects)):
        for i, obj2 in enumerate(mesh_objects):
            parent = obj2
            while True:
                if parent in root_objects:
                    break
                parent = parent.parent
            if (parent != root_objects[obj]):
                obj2.hide_render = True

        # for idx in range(len(root_objects)):
        id_mask_node = tree.nodes.new(type='CompositorNodeIDMask')
        id_mask_node.index = obj + 1
        tree.links.new(render_layers_node.outputs['IndexOB'], id_mask_node.inputs['ID value'])
        tree.links.new(id_mask_node.outputs['Alpha'], output_file_node_masks.inputs[0])
        output_file_node_masks.file_slots[0].path = f"mask_{cur_num:03d}_{1}_{obj + 1}_#"

        bpy.ops.render.render(write_still=True)
        for i, obj2 in enumerate(mesh_objects):
            obj2.hide_render = False


def save_images(object_file: str, num: str) -> None:
    """Saves rendered images of the object in the scene."""
    tmp_pth = os.path.join(args.output_dir, num)
    os.makedirs(tmp_pth, exist_ok=True)

    reset_scene()

    # load the object
    load_object(object_file)
    object_uid = os.path.basename(object_file).split(".")[0]
    normalize_scene()
    # breakpoint()

    # create an empty object to track
    empty = bpy.data.objects.new("Empty", None)
    empty.location = Vector((0.0, 0.0, 0.0))
    scene.collection.objects.link(empty)
    cam_constraint.target = empty

    # randomize_lighting()
    reset_lighting()
    bbox_min, bbox_max = scene_bbox()
    flg = 0
    if (bbox_max[2] - bbox_min[2] > 0.8):
        flg = 1
    for i in range(args.num_images):
        # # set the camera position
        elevation_lookat, azimuth_lookat, distance_lookat = randomize_lookat()
        tmp_x = distance_lookat * math.cos(math.radians(elevation_lookat)) * math.cos(math.radians(azimuth_lookat))
        tmp_y = distance_lookat * math.cos(math.radians(elevation_lookat)) * math.sin(math.radians(azimuth_lookat))
        tmp_z = distance_lookat * math.sin(math.radians(elevation_lookat))
        empty.location = Vector((tmp_x, tmp_y, tmp_z))
        # empty.location = Vector((0.0124, -0.0098, 0.0133))
        print(empty.location)
        cam_constraint.target = empty
        camera = randomize_camera(tmp_x, tmp_y, tmp_z, flg)

        # breakpoint()
        # render the image
        render_path = os.path.join(tmp_pth, object_uid, f"{i:03d}.png")
        render_mask(root_pth=os.path.join(tmp_pth, object_uid), cur_num=i)
        bpy.ops.render.render(write_still=True)

        render_instance(root_pth=os.path.join(tmp_pth, object_uid), cur_num=i)

        # save camera RT matrix
        RT = get_3x4_RT_matrix_from_blender(camera)
        RT_path = os.path.join(tmp_pth, object_uid, f"{i:03d}_1.npy")
        np.save(RT_path, RT)


def download_object(object_url: str) -> str:
    """Download the object and return the path."""
    # uid = uuid.uuid4()
    uid = object_url.split("/")[-1].split(".")[0]
    tmp_local_path = os.path.join("tmp-objects", f"{uid}.glb" + ".tmp")
    local_path = os.path.join("tmp-objects", f"{uid}.glb")
    # wget the file and put it in local_path
    os.makedirs(os.path.dirname(tmp_local_path), exist_ok=True)
    urllib.request.urlretrieve(object_url, tmp_local_path)
    os.rename(tmp_local_path, local_path)
    # get the absolute path
    local_path = os.path.abspath(local_path)
    return local_path


if __name__ == "__main__":
    try:
        start_i = time.time()
        if args.object_path.startswith("http"):
            local_path = download_object(args.object_path)
        else:
            local_path = args.object_path
        num = args.object_path.split('/')[-2]
        save_images(local_path, num)
        end_i = time.time()
        print("Finished", local_path, "in", end_i - start_i, "seconds")
        # delete the object if it was downloaded
        if args.object_path.startswith("http"):
            os.remove(local_path)
    except Exception as e:
        print("Failed to render", args.object_path)
        print(e)