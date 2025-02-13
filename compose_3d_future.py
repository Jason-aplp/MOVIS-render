import argparse
import math
import os
import random
import sys
import time
import urllib.request
from typing import Tuple
import json

import bpy
from mathutils import Vector
import numpy as np


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

def scene_meshes(lst):
    obj_lst = []
    for i in lst:
        obj_lst.append(bpy.data.objects[i])
    for obj in bpy.context.scene.objects.values():
        if (isinstance(obj.data, (bpy.types.Mesh))) and (obj in obj_lst):
            yield obj

def scene_bbox(obj_lst, single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes(obj_lst) if single_obj is None else [single_obj]:
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

def scene_root_objects(lst):
    obj_lst = []
    for i in lst:
        obj_lst.append(bpy.data.objects[i])
    for obj in bpy.context.scene.objects.values():
        if (not obj.parent) and (obj in obj_lst):
            yield obj

def return_root_objects(lst):
    obj_lst = []
    for i in lst:
        obj_lst.append(bpy.data.objects[i])
    for obj in bpy.context.scene.objects.values():
        if (not obj.parent) and (obj in obj_lst):
            return obj


def intersect(x, y, centers, b_box):
    num = len(b_box)
    xmin = x - b_box[num - 1][0] / 2.0
    xmax = x + b_box[num - 1][0] / 2.0
    ymin = y - b_box[num - 1][1] / 2.0
    ymax = y + b_box[num - 1][1] / 2.0
    tot_sum = 0
    for i in range(len(centers)):
        width = b_box[i][0] / 2.0
        height = b_box[i][1] / 2.0
        ref_xmin = centers[i][0] - width
        ref_xmax = centers[i][0] + width
        ref_ymin = centers[i][1] - height
        ref_ymax = centers[i][1] + height
        if (xmax < ref_xmin or xmin > ref_xmax or ymax < ref_ymin or ymin > ref_ymax):
            tot_sum += 1
        else:
            return True
    return False


def load_obj(obj_path):
    bpy.ops.import_scene.obj(filepath=obj_path)
    # join many same label objects e.g. bed
    bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
    bpy.ops.object.join()


def merge2(input_pth, output_pth):
    #import first glb and compute its bbox and normalize it 
    reset_scene()

    obj_num = len(input_pth)
    pth1 = input_pth[0]
    # bpy.ops.import_scene.obj(filepath=pth1)
    load_obj(pth1)

    if 'Camera' in bpy.data.objects.keys():
        bpy.data.objects.remove(bpy.data.objects['Camera'])
    if 'Cube' in bpy.data.objects.keys():
        bpy.data.objects.remove(bpy.data.objects['Cube'])
    if 'Light' in bpy.data.objects.keys():
        bpy.data.objects.remove(bpy.data.objects['Light'])

    #compute every object's components
    obj_list = []
    obj_lst_pre = []
    obj_lst_cur = bpy.data.objects.keys()
    obj_lst_tmp = []
    for i in obj_lst_cur:
        if (i not in obj_lst_pre):
            obj_lst_tmp.append(i)
    obj_list.append(obj_lst_tmp)
    obj_lst_pre = bpy.data.objects.keys()
    obj_lst_tmp = []
    for i in range(obj_num - 1):
        tmp_pth = input_pth[i + 1]
        # bpy.ops.import_scene.obj(filepath=tmp_pth)
        load_obj(tmp_pth)
        obj_lst_cur = bpy.data.objects.keys()
        for j in obj_lst_cur:
            if (j not in obj_lst_pre):
                obj_lst_tmp.append(j)
        obj_list.append(obj_lst_tmp)
        obj_lst_pre = bpy.data.objects.keys()
        obj_lst_tmp = []
    

    centers = []
    scales = []
    b_box = []
    for i in range(obj_num):
        bbox_min, bbox_max = scene_bbox(obj_list[i])
        # scale = 1 / max(bbox_max - bbox_min)
        scale = random.uniform(0.95, 1.05)
        # scales.append(tmp_scale)
        # scale = tmp_scale * scale
        for obj in scene_root_objects(obj_list[i]):
            obj.rotation_mode = 'XYZ'
            z_ran = random.uniform(0, math.pi)
            obj.rotation_euler[2] = z_ran
            obj.scale = obj.scale * scale
        bpy.context.view_layer.update()
        bbox_min, bbox_max = scene_bbox(obj_list[i])
        b_box.append((bbox_max[0] - bbox_min[0], bbox_max[1] - bbox_min[1]))
        offset = -(bbox_min + bbox_max) / 2
        offset[2] = -bbox_min[2]

        # print(bbox_min, bbox_max, offset)
        if (i == 0):
            x = random.uniform(-0.1, 0.1)
            y = random.uniform(-0.1, 0.1)
            centers.append((x, y))
            offset[0] += x
            offset[1] += y
        else:
            sum_x = 0
            sum_y = 0
            for j in range(len(centers)):
                sum_x += centers[j][0]
                sum_y += centers[j][1]
            sum_x = sum_x * 1.0 / len(centers)
            sum_y = sum_y * 1.0 / len(centers)
            dir_theta = random.uniform(0, math.pi*2)
            dx = math.cos(dir_theta)
            dy = math.sin(dir_theta)
            while True:
                #if exist intersection, return true
                if not (intersect(sum_x, sum_y, centers, b_box)):
                    break
                sum_x += dx * 0.05
                sum_y += dy * 0.05
            
            centers.append((sum_x, sum_y))
            offset[0] += sum_x
            offset[1] += sum_y
        
        for obj in scene_root_objects(obj_list[i]):
            obj.matrix_world.translation += offset
        bpy.context.view_layer.update()
        print(scene_bbox(obj_list[i]))

    empty = bpy.data.objects.new("Empty", None)
    node_name = 'Empty'
    for i in bpy.data.objects.keys():
        if i not in obj_lst_pre:
            node_name = i
    bpy.context.collection.objects.link(empty)

    for i in range(obj_num):
        son = return_root_objects(obj_list[i])
        parent = bpy.data.objects[node_name]
        son.parent = parent



    for i in bpy.data.objects.keys():
        if (i not in obj_lst_pre) and (i != node_name):
            son = bpy.data.objects[i]
            parent = bpy.data.objects[node_name]
            son.parent = parent
    bpy.ops.export_scene.gltf(filepath=output_pth)


def load_json(json_path):
    with open(json_path, 'r') as f:
        file_list = json.load(f)

    return file_list


parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    type=str,
    required=True,
    help="generate mode",
)
parser.add_argument(
    "--begin_num",
    type=int,
    required=True,
    help="generate begin num",
)
parser.add_argument(
    "--end_num",
    type=int,
    required=True,
    help="generate end num",
)
parser.add_argument(
    "--split_path",
    type=str,
    default='3D-split',
    help="path to obj files",
)
parser.add_argument(
    "--obj_root_path",
    type=str,
    default='3D-FUTURE',
    help="path to obj files",
)
parser.add_argument(
    "--save_root",
    type=str,
    default='3d_front_compose',
    help="path to save directory",
)
script_args = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(script_args)

# parse parameters
mode = args.mode
begin_num = args.begin_num
end_num = args.end_num
split_root_path = os.path.join(args.split_path, mode)
obj_root_path = args.obj_root_path
save_root_path = os.path.join(args.save_root, mode)
os.makedirs(save_root_path, exist_ok=True)

# get category prob
category_prob_dict = {
    'table': 0.5,
    'sofa': 0.5,
    'cabinet': 0.5,
    'cabinet1': 0.5,
    'night_stand': 0.3,
    'chair': 0.5,
    'chair1': 0.5,
    'bookshelf': 0.1,
    'bed': 0.4
}
norm_category_prob_dict = {}
sum_prob = 0
for i in category_prob_dict.keys():
    sum_prob += category_prob_dict[i]
for i in category_prob_dict.keys():
    norm_category_prob_dict[i] = category_prob_dict[i] / sum_prob
# print(norm_category_prob_dict)

# load all category list
all_object_dict = {}
for category_name in list(category_prob_dict.keys()):
    category_json_path = os.path.join(split_root_path, category_name + '.json')
    object_list = load_json(category_json_path)
    all_object_dict[category_name] = object_list

# generate scene
t1 = time.time()
print('Begin generate scene ...')
# split compose files every 5000 scenes
split_obj_num = 5000
for scene_idx in range(begin_num, end_num):

    dir_num = scene_idx // split_obj_num
    save_scene_root_path = os.path.join(save_root_path, f'{dir_num*split_obj_num:06d}_{(dir_num+1)*split_obj_num-1:06d}')
    os.makedirs(save_scene_root_path, exist_ok=True)

    # get category
    select_num = np.random.randint(3, 6)
    category_list = list(norm_category_prob_dict.keys())
    prob_list = list(norm_category_prob_dict.values())
    selected_category = np.random.choice(category_list, size=select_num, replace=False, p=prob_list)
    print(f'scene_idx: {scene_idx}, selected_category: {selected_category}')

    # get object
    scene_objects_path_list = []
    for category_name in selected_category:
        obj_idx = np.random.randint(len(all_object_dict[category_name]))
        obj_name = all_object_dict[category_name][obj_idx]
        obj_path = os.path.join(obj_root_path, obj_name, 'raw_model.obj')
        scene_objects_path_list.append(obj_path)

    output_pth = os.path.join(save_scene_root_path, f'{scene_idx}.glb')
    merge2(scene_objects_path_list, output_pth)

    print(f'{scene_idx} generate over')

t2 = time.time()
print(f'{end_num - begin_num + 1} done in {t2 - t1}s')
