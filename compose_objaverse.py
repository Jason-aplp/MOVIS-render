import argparse
import math
import os
import random
import sys
import time
import urllib.request
from typing import Tuple
import json
import numpy as np

import bpy
from mathutils import Vector


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
        if (xmax < ref_xmin  - 0.02 or xmin > ref_xmax + 0.02 or ymax < ref_ymin - 0.02 or ymin > ref_ymax + 0.02):
            tot_sum += 1
        else:
            return True
    return False

def load_obj(obj_path):
    ori_mesh_objs = [m for m in bpy.context.scene.objects if m.type == 'MESH']
    bpy.ops.import_scene.gltf(filepath=obj_path)
    cur_mesh_objs = [m for m in bpy.context.scene.objects if m.type == 'MESH']
    tmp_mesh_objs = [m for m in cur_mesh_objs if m not in ori_mesh_objs]
    # join many same label objects e.g. bed
    for i in bpy.context.selected_objects:
        if i in tmp_mesh_objs:
            bpy.context.view_layer.objects.active = i
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.join()

def merge2(input_pth, output_pth, dir):
    #import first glb and compute its bbox and normalize it 
    reset_scene()

    obj_num = len(input_pth)
    #import the first object
    # for i in range(obj_num):
    #     tmp_pth = dir + '/' + filter_path[input_pth[i]] + '.glb'
    #     # print(os.stat(tmp_pth).st_size/1024/1024)
    #     print(tmp_pth)
    #     if (os.stat(tmp_pth).st_size/1024/1024 > 30):
    #         # cur_num -= 1
    #         return False
    pth1 = dir + input_pth[0]
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
        tmp_pth = dir + input_pth[i + 1]
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
        category = input_pth[i].split('/')[0]
        # breakpoint()
        if category == 'desk':
            scale = 0.7 / (bbox_max[2] - bbox_min[2])
        elif category == 'chair':
            scale = 0.8 / (bbox_max[2] - bbox_min[2])
        elif category == 'bed':
            scale = 0.75 / (bbox_max[2] - bbox_min[2])
        elif category == 'sofa':
            scale = 0.65 / (bbox_max[2] - bbox_min[2])
        elif category == 'cabinet':
            scale = 0.9 / (bbox_max[2] - bbox_min[2])
        tmp_scale = random.uniform(0.95, 1.05)
        scales.append(tmp_scale)
        scale = tmp_scale * scale
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

    return True


parser = argparse.ArgumentParser()
parser.add_argument(
    "--train_path",
    type=str,
    help="Path to the object file",
)
parser.add_argument(
    "--valid_path",
    type=str,
    help="Path to the object file",
)
parser.add_argument("--output_dir", type=str, default="C_Obj")
parser.add_argument("--object_dir", type=str, default="/mnt/fillipo/ruijie/small_bench/")
parser.add_argument("--num", type=int, default=300)

script_args = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(script_args)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir, exist_ok=True)

bed_obj = os.listdir(os.path.join(args.object_dir, 'bed'))
bed_obj = ['bed/' + i for i in bed_obj]
cabinet_obj = os.listdir(os.path.join(args.object_dir, 'cabinet'))
cabinet_obj = ['cabinet/' + i for i in cabinet_obj]
chair_obj = os.listdir(os.path.join(args.object_dir, 'chair'))
chair_obj = ['chair/' + i for i in chair_obj]
desk_obj = os.listdir(os.path.join(args.object_dir, 'desk'))
desk_obj = ['desk/' + i for i in desk_obj]
sofa_obj = os.listdir(os.path.join(args.object_dir, 'sofa'))
sofa_obj = ['sofa/' + i for i in sofa_obj]
cnt = 0
item_dict = [bed_obj, cabinet_obj, chair_obj, chair_obj, desk_obj, sofa_obj]
item = np.array([0, 1, 2, 3, 4, 5])
prpr = np.array([0.15, 0.15, 0.2, 0.2, 0.25, 0.15])
prpr = prpr / prpr.sum()
cnt = 0
while(cnt < args.num):
    output_pth = args.output_dir + '/' + str(cnt) + '.glb'
    num = random.randint(3, 4)
    sample = np.random.choice(item, size=num, replace=False, p=prpr)
    ran = []
    for i in sample:
        r = random.randint(0, len(item_dict[i]) - 1)
        ran.append(item_dict[i][r])
    merge2(ran, output_pth, args.object_dir)
    cnt += 1
