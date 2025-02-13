# mode is train/test
# begin_num is the beginning scene id
# end_num is the end scene id (We compose 100k scenes in the paper)
# split_path is the path to train/test split
# obj_root_path is the path to 3D-FUTURE models
# save_root is the place to save composed files
blender-3.2.2-linux-x64/blender -b -P compose_3d_future.py -- \
    --mode train \
    --begin_num 0 \
    --end_num 100000 \
    --split_path 3D-split \
    --obj_root_path 3D-FUTURE \
    --save_root C3DFS