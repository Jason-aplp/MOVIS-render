# Rendering Code of MOVIS

Scripts to compose objects, render rgb images, depth maps and instance masks (both modal and amodal).

### Installation

1. Install Blender

```bash
wget https://download.blender.org/release/Blender3.2/blender-3.2.2-linux-x64.tar.xz
tar -xf blender-3.2.2-linux-x64.tar.xz
rm blender-3.2.2-linux-x64.tar.xz
```

2. Install Python dependencies

```bash
pip install -r requirements.txt
```

3. (Optional) If you are running rendering on a headless machine, you will need to start an xserver. To do this, run:

```bash
sudo apt-get install xserver-xorg
sudo python3 scripts/start_xserver.py start
```

### Compose and render 3D-FUTURE objects

1. Download 3D-FUTURE models as [3D-FUTURE](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future) and place them under `3D-FUTURE/`

2. Compose 3D-FUTURE objects

```bash
bash compose_3d_future.sh
```

The parameters are explained in the script.

3. Render composed 3D-FUTURE models
```bash
bash render.sh
```

We provide an example of composed objects in `example_c3dfs/`. If you wants to render your own composed objects, revise the content within `example.json` file.

After running the rendering script, you should get the following directory structure for each scene:
```
scene_id/
├── {view_id}_1.png (rgb image under view {view_id})
├── {view_id}_1.npy (camera pose of view {view_id})
├── depth_{view_id}_1.exr (depth map under view {view_id})
├── mask_{view_id}/
│   ├── mask_{view_id}_0_x_1.png (modal mask of object 'x' under view {view_id})
│   ├── mask_{view_id}_1_x_1.png (amodal mask of object 'x' under view {view_id})
```

4. Compose masks into mix_mask for training
```
python merge_mask.py --path example_mix.json
```
This will generate a `mixed_mask.png` file under every `scene_id/mask_{view_id}` with background as 1 and different instances as different id.

We provide an example file `example_mix.json` here, revise the content within to `/path/to/render_path/scene_id` accordingly.
### Compose and render Objaverse objects

1. Download filtered Objaverse models from [C-Obj](https://huggingface.co/datasets/JasonAplp/MOVIS_dataset/tree/main) (Our rendered version is also provided here) and place them under `Objaverse/` (or other places, but you need to revise the path below accordingly)

2. Compose 3D-FUTURE objects

```
blender-3.2.2-linux-x64/blender -b -P compose_objaverse.py -- \
    --output_dir C_obj \
    --object_dir Objaverse \
    --num 300
```

The `output_dir` stands for the path where you want to save the composed objects. The `object_dir` stands for the path where you downloaded the filtered object models. The `num` stands for how many objects you want to compose.

3. Render composed Objaverse model the same as 3D-FUTURE models.

### Acknowledgement

Some code are borrowed from [Objaverse-rendering](https://github.com/allenai/objaverse-rendering) and [Zero-1-to-3](https://github.com/cvlab-columbia/zero123). We would like to thank the authors of these work for publicly releasing their code.