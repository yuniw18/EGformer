import json
import numpy as np
import open3d as o3d
from imageio import imread
from PIL import Image
import time

from scipy.spatial.transform import Rotation as R

def custom_draw_geometry_with_key_callback(pcd, render_option_path):
    def change_background_to_black(vis):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        return False

    def load_render_option(vis):
        vis.get_render_option().load_from_json('render_option.json')
        return False
 
    def load_view_option(vis):
        vis.get_view_control().load_from_json('camera_option.json')
        return False


    key_to_callback = {}
    key_to_callback[ord("R")] = change_background_to_black
    key_to_callback[ord("N")] = load_render_option
    key_to_callback[ord("M")] = load_render_option
 
    
    o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback)


def get_uni_sphere_xyz(H, W):
    j, i = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    u = (i+0.5) / W * 2 * np.pi
    v = ((j+0.5) / H - 0.5) * np.pi
    z = -np.sin(v)
    c = np.cos(v)
    y = c * np.sin(u)
    x = c * np.cos(u)
    sphere_xyz = np.stack([x, y, z], -1)
    return sphere_xyz


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--img', default='sample/S3D_img.png',
                        help='Image texture in equirectangular format')
    parser.add_argument('--depth', default='sample/EGformer_depth.png',
                        help='Depth map')
    parser.add_argument('--scale', default=0.1, type=float,
                        help='Rescale the depth map')
    parser.add_argument('--crop_ratio', default=0, type=float,
                        help='Crop ratio for upper and lower part of the image')
    parser.add_argument('--crop_z_above', default=1.2, type=float,
                        help='Filter 3D point with z coordinate above')
    args = parser.parse_args()


    # Reading rgb-d
    rgb = imread(args.img)
    rgb = rgb[:,:,0:3]
    depth = np.array(Image.open(args.depth).convert("L")).astype(np.float32)
    depth = np.expand_dims(depth,axis=2)
    depth = depth * args.scale
    
    # Project to 3d
    H, W = rgb.shape[:2]
    xyz = depth * get_uni_sphere_xyz(H, W)
    xyzrgb = np.concatenate([xyz, rgb/255.], 2)

    # Crop the image and flatten
    if args.crop_ratio > 0:
        assert args.crop_ratio < 1
        crop = int(H * args.crop_ratio)
        xyzrgb = xyzrgb[crop:-crop]
    xyzrgb = xyzrgb.reshape(-1, 6)

    # Crop in 3d
    xyzrgb = xyzrgb[xyzrgb[:,2] <= args.crop_z_above]
    
    # Visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyzrgb[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(xyzrgb[:, 3:])
 
    custom_draw_geometry_with_key_callback(pcd,None)

