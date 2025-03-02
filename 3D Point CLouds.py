import cv2
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torchvision.transforms as transforms
import open3d as o3d
import copy
from sklearn.neighbors import NearestNeighbors
from lib.multi_depth_model_woauxi import RelDepthModel
from lib.net_tools import load_ckpt

from lib.SuperPoint_V2 import SuperPointFrontend

def parse_args():
    parser = argparse.ArgumentParser(description='Depth Map and Point Cloud Generator')
    parser.add_argument('--rgb_path', required=True, help='Path to RGB image')
    parser.add_argument('--load_ckpt', default='./res50.pth', help='Checkpoint path to load depth model')
    parser.add_argument('--backbone', default='resnext101', help='Backbone model type')
    # parser.add_argument('--output_dir', default='output', help='Output directory')
    parser.add_argument('--output_dir', help='Output directory')
    parser.add_argument('--fx', type=float, help='Focal length x (optional)')
    parser.add_argument('--fy', type=float, help='Focal length y (optional)')
    parser.add_argument('--cx', type=float, help='Principal point x (optional)')
    parser.add_argument('--cy', type=float, help='Principal point y (optional)')
    parser.add_argument('--depth_scale', type=float, default=1000.0, 
                       help='Scale factor to convert depth values to meters (default: 1000.0)')
    parser.add_argument('--skip', type=int, default=1, 
                       help='Sample every nth pixel for point cloud (default: 1)')
    parser.add_argument('--no_display', action='store_true', 
                       help='Don\'t display visualizations')
    parser.add_argument("--keypoint_weight", type=str, 
                        help="Path to pretrained weights file (default: superpoint_v1.pth).")
    parser.add_argument("--nms_dist", type=int, default=2, 
                        help="Non Maximum Suppression (NMS) distance (default: 4).")
    parser.add_argument("--conf_thresh", type=float, default=0.015, 
                        help="Detector confidence threshold (default: 0.015).")
    parser.add_argument("--cuda", action="store_true", 
                        help="Use cuda GPU to speed up network processing speed (default: False)")
    parser.add_argument("--output_path", type=str, default=None, 
                        help="Path to save output visualization (default: None, will display instead).")
    parser.add_argument("--display_scale", type=int, default=2, 
                        help="Factor to scale output visualization (default: 2).")
    parser.add_argument("--smooth", action='store_true',
                        help="Use Smoothing algorithm on the generated point cloud")
    parser.add_argument("--show_difference", action='store_true',
                        help="Show the outliers")
    parser.add_argument("--use_keypoints", action='store_true',
                        help="Smooth using keypoints")
    
    args = parser.parse_args()
    return args

def scale_torch(img):
    """Scale the image and output it in torch.tensor."""
    if len(img.shape) == 2:
        img = img[np.newaxis, :, :]
    if img.shape[2] == 3:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        img = transform(img)
    else:
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
    return img

def generate_depth_map(rgb_path, depth_model):
    """Generate depth map from RGB image."""
    rgb = cv2.imread(rgb_path)
    rgb_c = rgb[:, :, ::-1].copy()
    A_resize = cv2.resize(rgb_c, (448, 448))
    
    img_torch = scale_torch(A_resize)[None, :, :, :]
    pred_depth = depth_model.inference(img_torch).cpu().numpy().squeeze()
    pred_depth_ori = cv2.resize(pred_depth, (rgb.shape[1], rgb.shape[0]))
    
    # Normalize depth map
    pred_depth_normalized = (pred_depth_ori - pred_depth_ori.min()) / (pred_depth_ori.max() - pred_depth_ori.min())
    pred_depth_gray = (pred_depth_normalized * 255).astype(np.uint8)
    pred_depth_eq = cv2.equalizeHist(pred_depth_gray)
    
    return rgb, pred_depth_ori, pred_depth_normalized, pred_depth_gray, pred_depth_eq

def visualize_depth_maps(rgb, pred_depth_ori, pred_depth_normalized, pred_depth_gray, pred_depth_eq):
    """Visualize different versions of depth maps."""
    fig = plt.figure(figsize=(8, 10))
    
    gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1])  # First row taller

    # Top image spanning 2 columns
    ax1 = plt.subplot(gs[0, :])  # Row 0, span all columns
    ax1.imshow(rgb)
    ax1.set_title("Image")
    ax1.axis('off')
    
    ax2 = plt.subplot(gs[1, 0])
    ax3 = plt.subplot(gs[1, 1])
    ax4 = plt.subplot(gs[2, 0])
    ax5 = plt.subplot(gs[2, 1])

    im = ax2.imshow(pred_depth_ori, cmap='rainbow')
    ax2.set_title('Original Depth Map')
    ax2.axis('off')
    fig.colorbar(im, ax=ax2)
    
    im = ax3.imshow(pred_depth_normalized, cmap='rainbow')
    ax3.set_title('Normalized Depth Map')
    ax3.axis('off')
    fig.colorbar(im, ax=ax3)
    
    im = ax4.imshow(pred_depth_gray, cmap='gray')
    ax4.set_title('Grayscale Depth Map')
    ax4.axis('off')
    fig.colorbar(im, ax=ax4)
    
    im = ax5.imshow(pred_depth_eq, cmap='gray')
    ax5.set_title('Equalized Depth Map')
    ax5.axis('off')
    fig.colorbar(im, ax=ax5)

    # Show the figure
    plt.tight_layout()
    plt.show()


def depth_to_pointcloud(rgb, depth, fx=None, fy=None, cx=None, cy=None, depth_scale=1000.0, skip=1):
    """Convert depth map to point cloud."""
    height, width = depth.shape
    
    # Set default camera parameters if not provided
    if fx is None or fy is None:
        fx = fy = max(width, height) * 0.8
        print(fx, fy)
    if cx is None:
        cx = width / 2.0
        print(cx)
    if cy is None:
        cy = height / 2.0
        print(cy)

    # Create coordinate grid
    x, y = np.meshgrid(np.arange(0, width, skip), np.arange(0, height, skip))
    x = x.flatten()
    y = y.flatten()
    
    # Convert depth to meters and filter invalid values
    z = depth[y, x].astype(np.float32) / depth_scale
    valid_mask = (z > 0) & np.isfinite(z)
    x, y, z = x[valid_mask], y[valid_mask], z[valid_mask]

    # Convert to 3D coordinates
    x_3d = (x - cx) * z / fx
    y_3d = (y - cy) * z / fy
    z_3d = z

    # Get colors
    colors = rgb[y, x] / 255.0

    # Create point cloud
    points = np.stack([x_3d, y_3d, z_3d], axis=1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd


def process_image(
    image_path,
    weights_path,
    nms_dist=4,
    conf_thresh=0.015,
    cuda=False,
    output_path=None,
    display_scale=2,
):
    """Process a single image with SuperPoint."""
    # Read and preprocess image
    img = cv2.imread(image_path, 0)
    if img is None:
        raise Exception(f"Error reading image {image_path}")
    img = img.astype("float32") / 255.0

    # Initialize SuperPoint
    fe = SuperPointFrontend(
        weights_path=weights_path, nms_dist=nms_dist, conf_thresh=conf_thresh, cuda=cuda
    )

    # Process image
    pts, desc, heatmap = fe.run(img)

    # Visualize results
    out = (np.dstack((img, img, img)) * 255.0).astype("uint8")
    print(out.shape)

    # Draw points
    for pt in pts.T:
        pt1 = (int(round(pt[0])), int(round(pt[1])))
        cv2.circle(out, pt1, 1, (0, 255, 0), -1, lineType=16)

    # Resize output
    # out = cv2.resize(out, (display_scale*img.shape[1], display_scale*img.shape[0]))

    # Save or display result
    if output_path:
        cv2.imwrite(output_path, out)
        print(f"Output saved to {output_path}")

    return pts, desc, out


def smoothness(rgb, depth_map, point_cloud, keypoints, distance_threshold=0.01, ransac_n=5, num_iterations=1000, show_difference=False):
    
    if keypoints is None:
        # Fit a plane model using RANSAC
        plane_model, inliers = point_cloud.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )

        inlier_cloud = point_cloud.select_by_index(inliers)  
        outlier_cloud = point_cloud.select_by_index(inliers, invert=True)  # Outliers

        print("Number of inliers (cleaned points):", len(inlier_cloud.points))
        print("Number of outliers (removed points):", len(outlier_cloud.points))
        
        if show_difference:
            i_cloud = copy.deepcopy(inlier_cloud)
            o_cloud = copy.deepcopy(outlier_cloud)
            
            i_cloud.paint_uniform_color([0, 1, 0])
            o_cloud.paint_uniform_color([1, 0, 0])
            
            o3d.visualization.draw_geometries([i_cloud, o_cloud], window_name="Inliers (Green) and Outliers (Red)")
            
        o3d.visualization.draw_geometries([inlier_cloud], window_name="After RANSAC Smoothness")
        
        return inlier_cloud, outlier_cloud
    
    else:
        def lift_keypoints_to_3d(keypoints_2d, depth_map, K):
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            keypoints_3d = []
            h, w = depth_map.shape

            for (u, v) in keypoints_2d:
                u_int, v_int = int(round(u)), int(round(v))
                if 0 <= u_int < w and 0 <= v_int < h:
                    z = depth_map[v_int, u_int]
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy
                    keypoints_3d.append([x, y, z])

            return np.array(keypoints_3d, dtype=np.float32)
        
        points = np.asarray(point_cloud.points)
        colors = np.asarray(point_cloud.colors)
        
        depth_map = depth_map.astype(np.float32)
        
        height, width = depth_map.shape
        fx = fy = max(width, height) * 0.8
        cx = width / 2.0
        cy = height / 2.0

        K = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0, 0,  1 ]])
        
        keypoints_2d = keypoints
        
        keypoints_3d = lift_keypoints_to_3d(keypoints_2d, depth_map, K)
        print("Lifted 3D keypoints:\n", keypoints_3d)
        
        knn = NearestNeighbors(n_neighbors=16, algorithm='auto')
        knn.fit(points)  # Fit KNN on the point cloud

        # Find neighbors for each point
        distances, indices = knn.kneighbors(points)

        # Enhance point cloud by averaging neighbor positions
        enhanced_points = np.copy(points)
        for i in range(points.shape[0]):
            neighbor_pts = points[indices[i]]
            enhanced_points[i] = np.mean(neighbor_pts, axis=0)  # Smooth position


        pcd_original = o3d.geometry.PointCloud()
        pcd_original.points = o3d.utility.Vector3dVector(points)
        pcd_original.colors = o3d.utility.Vector3dVector(colors)

        pcd_enhanced = o3d.geometry.PointCloud()
        pcd_enhanced.points = o3d.utility.Vector3dVector(enhanced_points)
        pcd_enhanced.colors = o3d.utility.Vector3dVector(colors) 
        
        if show_difference:
            orig = copy.deepcopy(pcd_original).paint_uniform_color([1, 0, 0])  # Red color for original
            enhan = copy.deepcopy(pcd_enhanced).paint_uniform_color([0, 1, 0])  # Green color for enhanced

            o3d.visualization.draw_geometries([orig, enhan], window_name="Original (Red) vs Enhanced (Green)")
        
        o3d.visualization.draw_geometries([pcd_enhanced], window_name="Point Cloud after Smoothness")
        
        return pcd_original, pcd_enhanced
        

def main():
    args = parse_args()
    
    if args.output_dir:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize depth model
    depth_model = RelDepthModel(backbone=args.backbone)
    depth_model.eval()
    load_ckpt(args, depth_model, None, None)
    depth_model.to(device='cpu')
    
    # Generate depth maps
    rgb, pred_depth_ori, pred_depth_normalized, pred_depth_gray, pred_depth_eq = generate_depth_map(args.rgb_path, depth_model)
    
    if args.use_keypoints:
        # keypoint detector
        pts, desc, out = process_image(
            args.rgb_path,
            args.keypoint_weight,
            args.nms_dist,
            args.conf_thresh,
            args.cuda,
            args.output_path,
            args.display_scale,
        )
        
        cv2.imshow("SuperPoint Detection", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # print(keypoints_2d.shape)
    
        print(f"Detected {pts.shape[1]} keypoints")
    
    # Save depth maps
    # cv2.imwrite(os.path.join(args.output_dir, 'depth_grayscale.png'), pred_depth_gray)
    # cv2.imwrite(os.path.join(args.output_dir, 'depth_equalized.png'), pred_depth_eq)
    
    # Visualize depth maps
    if not args.no_display:
        visualize_depth_maps(rgb, pred_depth_ori, pred_depth_normalized, pred_depth_gray, pred_depth_eq)
    
    # Generate point cloud
    pcd = depth_to_pointcloud(
        rgb, pred_depth_eq,
        fx=args.fx, fy=args.fy,
        cx=args.cx, cy=args.cy,
        depth_scale=args.depth_scale,
        skip=args.skip
    )
    
    # Save point cloud
    # o3d.io.write_point_cloud(os.path.join(args.output_dir, 'pointcloud_2.ply'), pcd)
    
    if not args.no_display:
        o3d.visualization.draw_geometries([pcd])
        
    if args.smooth:
        
        if args.use_keypoints:
            keypoints_2d = pts[:2, :].T
            
            print("Smoothing Using KNN")
        
            original, enhanced = smoothness(
                rgb = rgb,
                depth_map = pred_depth_eq,
                point_cloud = pcd,
                keypoints=keypoints_2d,
                distance_threshold=0.03,
                ransac_n=3,
                num_iterations=1000,
                show_difference=args.show_difference)
        
        else:
            print("Smoothing Using RANSAC")
            
            original, enhanced = smoothness(
                rgb = rgb,
                depth_map = pred_depth_eq,
                point_cloud = pcd,
                keypoints=None,
                distance_threshold=0.03,
                ransac_n=3,
                num_iterations=1000,
                show_difference=args.show_difference)
        


if __name__ == '__main__':
    main()