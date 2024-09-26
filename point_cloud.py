import numpy as np
import open3d as o3d
from scipy.interpolate import griddata
import plotly.graph_objects as go

def create_point_cloud(mkpts0, mkpts1, baseline, center_distance):
    disparity = np.abs(mkpts0[:, 0] - mkpts1[:, 0])
    depth = (baseline * center_distance) / disparity
    # Create point cloud
    points = []
    for i in range(len(mkpts0)):
        x = mkpts0[i][0]
        y = mkpts0[i][1]
        z = depth[i]
        points.append([x, y, z])

    points = np.array(points)
    return points

def remove_outliers(points, outlier_indexes):
    return np.delete(points, outlier_indexes, axis=0)

def plot_3d_surface(points):
    # Создание облака точек в Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Визуализация точек в 3D
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=dict(
                    size=3,
                    color=points[:, 2],  # Можно использовать глубину для окраски
                ),
            )
        ]
    )

    # Extract x, y, z coordinates
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2] / 1000

    # Define grid over x and y
    grid_x, grid_y = np.mgrid[x.min():x.max(), y.min():y.max()]  # Adjust range and resolution as needed

    # Interpolate z values on the grid
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='linear')

    fig = go.Figure(data=[go.Surface(x=grid_x, y=grid_y, z=grid_z)])

    return fig
