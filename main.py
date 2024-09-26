import gradio as gr
from stereo_processing import loftr_feature_matching, find_outliers
from point_cloud import create_point_cloud, remove_outliers, plot_3d_surface

def loftr_and_depth(img1, img2, baseline, center_distance, distance_cam2_object, distance_cam1_object):
    mkpts0, mkpts1 = loftr_feature_matching(img1, img2)
    points = create_point_cloud(mkpts0, mkpts1, baseline, center_distance)
    outliers_indexes = find_outliers(points[:, 2])
    points_cleaned = remove_outliers(points, outliers_indexes)
    fig = plot_3d_surface(points_cleaned)
    return fig

inputs = [
    gr.Image(type="numpy", label="Изображение 1"),
    gr.Image(type="numpy", label="Изображение 2"),
    gr.Number(value=412, label="Базовое расстояние (мм)"),
    gr.Number(value=639, label="Расстояние до центра (мм)"),
    gr.Number(value=641, label="Расстояние от камеры 2 до объекта (мм)"),
    gr.Number(value=660, label="Расстояние от камеры 1 до объекта (мм)"),
]

outputs = [
    gr.Plot(label="3D облако точек"),
]

description = """
Загрузите пару изображений и при необходимости настройте параметры, затем нажмите 'Submit' для измерения геометрии.
- **Базовое расстояние**: Расстояние между двумя камерами (в мм).
- **Расстояние до центра**: Расстояние от центра между камерами до объекта (в мм).
- **Расстояние от камеры 2 до объекта**: Расстояние от камеры 2 до объекта (в мм).
- **Расстояние от камеры 1 до объекта**: Расстояние от камеры 1 до объекта (в мм).
"""

title = "Измерение геометрии на основе стереоизображений"

demo = gr.Interface(
    fn=loftr_and_depth,
    inputs=inputs,
    outputs=outputs,
    live=False,
    title=title,
    description=description
)

if __name__ == "__main__":
    demo.launch()
