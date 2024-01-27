from __future__ import absolute_import

import os
from typing import Optional
import numpy as np
import craft_text_detector.craft_utils as craft_utils
import craft_text_detector.file_utils as file_utils
import craft_text_detector.image_utils as image_utils
import craft_text_detector.predict as predict
import craft_text_detector.torch_utils as torch_utils

__version__ = "0.4.7"

__all__ = [
    "read_image",
    "load_craftnet_model",
    "load_refinenet_model",
    "get_prediction",
    "export_detected_regions",
    "export_extra_results",
    "empty_cuda_cache",
    "Craft",
]

read_image = image_utils.read_image
load_craftnet_model = craft_utils.load_craftnet_model
load_refinenet_model = craft_utils.load_refinenet_model
get_prediction = predict.get_prediction
export_detected_regions = file_utils.export_detected_regions
export_extra_results = file_utils.export_extra_results
empty_cuda_cache = torch_utils.empty_cuda_cache


def calculate_polygon_area(polygon):
    x = polygon[:, 0]
    y = polygon[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def calculate_polygons_area_ratio(polygons, image_width, image_height):
    total_area = image_width * image_height
    polygons_area = sum(calculate_polygon_area(poly) for poly in polygons)
    return polygons_area / total_area


def calculate_angle(p1, p2):
    """计算两点间线段与水平线的夹角（以度为单位）"""
    angle = np.arctan2(abs(p2[1] - p1[1]), abs(p2[0] - p1[0])) * 180 / np.pi
    return min(angle, 90 - angle)  # 返回与水平或垂直线较小的角度


def check_is_aligned(polygons):
    """检查polygons列表中是否水平的多边形"""
    for polygon in polygons:
        if len(polygon) == 4:  # 确保多边形由四个点组成
            # 计算所有相邻点对之间的距离，并找到最长边
            distances = [
                np.linalg.norm(np.array(polygon[i]) - np.array(polygon[(i + 1) % 4]))
                for i in range(4)
            ]
            longest_edge_index = np.argmax(distances)

            # 计算最长边与水平线的夹角
            p1, p2 = polygon[longest_edge_index], polygon[(longest_edge_index + 1) % 4]
            angle = calculate_angle(p1, p2)

            if angle < 5:
                return True
    return False


def calculate_vertical_range(height, percentage):
    """根据图像高度和百分比计算垂直范围"""
    margin = height * (1 - percentage) / 2
    return margin, height - margin


def is_within_vertical_range(polygon, height, percentage):
    """检查多边形中心点的y坐标是否在指定的垂直范围内"""
    # 计算多边形的中心点
    center = np.mean(polygon, axis=0)
    # 计算垂直范围
    lower_bound, upper_bound = calculate_vertical_range(height, percentage)
    # 检查中心点的y坐标是否在范围内
    return lower_bound <= center[1] <= upper_bound


def check_is_not_center(polygons, height, percentage):
    """检查polygons列表中的多边形中心点是否满足条件"""
    for polygon in polygons:
        if not is_within_vertical_range(polygon, height, percentage):
            return True
    return False


class Craft:
    def __init__(
        self,
        output_dir=None,
        rectify=True,
        export_extra=True,
        text_threshold=0.7,
        link_threshold=0.4,
        low_text=0.4,
        cuda=False,
        long_size=1280,
        refiner=True,
        crop_type="poly",
        weight_path_craft_net: Optional[str] = None,
        weight_path_refine_net: Optional[str] = None,
    ):
        """
        Arguments:
            output_dir: path to the results to be exported
            rectify: rectify detected polygon by affine transform
            export_extra: export heatmap, detection points, box visualization
            text_threshold: text confidence threshold
            link_threshold: link confidence threshold
            low_text: text low-bound score
            cuda: Use cuda for inference
            long_size: desired longest image size for inference
            refiner: enable link refiner
            crop_type: crop regions by detected boxes or polys ("poly" or "box")
        """
        self.craft_net = None
        self.refine_net = None
        self.output_dir = output_dir
        self.rectify = rectify
        self.export_extra = export_extra
        self.text_threshold = text_threshold
        self.link_threshold = link_threshold
        self.low_text = low_text
        self.cuda = cuda
        self.long_size = long_size
        self.refiner = refiner
        self.crop_type = crop_type

        # load craftnet
        self.load_craftnet_model(weight_path_craft_net)
        # load refinernet if required
        if refiner:
            self.load_refinenet_model(weight_path_refine_net)

    def load_craftnet_model(self, weight_path: Optional[str] = None):
        """
        Loads craftnet model
        """
        self.craft_net = load_craftnet_model(self.cuda, weight_path=weight_path)

    def load_refinenet_model(self, weight_path: Optional[str] = None):
        """
        Loads refinenet model
        """
        self.refine_net = load_refinenet_model(self.cuda, weight_path=weight_path)

    def unload_craftnet_model(self):
        """
        Unloads craftnet model
        """
        self.craft_net = None
        empty_cuda_cache()

    def unload_refinenet_model(self):
        """
        Unloads refinenet model
        """
        self.refine_net = None
        empty_cuda_cache()

    def detect_text(
        self, image, image_path=None, file_name=None, maxR=0.05, isOpEd=False
    ):
        """
        Arguments:
            image: path to the image to be processed or numpy array or PIL image

        Output:
            {
                "masks": lists of predicted masks 2d as bool array,
                "boxes": list of coords of points of predicted boxes,
                "boxes_as_ratios": list of coords of points of predicted boxes as ratios of image size,
                "polys_as_ratios": list of coords of points of predicted polys as ratios of image size,
                "heatmaps": visualization of the detected characters/links,
                "text_crop_paths": list of paths of the exported text boxes/polys,
                "times": elapsed times of the sub modules, in seconds
            }
        """

        if image_path is not None:
            print("Argument 'image_path' is deprecated, use 'image' instead.")
            image = image_path

        # perform prediction
        prediction_result = get_prediction(
            image=image,
            craft_net=self.craft_net,
            refine_net=self.refine_net,
            text_threshold=self.text_threshold,
            link_threshold=self.link_threshold,
            low_text=self.low_text,
            cuda=self.cuda,
            long_size=self.long_size,
        )

        # arange regions
        if self.crop_type == "box":
            regions = prediction_result["boxes"]
        elif self.crop_type == "poly":
            regions = prediction_result["polys"]
        else:
            raise TypeError("crop_type can be only 'polys' or 'boxes'")

        if type(image) == str:
            image = image_utils.read_image(image)

        height, width = image.shape[:2]

        # filiter here
        total = calculate_polygons_area_ratio(
            regions, image_width=width, image_height=height
        )

        # 当不是片头片尾的时候
        if not isOpEd:
            if not total >= maxR or not len(regions) > 3:
                print(f"文字占比或文字数量不足, r: {total}, n: {len(regions)}")
                return
            # 检查是否存在水平的多边形
            if not check_is_aligned(regions):
                print("不存在水平的多边形")
                return
            # 检查是否存在内容边界以外的多边形
            if not check_is_not_center(regions, height, 0.7):
                print("不存在内容边界以外的内容")
                return
        else:
            if not len(regions) > 3:
                print(f"数量不足, n: {len(regions)}")
                return

        # print("前置过滤完毕")
        # print(regions)
        # export if output_dir is given
        prediction_result["text_crop_paths"] = []
        if self.output_dir is not None:
            # export detected text regions
            if type(image) == str:
                file_name, file_ext = os.path.splitext(os.path.basename(image))
            else:
                file_name = file_name if file_name else "image"
            exported_file_paths = export_detected_regions(
                image=image,
                regions=regions,
                file_name=file_name,
                output_dir=self.output_dir,
                rectify=self.rectify,
            )
            prediction_result["text_crop_paths"] = exported_file_paths

            # export heatmap, detection points, box visualization
            if self.export_extra:
                export_extra_results(
                    image=image,
                    regions=regions,
                    heatmaps=prediction_result["heatmaps"],
                    file_name=file_name,
                    output_dir=self.output_dir,
                )

        # return prediction results
        return prediction_result
