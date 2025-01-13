#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import json
import numpy as np
import os
from collections import defaultdict
import cv2
import tqdm
from osgeo import gdal

from detectron2.data.datasets import register_coco_instances
register_coco_instances("adrs_train", {}, "/media/data1/ljt/Instance_RSAD_Data/training/labels.json", "/media/data1/ljt/Instance_RSAD_Data/training/images")
register_coco_instances("adrs_test_thermal", {}, "/media/data1/ljt/Instance_RSAD_Data/testing/Thermal/thermal.json", "/media/data1/ljt/Instance_RSAD_Data/testing/Thermal/images")
register_coco_instances("adrs_test_HSI", {}, "/media/data1/ljt/Instance_RSAD_Data/testing/HSI/HSI.json", "/media/data1/ljt/Instance_RSAD_Data/testing/HSI/images")
register_coco_instances("adrs_test_SAR", {}, "/media/data1/ljt/Instance_RSAD_Data/testing/SAR/SAR.json", "/media/data1/ljt/Instance_RSAD_Data/testing/SAR/images")

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer

def read_img(img_path: str):
    """
    Read imagery as ndarray
    :param img_path:
    :param gdal_read:
    :return:
    """
    dataset = gdal.Open(img_path)
    w, h = dataset.RasterXSize, dataset.RasterYSize
    img = dataset.ReadAsArray(0, 0, w, h)
    if len(img.shape) == 3:
        img = np.transpose(img, axes=(1, 2, 0))  # [c,h,w]->[h,w,c]
    return img


def write_img(img: np.ndarray, save_path: str):
    """
    Save ndarray as imagery
    :param img:
    :param save_path:
    :param gdal_write: 
    :return:
    """
    if 'int8' in img.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in img.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(img.shape) == 3:
        img = np.transpose(img, axes=(2, 0, 1))  # [h,w,c]->[c,h,w]
    elif len(img.shape) == 2:
        img = np.expand_dims(img, axis=0)

    img_bands, img_height, img_width = img.shape

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(save_path, int(img_width), int(img_height), int(img_bands), datatype)
    for i in range(img_bands):
        dataset.GetRasterBand(i + 1).WriteArray(img[i])
    del dataset

    
def create_instances(predictions, image_size):
    ret = Instances(image_size)

    score = np.asarray([x["score"] for x in predictions])
    chosen = (score > args.conf_threshold).nonzero()[0]
    score = score[chosen]
    bbox = np.asarray([predictions[i]["bbox"] for i in chosen]).reshape(-1, 4)
    bbox = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)

    labels = np.asarray([dataset_id_map(predictions[i]["category_id"]) for i in chosen])

    ret.scores = score
    ret.pred_boxes = Boxes(bbox)
    ret.pred_classes = labels

    try:
        ret.pred_masks = [predictions[i]["segmentation"] for i in chosen]
    except KeyError:
        pass
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script that visualizes the json predictions from COCO or LVIS dataset."
    )
    parser.add_argument("--input", help="JSON file produced by the model")
    parser.add_argument("--output",  help="output directory")
    parser.add_argument("--dataset", help="name of the dataset", default="adrs_test_thermal")
    parser.add_argument("--conf-threshold", default=0.35, type=float, help="confidence threshold")
    args = parser.parse_args()

    logger = setup_logger()

    with PathManager.open(args.input, "r") as f:
        predictions = json.load(f)

    pred_by_image = defaultdict(list)
    for p in predictions:
        pred_by_image[p["image_id"]].append(p)

    dicts = list(DatasetCatalog.get(args.dataset))
    metadata = MetadataCatalog.get(args.dataset)
    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):

        def dataset_id_map(ds_id):
            return metadata.thing_dataset_id_to_contiguous_id[ds_id]

    elif "lvis" in args.dataset:
        # LVIS results are in the same format as COCO results, but have a different
        # mapping from dataset category id to contiguous category id in [0, #categories - 1]
        def dataset_id_map(ds_id):
            return ds_id - 1

    else:
        raise ValueError("Unsupported dataset: {}".format(args.dataset))

    os.makedirs(args.output, exist_ok=True)

    for dic in tqdm.tqdm(dicts):
        if 'tif' in dic["file_name"]:
            img = read_img(dic["file_name"])
            if img.shape[2] > 200:
                img = img[:,:,[135,66,22]]
                img = (img - img.min())/(img.max() - img.min())
                img = img * 255
            elif img.shape[2] > 3:
                index1 = int(img.shape[2]/3)
                index2 = int(img.shape[2]/2)
                index3 = int(img.shape[2] - 1)
                img = img[:,:,[index1, index2, index3]]*255
        else:
            img = cv2.imread(dic["file_name"], cv2.IMREAD_COLOR)[:, :, ::-1]
        basename = os.path.basename(dic["file_name"])

        predictions = create_instances(pred_by_image[dic["image_id"]], img.shape[:2])
        vis = Visualizer(img, metadata)
        vis_pred = vis.draw_instance_predictions(predictions).get_image()

        vis = Visualizer(img, metadata)
        vis_gt = vis.draw_dataset_dict(dic).get_image()

        concat = np.concatenate((vis_pred, vis_gt), axis=1)
        cv2.imwrite(os.path.join(args.output, basename), concat[:, :, ::-1])
