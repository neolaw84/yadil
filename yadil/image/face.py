import glob
import pathlib
import math
import traceback as tb
from typing import Tuple, List, Union

import pandas as pd
from tqdm.auto import tqdm
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from insightface.app import FaceAnalysis

from yadil.image.face_model import model_points
from yadil.image.utils import shift_centroid_to_origin as shift_to_o


def scale_face(points, target_points):
    dist1 = np.linalg.norm(points[16] - points[0], ord=2)
    dist2 = np.linalg.norm(target_points[16] - target_points[0], ord=2)
    return points * dist2 / dist1


model_points = shift_to_o(model_points)

app = FaceAnalysis(allowed_modules=["detection", "genderage", "landmark_3d_68"], providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

X = 0
Y = 1

X1 = 0
Y1 = 1
X2 = 2
Y2 = 3


def _get_new_bbox_points(size=(256, 256)):
    return np.float32([[0, 0], [size[X], 0], [0, size[Y]], [size[X], size[Y]]])


def _get_width_and_height(bbox: List) -> Tuple:
    return bbox[X2] - bbox[X1], bbox[Y2] - bbox[Y1]


def _get_bbox_center_width_height(bbox: List) -> Tuple:
    """[summary]

    Args:
        bbox (List): [x, y, width, height]

    Returns:
        Tuple: (x, y)
    """
    width, height = _get_width_and_height(bbox)
    return (bbox[X] + width / 2.0, bbox[Y] + height / 2.0), width, height


def _rotate_image(image, angle, image_center):
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def combine_affine_transform(M1, M2):
    M1_ = np.vstack([M1, np.float32([[0, 0, 1]])])
    M2_ = np.vstack([M2, np.float32([[0, 0, 1]])])
    M_final = np.matmul(M2_, M1_)[0:2, :]
    return M_final


def _extract(
    img, f, size: Tuple = (256, 256), bbox_scale: float = 1.0, correct_rotate: bool = False, return_all: bool = False
):
    bbox = f["bbox"]

    center, width, height = _get_bbox_center_width_height(bbox)

    # scaling bbox
    if bbox_scale != 1.0:
        width = width * bbox_scale
        height = height * bbox_scale

    if width > height:
        height = width
    else:
        width = height
    bbox_points = np.float32(
        [
            [center[X] - width / 2.0, center[Y] - height / 2.0],
            [center[X] + width / 2.0, center[Y] - height / 2.0],
            [center[X] - width / 2.0, center[Y] + height / 2.0],
            [center[X] + width / 2.0, center[Y] + height / 2.0],
        ]
    )

    # correct rotation
    if correct_rotate:
        landmarks = f["landmark_3d_68"]
        landmarks = shift_to_o(landmarks)
        mps_norm = scale_face(model_points, landmarks)

        # these two reshapes are to make sure Rotation is happy
        mps = mps_norm.reshape((-1, 3))
        lmk = landmarks.reshape((-1, 3))
        rmat, rmsd = Rotation.align_vectors(a=lmk, b=mps)

        rotation = math.degrees(rmat.as_rotvec()[2])

        # rimg_r = _rotate_image(img, rotation, center)
        M1 = cv2.getRotationMatrix2D(center, rotation, 1.0)

    # M = cv2.getAffineTransform(src=bbox_points, dst=_get_new_bbox_points(size=size))
    M, _ = cv2.estimateAffinePartial2D(bbox_points, _get_new_bbox_points(size=size))
    if correct_rotate:
        M = combine_affine_transform(M1, M)
    rimg = cv2.warpAffine(img, M, dsize=size, flags=cv2.INTER_CUBIC)

    if return_all:
        return {
            "rimg": rimg,
            "M": M,
            "rotation": rotation if correct_rotate else None,
            "rmat": rmat if correct_rotate else None,
            "rmsd": rmsd if correct_rotate else None,
            "bbox": bbox,
            "bbox_points": bbox_points,
        }
    return rimg


def extract(
    img, size: Tuple = (256, 256), bbox_scale: float = 1.0, correct_rotate: bool = False, return_all: bool = False
):
    """extract cv2 image(s) of the faces from the given image.

    Args:
        img (cv2_image): input image
        size (Tuple, optional): result size (x, y). Only supports square shapes. Defaults to (256, 256).
        bbox_scale (float, optional): how tight or loose the bounding box. <1 is tight while >1 is loose. Defaults to 1.0.
        correct_rotate (bool, optional): whether to correct rotation of face w.r.t Z axis of face (not camera). Defaults to False.
        return_all (bool, optional): whether to return all attributes

        It processes in this order:
            * scale bbox
            * correct rotation and
            * resize to size

    Returns:
        List[cv2_image]: a list of cv2 image(s) of the faces from the given image.
    """

    # we will never support this
    assert size[X] == size[Y]


    faces = app.get(img)

    rimgs = [
        _extract(img, f, size=size, bbox_scale=bbox_scale, correct_rotate=correct_rotate, return_all=return_all)
        for f in faces
    ]

    gender_ages = [
        {"gender" : f["gender"], "age" : f["age"]} for f in faces
    ]
    det_scores = [f["det_score"] for f in faces]
    
    return rimgs, gender_ages, det_scores if return_all else rimgs


def _extract_pitch_yaw_roll(f, size: Tuple = (256, 256), bbox_scale: float = 1.0, correct_rotate: bool = False):
    bbox = f["bbox"]

    landmarks = f["landmark_3d_68"]
    landmarks = shift_to_o(landmarks)
    mps_norm = scale_face(model_points, landmarks)

    # these two reshapes are to make sure Rotation is happy
    mps = mps_norm.reshape((-1, 3))
    lmk = landmarks.reshape((-1, 3))
    rmat, rmsd = Rotation.align_vectors(a=lmk, b=mps)

    return rmat.as_rotvec()


def extract_pitch_yaw_roll(img):

    faces = app.get(img)
    rimgs = [_extract_pitch_yaw_roll(f) for f in faces]

    return rimgs


def extract_all(input_glob, output_dir, input_meta=None, output_meta=None):
    if input_meta:
        print ("reading input_meta {}".format(input_meta))
        df_all = pd.read_csv(input_meta, index_col=None)
    else:
        df_all = None

    uuid_to_url = {
        k: v for k, v in zip (df_all.uuid, df_all.url)
    } if df_all is not None else {}

    def get_url(uuid_):
        return uuid_to_url.get(uuid_, "")

    def create_df(row_list):
        temp_df = pd.DataFrame(row_list)
        temp_df[["M00", "M01", "M02", "M10", "M11", "M12"]] = pd.DataFrame(temp_df.M.tolist(), index=temp_df.index)
        temp_df[["pitch", "yaw", "roll"]] = pd.DataFrame(temp_df.rmat.tolist(), index=temp_df.index)
        temp_df[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]] = pd.DataFrame(temp_df.bbox.tolist(), index=temp_df.index)
        temp_df[["bbox_scaled_x1", "bbox_scaled_y1", "bbox_scaled_x2", "bbox_scaled_y2"]] = pd.DataFrame(
            temp_df.bbox_points.tolist(), index=temp_df.index
        )[
            [0, 1, 6, 7]
        ]  # (0, 1, x, x, x, x, 6, 7) for 4 points forming the box
        temp_df.drop(["M", "rmat", "bbox", "bbox_points"], axis=1, inplace=True)
        return temp_df[
            [
                "url", 
                "uuid", 
                "ofname",
                "M00",
                "M01",
                "M02",
                "M10",
                "M11",
                "M12",
                "pitch",
                "yaw",
                "roll",
                "bbox_x1",
                "bbox_y1",
                "bbox_x2",
                "bbox_y2",
                "bbox_scaled_x1",
                "bbox_scaled_y1",
                "bbox_scaled_x2",
                "bbox_scaled_y2",
                "gender", 
                "age", 
                "det_score"
            ]
        ]

    def remove_ext(f):
        try:
            p = pathlib.Path(f)
            return "_".join(p.name.split(".")[:-1])
        except:
            return ""

    row_list = []
    outdir = pathlib.Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    for f in tqdm(glob.glob(input_glob)):
        try:
            fname = pathlib.Path(f).name
            f_uuid = remove_ext(fname)
            f_url = get_url(uuid_=f_uuid) if df_all is not None else ""
            img = cv2.imread(f)
            results, gender_ages, det_scores = extract(img, bbox_scale=1.2, size=(256, 256), correct_rotate=True, return_all=True)
            for i, (r, ga, ds) in enumerate(zip(results, gender_ages, det_scores)):
                rimg = r["rimg"]
                M = r["M"].reshape(-1)
                rotation = r["rotation"]
                rmat = r["rmat"].as_rotvec()
                bbox = r["bbox"].reshape(-1)
                bbox_points = r["bbox_points"].reshape(-1)
                ofname = f_uuid + "-" + str(i).zfill(3) + ".jpg"
                cv2.imwrite(str(pathlib.Path.joinpath(outdir, ofname)), rimg)
                temp_dict = {
                    "url": f_url,
                    "uuid": f_uuid,
                    "ofname": ofname,
                    "M": M,
                    "rotation": rotation,
                    "rmat": rmat,
                    "bbox": bbox,
                    "bbox_points": bbox_points,
                    "gender": ga["gender"], 
                    "age": ga["age"], 
                    "det_score": ds
                }
                row_list.append(temp_dict)
            if len(row_list) >= 1024:
                temp_df = create_df(row_list=row_list)
                temp_df.to_csv(output_meta, header=True, index=False, mode="a+") if output_meta else print (temp_df)
                row_list = []
        except Exception as e:
            tb.print_tb(e.__traceback__)
            

    if row_list:
        temp_df = create_df(row_list)
        temp_df.to_csv(output_meta, header=False, index=False, mode="a+") if output_meta else print (temp_df)
