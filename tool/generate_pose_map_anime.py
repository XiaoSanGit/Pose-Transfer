import numpy as np
import pandas as pd
import json
import os
import cv2
MISSING_VALUE = -1
def normalization(img_dir, kps_dir, out_img_dir,out_kps_dir, img_size=(255,255)):
    # TODO center point should be the neck !!!!!!!!!!
    # Original image size is 1920,1080
    names = os.listdir(img_dir)
    cnt = len(names)
    for i in range(cnt):
        print('processing %d / %d ...' % (i, cnt))
        img_ori = cv2.imread(os.path.join(img_dir, names[i]))
        kp_array = np.load(os.path.join(kps_dir, names[i]))

        center_point = kp_array[1] # TODO neck is 1??
        center_x, center_y = center_point
        cropped = img_ori[0:1080, center_y - 540:center_y + 540] # 裁剪坐标为[y0:y1, x0:x1]
        normalized = cv2.resize(cropped,img_size)
        cv2.imwrite(os.path.join(out_img_dir, names[i]), normalized)

        cropped_kps = kp_array[0:1080, center_point - 540:center_point + 540]  # 裁剪坐标为[y0:y1, x0:x1]
        normalized_kps = cv2.resize(cropped_kps, img_size)
        np.save(os.path.join(out_kps_dir, names[i]), normalized_kps)

        # if vis is not None and kpss[idx].split("_")[3] in vis:
        #     img_name = "D:\\download_cache\\anime_data\\train\\" + ".".join(names[i].split(".")[:-1])
        #     plot_points(img_name, modified_kps, ".".join(names[i].split(".")[:-1]))

def load_pose_cords_from_strings(y_str, x_str):
    y_cords = json.loads(y_str)
    x_cords = json.loads(x_str)
    return np.concatenate([np.expand_dims(y_cords, -1), np.expand_dims(x_cords, -1)], axis=1)

def cords_to_map(cords, img_size, sigma=6):
    result = np.zeros(img_size + cords.shape[0:1], dtype='float32')
    for i, point in enumerate(cords):
        if point[0] == MISSING_VALUE or point[1] == MISSING_VALUE:
            continue
        xx, yy = np.meshgrid(np.arange(img_size[1]), np.arange(img_size[0]))
        result[..., i] = np.exp(-((yy - point[1]) ** 2 + (xx - point[0]) ** 2) / (2 * sigma ** 2))  # the location of kps is lighted up and decrease as distribution
    return result

def compute_pose(image_size, ori_dir, savePath):
    ori_kps = os.listdir(ori_dir)
    cnt = len(ori_kps)
    for i in range(cnt):
        print('processing %d / %d ...' %(i, cnt))
        kp_array = np.load(os.path.join(ori_dir,ori_kps[i]))
        name = ori_kps[i]
        print(savePath, name)
        file_name = os.path.join(savePath, name)
        pose = cords_to_map(kp_array, image_size)
        np.save(file_name, pose)

if __name__ == '__main__':
    # img_dir = r'../anime_data/train' #raw image path
    img_size = (256, 192)
    ori_dir = r'D:/download_cache/anime_data/normK_s'  # pose annotation path
    save_path = r'D:/download_cache/anime_data/trainK'  # path to store pose maps
    compute_pose(img_size, ori_dir, save_path)