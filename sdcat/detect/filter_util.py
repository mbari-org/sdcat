import hashlib
import uuid

import cv2
import pandas as pd
import torch
from sahi.postprocess.combine import nms

from sdcat.cluster.utils import crop_square_image
from sdcat.logger import warn, info


def process_image(image_filename, save_path_base, save_path_det_raw, save_path_det_filtered, save_path_det_roi,  save_path_viz, min_area, max_area, min_saliency, class_agnostic, save_roi, roi_size):
    try:
        img_color = cv2.imread(image_filename.as_posix())
        height, width = img_color.shape[:2]

        # Path to save final csv file with the detections
        pred_out_csv = save_path_det_filtered / f'{image_filename.stem}.csv'

        df_combined = pd.DataFrame()
        for csv_file in save_path_det_raw.rglob(f'{image_filename.stem}*csv'):
            df = pd.read_csv(csv_file, sep=',')
            df_combined = pd.concat([df_combined, df])

        if df_combined.empty:
            warn(f'No detections found in {image_filename}')
            return None

        df_combined = df_combined[(df_combined['area'] > min_area) & (df_combined['area'] < max_area)]
        df_combined = df_combined[(df_combined['saliency'] > min_saliency) | (df_combined['saliency'] == -1)]

        if df_combined.empty:
            warn(f'No detections found in {image_filename}')
            return None

        pred_list = torch.tensor(df_combined[['x', 'y', 'xx', 'xy', 'score']].values.tolist())
        nms_pred_idx = nms(pred_list, 'IOU', 0.1)

        df_final = df_combined.iloc[nms_pred_idx].reset_index(drop=True)
        df_final['saliency'] = df_combined['saliency'].iloc[nms_pred_idx].reset_index(drop=True)
        df_final['area'] = df_combined['area'].iloc[nms_pred_idx].reset_index(drop=True)

        pred_list = df_final[['x', 'y', 'xx', 'xy', 'score', 'class']].values
        for p in pred_list:
            if class_agnostic:
                img_color = cv2.rectangle(img_color, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), (81, 12, 51), 3)
                img_color = cv2.putText(img_color, f'{p[5]} {p[4]:.2f}', (int(p[0]), int(p[1])),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                md5_hash = hashlib.md5(p[5].encode())
                hex_color = md5_hash.hexdigest()[:6]
                r, g, b = (int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:], 16))
                color = (r % 256, g % 256, b % 256)
                img_color = cv2.rectangle(img_color, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), color, 3)
                img_color = cv2.putText(img_color, p[5], (int(p[0]), int(p[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        output_path = save_path_viz / f"{image_filename.stem}.jpg"
        cv2.imwrite(str(output_path), img_color)

        df_final['cluster'] = -1
        df_final['image_path'] = image_filename.as_posix()
        df_final['image_width'] = width
        df_final['image_height'] = height
        df_final['x'] /= width
        df_final['y'] /= height
        df_final['xx'] /= width
        df_final['xy'] /= height
        df_final['w'] = df_final['xx'] - df_final['x']
        df_final['h'] = df_final['xy'] - df_final['y']

        if save_roi:
            # Add in a column for the unique crop name for each detection with a unique id
            # create a unique uuid based on the md5 hash of the box in the row
            df_final['crop_path'] = df_combined.apply(lambda row: f"{save_path_det_roi}/{uuid.uuid5(uuid.NAMESPACE_DNS, str(row['x']) + str(row['y']) + str(row['xx']) + str(row['xy']))}.png", axis=1)
            # Crop the square image
            for index, row in df_final.iterrows():
                crop_square_image(row, roi_size)

        # Save DataFrame to CSV file including image_width and image_height columns
        info(f'Detections saved to {pred_out_csv}')
        df_final.to_csv(pred_out_csv.as_posix(), index=False, header=True)
        if save_roi: info(f"ROI crops saved in {save_path_det_roi}")

        save_stats = save_path_base / 'stats.txt'
        with save_stats.open('w') as sf:
            sf.write(f"Statistics for {image_filename}:\n")
            sf.write("----------------------------------\n")
            sf.write(f"Total number of bounding boxes: {df_final.shape[0]}\n")
            sf.write(
                f"Total number of images with (bounding box) detections found: {df_final['image_path'].nunique()}\n")
            sf.write(
                f"Average number of bounding boxes per image: {df_final.shape[0] / df_final['image_path'].nunique()}\n")
            sf.write(f"Average width of bounding boxes: {df_final['w'].mean() * width}\n")
            sf.write(f"Average height of bounding boxes: {df_final['h'].mean() * height}\n")
            sf.write(f"Average area of bounding boxes: {df_final['area'].mean()}\n")
            sf.write(f"Average score of bounding boxes: {df_final['score'].mean()}\n")
            sf.write(f"Average saliency of bounding boxes: {df_final['saliency'].mean()}\n")
        return len(df_final)
    except Exception as e:
        warn(f'Error processing {image_filename}: {e}')
        return 0