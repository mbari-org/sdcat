# sdcat, Apache-2.0 license
# Filename: sdcat/ml/saliency.py
# Description:  Miscellaneous saliency functions for detecting targets in images.
import multiprocessing
import tempfile

import cv2
import numpy as np
import torch
import pandas as pd
from sahi.postprocess.combine import nms

from sdcat.logger import debug, info
from pathlib import Path

save = True


def process_contour(out_csv_file: str, contours: np.ndarray, gray: np.ndarray, min_blob_area: int, max_blob_area: int,
                    scale: float = 1.0):
    """
    Process a single contour. Save the results to a csv file.
    This is a separate function, so it can be run in parallel.
    :param out_csv_file: The csv path to save the results
    :param contours:  List of contours
    :param gray: Grayscale image to use for extracting basic statistics
    :param min_blob_area: The minimum blob area to consider
    :param max_blob_area: The maximum blob area to consider
    :param scale: The scale factor used to rescale the image
    :return:
    """
    df = pd.DataFrame()
    for c in contours:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        mask = np.zeros(gray.shape, np.uint8)

        if min_blob_area < area < max_blob_area:
            cv2.drawContours(mask, [c], 0, 255, -1)
            masked_image = cv2.bitwise_and(gray, gray, mask=mask)
            mean_intensity = cv2.mean(masked_image, mask=mask)[0]
            max_intensity = np.max(masked_image[mask != 0])
            std_intensity = np.std(masked_image[mask != 0])

            # Small blobs that are very bright are likely reflections, so ignore them
            if max_intensity > 200. and area < 10000 and std_intensity < 30.:
                debug(
                    f'Skipping possible reflection blob: {area}, mean: {mean_intensity}, max: {max_intensity}, std: {std_intensity}, area: {area}')
                continue
            if std_intensity > 8.:
                debug(
                    f'Found blob area: {area}, mean: {mean_intensity}, max: {max_intensity}, std: {std_intensity}, area: {area}')
                df = pd.concat([df, pd.DataFrame({
                    'image_path': '',
                    'class': 'Unknown',
                    'score': 0.1,
                    'area': area,
                    'mean_intensity': mean_intensity,
                    'std_intensity': std_intensity,
                    'x': int(x / scale),
                    'y': int(y / scale),
                    'xx': int((x + w) / scale),
                    'xy': int((y + h) / scale),
                    'w': int(w / scale),
                    'h': int(h / scale),
                },
                    index=[0])])

    if len(df) > 0:
        # Save the df to a csv file for later processing
        df.to_csv(out_csv_file, mode='w', header=True, index=False)


def homomorphic_filtering(image_in: np.ndarray, alpha: float = 0.0003125, beta: float = 1.00):
    """
    Apply homomorphic filtering to an image to remove illumination artifacts
    :param image_in:
    :param alpha:
    :param beta:
    :return:
    """
    img_float32 = np.float32(image_in)
    log_image = np.log1p(img_float32)
    fft = np.fft.fft2(log_image)
    rows, cols = img_float32.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    r = 30
    center = (crow, ccol)
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
    mask[mask_area] = 0
    fft_shift = fft * mask
    img_back = np.fft.ifft2(fft_shift)
    img_back = np.exp(np.real(img_back))
    img_filtered = (img_back - np.min(img_back)) / (np.max(img_back) - np.min(img_back))
    img_filtered = np.power(img_filtered, alpha) * beta
    img_filtered = np.uint8(img_filtered * 255)
    return img_filtered


def extract_blobs(saliency_map: np.ndarray, img_gray: np.ndarray, img_color: np.ndarray, show=False) -> pd.DataFrame:
    """
    Extract blobs from a saliency map
    :param saliency_map: normalized saliency map
    :param img_gray: grayscale image used to extract basic statistics
    :param img_color: color image used when showing the results
    :param show: True to show the results
    :return: pandas dataframe of blobs
    """

    # Find contours in the salient object mask
    contours, _ = cv2.findContours(saliency_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

    # Show the contours on the image
    if show:
        result_image = img_color.copy()
        cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)
        cv2.imshow(f'{len(contours)} Contours', result_image)
        cv2.waitKey(0)

    # Iterate through the contours and filter out by size and standard deviation of intensity
    min_blob_area = 10
    max_blob_area = int(img_gray.shape[0] * img_gray.shape[1] / 2)  # 50% of the image area for the largest blob

    # Do the work in parallel to speed up the processing
    num_processes = min(multiprocessing.cpu_count(), len(contours) // 100)
    num_processes = max(1, num_processes)
    info(f'Using {num_processes} processes to compute {len(contours)} 100 at a time ...')

    # Work in a temporary directory
    df = pd.DataFrame()
    with tempfile.TemporaryDirectory() as temp_path:
        temp_path = Path(temp_path)
        gray = img_gray
        # process_contour((temp_path / f'det.csv').as_posix(), contours, gray, min_blob_area, max_blob_area) # useful for debugging
        with multiprocessing.Pool(num_processes) as pool:
            args = [((temp_path / f'{i}_det.csv').as_posix(),
                     contours[i:i + 100],
                     gray,
                     min_blob_area,
                     max_blob_area)
                    for i in range(0, len(contours), 100)]
            pool.starmap(process_contour, args)
            pool.close()

        # Combine the results
        info(f'Combining blob detection results')
        for csv in temp_path.glob('*.csv'):
            df = pd.concat([df, pd.read_csv(csv)])
            csv.unlink()

    if show:
        result_image = img_color.copy()
        for i, row in df.iterrows():
            x = row['x']
            y = row['y']
            w = row['w']
            h = row['h']
            std_intensity = row['std_intensity']
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (81, 12, 51), 2)
            std_intensity_int = np.round(std_intensity * 100)
            cv2.putText(result_image, f'{std_intensity_int}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow(f'{len(df)} Bounding Boxes around Salient Regions', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return df


def fine_grained_saliency_pyramid(image_path: str, show=False) -> np.ndarray:
    """
    Compute a fine-grained saliency map and normalize it
    :param image_path:  path to the image
    :param show:  True to show the results
    :return: normalized saliency map
    """
    # Load the original image
    image = cv2.imread(image_path)

    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Blur the saturation channel to remove noise
    img_hsv[:, :, 1] = cv2.GaussianBlur(img_hsv[:, :, 1], (15, 15), 0)

    # Build an image pyramid, starting with the original image, then downsampling it 2 times
    pyramid = [img_hsv]
    for i in range(1, 3):
        pyramid.append(cv2.pyrDown(pyramid[-1]))

    # Initialize an empty saliency map
    saliency_map = np.zeros_like(image[:, :, 0], dtype=np.float32)

    # Process each level of the pyramid
    for level, img in enumerate(pyramid):
        # Extract the hue, saturation, and value channels
        hue = img[:, :, 0]
        saturation = img[:, :, 1]
        value = img[:, :, 2]

        # Run homomorphic filtering on each channel
        filtered_hue = homomorphic_filtering(hue)
        filtered_saturation = saturation
        filtered_value = homomorphic_filtering(value)

        blurred_hue = cv2.GaussianBlur(filtered_hue, (3, 3), 0)
        blurred_saturation = cv2.GaussianBlur(filtered_saturation, (5, 5), 0)
        blurred_value = cv2.GaussianBlur(filtered_value, (3, 3), 0)

        # Calculate the center and surround regions for each channel
        center_hue = cv2.GaussianBlur(blurred_hue, (0, 0), 2)
        surround_hue = blurred_hue - center_hue

        center_saturation = cv2.GaussianBlur(blurred_saturation, (3, 3), 2)
        surround_saturation = blurred_saturation - center_saturation

        center_value = cv2.GaussianBlur(blurred_value, (0, 0), 2)
        surround_value = blurred_value - center_value

        # Normalize the values to the range [0, 255]
        center_hue = cv2.normalize(center_hue, None, 0, 255, cv2.NORM_MINMAX)
        surround_hue = cv2.normalize(surround_hue, None, 0, 255, cv2.NORM_MINMAX)

        center_saturation = cv2.normalize(center_saturation, None, 0, 255, cv2.NORM_MINMAX)
        surround_saturation = cv2.normalize(surround_saturation, None, 0, 255, cv2.NORM_MINMAX)

        center_value = cv2.normalize(center_value, None, 0, 255, cv2.NORM_MINMAX)
        surround_value = cv2.normalize(surround_value, None, 0, 255, cv2.NORM_MINMAX)

        # Combine the center and surround regions for each channel
        # Reduce the weight of the saturation channel
        saliency_hue = (center_hue - surround_hue) * (3 ** level)
        saliency_saturation = (center_saturation - surround_saturation) / (3 ** level)
        saliency_value = (center_value - surround_value) * (3 ** level)

        # Combine the saliency maps into the final saliency map
        saliency_level = (saliency_hue + saliency_saturation + saliency_value) / 3

        # Resize the saliency map to the original image size
        saliency_level = cv2.resize(saliency_level, (image.shape[1], image.shape[0]))

        # Accumulate the saliency map at each level
        saliency_map += saliency_level

    # Normalize the final saliency map
    saliency_map = cv2.normalize(saliency_map, None, 0, 255, cv2.NORM_MINMAX)

    # Blur the saliency map to remove noise
    saliency_map = cv2.GaussianBlur(saliency_map, (15, 15), 0)

    if show:
        # Display the original image and the saliency map
        cv2.imshow('Original Image', image)
        cv2.imshow('Fine-Grained Saliency Map', saliency_map.astype(np.uint8))
        if save: cv2.imwrite('my_saliency_map.jpg', saliency_map.astype(np.uint8))
        cv2.waitKey(0)

    # Threshold the saliency map
    mean, std = cv2.meanStdDev(saliency_map)
    info(f'Mean: {mean}, Std: {std}')
    _, saliency_map = cv2.threshold(saliency_map.astype(np.uint8), int(mean - 1.5 * std), 255,
                                    cv2.ADAPTIVE_THRESH_MEAN_C)

    if show:
        # Display the thresholded saliency map
        cv2.imshow('Thresholded Saliency Map', saliency_map.astype(np.uint8))
        cv2.waitKey(0)

    # Invert the saliency map so that the salient regions are white
    saliency_map = cv2.bitwise_not(saliency_map)

    # Connect the salient regions
    kernel = np.ones((5, 5), np.uint8)
    saliency_map = cv2.dilate(saliency_map, kernel, iterations=5)
    return saliency_map


if __name__ == '__main__':
    # Read all images from the tests/data/kelpflow directory
    # Get the path of the this file
    # test_path = Path(__file__).parent.parent.parent / 'tests' / 'data' / 'bird'
    # test_path = Path(__file__).parent.parent.parent / 'tests' / 'data' / 'whale'
    # test_path = Path(__file__).parent.parent.parent / 'tests' / 'data' / 'kelpflow'
    # test_path = Path( __file__).parent.parent.parent / 'tests' / 'data' / 'glare'
    # test_path = Path(__file__).parent.parent.parent / 'tests' / 'data' / 'dolphin'
    # test_path = Path(__file__).parent.parent.parent / 'tests' / 'data' / 'all'
    test_path = Path(__file__).parent.parent.parent / 'tests' / 'data' / 'jelly_DSC'
    show = True
    for image_path in test_path.glob('*.JPG'):
        saliency_map = fine_grained_saliency_pyramid(image_path.as_posix())
        img_gray = cv2.cvtColor(cv2.imread(image_path.as_posix()), cv2.COLOR_RGB2GRAY)
        img_color = cv2.imread(image_path.as_posix())
        df = extract_blobs(saliency_map, img_gray, img_color, show=show)

        print(f'Found {len(df)} blobs in {image_path}')

        # Convert the x, y, xx, xy, score columns to a list of Tensors that is Shape: [num_boxes,5].
        pred_list = df[['x', 'y', 'xx', 'xy', 'score']].values.tolist()
        pred_list = torch.tensor(pred_list)  # convert the list of Tensors to a list of Tensors
        print(f'Running NMS on {len(df)} predictions')
        nms_pred_idx = nms(pred_list, 'IOU', 0.1)  # run nms with 0.1 IOU
        nms_pred_list = pred_list[nms_pred_idx].tolist()
        df_all = pd.DataFrame(nms_pred_list, columns=['x', 'y', 'xx', 'xy', 'score'])
        print(f'{len(df_all)} predictions found after NMS')

        # Plot boxes on the input frame
        pred_list = df_all[['x', 'y', 'xx', 'xy', 'score']].values.tolist()
        for p in pred_list:
            # Draw a rectangle with red line borders of thickness of 3 px
            image_raw = cv2.rectangle(img_color,
                                      (int(p[0]), int(p[1])),
                                      (int(p[2]), int(p[3])), (0, 0, 255), 3)

        cv2.imshow('final', img_color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
