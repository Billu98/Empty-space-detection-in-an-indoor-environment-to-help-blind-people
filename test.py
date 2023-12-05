

import torch
# Save raw
# from zoedepth.utils.misc import save_raw_16bit
from PIL import Image
# from zoedepth.utils.misc import colorize
import numpy as np
import cv2
import argparse
# import os

yolo = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
# Zoe_N
model_zoe_n = torch.hub.load(".", "ZoeD_N", source="local", pretrained=True)
# sample prediction
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
zoe = model_zoe_n.to(DEVICE)


def predict_image(image):
    yolo_predictions = yolo([image])
    depth_numpy = zoe.infer_pil(Image.fromarray(image), pad_input=False)

    image = postprocess_predictions(image, yolo_predictions,
                                    depth_numpy, meter_threshold=3)
    return image


def postprocess_predictions(image, yolo_predictions, depth_map,
                            meter_threshold):
    # image = np.array(img)
    yolo_df = yolo_predictions.pandas().xyxy[0]
    predictions = yolo_df[yolo_df["name"].isin(["table", "chair", "door"])]
    predictions["width"] = predictions["xmax"] - predictions["xmin"]
    predictions["height"] = predictions["ymax"] - predictions["ymin"]
    predictions["center_x"] = predictions["xmin"] + predictions["width"]/2
    predictions["center_y"] = predictions["ymin"] + predictions["height"]/2
    predictions[predictions["center_x"] > depth_map.shape[1]]["center_x"] = depth_map.shape[1] - 1
    predictions[predictions["center_y"] > depth_map.shape[0]]["center_y"] = depth_map.shape[0] - 1
    h, w, c = image.shape

    cv2.line(image, (w//2, h), (w//2, 0), (255, 0, 0), 2)
    left_origin = (50, 50)
    right_origin = (w - 100, 50)
    image = cv2.putText(image, 'Left', left_origin,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                        cv2.LINE_AA)
    image = cv2.putText(image, 'Right', right_origin,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                        cv2.LINE_AA)
    distances = []
    for x, y in zip(predictions["center_x"].values.astype(int),
                    predictions["center_y"].values.astype(int)):
        print(image.shape, depth_map.shape, (y, x))
        distances.append(depth_map[y, x])

    # distances = [depth_map[x, y] ]
    predictions["distance"] = distances
    print("min max depth", depth_map.min(), depth_map.max())
    for i, row in predictions.iterrows():
        xmin = int(row["xmin"])
        xmax = int(row["xmax"])
        ymin = int(row["ymin"])
        ymax = int(row["ymax"])
        name = row["name"]
        score = row["confidence"]
        distance = row["distance"]
        print(xmin, xmax, ymin, ymax, name, score, distance)
        if distance <= meter_threshold:
            color = (0, 0, 255)

        else:
            color = (0, 255, 0)

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        image = cv2.putText(image, str(np.round(distance, 2)), (xmin,
                            ymin - 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 0), 2,
                            cv2.LINE_AA)

    return image


def run_demo_image(image_path, output_path):
    # To test on image please specify the path of image here
    image = cv2.imread(image_path)
    predicted_image = predict_image(image)
    cv2.imwrite(output_path, predicted_image)


def run_demo_video(video_path, output_path):
    # Read the video from specified path
    cam = cv2.VideoCapture(video_path)
    ret, frame = cam.read()
    h, w, c = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (w, h))
    i = -1
    while (ret):

        # reading from frame
        ret, frame = cam.read()
        i += 1
        if i % 20 != 0:
            continue
        if not ret:
            break

        # Sometimes the video is 180 degree rotated so normalize it first.
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        frame = predict_image(frame)
        out.write(frame)
        cv2.imshow("frame", frame)
        k = cv2.waitKey(1)
        if k == 27:
            break

    # Release all space and windows once done
    cam.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Demo for zoe depth')
    parser.add_argument('input_path', help='Input path either video or image, example /path/to/input/image.jpg or /path/to/input/video.mp4')
    parser.add_argument('out_path', help='Output path, example /path/to/output/out.jpg or /path/to/output/out.mp4')
    args = parser.parse_args()
    if args.input_path.endswith(".jpg") or args.input_path.endswith(".jpeg"):
        run_demo_image(args.input_path, args.out_path)
    elif args.input_path.endswith(".mp4") or args.input_path.endswith(".avi"):
        run_demo_video(args.input_path, args.out_path)
    else:
        print("Path not supported")


