import os
import cv2
import csv

from sort.sort import Sort
import numpy as np

# INPUT/OUTPUT DIRS
bmp_dir = "/home/tsiddi5/projects/def-bauer/tsiddi5/code/inference_output/better_cpu_run"
directory = "/home/tsiddi5/projects/def-bauer/tsiddi5/code/inference_output/better_cpu_run/labels"
frame_detect_output_dir = "/home/tsiddi5/projects/def-bauer/tsiddi5/code/1_object_tracking/tracked_objects_new.csv"


def extract_pixel_number(filename):
    # Assumes filename ends in ".txt" and is formatted like: img_left_rect_color_123.txt
    base = os.path.splitext(filename)[0]  # remove ".txt"
    number_str = base.split('_')[-1]      # get the last part after underscore
    return int(number_str)


# Get all matching files and sort by extracted number
txt_files = sorted(
    (f for f in os.listdir(directory)
     if f.endswith(".txt") and os.path.isfile(os.path.join(directory, f))),
    key=extract_pixel_number
)


def get_img_dims( image_path ):
    image_path = os.path.join(bmp_dir, image_path)
    # print(image_path)

    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    return [width, height]


def load_yolo_txt(file_path, img_width, img_height, default_confidence=0.99):
    """
    Reads a YOLO-format label file and converts to [x1, y1, x2, y2, confidence]
    """
    file_path = os.path.join( directory, file_path )

    detections = []
    class_ids = []
    with open(file_path, 'r') as f:
        for line in f:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            
            # Convert from normalized YOLO format to pixel coordinates
            x_center *= img_width
            y_center *= img_height
            width *= img_width
            height *= img_height
            
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            
            # Append as [x1, y1, x2, y2, confidence]
            detections.append([x1, y1, x2, y2, default_confidence])
            class_ids.append( int(class_id) )

    return class_ids, np.array(detections)



tracker = Sort()  # Initialize SORT

i=0
with open( frame_detect_output_dir, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["frame_number", "class_id", "object_id", "x1", "y1", "x2", "y2"])


    for f in txt_files:
        # finding bmp image equivalent to .txt prcoessed yolo file
        curr_frame = os.path.splitext(f)[0].split("/")[-1].split("_")[-1]

        bmp_path = os.path.splitext(f)[0] + ".bmp"
        bmp_path = bmp_path.replace("/labels/", "/")

        img_width, img_height = get_img_dims( bmp_path )
        class_ids, detections = load_yolo_txt(f,img_width,img_height)  # numpy array of shape [N, 5] → [x1, y1, x2, y2, score]
        

        np_array = np.array( detections )
        
        # Track objects
        tracked_objects = tracker.update(np_array)


        ind = 0
        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = map(int, obj)
            curr_class_id = class_ids[ind]
            ind+=1

            writer.writerow([curr_frame, curr_class_id, obj_id, x1, y1, x2, y2])


        i+=1
        if i==500:
            file.flush()
            i=0