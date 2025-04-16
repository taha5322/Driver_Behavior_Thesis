import os
import pandas as pd


# CONSTANTS
BASE_THRESHOLD = 100
CAR_THRESHOLD = 100
TRAFFIC_LIGHT_THRESH = 100
PEDES_THRESHOLD = 100



# STEP 0: EXTRACTING PIXEL NUMBERS

def extract_pixel_numbers_from_bmp(folder_path):
    pixel_numbers = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".bmp") and filename.startswith("img_left_rect_color_"):
            try:
                base = os.path.splitext(filename)[0]  # Remove .bmp
                number_str = base.split('_')[-1]      # Get the last part
                pixel_number = int(number_str)
                pixel_numbers.append(pixel_number)
            except ValueError:
                # In case the part after the last '_' is not an integer
                print(f"Warning: could not extract number from {filename}")

    return pixel_numbers

path = "/home/tsiddi5/projects/def-bauer/tsiddi5/code/inference_output/better_cpu_run"
frame_nums = extract_pixel_numbers_from_bmp(path)
frame_nums.sort()
frame_nums = set(frame_nums)

# print( frame_nums[13000:13010] )

# STEP 1: EXTRACTING VALID FRAMES W/ POG VALUES
def get_valid_gaze_frames(csv_path):
    # Read the CSV
    df = pd.read_csv(csv_path)

    # Initialize set for valid frame numbers
    valid_frames = set()

    i=0

    for index, row in df.iterrows():
        try:
            # Attempt to convert gaze coordinates to integers
            x = int(row['x_position'])
            y = int(row['y_position'])

            # If both succeed, add the frame number
            frame = int(row['f_number'])
            valid_frames.add(frame)

        except (ValueError, TypeError):
            # Skip rows where x or y are missing or invalid
            continue
        
    return valid_frames

path = "/home/tsiddi5/projects/def-bauer/tsiddi5/backup_gaze/copy_pog.csv"
valid_gaze_frames = get_valid_gaze_frames(path)


avail_frames = frame_nums.intersection(valid_gaze_frames)


# STEP 2: VALIDATING TIME WINDOWS
def is_valid_time_window(
    start_frame: int,
    end_frame: int,
    available_frames: set,
    fps: int = 30,
    max_missing_total_secs: float = 0.5,
    max_missing_consec_secs: float = 0.25,
) -> bool:
    """
    Check if a time window has acceptable levels of missing frames.

    Parameters:
    - start_frame, end_frame: define the window [start_frame, end_frame)
    - available_frames: set of frame numbers that actually exist
    - fps: frames per second (default = 30)
    - max_missing_total_secs: total allowed missing frames in seconds (default = 0.5s)
    - max_missing_consec_secs: max allowed consecutive missing frames in seconds (default = 0.25s)

    Returns:
    - True if the time window is valid, False if it should be discarded.
    """

    window_frames = list(range(start_frame, end_frame))
    present = [frame in available_frames for frame in window_frames]
    # print(window_frames)
    # print(present)

    # Calculate total number of missing frames
    missing_total = present.count(False)
    # print( missing_total )
    if missing_total > fps * max_missing_total_secs:
        return False

    # Check for longest consecutive gap
    max_consec_missing = 0
    current_gap = 0
    for is_present in present:
        if not is_present:
            current_gap += 1
            max_consec_missing = max(max_consec_missing, current_gap)
        else:
            current_gap = 0
    
    # print(max_consec_missing, fps * max_missing_consec_secs)

    if max_consec_missing > fps * max_missing_consec_secs:
        return False

    return True


# valid = is_valid_time_window(
#     start_frame=7012,
#     end_frame=7072,
#     available_frames=avail_frames,
#     fps=30,
#     max_missing_total_secs=1,
#     max_missing_consec_secs=0.25
# )

# print("Window valid?", valid)

# STEP 3: GAZE PATTERN MATCHING FOR VALID INTERVALS

# ideally should be estimate thresh based on object_id
def is_point_in_box(px, py, x1, y1, x2, y2, class_id, thresh=BASE_THRESHOLD):
    # try:
    #     assert all(isinstance(arg, int) for arg in [px, py, x1, y1, x2, y2]), \
    #         f"All arguments must be int {px, py, x1, y1, x2, y2}"
    # except:
    #     return False
    #print( px, py, x1, y1, x2, y2 )
    class_id = int(class_id)
    if class_id == 1:
        thresh = CAR_THRESHOLD
    elif class_id == 2:
        thresh = PEDES_THRESHOLD
        print( "pedestrian thresh" )
    elif class_id >= 3 and class_id <= 9:
        thresh = TRAFFIC_LIGHT_THRESH
        print("traffic light thresh")

    try: 
        px = int(px)
        py = int(py)
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        res = (x1 - thresh) <= px <= (x2 + thresh) and (y1 - thresh) <= py <= (y2 + thresh)
        # if res: 
        # #    print("true")
        # else:
        #     print("false")
        return res
    except:
        # print("excep", px, py)
        # print( px, py, x1, y1, x2, y2 )
        return False

def match_gaze_to_objects(
    gaze_df, tracked_df, maneuvers_df, available_frames,
    fps=30, durations=[2, 5, 10], min_fixation_frames=15
):
    results = []

    # Group tracked objects by frame
    objects_by_frame = tracked_df \
        .groupby('frame_number', group_keys=False) \
        .apply(lambda x: x.drop(columns=['frame_number']).to_dict('records')) \
        .to_dict()
    # tracked_df.groupby('frame_number').apply(lambda x: x.to_dict('records')).to_dict()
    # print(objects_by_frame)
    # available_frames = set(gaze_df['frame'])

    for _, maneuver in maneuvers_df.iterrows():
        maneuver_start = maneuver['Start']
        maneuver_type = maneuver['Type of the maneuver']

        # print("man_st, man_type", maneuver_start, maneuver_type  )


        for duration in durations:
            frame_start = maneuver_start - int(duration * fps)
            frame_end = maneuver_start

            if frame_start < 0:
                continue

            if not is_valid_time_window(frame_start, frame_end, available_frames, fps):
                # print("invalid time window")
                continue

            # print("valid time window with duration:",  duration )

            # Map gaze hits per obj_id
            gaze_hits = {}  # {obj_id: [frame1, frame2, ...]}

            for f in range(frame_start, frame_end):
                row = gaze_df[gaze_df['f_number'] == f]
                # print("curr frame", f)
                # print("row?",row)
                if row.empty or f not in objects_by_frame:
                    # print("empty row for this frame number in gaze_df")
                    continue

                px, py = row.iloc[0]['x_position'], row.iloc[0]['y_position']
                # print("gaze coors", px, py)
                # print( "objects by frame dict", objects_by_frame[f] )
                for obj in objects_by_frame[f]:
                    class_id = obj['class_id']
                    if is_point_in_box(px, py, obj['x1'], obj['y1'], obj['x2'], obj['y2'], class_id):
                        # print( objects_by_frame[f] )
                        # print(obj)
                        # print(gaze_hits)
                        obj_id = obj['object_id']
                        if obj_id not in gaze_hits:
                            gaze_hits[obj_id] = []
                        gaze_hits[obj_id].append(f)
                
                # print("gaze hits:",gaze_hits)
                # return None
                # if gaze_hits:
                #     print("")
                #     print("")
                #     print("hit",gaze_hits)
                # print(gaze_hits)
                # if type(px) == int and type(py)==int:

            # Evaluate fixations
            for obj_id, frames in gaze_hits.items():
                frames.sort()
                total_gaze_frames = len(frames)

                # Longest consecutive run of gaze
                max_consec = 1
                consec = 1
                for i in range(1, len(frames)):
                    if frames[i] == frames[i - 1] + 1:
                        consec += 1
                        max_consec = max(max_consec, consec)
                    else:
                        consec = 1

                if max_consec >= min_fixation_frames:
                    last_frame = frames[-1]
                    matched = [obj for obj in objects_by_frame.get(last_frame, []) if obj['object_id'] == obj_id]
                    if matched:
                        
                        class_id = matched[0]['class_id']
                        results.append({
                            'maneuver_start': maneuver_start,
                            'maneuver_type': maneuver_type,
                            'duration_prior_s': duration,
                            'fixated_obj_id': obj_id,
                            'fixated_class_id': class_id,
                            'fixation_frames': max_consec,
                            'total_gaze_frames': total_gaze_frames
                        })
                # print( "max consec", max_consec )

                # print("")   
            # return None
        #     print("")
        # print("")

    return pd.DataFrame(results)


maneuvers_df = pd.read_csv("/home/tsiddi5/projects/def-bauer/tsiddi5/backup_gaze/copy_maneuvers.csv")
maneuvers_df.columns = maneuvers_df.columns.str.strip()

gaze_df = pd.read_csv("/home/tsiddi5/projects/def-bauer/tsiddi5/backup_gaze/copy_pog.csv")
gaze_df.columns = gaze_df.columns.str.strip()

tracked_df = pd.read_csv("/home/tsiddi5/projects/def-bauer/tsiddi5/code/object_tracking/sort/tracked_objects_norm.csv")
tracked_df.columns = tracked_df.columns.str.strip()


final_df = match_gaze_to_objects(gaze_df, tracked_df, maneuvers_df, avail_frames)
# print(final_df)
final_df.to_csv("gaze_object_fixations_with_totals_norm.csv", index=False)
