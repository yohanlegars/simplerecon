import sys
import pyzed.sl as sl
import numpy as np
import cv2
from pathlib import Path
import os
import shutil
import argparse

# example command to extract every 50 images from svo and generate the format for running simplerecon eval.
"""
python3 svo2simplerecon.py \
--path_svo /path/to/filename.svo\
--output_path /path/to/images_output_dir\
--resolution 640 480
--step 50
"""

def progress_bar(percent_done, bar_length=50):

    done_length = int(bar_length * percent_done / 100)
    bar = '=' * done_length + '-' * (bar_length - done_length)
    sys.stdout.write('[%s] %f%s\r' % (bar, percent_done, '%'))
    sys.stdout.flush()


def generate_dataset_directory(root_path, scene):

    scene_path = os.path.join(root_path, "scans_test", scene)
    intrinsic_path = os.path.join(scene_path, "intrinsic")
    data_path = os.path.join(scene_path, "sensor_data")

    if os.path.exists(root_path):
        # Directory already exists, so delete it
        shutil.rmtree(root_path)
        print('Existing directory deleted.')

    # Create the directories
    os.makedirs(scene_path)
    os.makedirs(intrinsic_path)
    os.makedirs(data_path)

    print('New directory created.')

    return data_path, intrinsic_path

def get_matrix(pose):
    matrix = np.zeros((4,4))

    for i in range(4):
         for j in range(4):
              matrix[i,j] = pose[i,j]
    
    matrix[:,3] = matrix[:,3]*0.001
    
    return matrix

def get_intrinsic_left_cam(info_cam):

    fx = info_cam.camera_configuration.calibration_parameters.left_cam.fx
    fy = info_cam.camera_configuration.calibration_parameters.left_cam.fy
    cx = info_cam.camera_configuration.calibration_parameters.left_cam.cx
    cy = info_cam.camera_configuration.calibration_parameters.left_cam.cy

    intrinsic_mat = np.array([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    return intrinsic_mat, fx, fy, cx, cy

def get_num_frames(directory_path='', format = 'jpg'):
    
    jpg_count = 0

    for filename in os.listdir(directory_path):
        if filename.lower().endswith(".jpg"):
            jpg_count += 1
    
    return jpg_count

         
def main(args):

    svo_input_path = args.path_svo
    output_path_root = args.output_path

    init_params = sl.InitParameters()
    init_params.set_from_svo_file(str(svo_input_path))
    init_params.svo_real_time_mode = False
    init_params.coordinate_units = sl.UNIT.MILLIMETER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP # coordinate for simplerecon

    zed = sl.Camera()

    camera_pose = sl.Pose()
    pose_data = sl.Transform()

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        sys.stdout.write(repr(err))
        zed.close()
        exit()

    res = sl.Resolution(args.resolution[0], args.resolution[1])
   
    info_cam = zed.get_camera_information(resizer = res)

    intrinsic_left_cam_mat, fx, fy, cx, cy = get_intrinsic_left_cam(info_cam)

   
    py_transform = sl.Transform()
    tracking_parameters = sl.PositionalTrackingParameters(_init_pos = py_transform)
    err = zed.enable_positional_tracking(tracking_parameters)

    if err != sl.ERROR_CODE.SUCCESS:
         exit(1)

    left_image = sl.Mat()
    depth_image = sl.Mat()
   
    rt_param = sl.RuntimeParameters()

    sys.stdout.write("Converting SVO... Use Ctrl-C to interrupt conversion.\n")

    nb_frames = zed.get_svo_number_of_frames()
    
    svo_file = os.path.basename(args.path_svo)
    svo_name = svo_file.split("_")[1]
    svo_path = os.path.join(output_path_root, 'scans_test',svo_name)
    data_path, intrinsic_path = generate_dataset_directory(output_path_root, svo_name)

    intrinsic_depth = os.path.join(intrinsic_path,"intrinsic_depth.txt")
    intrinsic_color = os.path.join(intrinsic_path, "intrinsic_color.txt")

    np.savetxt(intrinsic_depth, intrinsic_left_cam_mat)
    np.savetxt(intrinsic_color, intrinsic_left_cam_mat)

    step_size = args.step
    frame_id = 0

    while True:
        if zed.grab(rt_param) == sl.ERROR_CODE.SUCCESS:
                svo_position = zed.get_svo_position()
               # zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.IMAGE)
                 
                 
                if svo_position % int(step_size) == 0 or svo_position == 0:

                    frame_id_str = f"{frame_id:06}"

                # Retrieve SVO images (left + depth)
                    zed.retrieve_image(left_image, sl.VIEW.LEFT, resolution = res)
                    zed.retrieve_measure(depth_image, sl.MEASURE.DEPTH, resolution = res)
                    zed.get_position(camera_pose, sl.REFERENCE_FRAME.WORLD)
                    image_timestamp = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
                    image_filename = os.path.join(data_path,"frame-%s.jpg" % (frame_id_str + '.color')) 
                    depth_filename = os.path.join(data_path, "frame-%s.png" % (frame_id_str + '.depth'))
                    pose_filename = os.path.join(data_path, "frame-%s.txt" % (frame_id_str + '.pose'))
                    pose = camera_pose.pose_data()
            
                    pose = get_matrix(pose)
                                    
                    left_rgb = cv2.cvtColor(left_image.get_data(), cv2.COLOR_RGBA2RGB)

                   
                    cv2.imwrite(str(image_filename), left_rgb)
                    cv2.imwrite(str(depth_filename), depth_image.get_data().astype(np.uint16))
                    np.savetxt(pose_filename, pose) 
               
                    progress_bar((svo_position+1) / nb_frames*100, 30)

                    frame_id +=1

        
                if svo_position >= (nb_frames -1):
                    sys.stdout.write("\nSVO end has been reached. Exiting now.\n")
                    break
    

    sys.stdout.write("Extraction Done")

    num_frames = get_num_frames(directory_path=data_path)

    height = args.resolution[0]
    width = args.resolution[1]
   
    colorToDepthExtrinsics = np.identity(4)

    
    with open(svo_path+'/'+svo_name+f".txt",'w') as f:
         f.write(f'colorHeight = {height}\n')
         f.write(f'colorToDepthExtrinsics = {" ".join(str(x) for array in colorToDepthExtrinsics for x in array)}\n')
         f.write(f'colorWidth = {width}\n')
         f.write(f'depthHeight = {height}\n')
         f.write(f'depthWidth = {width}\n')
         f.write(f'fx_color = {fx}\n')
         f.write(f'fx_depth = {fx}\n')
         f.write(f'fy_color = {fy}\n')
         f.write(f'fy_depth = {fy}\n')
         f.write(f'mx_color = {cx}\n')
         f.write(f'mx_depth = {cx}\n')
         f.write(f'my_color = {cy}\n')
         f.write(f'my_depth = {cy}\n')
         f.write(f'numColorFrames = {num_frames}\n')
         f.write(f'numDepthFrames = {num_frames}\n')
         f.write(f'numIMUmeasurements = {num_frames}\n')
         
         
         
if __name__ == "__main__":
     
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Script to extract left images from SVO file")

    # Add command-line arguments
    parser.add_argument("--path_svo", type=str, help="Path to the SVO file", required=True)
    parser.add_argument("--output_path", type=str, help="Path to the output images directory", required=True)
    parser.add_argument("--resolution", type=int, nargs=2, help="Resolution (width, height)", required=True)
    parser.add_argument("--step", type=int, help="Step for extraction", default=50)

    # Parse the command-line arguments
    args = parser.parse_args()

    main(args)

