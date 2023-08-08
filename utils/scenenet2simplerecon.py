import math
import numpy as np
import scenenet_pb2 as sn
import os
import shutil
from tqdm import tqdm


def position_to_np_array(position,homogenous=False):
    if not homogenous:
        return np.array([position.x,position.y,position.z])
    return np.array([position.x,position.y,position.z,1.0])

def normalize(v):
    return v/np.linalg.norm(v)

def interpolate_poses(start_pose,end_pose,alpha):
    assert alpha >= 0.0
    assert alpha <= 1.0
    camera_pose = alpha * position_to_np_array(end_pose.camera)
    camera_pose += (1.0 - alpha) * position_to_np_array(start_pose.camera)
    lookat_pose = alpha * position_to_np_array(end_pose.lookat)
    lookat_pose += (1.0 - alpha) * position_to_np_array(start_pose.lookat)
    timestamp = alpha * end_pose.timestamp + (1.0 - alpha) * start_pose.timestamp
    pose = sn.Pose()
    pose.camera.x = camera_pose[0]
    pose.camera.y = camera_pose[1]
    pose.camera.z = camera_pose[2]
    pose.lookat.x = lookat_pose[0]
    pose.lookat.y = lookat_pose[1]
    pose.lookat.z = lookat_pose[2]
    pose.timestamp = timestamp
    return pose

def world_to_camera_with_pose(view_pose):
    lookat_pose = position_to_np_array(view_pose.lookat)
    camera_pose = position_to_np_array(view_pose.camera)
    up = np.array([0,1,0])
    R = np.diag(np.ones(4))
    R[2,:3] = normalize(lookat_pose - camera_pose)
    R[0,:3] = normalize(np.cross(R[2,:3],up))
    R[1,:3] = -normalize(np.cross(R[0,:3],R[2,:3]))
    T = np.diag(np.ones(4))
    T[:3,3] = -camera_pose # -camera_pose 
    return np.linalg.inv(R.dot(T))

def camera_intrinsic_transform(vfov=45,hfov=60,pixel_width=320,pixel_height=240):
    camera_intrinsics = np.zeros((3,4))
    camera_intrinsics[2,2] = 1
    camera_intrinsics[0,0] = (pixel_width/2.0)/math.tan(math.radians(hfov/2.0))
    camera_intrinsics[0,2] = pixel_width/2.0
    camera_intrinsics[1,1] = (pixel_height/2.0)/math.tan(math.radians(vfov/2.0))
    camera_intrinsics[1,2] = pixel_height/2.0
    return camera_intrinsics


def image_size():
    #obtain image size
    height = 240
    width = 320
    return height, width

def colortodepth():
    
    #generate transform 1 by 12
    transform = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    return transform 

def num_frames():
    # number of images in a folder
    pass

if __name__ == "__main__":

    data_root_path = '/home/yohan-sl-intern/Documents/SceneNetRGBD-val/val'
    protobuf_path = 'data/scenenet_rgbd_val.pb'
    trajectories = sn.Trajectories()
    try:
        with open(protobuf_path,'rb') as f:
            trajectories.ParseFromString(f.read())
    except IOError:
        print('Scenenet protobuf data not found at location:{0}'.format(data_root_path))
        print('Please ensure you have copied the pb file to the data directory')

    filenames = os.listdir(data_root_path+'/0')

    sorted_filenames = sorted(filenames, key=lambda x: int(x))
  
    for i in tqdm(sorted_filenames[:10]): # 1000 for SceneNet val
        i = int(i)
        traj = trajectories.trajectories[i]
        render_path = traj.render_path 
        
     
        intrinsic_matrix = camera_intrinsic_transform()
      
        # generate pose folder with camera pose
        folder_path_pose = os.path.join(data_root_path,"0/"+render_path[2:5]+"/pose/")
        if not os.path.exists(folder_path_pose):
            os.makedirs(folder_path_pose)

        for idx,view in enumerate(traj.views):
            # Get camera pose
            ground_truth_pose = interpolate_poses(view.shutter_open,view.shutter_close,0.5)
            world_to_camera_matrix = world_to_camera_with_pose(ground_truth_pose)
            # Save camera pose in txt file
            with open(folder_path_pose+f"{idx}.txt", "w") as f:
                for row in world_to_camera_matrix[:,:]:
                    f.write(" ".join(str(x) for x in row) + "\n")
    
        # generate metadata txt file

        folder_path_metadata = os.path.join(data_root_path, "0/"+render_path[2:5]+'/')

        # intrinsics

        fx_color = intrinsic_matrix[0,0]
        fy_color = intrinsic_matrix[1,1]
        mx_color = intrinsic_matrix[0,2]
        my_color = intrinsic_matrix[1,2]

        fx_depth = fx_color
        fy_depth = fy_color
        mx_depth = mx_color
        my_depth = my_color

        colorHeight, colorWidth = image_size()
        depthHeight, depthWidth = image_size()
        
        # num images
        img_path = "/home/yohan-sl-intern/Documents/SceneNetRGBD-val/val/0/"+render_path[2:5]
        image_extension = (".jpg", ".png")
        numColorFrames = len([file for file in os.listdir(img_path+"/photo/") if file.endswith(image_extension)])
        numDepthFrames = numColorFrames

        # transform color to depth

        colorToDepthExtrinsics = colortodepth()
        numIMUmeasurements = 1632


        with open(folder_path_metadata + render_path[2:5]+ f".txt", 'w') as f:
            f.write(f'colorHeight = {colorHeight}\n')
            f.write(f'colorToDepthExtrinsics = {" ".join(str(x) for x in colorToDepthExtrinsics)}\n')
            f.write(f'colorWidth = {colorWidth}\n')
            f.write(f'depthHeight = {depthHeight}\n')
            f.write(f'depthWidth = {depthWidth}\n')
            f.write(f'fx_color = {fx_color}\n')
            f.write(f'fx_depth = {fx_depth}\n')
            f.write(f'fy_color = {fy_color}\n')
            f.write(f'fy_depth = {fy_depth}\n')
            f.write(f'mx_color = {mx_color}\n')
            f.write(f'mx_depth = {mx_depth}\n')
            f.write(f'my_color = {my_color}\n')
            f.write(f'my_depth = {my_depth}\n')
            f.write(f'numColorFrames = {numColorFrames}\n')
            f.write(f'numDepthFrames = {numDepthFrames}\n')
            f.write(f'numIMUmeasurements = {numIMUmeasurements}\n')


        # # Set the paths to the input and output directories

        input_dir = '/home/yohan-sl-intern/Documents/SceneNetRGBD-val'
        output_dir = '/home/yohan-sl-intern/Documents/SceneNetRGBD-val/scans_test'

        # set the path to "val/0/0" directory
        val_dir_path = os.path.join(input_dir, 'val', '0', render_path[2:5])
        scene_id_dir = '{:04d}'.format(int(render_path[2:5]))
        scene_id = '{:04}'.format(int(render_path[2:5]))

        scene_dir_path = os.path.join(output_dir, 'scene{}'.format(scene_id_dir))
        os.makedirs(scene_dir_path)

        # Copy the scene description file to the new directory
        scene_desc_path_old = os.path.join(val_dir_path, '{}.txt'.format(render_path[2:5]))
        scene_desc_path_new = os.path.join(scene_dir_path, 'scene{}.txt'.format(scene_id))
        shutil.copy(scene_desc_path_old, scene_desc_path_new)

        # Create a new dir for intrinsics files in the scene directory

        intrinsic_dir = os.path.join(scene_dir_path, 'intrinsic')
        os.makedirs(intrinsic_dir)
        
        # Read the scene description file from the txt file for the current scan

        metadata_file = os.path.join(scene_desc_path_new)
        with open(metadata_file, 'r') as f:
            lines = f.readlines()
            # print(lines[5].strip().split(' ')[2])
            fx = float(lines[5].strip().split(' ')[2])
            fy = float(lines[7].strip().split(' ')[2])
            cx = float(lines[8].strip().split(' ')[2])
            cy = float(lines[9].strip().split(' ')[2])

        K = np.array([[fx, 0, cx, 0],
                      [0, fy, cy, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        
        # Save the intrinsics matrix to the intrinsic directory as both depth and color
        np.savetxt(os.path.join(intrinsic_dir, 'intrinsic_depth.txt'), K)
        np.savetxt(os.path.join(intrinsic_dir, 'intrinsic_color.txt'), K)


        # Create a new dir for the sensor data in the scene directory

        sensor_data_dir_path = os.path.join(scene_dir_path, 'sensor_data')
        os.makedirs(sensor_data_dir_path)

        # Loop through the image and pose files in the "photo" and "pose" directories

        for subdir in ['photo', 'pose']:
            subdir_path = os.path.join(val_dir_path, subdir)
            print(val_dir_path)

            files = os.listdir(subdir_path)
            sorted_files = sorted(files, key=lambda x: int(x.split('.')[0]))

            # for i, filename in enumerate(sorted(os.listdir(subdir_path))):
            for i, filename in enumerate(sorted_files):

                # extract the frame id from the file name
                frame_id = str(i).zfill(4)
            
                # Copy the image or pose file to the sensor data directory with the new name
            
                if subdir == 'photo':
                    new_subdir = 'color'
                    file_ext = '.jpg'
                else:
                    new_subdir = 'pose'
                    file_ext = '.txt'
                
                old_file_path = os.path.join(subdir_path, filename)
                new_file_name = 'frame-{}.{}{}'.format(frame_id, new_subdir, file_ext)
                new_file_path = os.path.join(sensor_data_dir_path, new_file_name)
                shutil.copy(old_file_path, new_file_path)

        # Loop through the depth image files in the "depth" directory
        depth_dir_path = os.path.join(val_dir_path, 'depth')
        depth_files = os.listdir(depth_dir_path)
        sorted_files = sorted(depth_files, key=lambda x: int(x.split('.')[0]))

        for i, filename in enumerate(sorted_files):
            # Extract the frame ID from the file name
            frame_id = str(i).zfill(4)

            # Copy the depth image file to the sensor data directory with the new name
            old_file_path = os.path.join(depth_dir_path, filename)
            new_file_name = 'frame-{}.depth.png'.format(frame_id)
            new_file_path = os.path.join(sensor_data_dir_path, new_file_name)
            shutil.copy(old_file_path, new_file_path)

