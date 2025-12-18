import torch
# from diffusers import AutoPipelineForInpainting
# from diffusers.utils import load_image, make_image_grid
import h5py
import numpy as np
from tqdm import tqdm
import torchvision.transforms.functional as TVF
import matplotlib.pyplot as plt
import egomimic
import os
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from egomimic.utils.egomimicUtils import AlohaFK, JakaFK, ee_pose_to_cam_pixels, ARIA_INTRINSICS, EXTRINSICS, ee_pose_to_cam_frame, cam_frame_to_cam_pixels, draw_dot_on_frame
import cv2
import random
from PIL import Image

def get_bounds(binary_image):
    # gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Threshold the grayscale image to create a binary image
    # _,binary_image = cv2.threshold(gray_image, 254, 255, cv2.THRESH_BINARY)
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables to store max and min x and y values
    max_x = max_y = 0
    min_x = min_y = float('inf')

    if len(contours) == 0:
        return None, None, None, None

    # Loop through all contours to find max and min x and y values
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)
        min_x = min(min_x, x)
        min_y = min(min_y, y)

    return min_x, max_x, min_y, max_y

def line_on_hand(images, masks, arm):
    """
    Draw a line on the hand
    images: np.array of shape (n, h, w, c)
    masks: np.array of shape (n, h, w)
    arm: str, "left" or "right"
    """
    overlayed_imgs = np.zeros_like(images)
    for k, (image, mask) in enumerate(zip(images, masks)):
        min_x, max_x, min_y, max_y = get_bounds(mask.astype(np.uint8))
        if min_x is None:
            overlayed_imgs[k] = image
            continue

        gamma = 0.8
        alpha = 0.2
        scale = max_y - min_y
        min_x = int(max_x + gamma * (min_x - max_x))
        min_y = int(max_y + gamma * (min_y - max_y))
        max_x = int(max_x - scale * alpha)

        if arm == "right":
            line_image = cv2.line(image.copy(), (min_x,min_y),(max_x,max_y),color=(255,0,0), thickness=25)
        elif arm == "left":
            line_image = cv2.line(image.copy(), (min_x,max_y),(max_x,min_y),color=(255,0,0), thickness=25)
        else:
            raise ValueError(f"Invalid arm: {arm}")
        overlayed_imgs[k] = line_image
    
    return overlayed_imgs

def get_valid_points(points, img_shape):
    height, width = img_shape[:2]
    
    # Stack list of arrays into a single (N, 2) array
    # Assuming input is list of (1,2) arrays, we use vstack
    pts_arr = np.vstack(points) 
    
    # Create a boolean mask for X and Y bounds
    # x is column 0, y is column 1
    x_valid = (pts_arr[:, 0] >= 0) & (pts_arr[:, 0] < width)
    y_valid = (pts_arr[:, 1] >= 0) & (pts_arr[:, 1] < height)
    
    # Combine masks (Point must be valid in both X and Y)
    valid_mask = x_valid & y_valid
    
    # Apply mask to select points
    input_point = pts_arr[valid_mask]
    
    # Create labels (all 1s) matching the number of valid points
    input_label = np.ones(input_point.shape[0], dtype=int)
    
    # Handle the empty case if needed (NumPy usually handles empty slices fine, 
    # but if you specifically need empty arrays instead of (0,2) shape):
    if len(input_point) == 0:
        return np.array([]), np.array([])
        
    return input_point, input_label

def get_negative_prompts(all_joint_points, img_shape, num_points=10, box_offset=50):
    """
    Generates random negative points that are strictly OUTSIDE the bounding box 
    formed by the robot joints + an offset buffer.
    """
    h, w = img_shape[:2]
    
    # Stack all points to find the global min/max
    if isinstance(all_joint_points, list):
        # Filter out any empty arrays if segments were skipped
        valid_arrays = [x for x in all_joint_points if x.size > 0]
        if not valid_arrays:
            return np.empty((0, 2)), np.empty((0,))
        all_pts = np.vstack(valid_arrays)
    else:
        all_pts = all_joint_points

    if all_pts.shape[0] == 0:
            return np.empty((0, 2)), np.empty((0,))

    # 1. Calculate Bounding Box of the Robot
    min_x, min_y = np.min(all_pts, axis=0)
    max_x, max_y = np.max(all_pts, axis=0)
    
    # 2. Expand Box by Offset (The "Danger Zone")
    danger_x1 = max(0, min_x - box_offset)
    danger_y1 = max(0, min_y - box_offset)
    danger_x2 = min(w, max_x + box_offset*2.5)
    danger_y2 = min(h, max_y + box_offset)
    
    neg_points = []
    
    # 3. Generate Random Points OUTSIDE the Danger Zone
    # We try a set number of times to find valid points
    max_attempts = num_points * 5 
    attempts = 0
    
    while len(neg_points) < num_points and attempts < max_attempts:
        attempts += 1
        rx = np.random.randint(0, w)
        ry = np.random.randint(0, h)
        
        # Check if point is INSIDE the box
        is_inside_box = (rx >= danger_x1 and rx <= danger_x2) and \
                        (ry >= danger_y1 and ry <= danger_y2)
        
        if not is_inside_box:
            neg_points.append([rx, ry])
            
    if not neg_points:
        return np.empty((0, 2)), np.empty((0,))

    return np.array(neg_points), np.zeros(len(neg_points))


def interpolate_dense_path(points, density=5):
    """
    Interpolates points between an arbitrary number of sequential keypoints.
    
    Args:
        points: Numpy array of shape (N, 2) containing N keypoints (x, y).
        density: Number of intermediate points to generate between each pair.
        
    Returns:
        dense_points: (M, 2) array of original + interpolated points.
        labels: (M,) array of 1s (positive prompts).
    """
    # Ensure input is correct shape (N, 2)
    points = np.atleast_2d(points)
    
    if points.shape[0] < 2:
        # If only 1 point, cannot interpolate, return as is
        return points, np.array([1])

    dense_points_list = []

    # Iterate through each pair of points (0->1, 1->2, 2->3...)
    for i in range(len(points) - 1):
        p_start = points[i]
        p_end = points[i+1]
        
        # Create linear interpolation
        # num=density+2 includes the start and end points
        segment = np.linspace(p_start, p_end, num=density+2)
        
        # If this is not the first segment, remove the first point 
        # (because it was the last point of the previous segment)
        if i > 0:
            segment = segment[1:]
            
        dense_points_list.append(segment)

    # Stack all segments into one array (M, 2)
    dense_points = np.vstack(dense_points_list)
    
    # Generate labels (all 1s because they are all part of the robot)
    labels = np.ones(dense_points.shape[0])
    
    return dense_points, labels

class SAM:
    def __init__(self):
        sam2_checkpoint = os.path.join(
            os.path.dirname(egomimic.__file__), "resources/sam2.1_hiera_small.pt"
        )
        model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")

        predictor = SAM2ImagePredictor(sam2_model)

        self.predictor = predictor
        self.fk = AlohaFK()
    
    def get_robot_mask_line_batched_from_qpos(self, images, qpos, extrinsics, intrinsics, arm="right"):
        """
            images: tensor (B, H, W, 3)
            qpos: B, 7
        """
        px_dict = self.project_joint_positions_to_image(qpos, extrinsics, intrinsics, arm=arm)
        mask_images, line_images =  self.get_robot_mask_line_batched(images, px_dict, arm=arm)
        return mask_images, line_images
    
    def get_hand_mask_line_batched(self, imgs, ee_poses, intrinsics, arm, debug=False):
        ## both hands
        if ee_poses.shape[-1] == 6 and arm == "both":
            prompts_l = cam_frame_to_cam_pixels(ee_poses[:, :3], intrinsics)[:, :2]
            prompts_r = cam_frame_to_cam_pixels(ee_poses[:, 3:], intrinsics)[:, :2]

            masked_img_l, raw_masks_l = self.get_hand_mask_batched(imgs, prompts_l, neg_prompts=prompts_r)
            mask = np.arange(640)[None, :] < prompts_l[:, [0]] + 100
            mask = mask[:, None, :]
            raw_masks_l = raw_masks_l & mask

            masked_img_r, raw_masks_r = self.get_hand_mask_batched(imgs, prompts_r, neg_prompts=prompts_l)
            mask = np.arange(640)[None, :] > prompts_r[:, [0]] - 100
            mask = mask[:, None, :]
            raw_masks_r = raw_masks_r & mask


            masked_imgs = imgs[:].copy()
            masked_imgs[raw_masks_l] = 0
            masked_imgs[raw_masks_r] = 0
            # masked_imgs[raw_masks_l, 0] = 255
            # masked_imgs[raw_masks_r, 1] = 255
            raw_masks = raw_masks_l | raw_masks_r
            
            overlayed_imgs = line_on_hand(masked_imgs, raw_masks_r, "right")
            overlayed_imgs = line_on_hand(overlayed_imgs, raw_masks_l, "left")

        elif ee_poses.shape[-1] == 3:
            prompts_l = None
            prompts_r = cam_frame_to_cam_pixels(ee_poses, intrinsics)[:, :2]

            masked_imgs, raw_masks = self.get_hand_mask_batched(imgs, prompts_r)

            overlayed_imgs = line_on_hand(masked_imgs, raw_masks, arm)
        else:
            raise ValueError(f"Invalid shape for ee_poses: {ee_poses.shape}")
        
        #cv2 imsave the masked_img_l and masked_img_r, bgr to rgb as well
        if debug:
            breakpoint()
            for j in range(overlayed_imgs.shape[0]):
                overlayed_imgs[j] = cv2.cvtColor(overlayed_imgs[j], cv2.COLOR_BGR2RGB)
                if prompts_l is not None:
                    overlayed_imgs[j] = draw_dot_on_frame(overlayed_imgs[j], prompts_l[[j]], palette="Set1")
                if prompts_r is not None:
                    overlayed_imgs[j] = draw_dot_on_frame(overlayed_imgs[j], prompts_r[[j]], palette="Set2")
                cv2.imwrite(f"./overlays/overlayed_img_{j}.png", overlayed_imgs[j])
                cv2.imwrite(f"./overlays/masked_img_{j}.png", cv2.cvtColor(masked_imgs[j], cv2.COLOR_BGR2RGB))
                cv2.imwrite(f"./overlays/mask_{j}.png", raw_masks[j].astype(np.uint8) * 255)
        
        return overlayed_imgs, masked_imgs, raw_masks

    def get_hand_mask_batched(self, images, pos_prompts, neg_prompts=None):
        """
            images: tensor (B, H, W, 3)
            pos_prompts: tensor (B, 2)

            returns: raw_masks
        """
        masked_imgs = np.zeros_like(images)
        raw_masks = np.zeros((images.shape[0], 480, 640)).astype(bool)

        for k in range(images.shape[0]):
            img = images[k]

            if neg_prompts is not None:
                output = self.get_hand_mask(img, pos_prompts[[k]], neg_prompt=neg_prompts[[k]])
            else:
                output = self.get_hand_mask(img, pos_prompts[[k]])
            if output is None:
                continue
            else:
                masked_img, raw_mask = output

            raw_masks[k] = raw_mask
            masked_imgs[k] = masked_img

        return masked_imgs, raw_masks

    def get_hand_mask(self, img, pos_prompt, neg_prompt=None):
        """
        Image:
        Pos_prompt: (2) array containing x, y coordinates of the point
        Neg_prompt: (2) array containing x, y coordinates of the point
        
        Returns:
        Masked image, mask, score, logits
        """
        input_point = pos_prompt
        if input_point[0, 0] > 640 or input_point[0, 1] > 480 or input_point[0, 0] < 0 or input_point[0, 1] < 0:
            masked_img = img
            return None
        input_label = np.array([1])
        if neg_prompt is not None:
            input_point = np.concatenate([input_point, neg_prompt], axis=0)
            input_label = np.array([1, 0])

        masked_img, masks, scores, logits = self.get_mask(img.copy(), input_point, input_label)

        raw_mask = masks[0].astype(bool)
        return masked_img, raw_mask



    def get_mask(self, image, points, label):
        """
        Image:
        Points: (N, 2) array containing x, y coordinates of the points
        Label: (N) array containing 0 or 1 specifying negative or positive prompts
        
        Returns:
        Masked image, masks, scores, logits
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image)

        img_point = image.copy()
        for i, (point, lb) in enumerate(zip(points, label)):
            img_point = cv2.circle(
                img_point,
                (int(point[0]), int(point[1])),
                radius=5,
                color=(255, 255-(i*50), 0*lb),
            )
        cv2.imwrite("img_point.png", img_point)
        cv2.waitKey(1)

        masks, scores, logits = self.predictor.predict(
            point_coords=points,
            point_labels=label,
            multimask_output=False,
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]
        
        masked_img = image
        masked_img[masks[0] == 1] = 0
        cv2.imwrite("masked_img.png", image)
        return masked_img, masks, scores, logits
    



    def project_single_joint_position_to_image(self, qpos, extrinsics, intrinsics, arm="right"):
        joint_pos = self.fk.chain.forward_kinematics(qpos, end_only=False)
        fk_positions = joint_pos['vx300s/ee_gripper_link'].get_matrix()[:, :3, 3]
        wrist_positions = joint_pos['vx300s/wrist_link'].get_matrix()[:, :3, 3]
        elbow_positions = joint_pos['vx300s/upper_forearm_link'].get_matrix()[:, :3, 3]
        arm_positions = joint_pos['vx300s/ee_arm_link'].get_matrix()[:, :3, 3]
        lower_forearm_positions = joint_pos['vx300s/lower_forearm_link'].get_matrix()[:, :3, 3]


        fk_positions = ee_pose_to_cam_frame(fk_positions, extrinsics)[:, :3]
        wrist_positions = ee_pose_to_cam_frame(wrist_positions, extrinsics)[:, :3]
        elbow_positions = ee_pose_to_cam_frame(elbow_positions, extrinsics)[:, :3]
        arm_positions = ee_pose_to_cam_frame(arm_positions, extrinsics)[:, :3]
        lower_forearm_positions = ee_pose_to_cam_frame(lower_forearm_positions, extrinsics)[:, :3]

        px_val_gripper = cam_frame_to_cam_pixels(fk_positions, intrinsics)[:, :2]
        px_val_wrist = cam_frame_to_cam_pixels(wrist_positions, intrinsics)[:, :2]
        px_val_elbow = cam_frame_to_cam_pixels(elbow_positions, intrinsics)[:, :2]
        px_val_arm = cam_frame_to_cam_pixels(arm_positions, intrinsics)[:, :2]
        px_val_lower_forearm = cam_frame_to_cam_pixels(lower_forearm_positions, intrinsics)[:, :2]

        if arm == "right":
            px_dict = {
                "px_val_gripper_right": px_val_gripper,
                "px_val_wrist_right": px_val_wrist,
                "px_val_elbow_right": px_val_elbow,
                "px_val_arm_right": px_val_arm,
                "px_val_lower_forearm_right": px_val_lower_forearm,
            }
        elif arm == "left":
            px_dict = {
                "px_val_gripper_left": px_val_gripper,
                "px_val_wrist_left": px_val_wrist,
                "px_val_elbow_left": px_val_elbow,
                "px_val_arm_left": px_val_arm,
                "px_val_lower_forearm_left": px_val_lower_forearm,
            }
        else:
            raise ValueError("Arm must be either 'right' or 'left'")
        return px_dict

    def project_joint_positions_to_image(self, qpos, extrinsics, intrinsics, arm="right"):
        if arm == "both":
            ## process left
            px_dict_left = self.project_single_joint_position_to_image(qpos[:, :6], extrinsics["left"], intrinsics, arm="left")
            ## process right
            px_dict_right = self.project_single_joint_position_to_image(qpos[:, 7:13], extrinsics["right"], intrinsics, arm="right")
            return {**px_dict_left, **px_dict_right}
        elif arm == "right":
            if qpos.shape[1] == 14:
                qpos = qpos[:, 7:]
            return self.project_single_joint_position_to_image(qpos[:, :-1], extrinsics["right"], intrinsics, arm="right")
        elif arm == "left":
            if qpos.shape[1] == 14:
                qpos = qpos[:, :7]
            return self.project_single_joint_position_to_image(qpos[:, :-1], extrinsics["left"], intrinsics, arm="left")
        else:
            raise ValueError("Arm must be 'both, 'right' or 'left'")

    def get_robot_mask_line_batched(self, images, px_dict, arm="right"):
        line_images = np.zeros_like(images)
        mask_images = np.zeros_like(images)

        if arm == "both":
            pt1_left = px_dict["px_val_wrist_left"]
            pt2_left = px_dict["px_val_gripper_left"]
            pt3_left = px_dict["px_val_arm_left"] #(pt1_left + pt2_left)/2
            pt1_right = px_dict["px_val_wrist_right"]
            pt2_right = px_dict["px_val_gripper_right"]
            pt3_right = px_dict["px_val_arm_right"] #(pt1_right + pt2_right)/2
        elif arm == "left":
            pt1_left = px_dict["px_val_wrist_left"]   # link_6
            pt2_left = px_dict["px_val_gripper_left"] # gripper
            pt3_left = px_dict["px_val_arm_left"]     # link_2                (pt1_left + pt2_left)/2
        elif arm == "right":
            pt1_right = px_dict["px_val_wrist_right"]
            pt2_right = px_dict["px_val_gripper_right"]
            pt3_right =  px_dict["px_val_arm_right"] #(pt1_right + pt2_right)/2


        for i,image in enumerate(images[:]):
            # get the point between px_val_lower_forearm
            # if i == 100:
            #     break
            masked_img = image.copy()

            ## init arrays
            input_point_left = np.array([])
            input_label_left = np.array([])
            input_point_right = np.array([])
            input_label_right = np.array([])

            ## Get Valid Points
            if arm == "both" or arm == "left":
            ## process left
                left1, left2, left3 = pt1_left[[i]], pt2_left[[i]], pt3_left[[i]]
                input_point_left, input_label_left = get_valid_points((left1, left2, left3), line_images[0].shape)
            if arm == "both" or arm == "right":
                ## process right
                right1, right2, right3 = pt1_right[[i]], pt2_right[[i]], pt3_right[[i]]
                input_point_right, input_label_right = get_valid_points((right1, right2, right3), line_images[0].shape)

            # print("i", i)
            # print("left", left1, left2, left3)
            # print("right", right1, right2, right3)

            ## Set Input Points
            if input_point_left.size == 0 and input_point_right.size > 0:
                input_point = input_point_right
                input_label = input_label_right
            elif input_point_right.size == 0 and input_point_left.size > 0:
                input_point = input_point_left
                input_label = input_label_left
            elif  input_point_right.size > 0 and input_point_left.size > 0:       
                input_point = np.concatenate([input_point_left, input_point_right], axis=0)
                input_label = np.concatenate([input_label_left, input_label_right], axis=0)
            else:
                mask_images[i] = masked_img
                line_images[i] = masked_img.copy()
                continue
            
            if input_point_left.size > 0:
                masked_img, masks, scores, logits = self.get_mask(masked_img, input_point_left, input_label_left)
            if input_point_right.size > 0:
                masked_img, masks, scores, logits = self.get_mask(masked_img, input_point_right, input_label_right)
            # masked_img, masks, scores, logits = self.get_mask(image, input_point, input_label)

            line_img = masked_img.copy()

            if arm == "both" or arm == "left":
                ## draw left line
                line_img = cv2.line(
                    line_img,
                    (int(px_dict["px_val_gripper_left"][i, 0]), int(px_dict["px_val_gripper_left"][i, 1])),
                    (int(px_dict["px_val_elbow_left"][i, 0]), int(px_dict["px_val_elbow_left"][i, 1])),
                    color=(255, 0, 0),
                    thickness=25
                )
            if arm == "both" or arm == "right":
                ## draw right line
                line_img = cv2.line(
                    line_img,
                    (int(px_dict["px_val_gripper_right"][i, 0]), int(px_dict["px_val_gripper_right"][i, 1])),
                    (int(px_dict["px_val_elbow_right"][i, 0]), int(px_dict["px_val_elbow_right"][i, 1])),
                    color=(255, 0, 0),
                    thickness=25
                )


            mask_images[i] = masked_img
            line_images[i] = line_img

            # masked_img = draw_dot_on_frame(line_img, input_point, palette="tab10")
            # breakpoint()
            
            # for pt in input_point:
            #     image = cv2.circle(image, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)
            #     masked_img = cv2.circle(masked_img, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)
            #     line_img = cv2.circle(line_img, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)

            # # print("WRITING")
            # cv2.imwrite(f"/nethome/dpatel756/flash/egoPlay_unified/EgoPlay/mimicplay/scripts/masking/overlays/masked_img_{i}.png", cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB))


        return mask_images, line_images

class SAMJaka(SAM):

    def __init__(self, use_sam3=False):

        if not use_sam3:
            super().__init__()
        else:
            EFFICENT_SAM3 = True

            if EFFICENT_SAM3:
                self.model = None
                self.init_efficient_sam3()
            else:
                self.init_sam3()
            
        self.fk = JakaFK("jaka_s12_pinza_nera_real_joints.urdf")

    
    def init_efficient_sam3(self):
        from sam3.model_builder import build_efficientsam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        # Load model
        self.model = build_efficientsam3_image_model(
            checkpoint_path="/run/media/Dati/Sviluppo/Università/Tesi/EgoMimic/egomimic/resources/efficient_sam3_efficientvit_s_sa_1b_1p.pt",
            backbone_type="efficientvit",
            model_name="b0",
            #text_encoder_type="MobileCLIP-S1"
        )

        # Process image and predict
        self.predictor = Sam3Processor(self.model)

        self.get_mask = self.get_mask_efficient_sam3

    
    def init_sam3(self):
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        # Load the model
        sam3_root ='external/sam3'
        bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
        model = build_sam3_image_model(bpe_path=bpe_path)
        self.predictor = Sam3Processor(model)

        self.get_mask = self.get_mask_sam3
    
    def project_single_joint_position_to_image(self, qpos, extrinsics, intrinsics, arm="right"):
        # Forward Kinematics to get all link transforms
        joint_pos = self.fk.chain.forward_kinematics(qpos, end_only=False)
        
        base_positions     = joint_pos['Link_01'].get_matrix()[:, :3, 3]
        shoulder_positions = joint_pos['Link_02'].get_matrix()[:, :3, 3]
        elbow_positions    = joint_pos['Link_03'].get_matrix()[:, :3, 3]
        forearm_positions  = joint_pos['Link_04'].get_matrix()[:, :3, 3]
        wrist_positions    = joint_pos['Link_05'].get_matrix()[:, :3, 3]
        hand_positions     = joint_pos['Link_06'].get_matrix()[:, :3, 3]
        fk_positions       = joint_pos['custom_ee_link'].get_matrix()[:, :3, 3]

        # 2. Transform to Camera Frame
        base_positions     = ee_pose_to_cam_frame(base_positions, extrinsics)[:, :3]
        shoulder_positions = ee_pose_to_cam_frame(shoulder_positions, extrinsics)[:, :3]
        elbow_positions    = ee_pose_to_cam_frame(elbow_positions, extrinsics)[:, :3]
        forearm_positions  = ee_pose_to_cam_frame(forearm_positions, extrinsics)[:, :3]
        wrist_positions    = ee_pose_to_cam_frame(wrist_positions, extrinsics)[:, :3]
        hand_positions     = ee_pose_to_cam_frame(hand_positions, extrinsics)[:, :3]
        fk_positions       = ee_pose_to_cam_frame(fk_positions, extrinsics)[:, :3]

        # 3. Project to Pixels
        px_val_base     = cam_frame_to_cam_pixels(base_positions, intrinsics)[:, :2]
        px_val_shoulder = cam_frame_to_cam_pixels(shoulder_positions, intrinsics)[:, :2]
        px_val_elbow    = cam_frame_to_cam_pixels(elbow_positions, intrinsics)[:, :2]
        px_val_forearm  = cam_frame_to_cam_pixels(forearm_positions, intrinsics)[:, :2]
        px_val_wrist    = cam_frame_to_cam_pixels(wrist_positions, intrinsics)[:, :2]
        px_val_hand     = cam_frame_to_cam_pixels(hand_positions, intrinsics)[:, :2]
        px_val_gripper  = cam_frame_to_cam_pixels(fk_positions, intrinsics)[:, :2]

        # 4. Pack into Dictionary
        suffix = f"_{arm}" 
        px_dict = {
            f"px_val_base{suffix}":     px_val_base,     # Link_01
            f"px_val_shoulder{suffix}": px_val_shoulder, # Link_02
            f"px_val_elbow{suffix}":    px_val_elbow,    # Link_03
            f"px_val_forearm{suffix}":  px_val_forearm,  # Link_04
            f"px_val_wrist{suffix}":    px_val_wrist,    # Link_05
            f"px_val_hand{suffix}":     px_val_hand,     # Link_06
            f"px_val_gripper{suffix}":  px_val_gripper,  # Custom EE
        }
        
        return px_dict

    def get_robot_mask_line_batched(self, images, px_dict, arm="right"):
        line_images = np.zeros_like(images)
        mask_images = np.zeros_like(images)

        if arm == "both":
            pass
        elif arm == "left" or arm == "right":
            suffix = f"_{arm}"
            # Order: Base -> Shoulder -> Elbow -> Forearm -> Wrist -> Hand -> Gripper
            pt1 = px_dict[f"px_val_base{suffix}"]
            pt2 = px_dict[f"px_val_shoulder{suffix}"]
            pt3 = px_dict[f"px_val_elbow{suffix}"]
            pt4 = px_dict[f"px_val_forearm{suffix}"]
            pt5 = px_dict[f"px_val_wrist{suffix}"]
            pt6 = px_dict[f"px_val_hand{suffix}"]
            pt7 = px_dict[f"px_val_gripper{suffix}"]

        for i, image in enumerate(tqdm(images)):
            masked_img = image.copy()

            # Order: Base -> Shoulder -> Elbow -> Forearm -> Wrist -> Hand -> Gripper
            p_base     = pt1[[i]]
            p_shoulder = pt2[[i]]
            p_elbow    = pt3[[i]]
            p_forearm  = pt4[[i]]
            p_wrist    = pt5[[i]]
            p_hand     = pt6[[i]]
            p_gripper  = pt7[[i]]
            
            # Stack them in physical order
            keypoints = np.vstack([p_base, p_shoulder, p_elbow, p_forearm, p_wrist, p_hand]) # remove p_gripper
            
            input_point = np.array([])
            input_label = np.array([])

            # Interpolate if points are valid (not NaN)
            if not np.isnan(keypoints).any():
                # interpolation adds intermediate points for SAM to track thin links better
                density = 0  # Number of points between each keypoint
                input_point, input_label = interpolate_dense_path(keypoints, density=density)

                # Filter points that are out of image bounds
                input_point, input_label = get_valid_points(input_point, image.shape)

            if input_point.size > 0:
                # Add negative prompts (background) to help SAM differentiate
                neg_points, neg_labels = get_negative_prompts(
                    input_point, 
                    images[i].shape, 
                    num_points=0,   # Add 8 random background points
                    box_offset=60 
                )

                final_points = np.concatenate([input_point, neg_points], axis=0)
                final_labels = np.concatenate([input_label, neg_labels], axis=0)
                
                # Generate Mask
                masked_img, masks, scores, logits = self.get_mask(masked_img, final_points, final_labels)

            # Visualization: Draw simple skeleton line
            line_img = masked_img.copy()

            skeleton_points = [p_base, p_wrist]
            for j in range(len(skeleton_points) - 1):
                p_start = (int(skeleton_points[j][0, 0]), int(skeleton_points[j][0, 1]))
                p_end   = (int(skeleton_points[j+1][0, 0]), int(skeleton_points[j+1][0, 1]))
                cv2.line(line_img, p_start, p_end, color=(255, 0, 0), thickness=15)

            mask_images[i] = masked_img
            line_images[i] = line_img

        return mask_images, line_images
    

    def get_mask_sam3(self, image, points, label):

        # 1. Prepare Image
        # Ensure image is RGB for the model
        image_rgb = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        image_rgb = Image.fromarray(image_rgb, mode="RGB")
        inference_state = self.predictor.set_image(image_rgb)

        # 3. Run Inference with Text Prompt
        output = self.predictor.set_text_prompt(
            state=inference_state, 
            prompt="A red and silver articulated robotic arm in the foreground"
        )
        
        # masks shape is typically (N, H, W) or (N, 1, H, W)
        masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
        
        # 4. Process the Mask
        if len(masks) == 0:
            return image, masks, scores, boxes

        # We take the mask with the highest score (index 0 usually)
        best_mask_tensor = masks[0]
        
        # Convert Tensor to Numpy
        if isinstance(best_mask_tensor, torch.Tensor):
            best_mask = best_mask_tensor.cpu().numpy()
        else:
            best_mask = best_mask_tensor
            
        # Remove extra dimensions (e.g. 1x640x480 -> 640x480)
        best_mask = best_mask.squeeze()

        # Apply Mask to Image
        masked_img = image.copy()
        masked_img[best_mask] = 0  # Set background to black

        cv2.imwrite("masked_img.png", masked_img)

        return masked_img, masks, scores, boxes
    

    def get_mask_efficient_sam3(self, image, points, label):
        """
        Image:
        Points: (N, 2) array containing x, y coordinates of the points
        Label: (N) array containing 0 or 1 specifying negative or positive prompts
        
        Returns:
        Masked image, masks, scores, logits
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inference_state = self.predictor.set_image(image)

        img_point = image.copy()
        for i, (point, lb) in enumerate(zip(points, label)):
            img_point = cv2.circle(
                img_point,
                (int(point[0]), int(point[1])),
                radius=5,
                color=(255, 255-(i*50), 0*lb),
            )
        cv2.imwrite("img_point.png", img_point)
        cv2.waitKey(1)

        masks, scores, logits = self.model.predict_inst(
            inference_state,
            point_coords=points,
            point_labels=label,
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]
        
        masked_img = image
        masked_img[masks[0] == 1] = 0
        cv2.imwrite("masked_img.png", image)
        return masked_img, masks, scores, logits
