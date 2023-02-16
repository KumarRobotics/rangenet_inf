#!/usr/bin/env python3
from yaml import load
from load_model import Load_Model
import rospy
from sensor_msgs.msg import PointCloud2, PointField, Image, CameraInfo
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import time
import rospkg

class Inference:
    def __init__(self):
        rospack = rospkg.RosPack()
        base_path = rospack.get_path("rangenet_inf")
        model_path = rospy.get_param("~model_path", default=base_path + "/model") + "/"
        self.skip_count_ = rospy.get_param("~skip_count", default=0)
        # only relevant for panoramic inputs
        self.pano_fov_deg_ = rospy.get_param("~pano_fov_deg", default=90)
        self.counter_ = 0

        self.unproj_n_points = None
        self.full_data = None
        self.load_obj_ = Load_Model(model_path)
        self.model_ = self.load_obj_.load_model()

        # Set model cuda parameters
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_ = False
        if torch.cuda.is_available() and torch.cuda.device_count() > 0 and \
                rospy.get_param("~gpu", default=True):
            cudnn.benchmark = True
            cudnn.fastest = True
            self.gpu_ = True
            self.model_.cuda()
        else:
            self.model_.cpu()

        # Push model into eval mode
        self.model_.eval()
        if self.gpu_:
            torch.cuda.empty_cache()

        self.info_ = None
        self.intrinsic_proj_ = None
        self.pc_fields_ = self.make_fields()

        # get stats
        self.means_ = torch.tensor(self.load_obj_.arch_configs["dataset"]["sensor"]["img_means"], device=self.device_)[:, None, None]
        self.stds_ = 1 / torch.tensor(self.load_obj_.arch_configs["dataset"]["sensor"]["img_stds"], device=self.device_)[:, None, None]

        # subs and pubs
        self.scan_sub_ = rospy.Subscriber("~scan_image", Image, callback=self.scan_cb, queue_size=5)
        self.info_sub_ = rospy.Subscriber("~scan_camera_info", CameraInfo, callback=self.info_cb, queue_size=5)
        self.pc_pub_ = rospy.Publisher("~sem_point_cloud", PointCloud2, queue_size=30)
        self.sem_image_pub_ = rospy.Publisher("~sem_image", Image, queue_size=5)

    def make_fields(self):
        fields = []
        field = PointField()
        field.name = 'x'
        field.count = 1
        field.offset = 0
        field.datatype = PointField.FLOAT32
        fields.append(field)

        field = PointField()
        field.name = 'y'
        field.count = 1
        field.offset = 4
        field.datatype = PointField.FLOAT32
        fields.append(field)

        field = PointField()
        field.name = 'z'
        field.count = 1
        field.offset = 8
        field.datatype = PointField.FLOAT32
        fields.append(field)

        field = PointField()
        field.name = 'intensity'
        field.count = 1
        field.offset = 12
        field.datatype = PointField.FLOAT32
        fields.append(field)
        return fields

    def info_cb(self, msg):
        if self.info_ is None:
            self.info_ = msg

            azis = np.arange(2*np.pi, 0, -np.pi * 2 / msg.width)
            vfov = np.radians(self.pano_fov_deg_)
            elevs = np.arange(vfov/2, -vfov/2-0.0001, -vfov / (msg.height - 1))
            self.intrinsic_proj_ = np.empty((msg.height, msg.width, 3), dtype=np.float32)
            # x,y,z
            self.intrinsic_proj_[:,:,0] = np.cos(elevs[:, None]) * np.cos(azis)
            self.intrinsic_proj_[:,:,1] = np.cos(elevs[:, None]) * np.sin(azis)
            self.intrinsic_proj_[:,:,2] = np.sin(elevs[:, None])

    def compute_normals(self, points_xyz):
        vec1 = np.roll(points_xyz, 1, axis=0) - points_xyz
        vec2 = np.roll(points_xyz, 1, axis=1) - points_xyz
        normals = np.cross(vec1, vec2, axis=2)
        normal_norms = np.linalg.norm(normals, axis=2)
        nonzero = (normal_norms > 0)
        normals[nonzero, :] /= normal_norms[nonzero, None]

        return normals

    def scan_cb(self, msg):
        # handle skipping frames
        if self.counter_ < self.skip_count_:
            self.counter_ += 1
            return
        self.counter_ = 0

        if self.info_ is None:
            rospy.logwarn("[RangeNet] Waiting for info message...")
            return
        rospy.loginfo("[RangeNet] =======================")

        start_t = time.time()
        # store the header in case it get updated before the inference finishes
        header = msg.header
        if msg.header.frame_id == 'odom':
            # pano
            scan_data = np.frombuffer(msg.data, dtype=np.uint16).reshape(self.info_.height, self.info_.width, -1).copy()
            range_intensity = scan_data[:, :, :2].astype(np.float32)
            range_intensity[:, :, 0] /= self.info_.R[0]
            points_xyz = range_intensity[:, :, 0, None] * self.intrinsic_proj_
        else:
            # sweep
            scan_data = np.frombuffer(msg.data, dtype=np.float32).reshape(self.info_.height, self.info_.width, -1).copy()
            # destagger
            for row, shift in enumerate(self.info_.D):
                scan_data[row, :, :] = np.roll(scan_data[row, :, :], int(shift), axis=0)

            points_xyz = np.nan_to_num(scan_data[:, :, :3], nan=0.0)
            range_intensity = np.frombuffer(scan_data[:, :, 3].tobytes(), dtype=np.uint16).reshape(
                    *points_xyz.shape[:2], -1).astype(np.float32)
            range_intensity[:, :, 0] /= self.info_.R[0]

        mask = points_xyz[:, :, 0] != 0

        # prep final points
        proj_range = torch.from_numpy(range_intensity[:, :, 0])
        proj_normals = torch.from_numpy(self.compute_normals(points_xyz)).permute(2, 0, 1)
        proj_remission = torch.from_numpy(range_intensity[:, :, 1])
        proj_mask = torch.from_numpy(mask)
        proj_labels = []

        proj = torch.cat([proj_range.unsqueeze(0),
                          proj_normals,
                          proj_remission.unsqueeze(0)])

        rospy.loginfo(f"[RangeNet] preproc took: {time.time() - start_t} sec")

        with torch.no_grad():
            start_t = time.time()

            proj = torch.unsqueeze(proj, 0)
            proj_mask = torch.unsqueeze(proj_mask, 0)

            if self.gpu_:
                proj_in = proj.cuda()
                proj_mask = proj_mask.cuda()

            proj_in = (proj_in - self.means_) * self.stds_
            proj_in = proj_in * proj_mask.float()

            proj_output = self.model_(proj_in, proj_mask)
            # subtract 1 so 0 is first index
            proj_argmax = proj_output[0].argmax(dim=0) - 1

            pred_np = proj_argmax.cpu().numpy()

            rospy.loginfo(f"[RangeNet] inference took: {time.time() - start_t} sec")

            start_t = time.time()
            # restagger to agree with pano from driver
            pred_np_stagger = pred_np.copy()
            for row, shift in enumerate(self.info_.D):
                pred_np_stagger[row, :] = np.roll(pred_np[row, :], -int(shift), axis=0)

            # publish single channel sem image
            sem_img_msg = Image()
            sem_img_msg.header = msg.header
            sem_img_msg.encoding = "mono8"
            sem_img_msg.height = pred_np.shape[0]
            sem_img_msg.width = pred_np.shape[1]
            sem_img_msg.step = sem_img_msg.width
            sem_img_msg.data = pred_np_stagger.astype(np.uint8).tobytes()
            self.sem_image_pub_.publish(sem_img_msg)

            # publish as point cloud msg, where intensity encodes segmentation results
            pc_msg = PointCloud2()
            pc_msg.header = msg.header
            pc_msg.width = points_xyz.shape[0]*points_xyz.shape[1]
            pc_msg.height = 1
            pc_msg.point_step = 16
            pc_msg.row_step = pc_msg.width * pc_msg.point_step
            pc_msg.fields = self.pc_fields_

            full_data = np.concatenate((points_xyz, pred_np[:, :, None]), axis=2).astype(np.float32)
            pc_msg.data = full_data.tobytes()
            self.pc_pub_.publish(pc_msg)
            rospy.loginfo(f"[RangeNet] pub took: {time.time() - start_t} sec")

if __name__ == '__main__':
    rospy.init_node("inference_node")
    inf = Inference()
    rospy.spin()
