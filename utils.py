from __future__ import absolute_import, division, print_function

import os, shutil
import hashlib
import zipfile
import numpy as np
from six.moves import urllib
from collections import Counter


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def save_code(srcfile, log_path):
    #save depth_decoder.py to log, easy to evaluation
    #srcfile = "./networks/depth_decoder.py" for example, it is used in trainer
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(srcfile)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        shutil.copy(srcfile, os.path.join(log_path, fname))
        print ("copy %s -> %s"%(srcfile, os.path.join(log_path, fname)))

def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s


def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)

def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d

def download_model_if_doesnt_exist(model_name):
    """If pretrained kitti model doesn't exist, download and unzip it
    """
    # values are tuples of (<google cloud URL>, <md5 checksum>)
    download_paths = {
        "mono_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192.zip",
             "a964b8356e08a02d009609d9e3928f7c"),
        "stereo_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_640x192.zip",
             "3dfb76bcff0786e4ec07ac00f658dd07"),
        "mono+stereo_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_640x192.zip",
             "c024d69012485ed05d7eaa9617a96b81"),
        "mono_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_no_pt_640x192.zip",
             "9c2f071e35027c895a4728358ffc913a"),
        "stereo_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_no_pt_640x192.zip",
             "41ec2de112905f85541ac33a854742d1"),
        "mono+stereo_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_no_pt_640x192.zip",
             "46c3b824f541d143a45c37df65fbab0a"),
        "mono_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_1024x320.zip",
             "0ab0766efdfeea89a0d9ea8ba90e1e63"),
        "stereo_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_1024x320.zip",
             "afc2f2126d70cf3fdf26b550898b501a"),
        "mono+stereo_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_1024x320.zip",
             "cdc5fc9b23513c07d5b19235d9ef08f7"),
        }

    if not os.path.exists("models"):
        os.makedirs("models")

    model_path = os.path.join("models", model_name)

    def check_file_matches_md5(checksum, fpath):
        if not os.path.exists(fpath):
            return False
        with open(fpath, 'rb') as f:
            current_md5checksum = hashlib.md5(f.read()).hexdigest()
        return current_md5checksum == checksum

    # see if we have the model already downloaded...
    if not os.path.exists(os.path.join(model_path, "encoder.pth")):

        model_url, required_md5checksum = download_paths[model_name]

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("-> Downloading pretrained model to {}".format(model_path + ".zip"))
            urllib.request.urlretrieve(model_url, model_path + ".zip")

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("   Failed to download a file which matches the checksum - quitting")
            quit()

        print("   Unzipping model...")
        with zipfile.ZipFile(model_path + ".zip", 'r') as f:
            f.extractall(model_path)

        print("   Model unzipped to {}".format(model_path))


import collections
from torch._six import string_classes
import re
import torch

np_str_obj_array_pattern = re.compile(r'[SaUO]')
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")

def rmnone_collate(batch):
    batch_new = []
    for v in batch:
        if v is not None:
            batch_new.append(v)
    batch = batch_new
    if len(batch) == 0:
        return None
    else:
        return default_collate(batch)

def default_collate(batch):
    """Puts each data field into a tensor with outer dimension batch size"""
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))

def load_velodyne_points(filename):
    """Load 3D point cloud from KITTI file format
    (adapted from https://github.com/hunse/kitti)
    """
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # homogeneous
    return points


def read_calib_file(path):
    """Read KITTI calibration file
    (from https://github.com/hunse/kitti)
    """
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data

def sub2ind(matrixSize, rowSub, colSub):
    """Convert row, col matrix subscripts to linear indices
    """
    m, n = matrixSize
    return rowSub * (n-1) + colSub - 1


def generate_depth_map(calib_dir, velo_filename, cam=2, vel_depth=False):
    """Generate a depth map from velodyne data
    """
    # load calibration files
    cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
    velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
    velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

    # get image shape
    im_shape = cam2cam["S_rect_02"][::-1].astype(np.int32)

    # compute projection matrix velodyne->image plane
    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
    P_rect = cam2cam['P_rect_0'+str(cam)].reshape(3, 4)
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    # load velodyne points and remove all behind image plane (approximation)
    # each row of the velodyne data is forward, left, up, reflectance
    velo = load_velodyne_points(velo_filename)
    velo = velo[velo[:, 0] >= 0, :]

    # project the points to the camera
    velo_pts_im = np.dot(P_velo2im, velo.T).T
    velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis]

    if vel_depth:
        velo_pts_im[:, 2] = velo[:, 0]

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:, 0] < im_shape[1]) & (velo_pts_im[:, 1] < im_shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]

    # project to image
    depth = np.zeros((im_shape[:2]))
    depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]

    # find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    depth[depth < 0] = 0

    return depth

def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #torch.cuda.manual_seed(seed) # torch doc says that torch.manual_seed also work for CUDA
    #Speed-reproducibility tradeoff -> https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def worker_init():
    worker_seed = torch.utils.data.get_worker_info().seed % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def log_time(batch_idx, duration, losses):
    """
        Print a logging statement to the terminal
    """
    samples_per_sec = self.opt.batch_size * torch.cuda.device_count() / duration
    time_sofar = time.time() - self.start_time
    training_time_left = (self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
    print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                   " | loss: {:.5f} | time elapsed: {} | time left: {}"
    print(print_string.format(self.epoch, batch_idx, samples_per_sec, losses["loss/total_loss"].cpu().data,
                              sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

def log(writers, mode, losses, step):
    """
        Write an event to the tensorboard events file
    """
    writer = writers[mode]
    for l, v in losses.items():
        writer.add_scalar(l, v, step)

def log_img(writers, mode, inputs, outputs, val_idx, batch_size, epoch, target_sides):
    """
        Write an event to the tensorboard events file
    """
    writer = writers[mode]

    for j in range(min(4, batch_size)):  # write a maximum of four images
        for frame_id in ["l", "r"] + self.opt.novel_frame_ids:
            writer.add_image(
                "color_{}/{}".format(frame_id, epoch),
                inputs[("color", frame_id)][j].data, val_idx + j)

        if mode == "train":

            for frame_id in target_sides:
                writer.add_image(
                    "color_pred_{}/{}".format(frame_id, epoch),
                    outputs[("rgb_rec", frame_id)][j].data, val_idx + j)

            if "disp_pp" in outputs:
                writer.add_image(
                    "disp_pp/{}".format(epoch),
                    normalize_image(outputs["disp_pp"][j]), val_idx + j)

        writer.add_image(
            "disp/{}".format(epoch),
            normalize_image(outputs["disp"][j]), val_idx + j)

def save_opts(log_path):
    """
        Save options to disk so we know what we ran this experiment with
    """
    models_dir = log_path
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    to_save = self.opt.__dict__.copy() #aici trebuie sa modific pt ca eu am yaml

    with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
        json.dump(to_save, f, indent=2)

def save_model(folder_name, log_path, models, model_optimizer, height, width):
    """
        Save model weights to disk
    """
    save_folder = os.path.join(log_path, folder_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for model_name, model in models.items():
        save_path = os.path.join(save_folder, "{}.pth".format(model_name))
        to_save = model.module.state_dict()
        if model_name == 'encoder':
            # save the sizes - these are needed at prediction time
            to_save['height'] = height
            to_save['width'] = width
        torch.save(to_save, save_path)

    save_path = os.path.join(save_folder, "{}.pth".format("adam"))
    torch.save(model_optimizer.state_dict(), save_path)


def add_flip_right_inputs(inputs):
    new_inputs = {}
    new_inputs[("color", "l")] = torch.cat([inputs[("color", "l")], inputs[("color", "r")].flip(-1)], dim=0)
    new_inputs[("color", "r")] = torch.cat([inputs[("color", "r")], inputs[("color", "l")].flip(-1)], dim=0)
    new_inputs[("color_aug", "l")] = torch.cat([inputs[("color_aug", "l")], inputs[("color_aug", "r")].flip(-1)], dim=0)
    new_inputs[("color_aug", "r")] = torch.cat([inputs[("color_aug", "r")], inputs[("color_aug", "l")].flip(-1)], dim=0)
    grid_fliped = inputs["grid"].clone()
    grid_fliped[:, 0, :, :] *= -1.
    grid_fliped = grid_fliped.flip(-1)
    new_inputs["grid"] = torch.cat([inputs["grid"], grid_fliped], dim=0)
    new_inputs[("depth_gt", "l")] = torch.cat([inputs[("depth_gt", "l")], inputs[("depth_gt", "r")].flip(-1)], dim=0)
    new_inputs[("depth_gt", "r")] = torch.cat([inputs[("depth_gt", "r")], inputs[("depth_gt", "l")].flip(-1)], dim=0)

    new_inputs["K"] = inputs["K"].repeat(2, 1, 1)
    new_inputs["inv_K"] = inputs["inv_K"].repeat(2, 1, 1)

    new_inputs[("Rt", "l")] = inputs[("Rt", "l")].repeat(2, 1, 1)
    new_inputs[("Rt", "r")] = inputs[("Rt", "r")].repeat(2, 1, 1)

    # The the left +1/-1 frame becomes the right side, but it should not affect the training
    for novel_frame_id in get('novel_frame_ids'):
        new_inputs[("color", novel_frame_id)] = torch.cat(
            [inputs[("color", novel_frame_id)], inputs[("color", novel_frame_id)].flip(-1)], dim=0)
        new_inputs[("color_aug", novel_frame_id)] = torch.cat(
            [inputs[("color_aug", novel_frame_id)], inputs[("color_aug", novel_frame_id)].flip(-1)], dim=0)

    return new_inputs


def predict_poses(inputs):
    """
        Predict poses between input frames for monocular sequences.
    """
    outputs = {}
    # In this setting, we compute the pose to each source frame via a
    # separate forward pass through the pose network.
    outputs[("Rt", "r")] = inputs[("Rt", "r")]

    for f_i in get('novel_frame_ids'):

        if not get('use_colmap'):
            if f_i < 0:
                pose_inputs = [inputs[("color_aug", f_i)], inputs[("color_aug", "l")]]
            else:
                pose_inputs = [inputs[("color_aug", "l")], inputs[("color_aug", f_i)]]

            pose_inputs = [models["pose_encoder"](torch.cat(pose_inputs, 1))]

            axisangle, translation = models["pose_decoder"](pose_inputs, inputs["grid"])
            outputs[("axisangle", f_i)] = axisangle
            outputs[("translation", f_i)] = translation

            # Invert the matrix if the frame id is negative
            Rt = transformation_from_parameters(
                axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
        else:
            Rt = inputs[("Rt", f_i)].float()

        Rt_Rc = torch.zeros_like(Rt)

        gx0 = (inputs["grid"][:, 0, 0, -1] + inputs["grid"][:, 0, 0, 0]) / 2.
        gy0 = (inputs["grid"][:, 1, -1, 0] + inputs["grid"][:, 1, 0, 0]) / 2.
        f = (inputs["grid"][:, 0, 0, -1] - inputs["grid"][:, 0, 0, 0]) / 2.
        Rc_v = torch.stack([-gx0 / (2 * 0.58), -gy0 / (2 * 1.92), f], dim=1)
        Rc = torch.eye(3).cuda()
        Rc = Rc[None, :, :].repeat(Rc_v.shape[0], 1, 1)
        Rc[:, :, 2] = Rc_v
        outputs[("Rc", f_i)] = Rc
        Rt_Rc[:, :3, :3] = torch.matmul(Rc, torch.matmul(Rt[:, :3, :3], torch.inverse(Rc)))
        if get('use_colmap'):
            Rt_Rc[:, :3, 3:4] = torch.matmul(Rc, Rt[:, :3, 3:4])

        outputs[("Rt", f_i)] = Rt_Rc

    return outputs


def generate_post_process_disp(inputs):
    # set_eval()
    input_images = torch.cat([inputs[("color_aug", "l")], inputs[("color_aug", "l")].flip(-1)], dim=0)
    if get('num_ep') > 0:
        grid_fliped = inputs["grid"].clone()
        grid_fliped[:, 0, :, :] *= -1.
        grid_fliped = grid_fliped.flip(-1)
        input_grids = torch.cat([inputs["grid"], grid_fliped], dim=0)

    if get('model').get('net_type') == "ResNet":
        features = fixed_models["encoder"](input_images)
        outputs = fixed_models["depth"](features, input_grids)
    elif get('model').get('net_type') == "PladeNet":
        outputs = models["plade"](input_images, input_grids)
    elif get('model').get('net_type') == "FalNet":
        outputs = models["fal"](input_images)

    B, N, H, W = outputs["probability"].shape
    B = B // 2
    pix_coords = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
    pix_coords = torch.stack(pix_coords, dim=0).cuda().float()

    pix_coords_r = pix_coords[None, None, ...].expand(B, N, -1, -1, -1).clone()
    pix_coords_r[:, :, 0, :, :] += outputs["disp_layered"][:B, ...]
    pix_coords_r[:, :, 0, :, :] /= (W - 1)
    pix_coords_r[:, :, 1, :, :] /= (H - 1)
    pix_coords_r = (pix_coords_r - 0.5) * 2
    pix_coords_r = pix_coords_r.reshape(B * N, 2, H, W)
    pix_coords_r = pix_coords_r.permute(0, 2, 3, 1)

    pix_coords_l = pix_coords[None, None, ...].expand(B, N, -1, -1, -1).clone()
    pix_coords_l[:, :, 0, :, :] -= outputs["disp_layered"][B:, ...]
    pix_coords_l[:, :, 0, :, :] /= (W - 1)
    pix_coords_l[:, :, 1, :, :] /= (H - 1)
    pix_coords_l = (pix_coords_l - 0.5) * 2
    pix_coords_l = pix_coords_l.reshape(B * N, 2, H, W)
    pix_coords_l = pix_coords_l.permute(0, 2, 3, 1)

    # pll = outputs_stage1["probability"][:B, ...]
    pl = outputs["logits"][:B, ...].reshape(B * N, 1, H, W)
    plr = F.grid_sample(pl, pix_coords_r, padding_mode="zeros", align_corners=True).reshape(B, N, H, W)
    plr = softmax(plr)
    plr = plr.reshape(B * N, 1, H, W)
    o_l = F.grid_sample(plr, pix_coords_l, padding_mode="zeros", align_corners=True).reshape(B, N, H, W)
    o_l = o_l.sum(1, True)
    o_l[o_l > 1] = 1

    pfr = outputs["logits"][B:, ...].flip(-1).reshape(B * N, 1, H, W)
    pfrl = F.grid_sample(pfr, pix_coords_l, padding_mode="zeros", align_corners=True).reshape(B, N, H, W)
    pfrl = softmax(pfrl).reshape(B * N, 1, H, W)
    o_fr = F.grid_sample(pfrl, pix_coords_r, padding_mode="zeros", align_corners=True).reshape(B, N, H, W)
    o_fr = o_fr.sum(1, True)
    o_fr[o_fr > 1] = 1

    mean_disp = outputs["disp"][:B, ...] * 0.5 + outputs["disp"][B:, ...].flip(-1) * 0.5

    disp_pp = mean_disp * o_fr + outputs["disp"][:B, ...] * (1 - o_fr)
    disp_pp = disp_pp * o_l + outputs["disp"][-B:, ...].flip(-1) * (1 - o_l)

    mask_novel = F.grid_sample(outputs["probability"][:B, ...].reshape(B * N, 1, H, W), pix_coords_r,
                               padding_mode="zeros", align_corners=True).reshape(B, N, H, W)
    mask_novel = mask_novel.sum(1, True)
    mask_novel[mask_novel > 1] = 1
    return disp_pp.detach(), mask_novel.detach()


def val():
    """
        Validate the model on a single minibatch
    """

    # Convert all models to testing/evaluation mode
    for m in models.values():
        m.eval()
    num = 0
    metrics = {}
    with torch.no_grad():
        for batch_idx, inputs in enumerate(val_loader):
            # outputs, losses = process_batch(inputs)
            # losses.update(compute_depth_losses(inputs, outputs))
            for key, ipt in inputs.items():
                inputs[key] = ipt.to(device)

            # maybe we need use the same name for different model in models
            if get('model').get('net_type') == "ResNet":
                features = models["encoder"](inputs[("color_aug", "l")])
                outputs = models["depth"](features, inputs["grid"])
            elif get('model').get('net_type') == "PladeNet":
                outputs = models["plade"](inputs[("color_aug", "l")], inputs["grid"])
            elif get('model').get('net_type') == "FalNet":
                outputs = models["fal"](inputs[("color_aug", "l")])

            losses = compute_depth_losses(inputs, outputs)
            B = inputs[("color_aug", "l")].shape[0]
            num += B
            for k, v in losses.items():
                if k in metrics:
                    metrics[k] += v * B
                else:
                    metrics[k] = v * B

            '''if batch_idx % self.opt.log_img_frequency == 0'''
            if batch_idx % 250 == 0 and local_rank == 0:
                log_img("val", inputs, outputs, batch_idx)
            del inputs, outputs, losses
        # since the eval batch size is not the same
        # we need to sum them then mean
        num = torch.ones(1).cuda() * num
        dist.all_reduce(num, op=dist.ReduceOp.SUM)
        for k, v in metrics.items():
            dist.all_reduce(metrics[k], op=dist.ReduceOp.SUM)
            metrics[k] = metrics[k] / num
        if metrics["de/abs_rel"] < best_absrel:
            best_absrel = metrics["de/abs_rel"]
            if local_rank == 0:
                save_model("best_models")

        if local_rank == 0:
            log("val", metrics)
            print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
            print(("&{: 8.4f}  " * 7).format(*[metrics[k].cpu().data[0] for k in depth_metric_names]) + "\\\\")
            # write to log file
            print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"),
                  file=log_file)
            print(("&{: 8.4f}  " * 7).format(*[metrics[k].cpu().data[0] for k in depth_metric_names]) + "\\\\",
                  file=log_file)

    # Convert all models to training mode
    for m in models.values():
        m.train()


def pred_novel_images(inputs, outputs):
    """Generate the warped (reprojected) color images for a minibatch.
    Generated images are saved into the `outputs` dictionary.
    """

    B, N, H, W = outputs["probability"].shape

    source_side = "l"

    for target_side in target_sides:
        if get('warp_type') == "depth_warp":
            disps = outputs["disp_layered"]
            depths = 0.1 * 0.58 * W / disps
            T = inputs[("Rt", target_side)][:, None, :, :].expand(-1, N, -1, -1).reshape(B * N, 4, 4)
            cam_points = backproject_depth(depths.reshape(B * N, 1, H, W),
                                           inputs["inv_K"][:, None, :, :].expand(-1, N, -1, -1).reshape(B * N, 4, 4))
            pix_coords = project_3d(cam_points, inputs["K"][:, None, :, :].expand(-1, N, -1, -1).reshape(B * N, 4, 4),
                                    T)  # BN, H, W, 2

        elif get('warp_type') == "disp_warp":
            disps = outputs["disp_layered"]
            pix_coords = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
            pix_coords = torch.stack(pix_coords, dim=0).cuda().float()
            pix_coords = pix_coords[None, None, ...].expand(B, N, -1, -1, -1).clone()
            if target_side == "l":
                pix_coords[:, :, 0, :, :] -= disps
            elif target_side == "r":
                pix_coords[:, :, 0, :, :] += disps
            pix_coords[:, :, 0, :, :] /= (W - 1)
            pix_coords[:, :, 1, :, :] /= (H - 1)
            pix_coords = (pix_coords - 0.5) * 2
            pix_coords = pix_coords.reshape(B * N, 2, H, W)
            pix_coords = pix_coords.permute(0, 2, 3, 1)
            padding_mask = outputs["padding_mask"][:, :, None, :, :]

        elif get('warp_type') == "homography_warp":
            T = outputs[("Rt", target_side)][:, None, :, :].expand(-1, N, -1, -1).reshape(B * N, 4, 4)
            K = inputs["K"][:, None, :, :].expand(-1, N, -1, -1).reshape(B * N, 4, 4)
            inv_K = inputs["inv_K"][:, None, :, :].expand(-1, N, -1, -1).reshape(B * N, 4, 4)
            pix_coords, padding_mask = homography_warp(outputs["distance"], outputs["norm"], T, K, inv_K)

        if get('match_aug'):
            color_name = "color_aug"
        else:
            color_name = "color"

        features = torch.cat(
            [inputs[(color_name, source_side)][:, None].expand(-1, N, -1, -1, -1).reshape(B * N, 3, H, W), \
             outputs["logits"].reshape(B * N, 1, H, W)], dim=1)

        if get('use_mixture_loss'):
            features = torch.cat([features, outputs["sigma"].reshape(B * N, 1, H, W)], dim=1)

        rec_features = F.grid_sample(
            features,
            pix_coords,
            padding_mode="zeros",
            align_corners=True).reshape(B, N, -1, H, W)

        # only stereo could compute as this.
        rec_features = rec_features * padding_mask

        outputs[("rgb_rec_layered", target_side)] = rec_features[:, :, :3, ...]
        outputs[("logit_rec", target_side)] = rec_features[:, :, 3, ...]
        if get('render_probability'):
            # We read dists from output since the layered depth of stereo pair is the same.
            # otherwise we should recompute it.
            alpha = 1. - torch.exp(-F.relu(outputs[("logit_rec", target_side)][:, :-1, ...]) * outputs["dists"])
            ones = torch.ones_like(alpha[:, :1, ...])
            alpha = torch.cat([alpha, ones], dim=1)
            probability_rec = alpha * torch.cumprod(torch.cat([ones, 1. - alpha + 1e-10], dim=1), dim=1)[:, :-1, ...]
            outputs[("probability_rec", target_side)] = probability_rec
        else:
            outputs[("probability_rec", target_side)] = softmax(outputs[("logit_rec", target_side)])
        if get('use_mixture_loss'):
            sigma_rec = rec_features[:, :, 4, ...].clone()
            # sigma_rec[sigma_rec==0] = 1.
            sigma_rec = torch.clamp(sigma_rec, 0.01, 1.)
            outputs[("sigma_rec", target_side)] = sigma_rec
            outputs[("pi_rec", target_side)] = pi_rec = outputs[("probability_rec", target_side)]
            weights_rec = pi_rec / sigma_rec
            weights_rec = weights_rec / weights_rec.sum(1, True)
            outputs[("probability_rec", target_side)] = weights_rec
            outputs[("rgb_rec", target_side)] = (
                        outputs[("rgb_rec_layered", target_side)] * outputs[("probability_rec", target_side)][:, :,
                                                                    None]).sum(1)


def pred_self_images(inputs, outputs):
    """
    Generate the warped (reprojected) color images for a minibatch.
    Generated images are saved into the `outputs` dictionary.
    """
    disp = outputs["disp"]
    B, N, H, W = outputs["probability"].shape

    depth = 0.1 * 0.58 * W / disp
    T = inputs[("Rt", "r")]
    cam_points = backproject_depth(depth, inputs["inv_K"])
    pix_coords = project_3d(cam_points, inputs["K"], T)  # BN, H, W, 2

    if get('match_aug'):
        color_name = "color_aug"
    else:
        color_name = "color"

    features = inputs[(color_name, "r")]

    rec_features = F.grid_sample(
        features,
        pix_coords,
        padding_mode="border",
        align_corners=True)

    # only stereo could compute as this.
    # rec_features = rec_features * outputs["padding_mask"][:, :, None, ...]

    outputs["self_rec"] = rec_features
