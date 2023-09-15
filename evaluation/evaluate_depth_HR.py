from __future__ import absolute_import, division, print_function
import os, sys
sys.path.append('../')

from PIL import Image
import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

# from layers import disp_to_depth
from utils import readlines
import datasets
import networks
from config.conf import TrainConf

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "./splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return m_disp#r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate():
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    assert sum((eval_mono, eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    conf = TrainConf().conf
    get = lambda x: conf.get(x)

    if ext_disp_to_eval is None:

        get('load_weights_folder') = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(get('load_weights_folder')), \
            "Cannot find a folder at {}".format(get('load_weights_folder'))

        print("-> Loading weights from {}".format(get('load_weights_folder')))

        filenames = readlines(os.path.join(splits_dir, eval_split, "test_files.txt"))
        dataset = datasets.KITTIRAWDataset(get('kitti_path'), filenames,
                                           get('height'), get('width'),
                                           get('novel_frame_ids'), is_train=False, use_crop=False, use_colmap=False, img_ext=".png")
        dataloader = DataLoader(dataset, get('bs'), shuffle=False, num_workers=get('workers'),
                                pin_memory=True, drop_last=False)
        
        
        if get('net_type') == "ResNet":
            encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
            decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
            encoder_dict = torch.load(encoder_path)
            encoder = networks.ResnetEncoder(opt.num_resnet_layers, False)
            depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, 
                                                    opt.disp_levels, 
                                                    opt.disp_min, 
                                                    opt.disp_max, 
                                                    opt.num_ep, 
                                                    pe_type=opt.pos_emb_type,
                                                    use_denseaspp=opt.use_denseaspp, 
                                                    xz_levels=opt.xz_levels,
                                                    yz_levels=opt.yz_levels, 
                                                    use_mixture_loss=opt.use_mixture_loss, 
                                                    render_probability=opt.render_probability, 
                                                    plane_residual=opt.plane_residual)

            model_dict = encoder.state_dict()
            encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
            depth_decoder.load_state_dict(torch.load(decoder_path))
            
            encoder.cuda()
            encoder.eval()
            depth_decoder.cuda()
            depth_decoder.eval()
        elif opt.net_type == "PladeNet":
            model = networks.PladeNet(False, 
                                      opt.disp_levels, 
                                      opt.disp_min, 
                                      opt.disp_max, 
                                      opt.num_ep, 
                                      xz_levels=opt.xz_levels, 
                                      use_mixture_loss=opt.use_mixture_loss, 
                                      render_probability=opt.render_probability, 
                                      plane_residual=opt.plane_residual)
            model.load_state_dict(torch.load(os.path.join(opt.load_weights_folder, "plade.pth")))
            model.cuda()
            model.eval()
        elif opt.net_type == "FalNet":
            model = networks.FalNet(False, opt.height, opt.width, opt.disp_levels, opt.disp_min, opt.disp_max)
            model.load_state_dict(torch.load(os.path.join(opt.load_weights_folder, "fal.pth")))
            model.cuda()
            model.eval()

        pred_disps = []
        probabilities_max = []

        print("-> Computing predictions with size {}x{}".format(
            opt.width, opt.height))
        
        grid = torch.meshgrid(torch.linspace(-1, 1, opt.width), torch.linspace(-1, 1, opt.height), indexing="xy")
        grid = torch.stack(grid, dim=0)

        i = 0

        with torch.no_grad():
            for data in dataloader:
                input_color = data[("color", "l")].cuda()

                if post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                grids = grid[None, ...].expand(input_color.shape[0], -1, -1, -1).cuda()
                
                if opt.net_type == "ResNet":
                    output = depth_decoder(encoder(input_color), grids)
                elif opt.net_type == "FalNet":
                    output = model(input_color)
                elif opt.net_type == "PladeNet":
                    output = model(input_color, grids)

                pred_disp = output["disp"][:, 0].cpu().numpy()

                if post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_disps.append(pred_disp)
                probabilities_max.append(output["probability"].amax(1).mean(-1).mean(-1).cpu().numpy())

        pred_disps = np.concatenate(pred_disps)
        probabilities_max = np.concatenate(probabilities_max)
        print(probabilities_max.mean())
        
    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]

    if save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    if no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    elif eval_split == 'benchmark':
        save_dir = os.path.join(opt.load_weights_folder, "benchmark_predictions")
        print("-> Saving out benchmark predictions to {}".format(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(len(pred_disps)):
            disp_resized = cv2.resize(pred_disps[idx], (1216, 352))
            depth = STEREO_SCALE_FACTOR / disp_resized
            depth = np.clip(depth, 0, 80)
            depth = np.uint16(depth * 256)
            save_path = os.path.join(save_dir, "{:010d}.png".format(idx))
            cv2.imwrite(save_path, depth)

        print("-> No ground truth is available for the KITTI benchmark, so not evaluating. Done.")
        quit()

    gt_path = os.path.join(splits_dir, eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    print("-> Evaluating")

    if eval_stereo:
        print("   Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        disable_median_scaling = True
        pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []

    for i in range(pred_disps.shape[0]):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]

        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 0.1 * 0.58 * opt.width / (pred_disp)

        if eval_split == "eigen_raw" or eval_split == "eigen_improved":
            gt_depth[gt_depth < MIN_DEPTH] = MIN_DEPTH
            gt_depth[gt_depth > MAX_DEPTH] = MAX_DEPTH
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            #the following crop is used in FalNet and PladeNet, which has a slight unfair improvement than Eigen crop.
            # crop = np.array([gt_height - 219, gt_height - 4,
            #                  44,  1180]).astype(np.int32)
            crop_mask = np.zeros(gt_depth.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        else:
            mask = gt_depth > 0

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth *= pred_depth_scale_factor
        if not disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            # ratio = np.mean(gt_depth) / np.mean(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))

    if not disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.5f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == "__main__":
    eval_stereo = False  # if set evaluates in stereo mode
    eval_mono = True  # if set evaluates in mono mode
    no_eval = False  # if set disables evaluation
    eval_split = "eigen_raw"  # which split to run eval on; choices=["eigen_raw", "eigen_improved", "eigen_benchmark", "benchmark", "odom_9", "odom_10", "city"]
    disable_median_scaling = True #if set disables median scaling in evaluation
    pred_depth_scale_factor = 1 #if set multiplies predictions by this number
    ext_disp_to_eval = "" #optional path to a .npy disparities file to evaluate
    save_pred_disps = False #if set saves predicted disparities
    eval_eigen_to_benchmark = False #if set assume we are loading eigen results from npy but we want to evaluate using the new benchmark
    eval_out_dir = "" #if set will output the disparities to this folder
    post_process = False #if set will perform the flipping post processing from the original monodepth paper


    evaluate()
