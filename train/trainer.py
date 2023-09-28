from __future__ import absolute_import, division, print_function
import copy
import random

import numpy as np
import time

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from tensorboardX import SummaryWriter

import json

from utils import *
from utils import *
from networks.layers import *

import datasets as datasets
import networks
from IPython import embed
from config.conf import TrainConf


def train():

    #PREPARING LOGING DIRECTORIES
    log_dir = os.path.join(get('tensorboard_path'), get('model').get('net_type'))
    log_path = os.path.join(log_dir, get('model').get('name'))


    #SETTING UP THE DISTRIBUTED TRAINING ENVIRONMENT FOR PYTORCH
    dist.init_process_group(backend='nccl') #'init_process_group' is a function used to initialize the distributed training environment.
                                            #backend='nccl' specifies that the NCCL backend should be used for communication between processes.
                                            #NCCL is optimized for multi-GPU training and is commonly used with NVIDIA GPUs
    local_rank = int(os.environ['LOCAL_RANK'])# This line retrieves the local rank of the current process from the LOCAL_RANK environment
                                              # variable. In distributed training, each process typically has a unique local rank, which
                                              # can be used to assign specific tasks to each process.
    batch_size = get('batch_size') // torch.cuda.device_count()
    torch.cuda.set_device(local_rank) #This line sets the current CUDA device (GPU) to be used for the current process based on its local
                                      # rank. In a multi-GPU setup, each process is typically responsible for training on a specific GPU.
    init_seeds(1+local_rank)


    #SAVING THE CODE CONTAINING THE ARCHITECTURES AND CONFIGS FOR A EASY EVALUATION LATER ON
    if dist.get_rank() == 0: # rank of 0 typically corresponds to the master process or the process
                             # that performs additional tasks beyond model training.
        save_code("./trainer.py", log_path)
        save_code("./config/train_conf.yaml", log_path)
        if get('model').get('net_type') == "ResNet":
            save_code("./networks/depth_decoder.py", log_path)
        elif get('model').get('net_type') == "PladeNet":
            save_code("./networks/plade_net.py", log_path)
        elif get('model').get('net_type') == "FalNet":
            save_code("./networks/fal_net.py", log_path)

    # CHECKING HEIGHT AND WIDTH ARE MULTIPLES OF 32
    assert get('im_sz')[0] % 32 == 0, "'height' must be a multiple of 32"
    assert get('im_sz')[1] % 32 == 0, "'width' must be a multiple of 32"

    models = {}
    parameters_to_train = []
    if get('use_cuda'):
        device = torch.device("cuda")
    else
        device = torch.device("cpu")

    #CREATE THE MODELS
    models = {}
    if get('model').get('net_type') == "ResNet":
        print("train ResNet")
        models["encoder"] = networks.ResnetEncoder(get('model').get('num_resnet_layers'), True)
        models["depth"] = networks.DepthDecoder(models["encoder"].num_ch_enc,
                                                get('disp_levels'),
                                                get('disp_min'),
                                                get('disp_max'),
                                                get('num_ep'),
                                                pe_type=get('pos_emb_type'),
                                                use_denseaspp=get('use_denseaspp'),
                                                xz_levels=get('ground_planes'),
                                                yz_levels=get('vertical_planes'),
                                                use_mixture_loss=get('use_mixture_loss'),
                                                render_probability=get('render_probability'),
                                                plane_residual=get('plane_residual'))

    elif get('model').get('net_type') == "PladeNet":
        print("train PladeNet")
        models["plade"] = networks.PladeNet(False,
                                            get('disp_levels'),
                                            get('disp_min'),
                                            get('disp_max'),
                                            get('num_ep'),
                                            xz_levels=get('ground_planes'),
                                            use_mixture_loss=get('use_mixture_loss'),
                                            render_probability=get('render_probability'),
                                            plane_residual=get('plane_residual'))

    elif get('model').get('net_type') == "FalNet":
        print("train FalNet")
        models["fal"] = networks.FalNet(False, get('im_sz')[0], get('im_sz')[1], get('disp_levels'), get('disp_min'), get('disp_max'))

    else:
        print("undefined model type")
        quit()

    #CREATE THE POSE ENCODER AND DECODER TO ESTIMATE THE POSE
    if len(get('novel_frame_ids')) > 0 and not get('use_colmap'): #ce legatura are colmapul cu asta?
        models["pose_encoder"] = networks.ResnetPoseEncoder(18, True, 2)
        models["pose_decoder"] = networks.PoseDecoder(models["pose_encoder"].num_ch_enc, num_input_features=1, num_frames_to_predict_for=1, num_ep=8)

    #CONVERT THE MODELS ON DEVICE, CONVERT BATCHNORM TO SYNCHRONIZED BATCHNORM, ENABLE PARALLELIZATION AND COMPLETE THE LIST OF PARAMETERS TO TRAIN
    for model_name, model in models.items():
        model = model.to(device)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model) #converts batch normalization layers in the model to synchronized batch normalization
                                                               # layers. Synchronized batch normalization is often used in distributed training to
                                                               # ensure consistent statistics across multiple GPUs or nodes
        models[model_name] = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        parameters_to_train += list(models[model_name].parameters())

    #CAND FAC SELF-DISTILLATION, ENCODER-UL SI DECODER-UL LE COPIEZ
    if get('lambda_self_distillation') > 0:
        fixed_models = {}
        fixed_models["encoder"] = copy.deepcopy(models["encoder"].module).eval()
        fixed_models["depth"] = copy.deepcopy(models["depth"].module).eval()

    #LOAD THE PRETRAINED MODELS IF IT IS THE CASE
    if get('load_weights_folder') is not None:
        load_model() # -> asta o voi face from_pretrained

    #DEFINE THE OPTIMIZER AND LEARNING RATE SCHEDULER
    model_optimizer = optim.Adam(parameters_to_train, get('adamw').get('lr'), betas=(get('adamw').get('beta_1'), get('adamw').get('beta_2')))
    model_lr_scheduler = optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=get('milestones'), gamma=0.5)

    print("Training model named:\n  ", get('model').get('name'))
    print("Models and tensorboard events files are saved to:\n  ", log_dir)
    print("Training is using:\n  ", device)

    ###########################
    #SOME ADDITIONAL STEPS
    flip_right=get('flip_right')
    if get('use_mom'): #why??
        flip_right = True
    if flip_right:  #why??
        batch_size = batch_size // 2
    if not get('no_stereo'):
        target_sides = ["r"] + get('novel_frame_ids')
    else:
        target_sides = get('novel_frame_ids')
    ###########################

    #PREPARING DATASETS WITH DATALOADERS
    datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                     "kitti_odom": datasets.KITTIOdomDataset}
    dataset = datasets_dict[get('dataset')]

    fpath = os.path.join(os.path.dirname(__file__), "./splits", get('split'), "{}_files.txt")

    train_filenames = readlines(fpath.format("train"))
    val_filenames = readlines(fpath.format("val"))
    img_ext = '.png' if get('png') else '.jpg'

    num_train_samples = len(train_filenames)
    num_total_steps = num_train_samples // (batch_size * torch.cuda.device_count()) * get('epochs')

    train_dataset = dataset(get('data_path'), train_filenames, get('im_sz')[0], get('im_sz')[1],
                            get('novel_frame_ids'), is_train=True, use_crop=not get('no_crop'),
                            use_colmap=get('use_colmap'), colmap_path=get('colmap_path'), img_ext=img_ext)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size, False, num_workers=get('workers'),
                              sampler=train_sampler, pin_memory=True, drop_last=True,
                              worker_init_fn=worker_init, collate_fn=rmnone_collate)

    val_dataset = dataset(GET('data_path'), val_filenames, get('im_sz')[0], get('im_sz')[1],
                              get('novel_frame_ids'), is_train=False, use_crop=False,
                              use_colmap=False, img_ext=img_ext)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_loader = DataLoader(val_dataset, batch_size, False, num_workers=get('workers'),
                            sampler=val_sampler, pin_memory=True, drop_last=False)

    #DEFINE SSIM, BACKPROJECT_DEPTH, PROJECT_3D AND HOMOGRAPHY_WARP
    if get('use_ssim'):
        ssim = SSIM()
        ssim.to(device)

    backproject_depth = BackprojectDepth(get('im_sz')[0] , get('im_sz')[1])
    backproject_depth.to(device)

    project_3d = Project3D(get('im_sz')[0] , get('im_sz')[1])
    project_3d.to(device)

    homography_warp = HomographyWarp(get('im_sz')[0] , get('im_sz')[1])
    homography_warp.to(device)

    if get('pc_net') == "vgg19":
        pc_net = Vgg19_pc()
    elif get('pc_net') == "resnet18":
        pc_net = Resnet18_pc().cuda()
    if get('use_cuda'):
        pc_net = pc_net.cuda()

    softmax = nn.Softmax(1)

    depth_metric_names = ["de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

    if dist.get_rank() == 0:
        writers = {}
        for mode in ["train", "val"]:
            writers[mode] = SummaryWriter(os.path.join(log_path, mode))
        print("Using split:\n  ", get('split'))
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))
        save_opts()

        log_file = open(os.path.join(log_path, "logs.log"), 'w')

    best_absrel = 10.

    #Run the entire training pipeline
    epoch = 0
    for epoch in range(get('start_epoch')):
        model_lr_scheduler.step()
    step = 0
    start_time = time.time()
    for epoch in range(get('start_epoch'), get('num_epochs')):
        #RUN A SINGLE EPOCH OF TRAINING AND VALIDATION
        print("Training")
        train_sampler.set_epoch(epoch)
        #CONVERT ALL MODELS TO TRAINING MODE
        for m in models.values():
            m.train()
        for batch_idx, inputs in enumerate(train_loader):
            if inputs is None:
                model_optimizer.zero_grad()
                model_optimizer.step()
                step += 1
                continue

            before_op_time = time.time()

            if  get('flip_right'):
                inputs = add_flip_right_inputs(inputs)

            #Pass a minibatch through the network and generate images and losses
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

            outputs.update(predict_poses(inputs))

            pred_novel_images(inputs, outputs)

            if get('use_mom') and inputs[("color", "l")].shape[0] == batch_size * 2:
                mirror_occlusion_mask(outputs)

            if get('lambda_self_distillation') > 0.:
                with torch.no_grad():
                    outputs["disp_pp"], outputs["mask_novel"] = generate_post_process_disp(inputs)

            if get('alpha_self') > 0.:
                pred_self_images(inputs, outputs)

            losses = compute_losses(inputs, outputs)

            return outputs, losses

            ###
            outputs, losses = process_batch(inputs)

            model_optimizer.zero_grad()
            losses["loss/total_loss"].backward()
            model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % 100 == 0 and step < 500  # self.opt.log_frequency
            late_phase = step % 500 == 0  # self.opt.log_frequency == 0

            if early_phase or late_phase:
                if dist.get_rank() == 0:
                    log_time(batch_idx, duration, losses)

                    losses.update(compute_depth_losses(inputs, outputs))

                    log("train", losses)

            step += 1

            if batch_idx == 0 and dist.get_rank() == 0:
                log_img("train", inputs, outputs, batch_idx)

        val()
        model_lr_scheduler.step()

        if dist.get_rank() == 0:
            save_model("last_models")



if __name__ == '__main__':
    conf = TrainConf().conf
    get = lambda x: conf.get(x)

    train()