import torch

def compute_perceptual_loss(pred, target, pc_net, source=None):
    pred_vgg = pc_net(pred) #the type of net to compute pc loss; choices=["vgg19", "resnet18"]) -> in principiu e vgg19
    target_vgg = pc_net(target)
    if source is not None:
        source_vgg = pc_net(source)

    loss_pc = 0
    for i in range(3):
        l_p = ((pred_vgg[i] - target_vgg[i]) ** 2).mean(1, True)
        if source is not None:  # automask
            l_p_auto = ((source_vgg[i] - target_vgg[i]) ** 2).mean(1, True)
            l_p, _ = torch.cat([l_p, l_p_auto], dim=1).min(1, True)
        loss_pc += l_p.mean()
    return loss_pc

def compute_reprojection_loss(pred, target, use_ssim):
    """
        Computes reprojection loss between a batch of predicted and target images
    """
    abs_diff = torch.abs(target - pred)
    l1_loss = abs_diff.mean(1, True)

    if use_ssim:
        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss
    else:
        reprojection_loss = l1_loss

    return reprojection_loss

def get_smooth_loss_disp(disp, img, gamma=1):
    """
        Computes the smoothness loss for a disparity image.
        The color image is used for edge-aware smoothness.
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-gamma*grad_img_x)
    grad_disp_y *= torch.exp(-gamma*grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()

def compute_losses(inputs, outputs):
    """Compute the reprojection and smoothness losses for a minibatch
    """
    B, N, H, W = outputs["probability"].shape
    losses = {}
    losses["loss/ph_loss"] = 0
    losses["loss/pc_loss"] = 0
    if self.opt.alpha_self > 0.:
        losses["loss/self_loss"] = 0
    losses["loss/total_loss"] = 0

    if self.opt.match_aug:
        color_name = "color_aug"
    else:
        color_name = "color"

    for target_side in self.target_sides:
        total_loss = 0

        pred = outputs[("rgb_rec", target_side)]

        target = inputs[(color_name, target_side)]

        if "mask_novel" in outputs.keys():
            mask = outputs["mask_novel"]
            pred = pred * mask + target * (1. - mask)

        if self.opt.use_mixture_loss:
            error = torch.abs(outputs[("rgb_rec_layered", target_side)] - target[:, None]).mean(2)
            ph_loss = multimodal_loss(error, outputs[("sigma_rec", target_side)], outputs[("pi_rec", target_side)],
                                      dist='lap')  # .mean()
            if self.opt.automask:
                error_auto = torch.abs(inputs[(color_name, "l")][:, None] - target[:, None]).mean(2)
                ph_loss_auto = multimodal_loss(error_auto, outputs[("sigma_rec", target_side)].detach(),
                                               outputs[("pi_rec", target_side)].detach(), dist='lap')
                ph_loss, _ = torch.cat([ph_loss, ph_loss_auto], dim=1).min(1, True)
            if "mask_novel" in outputs.keys():
                ph_loss = ph_loss * mask
        else:
            ph_loss = torch.abs(pred - target).mean(1, True)
            if self.opt.automask:
                ph_loss_auto = torch.abs(inputs[(color_name, "l")] - target).mean(1, True)
                ph_loss, _ = torch.cat([ph_loss, ph_loss_auto], dim=1).min(1, True)
        ph_loss = ph_loss.mean()
        losses["loss/ph_loss"] += ph_loss
        total_loss += ph_loss

        if not self.opt.automask:
            pc_loss = self.compute_perceptual_loss(pred, target).mean()
        else:
            pc_loss = self.compute_perceptual_loss(pred, target, inputs[(color_name, "l")]).mean()
        losses["loss/pc_loss"] += pc_loss
        total_loss += self.opt.lambda_pc * pc_loss

        if self.opt.alpha_self > 0.:
            self_loss = self.compute_reprojection_loss(outputs[("self_rec", target_side)],
                                                       inputs[(color_name, "l")]).mean()
            losses["loss/self_loss"] += self_loss
            total_loss += self.opt.alpha_self * self_loss

        if self.opt.lambda_self_distillation > 0:
            disp_loss = torch.abs(outputs["disp"] - outputs["disp_pp"]).mean()
            losses["loss/disp_loss"] = disp_loss
            total_loss += self.opt.lambda_self_distillation * disp_loss

        losses["loss/total_loss"] += total_loss

    for k, v in losses.items():
        v /= len(self.target_sides)

    smooth_loss = get_smooth_loss_disp(outputs["disp"][..., int(0.2 * W):],
                                       inputs[("color", "l")][..., int(0.2 * W):], gamma=self.opt.gamma_smooth)
    losses["loss/smooth_loss"] = smooth_loss

    losses["loss/total_loss"] += self.opt.lambda_smooth * smooth_loss

    return losses

def compute_depth_losses(inputs, outputs):
    """Compute depth metrics, to allow monitoring during training

    This isn't particularly accurate as it averages over the entire batch,
    so is only used to give an indication of validation performance
    """
    depth_pred = outputs["depth"].detach()
    # depth_pred = torch.clamp(F.interpolate(
    #     depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80).detach()
    depth_pred = depth_pred * 2. / (inputs["grid"][:, 0:1, :, -1:] - inputs["grid"][:, 0:1, :, 0:1])
    depth_pred = torch.clamp(depth_pred, 1e-3, 80)

    depth_gt = inputs[("depth_gt", "l")]
    B, _, H, W = depth_gt.shape

    mask = depth_gt > 0

    # garg/eigen crop
    crop_mask = torch.zeros_like(mask)
    crop_mask[:, :, int(0.40810811 * H):int(0.99189189 * H), int(0.03594771 * W):int(0.96405229 * W)] = 1
    mask = mask * crop_mask

    depth_gt = torch.clamp(depth_gt[mask], 1e-3, 80)
    depth_pred = depth_pred[mask]
    if self.opt.no_stereo:
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)
    else:
        depth_pred *= 5.4

    depth_errors = compute_depth_errors(depth_gt, depth_pred)

    losses = {}
    for i, metric in enumerate(self.depth_metric_names):
        # losses[metric] = np.array(depth_errors[i].cpu())
        losses[metric] = depth_errors[i]
    return losses

def mirror_occlusion_mask(outputs):
    with torch.no_grad():
        B, N, H, W = outputs["probability"].shape
        B = B // 2
        pll = outputs["probability"][:B, ...]
        prr = outputs["probability"][B:, ...].flip(-1)
        plr = outputs["probability_rec"][:B, ...]
        prl = outputs["probability_rec"][B:, ...].flip(-1)

        pl = torch.stack([pll, prl], dim=2).reshape(B * N, 2, H, W)
        pr = torch.stack([prr, plr], dim=2).reshape(B * N, 2, H, W)

        pix_coords_r = self.pix_coords_r.expand(B, -1, -1, -1, -1).reshape(B * N, 2, H, W).permute(0, 2, 3, 1)
        o_r = F.grid_sample(
            pl,
            pix_coords_r,
            padding_mode="zeros", align_corners=True).reshape(B, N, 2, H, W)
        o_r = o_r.sum(1)
        o_r = o_r[:, 0] * o_r[:, 1]
        o_r[o_r > 1] = 1
        o_r = o_r.unsqueeze(1)

        pix_coords_l = self.pix_coords_l.expand(B, -1, -1, -1, -1).reshape(B * N, 2, H, W).permute(0, 2, 3, 1)
        o_l = F.grid_sample(
            pr,
            pix_coords_l,
            padding_mode="zeros", align_corners=True).reshape(B, N, 2, H, W)
        o_l = o_l.sum(1)
        o_l = o_l[:, 0] * o_l[:, 1]
        o_l[o_l > 1] = 1
        o_l = o_l.unsqueeze(1)

        outputs["mask_novel"] = torch.cat([o_r, o_l.flip(-1)], dim=0)
        outputs["mask_novel"] = outputs["mask_novel"].detach()