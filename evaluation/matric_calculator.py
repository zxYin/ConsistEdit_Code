import torch
from torchvision.transforms import Resize
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from torchmetrics.multimodal import CLIPScore
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.regression import MeanSquaredError
import cv2

class VitExtractor:
    BLOCK_KEY = 'block'
    ATTN_KEY = 'attn'
    PATCH_IMD_KEY = 'patch_imd'
    QKV_KEY = 'qkv'
    KEY_LIST = [BLOCK_KEY, ATTN_KEY, PATCH_IMD_KEY, QKV_KEY]

    def __init__(self, model_name, device):
        self.model = torch.hub.load('facebookresearch/dino:main', model_name).to(device)
        self.model.eval()
        self.model_name = model_name
        self.hook_handlers = []
        self.layers_dict = {}
        self.outputs_dict = {}
        for key in VitExtractor.KEY_LIST:
            self.layers_dict[key] = []
            self.outputs_dict[key] = []
        self._init_hooks_data()
        self.device=device

    def _init_hooks_data(self):
        self.layers_dict[VitExtractor.BLOCK_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.layers_dict[VitExtractor.ATTN_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.layers_dict[VitExtractor.QKV_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.layers_dict[VitExtractor.PATCH_IMD_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        for key in VitExtractor.KEY_LIST:
            # self.layers_dict[key] = kwargs[key] if key in kwargs.keys() else []
            self.outputs_dict[key] = []

    def _register_hooks(self, **kwargs):
        for block_idx, block in enumerate(self.model.blocks):
            if block_idx in self.layers_dict[VitExtractor.BLOCK_KEY]:
                self.hook_handlers.append(block.register_forward_hook(self._get_block_hook()))
            if block_idx in self.layers_dict[VitExtractor.ATTN_KEY]:
                self.hook_handlers.append(block.attn.attn_drop.register_forward_hook(self._get_attn_hook()))
            if block_idx in self.layers_dict[VitExtractor.QKV_KEY]:
                self.hook_handlers.append(block.attn.qkv.register_forward_hook(self._get_qkv_hook()))
            if block_idx in self.layers_dict[VitExtractor.PATCH_IMD_KEY]:
                self.hook_handlers.append(block.attn.register_forward_hook(self._get_patch_imd_hook()))

    def _clear_hooks(self):
        for handler in self.hook_handlers:
            handler.remove()
        self.hook_handlers = []

    def _get_block_hook(self):
        def _get_block_output(model, input, output):
            self.outputs_dict[VitExtractor.BLOCK_KEY].append(output)

        return _get_block_output

    def _get_attn_hook(self):
        def _get_attn_output(model, inp, output):
            self.outputs_dict[VitExtractor.ATTN_KEY].append(output)

        return _get_attn_output

    def _get_qkv_hook(self):
        def _get_qkv_output(model, inp, output):
            self.outputs_dict[VitExtractor.QKV_KEY].append(output)

        return _get_qkv_output

    # TODO: CHECK ATTN OUTPUT TUPLE
    def _get_patch_imd_hook(self):
        def _get_attn_output(model, inp, output):
            self.outputs_dict[VitExtractor.PATCH_IMD_KEY].append(output[0])

        return _get_attn_output

    def get_feature_from_input(self, input_img):  # List([B, N, D])
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.BLOCK_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    def get_qkv_feature_from_input(self, input_img):
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.QKV_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    def get_attn_feature_from_input(self, input_img):
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.ATTN_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    def get_patch_size(self):
        return 8 if "8" in self.model_name else 16

    def get_width_patch_num(self, input_img_shape):
        b, c, h, w = input_img_shape
        patch_size = self.get_patch_size()
        return w // patch_size

    def get_height_patch_num(self, input_img_shape):
        b, c, h, w = input_img_shape
        patch_size = self.get_patch_size()
        return h // patch_size

    def get_patch_num(self, input_img_shape):
        patch_num = 1 + (self.get_height_patch_num(input_img_shape) * self.get_width_patch_num(input_img_shape))
        return patch_num

    def get_head_num(self):
        if "dino" in self.model_name:
            return 6 if "s" in self.model_name else 12
        return 6 if "small" in self.model_name else 12

    def get_embedding_dim(self):
        if "dino" in self.model_name:
            return 384 if "s" in self.model_name else 768
        return 384 if "small" in self.model_name else 768

    def get_queries_from_qkv(self, qkv, input_img_shape):
        patch_num = self.get_patch_num(input_img_shape)
        head_num = self.get_head_num()
        embedding_dim = self.get_embedding_dim()
        q = qkv.reshape(patch_num, 3, head_num, embedding_dim // head_num).permute(1, 2, 0, 3)[0]
        return q

    def get_keys_from_qkv(self, qkv, input_img_shape):
        patch_num = self.get_patch_num(input_img_shape)
        head_num = self.get_head_num()
        embedding_dim = self.get_embedding_dim()
        k = qkv.reshape(patch_num, 3, head_num, embedding_dim // head_num).permute(1, 2, 0, 3)[1]
        return k

    def get_values_from_qkv(self, qkv, input_img_shape):
        patch_num = self.get_patch_num(input_img_shape)
        head_num = self.get_head_num()
        embedding_dim = self.get_embedding_dim()
        v = qkv.reshape(patch_num, 3, head_num, embedding_dim // head_num).permute(1, 2, 0, 3)[2]
        return v

    def get_keys_from_input(self, input_img, layer_num):
        qkv_features = self.get_qkv_feature_from_input(input_img)[layer_num]
        keys = self.get_keys_from_qkv(qkv_features, input_img.shape)
        return keys

    def get_keys_self_sim_from_input(self, input_img, layer_num):
        keys = self.get_keys_from_input(input_img, layer_num=layer_num)
        h, t, d = keys.shape
        concatenated_keys = keys.transpose(0, 1).reshape(t, h * d)
        ssim_map = self.attn_cosine_sim(concatenated_keys[None, None, ...])
        return ssim_map
    
    def attn_cosine_sim(self,x, eps=1e-08):
        x = x[0]  # TEMP: getting rid of redundant dimension, TBF
        norm1 = x.norm(dim=2, keepdim=True)
        factor = torch.clamp(norm1 @ norm1.permute(0, 2, 1), min=eps)
        sim_matrix = (x @ x.permute(0, 2, 1)) / factor
        return sim_matrix
    

class LossG(torch.nn.Module):
    def __init__(self, cfg,device):
        super().__init__()

        self.cfg = cfg
        self.device=device
        self.extractor = VitExtractor(model_name=cfg['dino_model_name'], device=device)

        imagenet_norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        global_resize_transform = Resize(cfg['dino_global_patch_size'], max_size=480)

        self.global_transform = transforms.Compose([global_resize_transform,
                                                    imagenet_norm
                                                    ])

        self.lambdas = dict(
            lambda_global_cls=cfg['lambda_global_cls'],
            lambda_global_ssim=0,
            lambda_entire_ssim=0,
            lambda_entire_cls=0,
            lambda_global_identity=0
        )

    def update_lambda_config(self, step):
        if step == self.cfg['cls_warmup']:
            self.lambdas['lambda_global_ssim'] = self.cfg['lambda_global_ssim']
            self.lambdas['lambda_global_identity'] = self.cfg['lambda_global_identity']

        if step % self.cfg['entire_A_every'] == 0:
            self.lambdas['lambda_entire_ssim'] = self.cfg['lambda_entire_ssim']
            self.lambdas['lambda_entire_cls'] = self.cfg['lambda_entire_cls']
        else:
            self.lambdas['lambda_entire_ssim'] = 0
            self.lambdas['lambda_entire_cls'] = 0

    def forward(self, outputs, inputs):
        self.update_lambda_config(inputs['step'])
        losses = {}
        loss_G = 0

        if self.lambdas['lambda_global_ssim'] > 0:
            losses['loss_global_ssim'] = self.calculate_global_ssim_loss(outputs['x_global'], inputs['A_global'])
            loss_G += losses['loss_global_ssim'] * self.lambdas['lambda_global_ssim']

        if self.lambdas['lambda_entire_ssim'] > 0:
            losses['loss_entire_ssim'] = self.calculate_global_ssim_loss(outputs['x_entire'], inputs['A'])
            loss_G += losses['loss_entire_ssim'] * self.lambdas['lambda_entire_ssim']

        if self.lambdas['lambda_entire_cls'] > 0:
            losses['loss_entire_cls'] = self.calculate_crop_cls_loss(outputs['x_entire'], inputs['B_global'])
            loss_G += losses['loss_entire_cls'] * self.lambdas['lambda_entire_cls']

        if self.lambdas['lambda_global_cls'] > 0:
            losses['loss_global_cls'] = self.calculate_crop_cls_loss(outputs['x_global'], inputs['B_global'])
            loss_G += losses['loss_global_cls'] * self.lambdas['lambda_global_cls']

        if self.lambdas['lambda_global_identity'] > 0:
            losses['loss_global_id_B'] = self.calculate_global_id_loss(outputs['y_global'], inputs['B_global'])
            loss_G += losses['loss_global_id_B'] * self.lambdas['lambda_global_identity']

        losses['loss'] = loss_G
        return losses

    def calculate_global_ssim_loss(self, outputs, inputs):
        loss = 0.0
        for a, b in zip(inputs, outputs):  # avoid memory limitations
            a = self.global_transform(a)
            b = self.global_transform(b)
            with torch.no_grad():
                target_keys_self_sim = self.extractor.get_keys_self_sim_from_input(a.unsqueeze(0), layer_num=11)
            keys_ssim = self.extractor.get_keys_self_sim_from_input(b.unsqueeze(0), layer_num=11)
            loss += F.mse_loss(keys_ssim, target_keys_self_sim)
        return loss

    def calculate_crop_cls_loss(self, outputs, inputs):
        loss = 0.0
        for a, b in zip(outputs, inputs):  # avoid memory limitations
            a = self.global_transform(a).unsqueeze(0).to(self.device)
            b = self.global_transform(b).unsqueeze(0).to(self.device)
            cls_token = self.extractor.get_feature_from_input(a)[-1][0, 0, :]
            with torch.no_grad():
                target_cls_token = self.extractor.get_feature_from_input(b)[-1][0, 0, :]
            loss += F.mse_loss(cls_token, target_cls_token)
        return loss

    def calculate_global_id_loss(self, outputs, inputs):
        loss = 0.0
        for a, b in zip(inputs, outputs):
            a = self.global_transform(a)
            b = self.global_transform(b)
            with torch.no_grad():
                keys_a = self.extractor.get_keys_from_input(a.unsqueeze(0), 11)
            keys_b = self.extractor.get_keys_from_input(b.unsqueeze(0), 11)
            loss += F.mse_loss(keys_a, keys_b)
        return loss
    

class MetricsCalculator:
    def __init__(self, device) -> None:
        self.device=device
        self.clip_metric_calculator = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14").to(device)
        self.psnr_metric_calculator = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.lpips_metric_calculator = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to(device)
        self.mse_metric_calculator = MeanSquaredError().to(device)
        self.ssim_metric_calculator = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.structure_distance_metric_calculator = LossG(cfg={
                            'dino_model_name': 'dino_vitb8', # ['dino_vitb8', 'dino_vits8', 'dino_vitb16', 'dino_vits16']
                            'dino_global_patch_size': 224,
                            'lambda_global_cls': 10.0,
                            'lambda_global_ssim': 1.0,
                            'lambda_global_identity': 1.0,
                            'entire_A_every':75,
                            'lambda_entire_cls':10,
                            'lambda_entire_ssim':1.0
                        },device=device)
    
    def calculate_clip_similarity(self, img, txt, mask=None):
        img = np.array(img)
        
        if mask is not None:
            mask = np.array(mask)
            img = np.uint8(img * mask)
            
        img_tensor=torch.tensor(img).permute(2,0,1).to(self.device)
        
        score = self.clip_metric_calculator(img_tensor, txt)
        score = score.cpu().item()
        
        return score

    def _rgb_to_yuv(self, image):
        """RGB转YUV色彩空间"""
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        r, g, b = image[0], image[1], image[2]
        # y = 0.299 * r + 0.587 * g + 0.114 * b
        # u = 0.492 * (b - y)
        # v = 0.877 * (r - y)
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.14713 * r - 0.28886 * g + 0.436 * b
        v =  0.615 * r - 0.51499 * g - 0.10001 * b
        return torch.stack([y, u, v], dim=0).to(dtype=torch.float32, device=self.device)

    def _calc_gradient(self, image):
        """计算图像梯度"""
        image = image.mean(dim=1, keepdim=True)  # 灰度化
        
        # Sobel算子
        sobel_x = torch.tensor([[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]]], 
                              dtype=torch.float32).to(self.device)
        sobel_y = torch.tensor([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]], 
                              dtype=torch.float32).to(self.device)
        
        grad_x = F.conv2d(image, sobel_x, padding=1)
        grad_y = F.conv2d(image, sobel_y, padding=1)
        
        magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        return magnitude

    def calculate_gradient_sim(self, img1, img2):
        """计算梯度相似性"""
        img1_yuv = self._rgb_to_yuv(img1).unsqueeze(0)
        img2_yuv = self._rgb_to_yuv(img2).unsqueeze(0)
        y1 = img1_yuv[:, 0:1]
        y2 = img2_yuv[:, 0:1]
        
        grad1 = self._calc_gradient(y1)
        grad2 = self._calc_gradient(y2)
        
        # 归一化
        max_grad1 = torch.max(grad1)
        max_grad2 = torch.max(grad2)
        max_val = max(max_grad1.item(), max_grad2.item(), 1e-6)
        
        grad1 = grad1 / max_val
        grad2 = grad2 / max_val
        
        return F.cosine_similarity(grad1.flatten(), grad2.flatten(), dim=0).item()

    def calculate_contrast_sim(self, img1, img2, kernel_size=7, sigma=1.0):
        """计算局部对比度相似性"""
        img1_yuv = self._rgb_to_yuv(img1).unsqueeze(0)
        img2_yuv = self._rgb_to_yuv(img2).unsqueeze(0)
        y1 = img1_yuv[:, 0:1]
        y2 = img2_yuv[:, 0:1]

        x = torch.arange(kernel_size).float() - kernel_size//2
        y = torch.arange(kernel_size).float() - kernel_size//2
        x, y = torch.meshgrid(x, y, indexing='ij')
        
        kernel = torch.exp(-(x**2 + y**2)/(2*sigma**2))
        kernel /= kernel.sum()
        gaussian_kernel = kernel.view(1, 1, kernel_size, kernel_size).to(dtype=torch.float32, device=self.device)
        
        # 应用高斯模糊
        y1_smooth = F.conv2d(y1, gaussian_kernel, padding=kernel_size//2)
        y2_smooth = F.conv2d(y2, gaussian_kernel, padding=kernel_size//2)
        
        # 计算局部对比度 (原始 - 平滑)
        local_contrast1 = y1 - y1_smooth
        local_contrast2 = y2 - y2_smooth
        
        # 归一化
        max_contrast1 = torch.max(torch.abs(local_contrast1))
        max_contrast2 = torch.max(torch.abs(local_contrast2))
        max_val = max(max_contrast1.item(), max_contrast2.item(), 1e-6)
        
        local_contrast1 = local_contrast1 / max_val
        local_contrast2 = local_contrast2 / max_val
        
        # 计算相似性
        return F.cosine_similarity(local_contrast1.flatten(), local_contrast2.flatten(), dim=0).item()

    def calculate_tile_correlation(self, img1, img2):
        """改进的区域相关性分析"""
        img1_yuv = self._rgb_to_yuv(img1).unsqueeze(0)
        img2_yuv = self._rgb_to_yuv(img2).unsqueeze(0)
        y1 = img1_yuv[:, 0:1]
        y2 = img2_yuv[:, 0:1]
        
        H, W = y1.shape[2], y1.shape[3]
        
        # 自适应网格大小
        grid_size = min(4, max(1, min(H, W) // 64))
        
        tile_h = max(1, H // grid_size)
        tile_w = max(1, W // grid_size)
        
        # 收集区域平均值
        tile_values1 = []
        tile_values2 = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                start_h = i * tile_h
                end_h = min(H, (i+1) * tile_h)
                start_w = j * tile_w
                end_w = min(W, (j+1) * tile_w)
                
                # 确保区域有效
                if start_h < end_h and start_w < end_w and (end_h - start_h) > 4 and (end_w - start_w) > 4:
                    tile1 = y1[:, :, start_h:end_h, start_w:end_w].mean()
                    tile2 = y2[:, :, start_h:end_h, start_w:end_w].mean()
                    tile_values1.append(tile1.item())
                    tile_values2.append(tile2.item())
        
        tile_tensor1 = torch.tensor(tile_values1)
        tile_tensor2 = torch.tensor(tile_values2)
        
        # 皮尔逊相关系数
        mean1 = tile_tensor1.mean()
        mean2 = tile_tensor2.mean()
        num = torch.sum((tile_tensor1 - mean1) * (tile_tensor2 - mean2))
        den = torch.sqrt(torch.sum((tile_tensor1 - mean1)**2) * torch.sum((tile_tensor2 - mean2)**2))
        
        if den.abs() < 1e-6:
            return 0.0
            
        return (num / den).item()

    def calculate_bhattacharyya_sim(self, img1, img2):
        """使用Hellinger距离测量亮度分布的相似性"""
        img1_yuv = self._rgb_to_yuv(img1).unsqueeze(0)
        img2_yuv = self._rgb_to_yuv(img2).unsqueeze(0)
        y1 = img1_yuv[:, 0].flatten()
        y2 = img2_yuv[:, 0].flatten()
        
        # 创建直方图
        hist1 = torch.histc(y1, bins=256, min=0, max=1) / max(1, len(y1))
        hist2 = torch.histc(y2, bins=256, min=0, max=1) / max(1, len(y2))
        
        # 添加平滑避免零
        eps = 1e-10
        hist1 = hist1 + eps
        hist2 = hist2 + eps
        hist1 /= hist1.sum()
        hist2 /= hist2.sum()
        
        # 计算Bhattacharyya系数
        bc = torch.sqrt(hist1 * hist2).sum()
        
        # 转换为相似性分数 (距离越小，相似性越高)
        return bc.item()

  
    def calculate_psnr(self, img_pred, img_gt, mask_pred=None, mask_gt=None):
        img_pred = np.array(img_pred).astype(np.float32)/255
        img_gt = np.array(img_gt).astype(np.float32)/255
        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."

        if mask_pred is not None:
            mask_pred = np.array(mask_pred).astype(np.float32)
            img_pred = img_pred * mask_pred
        if mask_gt is not None:
            mask_gt = np.array(mask_gt).astype(np.float32)
            img_gt = img_gt * mask_gt
            
        img_pred_tensor=torch.tensor(img_pred).permute(2,0,1).unsqueeze(0).to(self.device)
        img_gt_tensor=torch.tensor(img_gt).permute(2,0,1).unsqueeze(0).to(self.device)
            
        score = self.psnr_metric_calculator(img_pred_tensor,img_gt_tensor)
        score = score.cpu().item()
        
        return score
    
    def calculate_lpips(self, img_pred, img_gt, mask_pred=None, mask_gt=None):
        img_pred = np.array(img_pred).astype(np.float32)/255
        img_gt = np.array(img_gt).astype(np.float32)/255
        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."

        if mask_pred is not None:
            mask_pred = np.array(mask_pred).astype(np.float32)
            img_pred = img_pred * mask_pred
        if mask_gt is not None:
            mask_gt = np.array(mask_gt).astype(np.float32)
            img_gt = img_gt * mask_gt
            
        img_pred_tensor=torch.tensor(img_pred).permute(2,0,1).unsqueeze(0).to(self.device)
        img_gt_tensor=torch.tensor(img_gt).permute(2,0,1).unsqueeze(0).to(self.device)
            
        score =  self.lpips_metric_calculator(img_pred_tensor*2-1,img_gt_tensor*2-1)
        score = score.cpu().item()
        
        return score
    
    def calculate_mse(self, img_pred, img_gt, mask_pred=None, mask_gt=None):
        img_pred = np.array(img_pred).astype(np.float32)/255
        img_gt = np.array(img_gt).astype(np.float32)/255
        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."

        if mask_pred is not None:
            mask_pred = np.array(mask_pred).astype(np.float32)
            img_pred = img_pred * mask_pred
        if mask_gt is not None:
            mask_gt = np.array(mask_gt).astype(np.float32)
            img_gt = img_gt * mask_gt
            
        img_pred_tensor=torch.tensor(img_pred).permute(2,0,1).to(self.device)
        img_gt_tensor=torch.tensor(img_gt).permute(2,0,1).to(self.device)
            
        score =  self.mse_metric_calculator(img_pred_tensor.contiguous(),img_gt_tensor.contiguous())
        score = score.cpu().item()
        
        return score

    def calculate_canny_ssim(self, img_pred, img_gt, mask_pred=None, mask_gt=None):
        low_threshold = 100
        high_threshold = 200
        pred_canny = cv2.Canny(np.array(img_pred)[..., ::-1], low_threshold, high_threshold)
        gt_canny = cv2.Canny(np.array(img_gt)[..., ::-1], low_threshold, high_threshold)

        img_pred = np.array(pred_canny).astype(np.float32)/255
        img_gt = np.array(gt_canny).astype(np.float32)/255

        img_pred = np.repeat(img_pred[:, :, np.newaxis], 3, axis=2)
        img_gt = np.repeat(img_gt[:, :, np.newaxis], 3, axis=2)

        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."

        if mask_pred is not None:
            mask_pred = np.array(mask_pred).astype(np.float32)
            img_pred = img_pred * mask_pred
        if mask_gt is not None:
            mask_gt = np.array(mask_gt).astype(np.float32)
            img_gt = img_gt * mask_gt
            
        img_pred_tensor=torch.tensor(img_pred).permute(2,0,1).unsqueeze(0).to(self.device)
        img_gt_tensor=torch.tensor(img_gt).permute(2,0,1).unsqueeze(0).to(self.device)
            
        score =  self.ssim_metric_calculator(img_pred_tensor,img_gt_tensor)
        score = score.cpu().item()
        
        return score
    
    def calculate_ssim(self, img_pred, img_gt, mask_pred=None, mask_gt=None):
        img_pred = np.array(img_pred).astype(np.float32)/255
        img_gt = np.array(img_gt).astype(np.float32)/255
        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."

        if mask_pred is not None:
            mask_pred = np.array(mask_pred).astype(np.float32)
            img_pred = img_pred * mask_pred
        if mask_gt is not None:
            mask_gt = np.array(mask_gt).astype(np.float32)
            img_gt = img_gt * mask_gt
            
        img_pred_tensor=torch.tensor(img_pred).permute(2,0,1).unsqueeze(0).to(self.device)
        img_gt_tensor=torch.tensor(img_gt).permute(2,0,1).unsqueeze(0).to(self.device)
            
        score =  self.ssim_metric_calculator(img_pred_tensor,img_gt_tensor)
        score = score.cpu().item()
        
        return score
    
        
    def calculate_structure_distance(self, img_pred, img_gt, mask_pred=None, mask_gt=None, use_gpu = True):
        img_pred = np.array(img_pred).astype(np.float32)
        img_gt = np.array(img_gt).astype(np.float32)
        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."

        if mask_pred is not None:
            mask_pred = np.array(mask_pred).astype(np.float32)
            img_pred = img_pred * mask_pred
        if mask_gt is not None:
            mask_gt = np.array(mask_gt).astype(np.float32)
            img_gt = img_gt * mask_gt

        
        img_pred = torch.from_numpy(np.transpose(img_pred, axes=(2, 0, 1))).to(self.device)
        img_gt = torch.from_numpy(np.transpose(img_gt, axes=(2, 0, 1))).to(self.device)
        img_pred = torch.unsqueeze(img_pred, 0)
        img_gt = torch.unsqueeze(img_gt, 0)
        
        structure_distance = self.structure_distance_metric_calculator.calculate_global_ssim_loss(img_gt, img_pred)
        
        return structure_distance.data.cpu().numpy()

