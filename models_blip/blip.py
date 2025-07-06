import warnings
warnings.filterwarnings("ignore")
try:
    from models_blip.vit import VisionTransformer, interpolate_pos_embed
    from models_blip.med import BertConfig, BertModel, BertLMHeadModel
    from models_blip.vit import Block
except:
    from vit import VisionTransformer, interpolate_pos_embed
    from med import BertConfig, BertModel, BertLMHeadModel
    from vit import Block
from transformers import BertTokenizer
import torch
from torch import nn
import torch.nn.functional as F
import os
from urllib.parse import urlparse
from timm.models.hub import download_cached_file
import ImageReward
import timm
import torch
import time
import random
import numpy as np
from torch.fft import fft2, fftshift
class BLIP_Base(nn.Module):
    def __init__(self,
                 med_config='configs/med_config.json',
                 image_size=224,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 fix_rate=0.5
                 ):
        super().__init__()
        self.fix_rate = fix_rate
        self.reward = ImageReward.load("ImageReward-v1.0")
        self.visual_encoder=timm.create_model("vit_base_patch16_224", pretrained=True)
    def forward(self, image, captions, mode):
        if isinstance(captions, str):
            captions = [captions]
        prompts = []
        answers = []
        for caption in captions:
            separator = "|||"
            if separator in caption:
                prompt, answer = caption.split(separator, 1)
                prompts.append(prompt.strip())
                answers.append(answer.strip())
            else:
                prompts.append(caption.strip())
                answers.append("")
        self.reward.to(image.device)
        resized_images = image[:, 3:, :, :]
        image = image[:, :3, :, :]
        assert mode in ['image', 'text', 'multimodal'], "mode parameter must be image, text, or multimodal"
        image_embeds = self.reward.blip.visual_encoder(resized_images)
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(resized_images.device)
        text_input = self.reward.blip.tokenizer(prompts, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(image.device)
        text_input_answer = self.reward.blip.tokenizer(answers, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(image.device)
        if mode=='image':
            pass
        elif mode=='text':
            pass
        elif mode=='multimodal':
            image_features = self.visual_encoder(image)
            text_output = self.reward.blip.text_encoder(text_input.input_ids,
                                                     attention_mask = text_input.attention_mask,
                                                     encoder_hidden_states = image_embeds,
                                                     encoder_attention_mask = image_atts,
                                                     return_dict = True,
                                                    )
            text_output_answer = self.reward.blip.text_encoder(text_input_answer.input_ids,
                                                     attention_mask = text_input_answer.attention_mask,
                                                     encoder_hidden_states = image_embeds,
                                                     encoder_attention_mask = image_atts,
                                                     return_dict = True,
                                                    )
            text_features = text_output.last_hidden_state
            text_features_answer=text_output_answer.last_hidden_state
            return text_features,text_features_answer,image_features
class BLIP_Decoder(nn.Module):
    def __init__(self,
                 med_config = 'configs/med_config.json',
                 image_size = 224,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,
                 prompt = 'a picture of ',
                 ):
        super().__init__()
        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_decoder = BertLMHeadModel(config=med_config)
        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids)-1
    def forward(self, image, caption):
        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        text = self.tokenizer(caption, padding='longest', truncation=True, max_length=40, return_tensors="pt").to(image.device)
        text.input_ids[:,0] = self.tokenizer.bos_token_id
        decoder_targets = text.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100)
        decoder_targets[:,:self.prompt_length] = -100
        decoder_output = self.text_decoder(text.input_ids,
                                           attention_mask = text.attention_mask,
                                           encoder_hidden_states = image_embeds,
                                           encoder_attention_mask = image_atts,
                                           labels = decoder_targets,
                                           return_dict = True,
                                          )
        loss_lm = decoder_output.loss
        return loss_lm
    def generate(self, image, sample=False, num_beams=3, max_length=30, min_length=10, top_p=0.9, repetition_penalty=1.0):
        image_embeds = self.visual_encoder(image)
        if not sample:
            image_embeds = image_embeds.repeat_interleave(num_beams,dim=0)
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask":image_atts}
        prompt = [self.prompt] * image.size(0)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(image.device)
        input_ids[:,0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1]
        if sample:
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                  max_length=max_length,
                                                  min_length=min_length,
                                                  do_sample=True,
                                                  top_p=top_p,
                                                  num_return_sequences=1,
                                                  eos_token_id=self.tokenizer.sep_token_id,
                                                  pad_token_id=self.tokenizer.pad_token_id,
                                                  repetition_penalty=1.1,
                                                  **model_kwargs)
        else:
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                  max_length=max_length,
                                                  min_length=min_length,
                                                  num_beams=num_beams,
                                                  eos_token_id=self.tokenizer.sep_token_id,
                                                  pad_token_id=self.tokenizer.pad_token_id,
                                                  repetition_penalty=repetition_penalty,
                                                  **model_kwargs)
        captions = []
        for output in outputs:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)
            captions.append(caption[len(self.prompt):])
        return captions

def blip_feature_extractor(pretrained='',**kwargs):
    model = BLIP_Base(**kwargs)
    return model
def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    return tokenizer
def create_vit(vit, image_size, use_grad_checkpointing=False, ckpt_layer=0, drop_path_rate=0):
    assert vit in ['base', 'large'], "vit parameter must be base or large"
    if vit=='base':
        vision_width = 768
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=12,
                                           num_heads=12, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                           drop_path_rate=0 or drop_path_rate
                                          )
    elif vit=='large':
        vision_width = 1024
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=24,
                                           num_heads=16, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                           drop_path_rate=0.1 or drop_path_rate
                                          )
    return visual_encoder, vision_width
def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")
def load_checkpoint(model,url_or_filename):
    if is_url(url_or_filename):
        cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu')
    elif os.path.isfile(url_or_filename):
        checkpoint = torch.load(url_or_filename, map_location='cpu')
    else:
        raise RuntimeError('checkpoint url or path is invalid')
    state_dict = checkpoint['model']
    state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)
    if 'visual_encoder_m.pos_embed' in model.state_dict().keys():
        state_dict['visual_encoder_m.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                                         model.visual_encoder_m)
    for key in model.state_dict().keys():
        if key in state_dict.keys():
            if state_dict[key].shape!=model.state_dict()[key].shape:
                del state_dict[key]
    msg = model.load_state_dict(state_dict,strict=False)
    print('load checkpoint from %s'%url_or_filename)
    return model,msg
class IQAPooling(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.conv_avg = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.channel_adjust = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels)
        )
    def weighted_adaptive_pool2d(self,feature, weight):
        weight_sum = torch.sum(weight, dim=(2, 3), keepdim=True)
        weight_norm = weight / weight_sum
        weight_norm = weight_norm.expand_as(feature)
        weighted_feature = feature * weight_norm +0.4*feature
        output = torch.sum(weighted_feature, dim=(2, 3), keepdim=True)
        return output
    def forward(self, x,x1_csf):
        x1_csf.to(x.device)
        B, C, h,w = x.shape
        x_spatial = x.view(B, C, h, w).contiguous()
        x=x.view(B, C, h*w).contiguous()
        x1_csf_resized = F.interpolate(x1_csf, size=(x_spatial.shape[-1], x_spatial.shape[-1]), mode='bilinear', align_corners=False)
        quality_weights = self.conv_avg(x_spatial)
        channel_weights = self.channel_adjust(x.mean(dim=-1)).sigmoid()
        weighted = x_spatial * quality_weights
        weighted = weighted * channel_weights.view(B, C, 1, 1).contiguous()
        output=self.weighted_adaptive_pool2d(weighted,x1_csf_resized)
        return output.view(B, C, 1).contiguous()
def get_vit_feature(save_output):
    feat = torch.cat(
        (
            save_output.outputs[8][:, 1:, :],
            save_output.outputs[9][:, 1:, :],
            save_output.outputs[10][:, 1:, :],
            save_output.outputs[11][:, 1:, :]
        ),
        dim=2
    )
    return feat
import torch
import torch.nn as nn
class SaveOutput:
    def __init__(self):
        self.outputs = []
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
    def clear(self):
        self.outputs = []
from torch import einsum
from einops import rearrange, repeat
from inspect import isfunction
def exists(val):
    return val is not None
def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d
class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x, context=None, mask=None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)
        attn = sim.softmax(dim=-1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)
class Projection(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, 1)
        )
    def forward(self, x):
        return self.project(x.unsqueeze(0)).squeeze(0)
class IQARegression(nn.Module):
    def __init__(self, inchannels=768, outchannels=512):
        super().__init__()
        self.down_channel= nn.Conv2d(inchannels*4 , 768, kernel_size=1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        scale = inchannels ** -0.5
        self.cross_attention = CrossAttention(outchannels)
        self.cross_attention_text = CrossAttention(outchannels)
        self.norm1 = nn.LayerNorm(outchannels)
        self.norm2 = nn.LayerNorm(outchannels)
        self.norm3 = nn.LayerNorm(outchannels)
        self.proj = nn.Parameter(scale * torch.randn(inchannels, outchannels))
        self.proj_pool=nn.Linear(35, 1)
        self.gating_network = nn.Sequential(
            nn.Linear(outchannels*2, 4),
            nn.Softmax(dim=1)
        )
        proj_experts = []
        self.proj_nums = 4
        for i in range(self.proj_nums):
            proj_experts.append(Projection(2*outchannels, outchannels))
        self.proj_experts = nn.Sequential(*proj_experts)
        self.iqapool=IQAPooling(512)
    def forward(self, x, text_features,text_features_answer,crop_image_csf):
        f_dis = self.down_channel(x)
        f_dis = self.conv(f_dis)
        B, C, W, H = f_dis.shape
        L = W*H
        f_dis = f_dis.view(B, C, L).permute(0, 2, 1).contiguous()
        text_features = text_features @ self.proj
        text_features_answer =text_features_answer @ self.proj
        f_dis = self.norm1(f_dis)
        f_dis = f_dis + self.cross_attention(f_dis, self.norm2(text_features))
        consistency_text_features = self.cross_attention_text(text_features_answer,text_features)
        consistency_text_features=self.proj_pool(consistency_text_features.permute(0,2,1).contiguous()).squeeze(-1)
        f_dis = f_dis.permute(0, 2, 1).view(B, C, W, H).contiguous()
        f_dis=self.iqapool(f_dis,crop_image_csf).unsqueeze(-1)
        f_dis = f_dis.view(f_dis.size(0), -1).contiguous()
        f_dis=torch.cat((f_dis,self.norm3(consistency_text_features)),1)
        gating_weight = self.gating_network(f_dis)
        gating_weight_value, gating_weight_index = torch.topk(gating_weight, k=3, dim=1)
        preds = torch.tensor([]).to(f_dis.device)
        for i in range(f_dis.size(0)):
            preds_one = torch.tensor([]).to(f_dis.device)
            for j in gating_weight_index[i]:
                _pred = self.proj_experts[j](f_dis[i])
                preds_one = torch.cat((preds_one, _pred.unsqueeze(0)), 0)
            preds = torch.cat((preds, preds_one.unsqueeze(0)), 0)
        pred = torch.sum(preds * gating_weight_value.unsqueeze(2), dim=1)
        return pred
class BLIPRegressionModel(nn.Module):
    def __init__(self, pretrained='', **kwargs):
        super().__init__()
        self.blip_encoder = blip_feature_extractor(med_config='', image_size=224, vit='base')
        self.init_saveoutput()
        self.cross_attention = CrossAttention(512)
        self.down_channel= nn.Conv2d(768*4 , 768, kernel_size=1)
        self.regressor = IQARegression()
    def compute_csf_weight(self, x_spatial):
        B, C, H, W = x_spatial.shape
        device = x_spatial.device
        patch_size = 16
        x_gray = x_spatial.mean(dim=1, keepdim=True)
        x_gray = (x_gray - x_gray.min()) / (x_gray.max() - x_gray.min() + 1e-6)
        h_patches = H // patch_size
        w_patches = W // patch_size
        temp_output = torch.zeros(B, 1, h_patches, w_patches, device=device)
        for i in range(h_patches):
            for j in range(w_patches):
                patch = x_gray[:, :,
                            i*patch_size:(i+1)*patch_size,
                            j*patch_size:(j+1)*patch_size]
                fft_patch = fftshift(fft2(patch, dim=(-2, -1)), dim=(-2, -1))
                magnitude = torch.abs(fft_patch)
                u = torch.fft.fftshift(torch.fft.fftfreq(patch_size, d=0.02, device=device))
                v = torch.fft.fftshift(torch.fft.fftfreq(patch_size, d=0.02, device=device))
                uu, vv = torch.meshgrid(u, v, indexing='ij')
                freq = torch.sqrt(uu**2 + vv**2)
                csf_weights = 2.6 * (0.0192 + 0.114 * freq) * torch.exp(-(0.114 * freq)**1.1)
                temp_output[:, :, i, j] = torch.sum(magnitude * csf_weights)
        sensitivity_map = F.interpolate(temp_output,
                                    size=(H, W),
                                    mode='bilinear',
                                    align_corners=False)
        sensitivity_map = (sensitivity_map - sensitivity_map.min()) / (sensitivity_map.max() - sensitivity_map.min() + 1e-6)
        return sensitivity_map
    def init_saveoutput(self):
        self.save_output = SaveOutput()
        hook_handles = []
        for layer in self.blip_encoder.visual_encoder.blocks:
            handle = layer.register_forward_hook(self.save_output)
            hook_handles.append(handle)
    def forward(self, image, caption):
        crop_image=image[:, :3, :, :]
        crop_image_csf=self.compute_csf_weight(crop_image)
        text_features_answer,text_features,_=self.blip_encoder(image, caption, mode='multimodal')
        vit_dis = get_vit_feature(self.save_output)
        self.save_output.outputs.clear()
        B = vit_dis.shape[0]
        feat = vit_dis.transpose(1, 2).contiguous()
        feat = feat.view(B, 3072, 14, 14).contiguous()
        scores = self.regressor(feat, text_features,text_features_answer,crop_image_csf)
        return scores
def build_agiqa_model(config,device):
    model=BLIPRegressionModel().to(device)
    return model