import einops
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)
from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

import torch
import torch.nn as nn
from ldm.modules.attention import BasicTransformerBlock
from ldm.modules.diffusionmodules.util import checkpoint, FourierEmbedder
import torch.nn.functional as F
from ldm.modules.encoders.modules import FrozenDinoV2EncoderFeatures

from datasets.data_transfer import prepare_batch_hetero, get_clip_feature,project
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
from PIL import Image

# 走这里，不走UNet的forward, 但是用UNet的structure, 修改在UNet的structure改
class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None,objs=None, only_mid_control=False, **kwargs):
        hs = []
        # import pdb; pdb.set_trace()
        # with torch.no_grad():   # 为什么no_grad? 不用会怎么样？
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context,objs)
            hs.append(h)
        h = self.middle_block(h, emb, context,objs)

        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context,objs)

        h = h.type(x.dtype)
        return self.out(h)


class ControlNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch
    
    

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context,objs,**kwargs): #负责模块融合, objs为 image grounding token, 对应high frequency maps
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb) # 1,1280
        # import pdb; pdb.set_trace()
        # 1,320,64,64
        guided_hint = self.input_hint_block(hint, emb, context,objs)  # hint不能加text grounding
        outs = []

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                # skip the first layer
                h = guided_hint
                guided_hint = None
            else:
                h_new = module(h, emb, context,objs) #这里应该不需要grounding?
                h =  h_new 
            outs.append(zero_conv(h, emb, context,objs))

        h_new = self.middle_block(h, emb, context,objs)  
        outs.append(self.middle_block_out(h_new, emb, context,objs)) 
        return outs


class ControlLDM(LatentDiffusion):

    def __init__(self, control_stage_config, control_key, only_mid_control, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        # self.control_model_txt = instantiate_from_config(control_stage_config_txt)  # add control_stage_config_txt，与controlnet一致
        # import pdb; pdb.set_trace()
        # self.grounding_model = instantiate_from_config(grounding_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13
        self.position_net = PositionNet_txt(in_dim=768, out_dim=1024)
        self.position_net_image = PositionNet_dino_image3(in_dim=1024, out_dim=1024)  #试一试不对patch只对image ground
        self.position_net_textimage = PositionNet_dino_textimage(in_dim_txt=768, in_dim_image=1024, out_dim=1024)
        # self.DinoFeatureExtractor = FrozenDinoV2EncoderFeatures
        # import pdb; pdb.set_trace()
        # self.DinoFeatureExtractor = instantiate_from_config(config={'target': 'ldm.modules.encoders.modules.FrozenDinoV2Encoder', 'weight': 'checkpoints/dinov2_vitg14_pretrain.pth'})
        '''
        version = "openai/clip-vit-large-patch14"
        self.clip_model = CLIPModel.from_pretrained(version).cuda()
        self.processor = CLIPProcessor.from_pretrained(version)
        self.transform_to_pil = transforms.ToPILImage()
        # import pdb; pdb.set_trace()
        self.get_clip_feature = get_clip_feature(model=self.clip_model, processor=self.processor,input=None, is_image=True)
        self.projection_matrix = torch.load('projection_matrix').cuda()
        '''
    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        # print(batch.keys()) #dict_keys(['id', 'jpg', 'boxes', 'masks', 'image_masks', 'text_masks', 'text_embeddings', 'image_embeddings', 'caption'])
                            #  "box_ref", "image_embeddings_ref", 'image_mask_ref', 'text_mask_ref'
        
        # 3.31 判断是否用mvimgnet:
        
        # has_embeddings = 'image_embeddings' in batch.keys() and 'text_embeddings' in batch.keys()
        # if has_embeddings: 
        #     batch['jpg'] = batch['jpg'].permute(0,2,3,1)
        #     batch['ref'] = batch['ref'].permute(0,2,3,1) #torch.randn(batch['jpg'].shape[0],224,224,3)
        #     # batch['ref'] = torch.zeros(batch['jpg'].shape[0],224,224,3)
        #     batch['hint'] = torch.zeros(batch['jpg'].shape[0],512,512,4)
        # else:
        #     # load CLIP text encoder
        #     batch = prepare_batch_hetero(self.clip_model, self.processor, batch)
        
        if 'hint' in batch.keys():
            # 如果没法dataloader层面解决，就用这段代码：
            
            # image_embeddings = []
            # for i in range(batch['jpg'].shape[0]):
            #     ref = batch['ref'][i]
            #     # ref_embedding = self.get_clip_feature(model=self.clip_model, processor=self.processor, input=ref.permute(2,0,1), is_image=True)
            #     ref_embedding = get_clip_feature(model=self.clip_model, processor=self.processor, input=ref.permute(2,0,1), is_image=True)
            #     image_embeddings.append(torch.cat((ref_embedding, torch.zeros((29, 768)).cuda()), dim=0)) 
            # batch['image_embeddings'] = torch.stack(image_embeddings, dim=0)            
            pass
            
        # batch['ref'] = torch.zeros(batch['jpg'].shape[0],224,224,3)
        else: 
            batch['jpg'] = batch['jpg'].permute(0,2,3,1)
            batch['ref'] = batch['ref'].permute(0,2,3,1) #torch.randn(batch['jpg'].shape[0],224,224,3)
            batch['hint'] = torch.zeros(batch['jpg'].shape[0],512,512,4)   
        
        
        x, c_txt = super().get_input(batch, self.first_stage_key, cond_key='txt', *args, **kwargs) # [16, 77, 1024]
        # import pdb; pdb.set_trace()
        c_txt_ground = self.position_net(batch['boxes'].float() , batch['masks'].float() , batch['text_embeddings'].float() ) #新维度 [B, 30, 1024], grounding token, 30是允许的最多bbox数
        c_txt_all = torch.cat((c_txt,c_txt_ground ),dim=1)  #[B,334,1024]
        #最大的bbox, bbox object 加 grounding
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs) #'jpg'  c.shape[16,257,1024] 原本就是jpg
        # import pdb; pdb.set_trace()
        image_embeddings_ref = self.DinoFeatureExtractor.encode(batch['ref'].permute(0,3,1,2).float())
        # c_img_ground = self.position_net_image(batch['box_ref'], batch['image_mask_ref'], image_embeddings_ref) # [8, 1, 1024]
        '''
        c_image_ground = torch.cat([image_embeddings_ref, c_txt_ground[:,:1,:]],dim=1)  # 取第0个是因为对应最大的ref image
        '''

        '''
        # import pdb; pdb.set_trace()
        c_image_ground_new = self.position_net_image(batch['box_ref'], batch['image_mask_ref'], image_embeddings_ref)
        import pdb; pdb.set_trace()
        # 以下为text和image联合ground
        c_textimage_ground = self.position_net_textimage(batch['box_ref'], batch['image_mask_ref'], batch['text_embeddings'].float(),image_embeddings_ref)
        '''
        # 3.31尝试：用原来的c
        # import pdb; pdb.set_trace()
        
        # 4.4 假如只用text grounding
        
        # c = torch.zeros(batch['jpg'].shape[0],257,1024).to(self.device)
        # c = c_image_ground_new
        
        # import pdb; pdb.set_trace()
        c_txt_ground1 = self.position_net(torch.zeros(batch['boxes'].shape).cuda(), torch.zeros(batch['masks'].shape).cuda(), torch.zeros(batch['text_embeddings'].shape).cuda()) #新维度 [B, 30, 1024], grounding token, 30是允许的最多bbox数
        self.txt_ground1 = c_txt_ground1[:,:1,:]
        # c = torch.cat((c,c_txt,c_txt_ground ),dim=1)  #[B,334,1024]

        # add txt?
        
        #进行concat:
        # c = torch.cat((c,c_txt,c_txt_ground ),dim=1)  #[B,334,1024]
        # import pdb; pdb.set_trace()
        c = torch.cat((c,c_txt),dim=1)  #[B,334,1024]
        # import pdb; pdb.set_trace()
        control = batch[self.control_key]  #'hint'  hint替换为layout? 本身就包含了layout信息
        
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float() #([B, 4, 512, 512])
        self.time_steps = batch['time_steps']
        return x, dict(c_crossattn=[c], c_concat=[control], objs = [c_txt_ground])  # 3.23 是不是ground的问题？不拿c_txt_all试 # add ground

    # TODO: 3.14 需要在sample函数中加上Objs?
    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        # import pdb; pdb.set_trace()
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)
        objs = torch.cat(cond['objs'], 1) #cond['objs'] 第二次就没了？

        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt, objs = objs)  #这里control concat
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, objs=objs, only_mid_control=self.only_mid_control)
        return eps
    # 注意这里做uncontional 的shape修改
    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        # import pdb; pdb.set_trace()
        uncond =  self.get_learned_conditioning([ torch.zeros((1,3,224,224)) ] * N)
        uncond_txt = self.get_learned_conditioning_txt([""] * N)
        uncond = torch.cat((uncond,uncond_txt),dim=1)
        
        # uncond = torch.cat([uncond, self.txt_ground1.cuda()],dim=1)
        # 3.31试：还原原来的
        # uncond = torch.cat([uncond, torch.zeros(N,1,1024).cuda()],dim=1)
        
        
        return uncond

    @torch.no_grad()
    def log_images(self, batch, N=16, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None   #batch.keys()： ref, jpg,hint,extra_sizes, tar_box_yyxx_crop, time_steps
        
        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        # import pdb; pdb.set_trace()
        c_cat, c, c_grounding = c["c_concat"][0][:N], c["c_crossattn"][0][:N], c["objs"][0][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        # import pdb; pdb.set_trace()
        log["reconstruction"] = self.decode_first_stage(z) 

        # ==== visualize the shape mask or the high-frequency map ====
        guide_mask = (c_cat[:,-1,:,:].unsqueeze(1) + 1) * 0.5
        guide_mask = torch.cat([guide_mask,guide_mask,guide_mask],1) #[B,2,512,512]
        HF_map  = c_cat[:,:3,:,:] #* 2.0 - 1.0
        # res = c_cat[:,3,:,:]
        # # import pdb; pdb.set_trace()
        log["control"] = HF_map   #用high frequency map做control
        # log['res'] = torch.cat([res, res, res], dim=1)

        cond_image = batch[self.cond_stage_key].cpu().numpy().copy()   #[16,224,224,3]
        log["conditioning"] = torch.permute( torch.tensor(cond_image), (0,3,1,2)) * 2.0 - 1.0  
        # import pdb; pdb.set_trace()
        layout_image = batch['layout'].cpu().numpy().copy()
        log["layout"] = torch.permute(torch.tensor(layout_image), (0,3,1,2))

        layout_all = batch['layout_all'].cpu().numpy().copy()
        log["all_layout"] = torch.permute(torch.tensor(layout_all), (0,3,1,2))
        
        log["txt"] = log_txt_as_img((512, 512), batch["txt"], size=16)
        
        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid
        
        if unconditional_guidance_scale > 1.0:
            # import pdb; pdb.set_trace()
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_grounding = c_grounding
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross],"objs":[uc_grounding]}
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c],"objs":[c_grounding]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg #* 2.0 - 1.0
        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        # import pdb; pdb.set_trace()
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        # import pdb; pdb.set_trace()
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        try: #判断一下
            params += list(self.cond_stage_model.projector.parameters()) #这里是针对dinov2加的
        except:
            pass
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()


class PositionNet_txt(nn.Module):
    def __init__(self,  in_dim, out_dim, fourier_freqs=8):
        super().__init__()
        # import pdb; pdb.set_trace()
        self.in_dim = in_dim
        self.out_dim = out_dim 

        self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        self.position_dim = fourier_freqs*2*4 # 2 is sin&cos, 4 is xyxy 
      
        self.linears = nn.Sequential(
            nn.Linear( self.in_dim + self.position_dim, 512),
            nn.SiLU(),
            nn.Linear( 512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )
        
        self.null_positive_feature = torch.nn.Parameter(torch.zeros([self.in_dim]))
        self.null_position_feature = torch.nn.Parameter(torch.zeros([self.position_dim]))
  

    def forward(self, boxes, masks, positive_embeddings):
        B, N, _ = boxes.shape 
        masks = masks.unsqueeze(-1)
        # import pdb; pdb.set_trace()
        # embedding position (it may includes padding as placeholder)
        xyxy_embedding = self.fourier_embedder(boxes) # B*N*4 --> B*N*C

        # learnable null embedding 
        positive_null = self.null_positive_feature.view(1,1,-1)
        xyxy_null =  self.null_position_feature.view(1,1,-1)

        # replace padding with learnable null embedding 
        positive_embeddings = positive_embeddings*masks + (1-masks)*positive_null
        xyxy_embedding = xyxy_embedding*masks + (1-masks)*xyxy_null

        objs = self.linears(  torch.cat([positive_embeddings, xyxy_embedding], dim=-1)  )
        assert objs.shape == torch.Size([B,N,self.out_dim])        
        return objs

class PositionNet_dino_image(nn.Module):
    def __init__(self, in_dim, out_dim, fourier_freqs=8):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim 

        self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        self.position_dim = fourier_freqs*2*4 # 2 is sin&cos, 4 is xyxy 

        # -------------------------------------------------------------- #
        self.linears_text = nn.Sequential(
            nn.Linear( self.in_dim + self.position_dim, 512),
            nn.SiLU(),
            nn.Linear( 512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )

        self.linears_image = nn.Sequential(
            nn.Linear( self.in_dim + self.position_dim, 512),
            nn.SiLU(),
            nn.Linear( 512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )
        
        # -------------------------------------------------------------- #
        self.null_text_feature = torch.nn.Parameter(torch.zeros([self.in_dim]))
        self.null_image_feature = torch.nn.Parameter(torch.zeros([self.in_dim]))
        self.null_position_feature = torch.nn.Parameter(torch.zeros([self.position_dim]))
  

    def forward(self, boxes, image_masks, image_embeddings):
        B, N, _ = boxes.shape 
        # masks = masks.unsqueeze(-1) # B*N*1 
        # text_masks = text_masks.unsqueeze(-1) # B*N*1 
        image_masks = image_masks.unsqueeze(-1) # B*N*1
        
        # embedding position (it may includes padding as placeholder)
        xyxy_embedding = self.fourier_embedder(boxes) # B*N*4 --> B*N*C

        # learnable null embedding 
        # text_null  = self.null_text_feature.view(1,1,-1) # 1*1*C  (C=768)
        image_null = self.null_image_feature.view(1,1,-1) # 1*1*C
        xyxy_null  = self.null_position_feature.view(1,1,-1) # 1*1*C

        # replace padding with learnable null embedding 
        # import pdb; pdb.set_trace()
        # text_embeddings  = text_embeddings*text_masks  + (1-text_masks)*text_null       # [2,30,768]
        image_embeddings = image_embeddings*image_masks + (1-image_masks)*image_null    # [2,30,768]
        xyxy_embedding = xyxy_embedding*image_masks+ (1-image_masks)*xyxy_null                     # [2, 30, 64]

        # objs_text  = self.linears_text(  torch.cat([text_embeddings, xyxy_embedding], dim=-1)  ) # [2, 30, 768]
        # import pdb; pdb.set_trace()
        # 需要将xyxy_embedding复制到每一个对应的位置，因此需要repeat扩展
        xyxy_repeated = xyxy_embedding.repeat(1, 257, 1)  # 结果形状为[4, 257, 64]

        # 然后，沿dim=-1连接image_embedding和xyxy_repeated

        grounded_image_embeddings = torch.cat((image_embeddings, xyxy_repeated), dim=-1)  # 结果形状为[4, 257, 1088]
        objs_image = self.linears_image(grounded_image_embeddings)
        
        # objs_image = self.linears_image( torch.cat([image_embeddings,xyxy_embedding], dim=-1)  ) # [2, 30, 768]
        # objs = torch.cat( [objs_text,objs_image], dim=1 )  # [2, 30, 768]

        # assert objs_image.shape == torch.Size([B,N,self.out_dim])        
        return objs_image

class PositionNet_dino_image2(nn.Module):
    def __init__(self, in_dim, out_dim, fourier_freqs=8):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim 

        self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        self.position_dim = fourier_freqs*2*4 # 2 is sin&cos, 4 is xyxy 

        # -------------------------------------------------------------- #
        self.linears_text = nn.Sequential(
            nn.Linear( self.in_dim + self.position_dim, 512),
            nn.SiLU(),
            nn.Linear( 512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )

        self.linears_image = nn.Sequential(
            nn.Linear( self.in_dim + self.position_dim, 512),
            nn.SiLU(),
            nn.Linear( 512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )
        
        # -------------------------------------------------------------- #
        self.null_text_feature = torch.nn.Parameter(torch.zeros([self.in_dim]))
        self.null_image_feature = torch.nn.Parameter(torch.zeros([self.in_dim]))
        self.null_position_feature = torch.nn.Parameter(torch.zeros([self.position_dim]))
  

    def forward(self, boxes, image_masks, image_embeddings):
        B, N, _ = boxes.shape 
        # masks = masks.unsqueeze(-1) # B*N*1 
        # text_masks = text_masks.unsqueeze(-1) # B*N*1 
        image_masks = image_masks.unsqueeze(-1) # B*N*1
        
        # embedding position (it may includes padding as placeholder)
        xyxy_embedding = self.fourier_embedder(boxes) # B*N*4 --> B*N*C

        # learnable null embedding 
        # text_null  = self.null_text_feature.view(1,1,-1) # 1*1*C  (C=768)
        image_null = self.null_image_feature.view(1,1,-1) # 1*1*C
        xyxy_null  = self.null_position_feature.view(1,1,-1) # 1*1*C

        # replace padding with learnable null embedding 
        # import pdb; pdb.set_trace()
        # text_embeddings  = text_embeddings*text_masks  + (1-text_masks)*text_null       # [2,30,768]
        image_embeddings = image_embeddings*image_masks + (1-image_masks)*image_null    # [2,30,768]
        xyxy_embedding = xyxy_embedding*image_masks+ (1-image_masks)*xyxy_null                     # [2, 30, 64]

        # objs_text  = self.linears_text(  torch.cat([text_embeddings, xyxy_embedding], dim=-1)  ) # [2, 30, 768]
        # import pdb; pdb.set_trace()
        # 需要将xyxy_embedding复制到每一个对应的位置，因此需要repeat扩展
        # 4.3 new version:

        xyxy_filled = torch.zeros([xyxy_embedding.shape[0], 256, 64]).cuda()  # 结果形状为[4, 257, 64]
        xyxy_repeated = torch.cat([xyxy_embedding, xyxy_filled], dim=1)

        # 然后，沿dim=-1连接image_embedding和xyxy_repeated
        
        grounded_image_embeddings = torch.cat([image_embeddings, xyxy_repeated], dim=-1)  # 结果形状为[4, 257, 1088]
        objs_image = self.linears_image(grounded_image_embeddings)
        
        # objs_image = self.linears_image( torch.cat([image_embeddings,xyxy_embedding], dim=-1)  ) # [2, 30, 768]
        # objs = torch.cat( [objs_text,objs_image], dim=1 )  # [2, 30, 768]

        # assert objs_image.shape == torch.Size([B,N,self.out_dim])        
        return objs_image

class PositionNet_dino_image3(nn.Module):
    def __init__(self, in_dim, out_dim, fourier_freqs=8):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim 

        self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        self.position_dim = fourier_freqs*2*4 # 2 is sin&cos, 4 is xyxy 

        # -------------------------------------------------------------- #
        self.linears_text = nn.Sequential(
            nn.Linear( self.in_dim + self.position_dim, 512),
            nn.SiLU(),
            nn.Linear( 512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )

        self.linears_image = nn.Sequential(
            nn.Linear( self.in_dim + self.position_dim, 512),
            nn.SiLU(),
            nn.Linear( 512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )
        
        # -------------------------------------------------------------- #
        self.null_text_feature = torch.nn.Parameter(torch.zeros([self.in_dim]))
        self.null_image_feature = torch.nn.Parameter(torch.zeros([self.in_dim]))
        self.null_position_feature = torch.nn.Parameter(torch.zeros([self.position_dim]))
  

    def forward(self, boxes, image_masks, image_embeddings):
        B, N, _ = boxes.shape 
        # masks = masks.unsqueeze(-1) # B*N*1 
        # text_masks = text_masks.unsqueeze(-1) # B*N*1 
        image_masks = image_masks.unsqueeze(-1) # B*N*1
        
        # embedding position (it may includes padding as placeholder)
        xyxy_embedding = self.fourier_embedder(boxes) # B*N*4 --> B*N*C

        # learnable null embedding 
        # text_null  = self.null_text_feature.view(1,1,-1) # 1*1*C  (C=768)
        image_null = self.null_image_feature.view(1,1,-1) # 1*1*C
        xyxy_null  = self.null_position_feature.view(1,1,-1) # 1*1*C

        # replace padding with learnable null embedding 
        # import pdb; pdb.set_trace()
        # text_embeddings  = text_embeddings*text_masks  + (1-text_masks)*text_null       # [2,30,768]
        image_embeddings = image_embeddings*image_masks + (1-image_masks)*image_null    # [2,30,768]
        xyxy_embedding = xyxy_embedding*image_masks+ (1-image_masks)*xyxy_null                     # [2, 30, 64]

        # 4.3 new version:
        
        ground_embedding = torch.cat([image_embeddings[:,:1,:], xyxy_embedding], dim=-1)
        ground_embedding = self.linears_image(ground_embedding)
        objs_image = torch.cat([ground_embedding, image_embeddings[:,1:,:]], dim=1)


        # assert objs_image.shape == torch.Size([B,N,self.out_dim])        
        return objs_image


class PositionNet_dino_textimage(nn.Module):
    def __init__(self, in_dim_txt, in_dim_image, out_dim, fourier_freqs=8):
        super().__init__()
        self.in_dim_txt = in_dim_txt
        self.in_dim_image = in_dim_image
        self.out_dim = out_dim 

        self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        self.position_dim = fourier_freqs*2*4 # 2 is sin&cos, 4 is xyxy 

        # -------------------------------------------------------------- #
        self.linears_text = nn.Sequential(
            nn.Linear( self.in_dim_txt + self.position_dim, 512),
            nn.SiLU(),
            nn.Linear( 512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )
        
        self.linears_text1 = nn.Sequential(
            nn.Linear( self.in_dim_txt + self.position_dim, 512),
            nn.SiLU(),
            nn.Linear( 512, 512),
            nn.SiLU(),
            nn.Linear(512, self.in_dim_txt),
        )
        self.linears_text2 = nn.Sequential(
            nn.Linear( self.in_dim_txt, 512),
            nn.SiLU(),
            nn.Linear( 512, 512),
            nn.SiLU(),
            nn.Linear(512, self.out_dim),
        )
        self.linears_image = nn.Sequential(
            nn.Linear( self.in_dim_image + self.position_dim, 512),
            nn.SiLU(),
            nn.Linear( 512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )
        
        # -------------------------------------------------------------- #
        self.null_text_feature = torch.nn.Parameter(torch.zeros([self.in_dim_txt]))
        self.null_image_feature = torch.nn.Parameter(torch.zeros([self.in_dim_image]))
        self.null_position_feature = torch.nn.Parameter(torch.zeros([self.position_dim]))
  

    def forward(self, boxes, masks, text_embeddings,image_embeddings):
        B, N, _ = boxes.shape 
        # masks = masks.unsqueeze(-1) # B*N*1 
        text_masks = masks.unsqueeze(-1) # B*N*1 
        image_masks = masks.unsqueeze(-1) # B*N*1
        
        # embedding position (it may includes padding as placeholder)
        xyxy_embedding = self.fourier_embedder(boxes) # B*N*4 --> B*N*C

        # learnable null embedding 
        text_null  = self.null_text_feature.view(1,1,-1) # 1*1*C  (C=768)
        image_null = self.null_image_feature.view(1,1,-1) # 1*1*C
        xyxy_null  = self.null_position_feature.view(1,1,-1) # 1*1*C

        # replace padding with learnable null embedding 
        # import pdb; pdb.set_trace()
        text_embeddings  = text_embeddings*text_masks  + (1-text_masks)*text_null       # [2,30,768]
        image_embeddings = image_embeddings*image_masks + (1-image_masks)*image_null    # [2,30,768]
        xyxy_embedding = xyxy_embedding*image_masks+ (1-image_masks)*xyxy_null                     # [2, 30, 64]
        

        # import pdb; pdb.set_trace()
        grounding_text = torch.cat([text_embeddings[:,:1,:], xyxy_embedding], dim=-1)
        ground_text = self.linears_text1(grounding_text)
        objs_text = torch.cat([ ground_text,text_embeddings[:,1:,:]], dim=1)
        objs_text = self.linears_text2(objs_text)
        # objs_text  = torch.cat([objs_text, xyxy_embedding], dim=-1)
        # objs_image = self.linears_image( torch.cat([image_embeddings,xyxy_embedding], dim=-1)  ) # [2, 30, 768]

        # 4.3 new version:
        
        ground_embedding = torch.cat([image_embeddings[:,:1,:], xyxy_embedding], dim=-1)
        ground_embedding = self.linears_image(ground_embedding)

        objs_image = torch.cat([ground_embedding, image_embeddings[:,1:,:]], dim=1)

        objs = torch.cat( [objs_text,objs_image], dim=1 )  # [2, 30, 768]

        # assert objs_image.shape == torch.Size([B,N,self.out_dim])        
        return objs_image

class PositionNet_txt_image(nn.Module):
    def __init__(self, in_dim, out_dim, fourier_freqs=8):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim 

        self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        self.position_dim = fourier_freqs*2*4 # 2 is sin&cos, 4 is xyxy 

        # -------------------------------------------------------------- #
        self.linears_text = nn.Sequential(
            nn.Linear( self.in_dim + self.position_dim, 512),
            nn.SiLU(),
            nn.Linear( 512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )

        self.linears_image = nn.Sequential(
            nn.Linear( self.in_dim + self.position_dim, 512),
            nn.SiLU(),
            nn.Linear( 512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )
        
        # -------------------------------------------------------------- #
        self.null_text_feature = torch.nn.Parameter(torch.zeros([self.in_dim]))
        self.null_image_feature = torch.nn.Parameter(torch.zeros([self.in_dim]))
        self.null_position_feature = torch.nn.Parameter(torch.zeros([self.position_dim]))
  

    def forward(self, boxes, masks, text_masks, image_masks, text_embeddings, image_embeddings):
        B, N, _ = boxes.shape 
        masks = masks.unsqueeze(-1) # B*N*1 
        text_masks = text_masks.unsqueeze(-1) # B*N*1 
        image_masks = image_masks.unsqueeze(-1) # B*N*1
        
        # embedding position (it may includes padding as placeholder)
        xyxy_embedding = self.fourier_embedder(boxes) # B*N*4 --> B*N*C

        # learnable null embedding 
        text_null  = self.null_text_feature.view(1,1,-1) # 1*1*C  (C=768)
        image_null = self.null_image_feature.view(1,1,-1) # 1*1*C
        xyxy_null  = self.null_position_feature.view(1,1,-1) # 1*1*C

        # replace padding with learnable null embedding 
        # import pdb; pdb.set_trace()
        text_embeddings  = text_embeddings*text_masks  + (1-text_masks)*text_null       # [2,30,768]
        image_embeddings = image_embeddings*image_masks + (1-image_masks)*image_null    # [2,30,768]
        xyxy_embedding = xyxy_embedding*masks + (1-masks)*xyxy_null                     # [2, 30, 64]

        objs_text  = self.linears_text(  torch.cat([text_embeddings, xyxy_embedding], dim=-1)  ) # [2, 30, 768]
        objs_image = self.linears_image( torch.cat([image_embeddings,xyxy_embedding], dim=-1)  ) # [2, 30, 768]
        objs = torch.cat( [objs_text,objs_image], dim=1 )  # [2, 30, 768]

        assert objs.shape == torch.Size([B,N*2,self.out_dim])        
        return objs