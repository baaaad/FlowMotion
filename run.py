import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import WanPipeline, AutoencoderKLWan, UniPCMultistepScheduler
from diffusers.video_processor import VideoProcessor
from diffusers.utils import export_to_video
from PIL import Image
import argparse
import os
import imageio
from tqdm import tqdm
import random
import numpy as np


@torch.no_grad()
def latents_to_video(latents: torch.Tensor, vae: AutoencoderKLWan, device):
    """Convert latents back to video frames"""
    latents_mean = (
        torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device, vae.dtype)
    )
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
        device, vae.dtype
    )
    
    latents = latents / latents_std + latents_mean
    
    with torch.no_grad():
        decoded_frames = vae.decode(latents, return_dict=False)[0]
    
    return decoded_frames



class FlowMotionProcessor:
    """Main class for video motion transfer using FlowMotion technique"""
    
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.set_seed(args.seed)
        
        # Load models
        self.vae = AutoencoderKLWan.from_pretrained(
            args.model_id, subfolder="vae", torch_dtype=torch.float16
        )
        self.scheduler = UniPCMultistepScheduler(
            prediction_type='flow_prediction',
            use_flow_sigmas=True,
            num_train_timesteps=1000,
            flow_shift=args.flow_shift
        )
        self.pipe = WanPipeline.from_pretrained(
            args.model_id, vae=self.vae, torch_dtype=torch.float16
        ).to(self.device)
        
        # Save GPU cost
        # self.pipe.enable_model_cpu_offload()


        # Video processor
        vae_scale_factor_spatial = 2 ** len(self.pipe.vae.temperal_downsample) if getattr(self.pipe, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=vae_scale_factor_spatial)

    def set_seed(self, seed: int = 42):
        """Set random seed for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)

    @torch.no_grad()
    def latents_to_video(self, latents: torch.Tensor):
        """Convert latents back to video frames"""
        vae = self.pipe.vae
        latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(self.device, vae.dtype)
        latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(self.device, vae.dtype)
        
        latents = latents / latents_std + latents_mean
        decoded_frames = vae.decode(latents, return_dict=False)[0]
        return decoded_frames

    def load_video(self, file_path: str):
        """Load video and extract frames"""
        vid = imageio.get_reader(file_path)
        fps = vid.get_meta_data()['fps']
        images = [Image.fromarray(frame) for frame in vid]
        return images, fps

    @torch.no_grad()
    def calc_velocity(self, latents, prompt_embeds, negative_prompt_embeds, guidance, t):
        """Calculate velocity prediction from transformer"""
        timestep = t.expand(latents.shape[0])
        dtype = latents.dtype
        latents = latents.to(self.pipe.transformer.dtype)
        
        noise_pred = self.pipe.transformer(
            hidden_states=latents,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]
        
        if self.pipe.do_classifier_free_guidance:
            noise_pred_uncond = self.pipe.transformer(
                hidden_states=latents,
                timestep=timestep,
                encoder_hidden_states=negative_prompt_embeds,
                return_dict=False,
            )[0]
            noise_pred = guidance * (noise_pred - noise_pred_uncond) + noise_pred_uncond
        
        return noise_pred.to(dtype)

    def compute_motion_difference(self, latents):
        """Compute motion representation differences between frames"""
        latent_frames = latents[0].permute(1, 2, 3, 0)  # [frames, height, width, channels]
        frames, height, width, channels = latent_frames.shape
        
        diff_maps = []
        
        # Consecutive frame differences
        if frames > 1:
            consecutive_diffs = torch.abs(latent_frames[1:] - latent_frames[:-1])
            diff_maps.append(consecutive_diffs)
        
        # Frame-to-all-others differences
        for center_idx in range(frames):
            other_indices = [i for i in range(frames) if i != center_idx]
            if other_indices:
                other_frames = latent_frames[other_indices]
                center_frame_expanded = latent_frames[center_idx].unsqueeze(0).expand_as(other_frames)
                diffs = torch.abs(center_frame_expanded - other_frames)
                diff_maps.append(diffs)
        
        return torch.stack(diff_maps, dim=0)

    def decompose_velocity(self, v_tar, v_src, epsilon=1e-8):
        """Decompose target velocity into parallel and perpendicular components relative to source"""
        dot_product = torch.sum(v_tar * v_src, dim=1, keepdim=True)
        norm_src_sq = torch.sum(v_src * v_src, dim=1, keepdim=True)
        
        proj_scalar = dot_product / (norm_src_sq + epsilon)
        proj_vector = proj_scalar * v_src
        perp_vector = v_tar - proj_vector
        
        return proj_vector, perp_vector

    def process(self):
        """Main processing pipeline"""
        args = self.args
        
        # Encode prompts
        with torch.no_grad():
            src_prompt_embeds, src_negative_prompt_embeds = self.pipe.encode_prompt(
                prompt="",  # empty source prompt
                negative_prompt="",
                device=self.device,
            )
            
            self.pipe._guidance_scale = args.target_guidance_scale
            tar_prompt_embeds, tar_negative_prompt_embeds = self.pipe.encode_prompt(
                prompt=args.target_prompt,
                negative_prompt="",
                device=self.device,
            )

        # Load and process source video
        video, fps = self.load_video(args.video_path)

        
        video_tensor = self.video_processor.preprocess_video(
            video, height=args.height, width=args.width
        ).to(self.device, torch.float16)

        # Encode source video to latents
        with torch.no_grad():
            encoded_frames = self.pipe.vae.encode(video_tensor)[0].sample()
            latents_mean = torch.tensor(self.pipe.vae.config.latents_mean).view(1, self.pipe.vae.config.z_dim, 1, 1, 1).to(self.device, torch.float16)
            latents_std = 1.0 / torch.tensor(self.pipe.vae.config.latents_std).view(1, self.pipe.vae.config.z_dim, 1, 1, 1).to(self.device, torch.float16)
            encoded_frames = (encoded_frames - latents_mean) * latents_std

        # Prepare latents and timesteps
        clean_source_latents = encoded_frames.clone().to(self.device).to(torch.float32)

        self.pipe.scheduler.set_timesteps(args.T_steps, device=self.device)
        timesteps = self.pipe.scheduler.timesteps

        # Initialize target latent
        random_noise = torch.randn_like(clean_source_latents).to(self.device)
        target_latent = random_noise.clone()

        # Denoising process with guidance
        for i, t in tqdm(enumerate(timesteps)):
            t_i = t / 1000
            t_next_i = timesteps[i + 1] / 1000 if i + 1 < len(timesteps) else torch.zeros_like(t_i).to(t_i.device)
            timestep = t.expand(target_latent.shape[0])

            # Apply guidance during early steps
            if i < args.guidance_step:
                target_latent = self._apply_guidance_step(
                    target_latent, clean_source_latents, random_noise, t_i, timestep,
                    src_prompt_embeds, src_negative_prompt_embeds,
                    tar_prompt_embeds, tar_negative_prompt_embeds,
                    i
                )

            # Standard denoising step
            with torch.no_grad():
                Vt_tar = self.calc_velocity(
                    target_latent, tar_prompt_embeds, tar_negative_prompt_embeds,
                    args.target_guidance_scale, timestep
                )
                target_latent = target_latent.to(torch.float32) + (t_next_i - t_i) * Vt_tar
                target_latent = target_latent.to(Vt_tar.dtype)


        # Save result
        self._save_result(target_latent, fps, args.output_dir, args.guidance_type, args.seed)

    def _apply_guidance_step(self, target_latent, clean_source_latents, random_noise, 
                           t_i, timestep, src_prompt_embeds, src_negative_prompt_embeds,
                           tar_prompt_embeds, tar_negative_prompt_embeds, step_idx):
        """Apply optimization guidance at current step"""
        args = self.args
        
        target_latent_opt = target_latent.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([target_latent_opt], lr=args.lr_base)

        # Calculate average velocity
        v_mean = None
        if step_idx > 0:
            v_mean = (target_latent - random_noise) / (t_i - self.pipe.scheduler.timesteps[0]/1000)

        # Calculate source motion representation
        if args.guidance_type == "latent_prediction":
            fwd_noise = torch.randn_like(clean_source_latents).to(self.device)
            source_latent = (1-t_i)*clean_source_latents + (t_i)*fwd_noise
            Vt_src = self.calc_velocity(
                source_latent, src_prompt_embeds, src_negative_prompt_embeds,
                args.source_guidance_scale, timestep
            )
            source_latent_prediction = source_latent.to(torch.float32) + (torch.zeros_like(t_i).to(t_i.device) - t_i) * Vt_src
            source_motion_representation = source_latent_prediction.to(Vt_src.dtype)


        else:  # "clean_latent"
            source_motion_representation = clean_source_latents

        # Calculate frame-wise differences
        diff_source = self.compute_motion_difference(source_motion_representation)

        # Optimization loop
        for xxx in range(args.optimization_step):
            optimizer.zero_grad()
            
            Vt_tar = self.calc_velocity(
                target_latent_opt, tar_prompt_embeds, tar_negative_prompt_embeds,
                args.target_guidance_scale, timestep
            )
            
            # Velocity regulation
            if v_mean is not None:
                proj, perp = self.decompose_velocity(Vt_tar, v_mean)
                Vt_tar = proj + args.regulate_scale * perp

            # Calculate target motion representation
            target_latent_prediction = target_latent_opt.to(torch.float32) + (torch.zeros_like(t_i).to(t_i.device) - t_i) * Vt_tar
            target_motion_representation = target_latent_prediction.to(Vt_tar.dtype)

            diff_target = self.compute_motion_difference(target_motion_representation)
            
            # Calculate losses
            loss_la = args.alpha * F.mse_loss(source_motion_representation, target_motion_representation, reduction='mean')
            loss_da = args.beta * F.mse_loss(diff_source, diff_target, reduction='mean')
            loss = loss_la + loss_da

            
            loss.backward(retain_graph=False)
            optimizer.step()

        return target_latent_opt.detach().clone()

    def _save_result(self, latents, fps, output_dir, guidance_type, seed):
        """Save generated video"""
        os.makedirs(output_dir, exist_ok=True)
        
        with torch.no_grad():
            latents = latents.to(self.pipe.vae.dtype)
            video = self.latents_to_video(latents)
            video = self.video_processor.postprocess_video(video)[0]
            
            filename = os.path.join(output_dir, f"{guidance_type}_{seed}.mp4")
            export_to_video(video, filename, fps=fps)
            print(f"Video saved as {filename}")


def main():
    parser = argparse.ArgumentParser(description='FlowMotion Video Motion Transfer')
    parser.add_argument('--model_id', type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--T_steps', type=int, default=50)
    parser.add_argument('--source_guidance_scale', type=float, default=1.5)
    parser.add_argument('--target_guidance_scale', type=float, default=6.0)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--width', type=int, default=720)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--target_prompt', type=str, default="A penguin is walking along a frozen coastline.")
    parser.add_argument('--flow_shift', type=float, default=3.0)
    parser.add_argument('--video_path', type=str, default="./data/49f/hike.mp4")
    parser.add_argument('--output_dir', type=str, default="./results")
    parser.add_argument('--guidance_step', type=int, default=10)
    parser.add_argument('--lr_base', type=float, default=0.003)
    parser.add_argument('--optimization_step', type=int, default=3)
    parser.add_argument('--regulate_scale', type=float, default=0.1)
    parser.add_argument('--alpha', type=int, default=20)
    parser.add_argument('--beta', type=int, default=5)
    parser.add_argument('--guidance_type', type=str, default="latent_prediction", choices=["latent_prediction", "clean_latent"])
    
    args = parser.parse_args()
    
    processor = FlowMotionProcessor(args)
    processor.process()


if __name__ == "__main__":
    main()
