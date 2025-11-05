import torch
import torch.nn as nn
import lightning.pytorch as pl
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, StableDiffusionPipeline
import torch.nn.functional as F

from EEGClip.clip_models import EEGEncoder, ProjectionHead
from EEGClip.causal_reasoning import CausalReasoningEngine, compute_causal_loss


class EEGToTextGenerator(pl.LightningModule):
    """
    Model that generates text from EEG signals with causal reasoning
    """
    def __init__(
        self,
        eeg_model_emb_dim=128,
        text_decoder_name="gpt2",  # or other suitable text generation model
        n_chans=21,
        eeg_model_pretrained=False,
        eeg_model_trainable=True,
        projected_emb_dim=64,
        dropout_rate=0.1,
        num_fc_layers=1,
        lr=1e-4,
        causal_reasoning=True,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.causal_reasoning = causal_reasoning
        
        # EEG encoder (same as in EEGClipModel)
        self.eeg_encoder = EEGEncoder(
            eeg_model_emb_dim=eeg_model_emb_dim,
            n_chans=n_chans,
            eeg_model_pretrained=eeg_model_pretrained,
            eeg_model_trainable=eeg_model_trainable,
        )
        
        # Projection to match text decoder dimensions
        self.eeg_projection = ProjectionHead(
            input_dim=eeg_model_emb_dim,
            output_dim=projected_emb_dim,
            dropout_rate=dropout_rate,
            num_fc_layers=num_fc_layers,
            transpose=True,
        )
        
        # Causal reasoning module
        if self.causal_reasoning:
            self.causal_module = CausalReasoningEngine(
                eeg_model_emb_dim=eeg_model_emb_dim,
                projected_emb_dim=projected_emb_dim,
                num_eeg_channels=n_chans
            )
        
        # Text decoder
        self.text_decoder = AutoModelForCausalLM.from_pretrained(text_decoder_name)
        self.tokenizer = AutoTokenizer.from_pretrained(text_decoder_name)
        
        # Add a linear layer to map EEG features to decoder input embeddings
        decoder_input_dim = projected_emb_dim * 2 if self.causal_reasoning else projected_emb_dim
        self.eeg_to_text = nn.Linear(decoder_input_dim, self.text_decoder.config.n_embd)
        
        # Add special tokens to tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, eeg_batch, text_batch=None):
        # Encode EEG
        eeg_features = self.eeg_encoder(eeg_batch)  # [B, N_pred, Enc_size]
        eeg_features_proj = self.eeg_projection(eeg_features)  # [B, N_pred, projected_emb_dim]
        eeg_features_mean = torch.mean(eeg_features_proj, dim=1)  # [B, projected_emb_dim]
        
        # Apply causal reasoning if enabled
        causal_loss = 0
        if self.causal_reasoning:
            causal_matrix, counterfactual_features, interventions, causal_effects = self.causal_module(eeg_features)
            # Use causal effects as additional features
            combined_features = torch.cat([eeg_features_mean, causal_effects], dim=-1)
            # Compute causal loss
            causal_loss = compute_causal_loss(causal_matrix, interventions, causal_effects)
        else:
            combined_features = eeg_features_mean
            
        # Map to text decoder embedding space
        eeg_text_embedding = self.eeg_to_text(combined_features)  # [B, n_embd]
        
        if text_batch is not None:
            # Training mode: use teacher forcing
            inputs = self.tokenizer(
                text_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(eeg_batch.device)
            
            # Forward through decoder
            outputs = self.text_decoder(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                labels=inputs.input_ids,  # For computing loss
            )
            
            loss = outputs.loss
            
            # Add causal reasoning loss if enabled
            if self.causal_reasoning:
                loss = loss + 0.1 * causal_loss
            
            return loss
        else:
            # Generation mode
            return self.generate_text(eeg_text_embedding)

    def generate_text(self, eeg_text_embedding, max_length=100):
        # Generate text from EEG embedding
        generated = self.text_decoder.generate(
            inputs_embeds=eeg_text_embedding.unsqueeze(1),
            max_length=max_length,
            num_beams=5,
            early_stopping=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        return self.tokenizer.decode(generated[0], skip_special_tokens=True)

    def training_step(self, batch, batch_idx):
        eeg_batch, text_batch, _ = batch
        loss = self.forward(eeg_batch, text_batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        eeg_batch, text_batch, _ = batch
        loss = self.forward(eeg_batch, text_batch)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer


class EEGToImageGenerator(pl.LightningModule):
    """
    Model that generates images from EEG signals using diffusion models with causal reasoning
    """
    def __init__(
        self,
        eeg_model_emb_dim=128,
        n_chans=21,
        eeg_model_pretrained=False,
        eeg_model_trainable=True,
        projected_emb_dim=64,
        dropout_rate=0.1,
        num_fc_layers=1,
        lr=1e-4,
        num_train_timesteps=1000,
        causal_reasoning=True,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.causal_reasoning = causal_reasoning
        
        # EEG encoder (same as in EEGClipModel)
        self.eeg_encoder = EEGEncoder(
            eeg_model_emb_dim=eeg_model_emb_dim,
            n_chans=n_chans,
            eeg_model_pretrained=eeg_model_pretrained,
            eeg_model_trainable=eeg_model_trainable,
        )
        
        # Projection to image generation space
        self.eeg_projection = ProjectionHead(
            input_dim=eeg_model_emb_dim,
            output_dim=projected_emb_dim,
            dropout_rate=dropout_rate,
            num_fc_layers=num_fc_layers,
            transpose=True,
        )
        
        # Causal reasoning module
        if self.causal_reasoning:
            self.causal_module = CausalReasoningEngine(
                eeg_model_emb_dim=eeg_model_emb_dim,
                projected_emb_dim=projected_emb_dim,
                num_eeg_channels=n_chans
            )
        
        # Map EEG features to a larger dimension suitable for image generation
        context_dim = projected_emb_dim * 2 if self.causal_reasoning else projected_emb_dim
        self.eeg_to_context = nn.Linear(context_dim, 768)  # 768 matches the context dim of many diffusion models
        
        # Load pre-trained diffusion model components
        self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
        self.noise_scheduler = DDPMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
        
        # Freeze the VAE and UNet parameters initially
        for param in self.vae.parameters():
            param.requires_grad = False
            
        for param in self.unet.parameters():
            param.requires_grad = False

    def forward(self, eeg_batch):
        # Encode EEG
        eeg_features = self.eeg_encoder(eeg_batch)  # [B, N_pred, Enc_size]
        eeg_features_proj = self.eeg_projection(eeg_features)  # [B, N_pred, projected_emb_dim]
        eeg_features_mean = torch.mean(eeg_features_proj, dim=1)  # [B, projected_emb_dim]
        
        # Apply causal reasoning if enabled
        causal_loss = 0
        if self.causal_reasoning:
            causal_matrix, counterfactual_features, interventions, causal_effects = self.causal_module(eeg_features)
            # Use causal effects as additional features
            combined_features = torch.cat([eeg_features_mean, causal_effects], dim=-1)
            # Compute causal loss
            causal_loss = compute_causal_loss(causal_matrix, interventions, causal_effects)
        else:
            combined_features = eeg_features_mean
        
        # Map to context dimension
        context = self.eeg_to_context(combined_features)  # [B, 768]
        
        # Generate random noise
        batch_size = eeg_batch.shape[0]
        latent_shape = (batch_size, self.unet.config.in_channels, 64, 64)
        noise = torch.randn(latent_shape, device=eeg_batch.device)
        
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, 
                                 (batch_size,), device=eeg_batch.device).long()
        
        # Add noise to the latents
        noisy_latents = self.noise_scheduler.add_noise(noise, noise, timesteps)
        
        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, context.unsqueeze(1)).sample
        
        return noise_pred, causal_loss

    def training_step(self, batch, batch_idx):
        eeg_batch, text_batch, _ = batch
        
        # Forward pass
        noise_pred, causal_loss = self.forward(eeg_batch)
        
        # Encode EEG
        eeg_features = self.eeg_encoder(eeg_batch)  # [B, N_pred, Enc_size]
        eeg_features_proj = self.eeg_projection(eeg_features)  # [B, N_pred, projected_emb_dim]
        eeg_features_mean = torch.mean(eeg_features_proj, dim=1)  # [B, projected_emb_dim]
        
        # Apply causal reasoning if enabled
        if self.causal_reasoning:
            causal_matrix, counterfactual_features, interventions, causal_effects = self.causal_module(eeg_features)
            # Use causal effects as additional features
            combined_features = torch.cat([eeg_features_mean, causal_effects], dim=-1)
            # Compute causal loss
            causal_loss = compute_causal_loss(causal_matrix, interventions, causal_effects)
        else:
            combined_features = eeg_features_mean
            causal_loss = 0
        
        # Map to context dimension
        context = self.eeg_to_context(combined_features)  # [B, 768]
        
        # For training, we would need target images
        # As a simplified approach, we'll generate random images and train the model to denoise them
        batch_size = eeg_batch.shape[0]
        image_shape = (batch_size, 3, 512, 512)
        target_images = torch.randn(image_shape, device=eeg_batch.device)
        
        # Encode images to latent space
        with torch.no_grad():
            latents = self.vae.encode(target_images).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        batch_size = latents.shape[0]
        
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, 
                                 (batch_size,), device=eeg_batch.device).long()
        
        # Add noise to the latents according to the noise magnitude at each timestep
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, context.unsqueeze(1)).sample
        
        # Calculate loss
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        
        # Add causal reasoning loss if enabled
        if self.causal_reasoning:
            loss = loss + 0.1 * causal_loss
        
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        eeg_batch, text_batch, _ = batch
        
        # Encode EEG
        eeg_features = self.eeg_encoder(eeg_batch)  # [B, N_pred, Enc_size]
        eeg_features_proj = self.eeg_projection(eeg_features)  # [B, N_pred, projected_emb_dim]
        eeg_features_mean = torch.mean(eeg_features_proj, dim=1)  # [B, projected_emb_dim]
        
        # Apply causal reasoning if enabled
        if self.causal_reasoning:
            causal_matrix, counterfactual_features, interventions, causal_effects = self.causal_module(eeg_features)
            # Use causal effects as additional features
            combined_features = torch.cat([eeg_features_mean, causal_effects], dim=-1)
            # Compute causal loss
            causal_loss = compute_causal_loss(causal_matrix, interventions, causal_effects)
        else:
            combined_features = eeg_features_mean
            causal_loss = 0
        
        # Map to context dimension
        context = self.eeg_to_context(combined_features)  # [B, 768]
        
        # For validation, we'll generate random images and compute loss
        batch_size = eeg_batch.shape[0]
        image_shape = (batch_size, 3, 512, 512)
        target_images = torch.randn(image_shape, device=eeg_batch.device)
        
        # Encode images to latent space
        with torch.no_grad():
            latents = self.vae.encode(target_images).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        batch_size = latents.shape[0]
        
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, 
                                 (batch_size,), device=eeg_batch.device).long()
        
        # Add noise to the latents according to the noise magnitude at each timestep
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, context.unsqueeze(1)).sample
        
        # Calculate loss
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        
        # Add causal reasoning loss if enabled
        if self.causal_reasoning:
            loss = loss + 0.1 * causal_loss
        
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def generate_image(self, eeg_batch, num_inference_steps=50):
        """
        Generate images from EEG signals using the diffusion model
        """
        # Encode EEG
        eeg_features = self.eeg_encoder(eeg_batch)  # [B, N_pred, Enc_size]
        eeg_features_proj = self.eeg_projection(eeg_features)  # [B, N_pred, projected_emb_dim]
        eeg_features_mean = torch.mean(eeg_features_proj, dim=1)  # [B, projected_emb_dim]
        
        # Apply causal reasoning if enabled
        if self.causal_reasoning:
            causal_matrix, counterfactual_features, interventions, causal_effects = self.causal_module(eeg_features)
            # Use causal effects as additional features
            combined_features = torch.cat([eeg_features_mean, causal_effects], dim=-1)
        else:
            combined_features = eeg_features_mean
        
        # Map to context dimension
        context = self.eeg_to_context(combined_features)  # [B, 768]
        
        # Create random noise
        batch_size = eeg_batch.shape[0]
        latent_shape = (batch_size, self.unet.config.in_channels, 64, 64)
        latents = torch.randn(latent_shape, device=eeg_batch.device)
        
        # Set timesteps for denoising
        self.noise_scheduler.set_timesteps(num_inference_steps)
        latents = latents * self.noise_scheduler.init_noise_sigma
        
        # Denoising loop
        for t in self.noise_scheduler.timesteps:
            # Expand the latents if we are doing classifier-free guidance
            latent_model_input = torch.cat([latents] * 2) if context is not None else latents
            latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, t)
            
            # Predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input, 
                    t, 
                    encoder_hidden_states=context.unsqueeze(1)
                ).sample
                
            # Compute the previous noisy sample x_t -> x_t-1
            latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample
            
        # Scale and decode the image
        latents = 1 / self.vae.config.scaling_factor * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        
        return image