import torch
import pytorch_lightning as pl
import wandb
import math
from math import log10
from torchvision.transforms import Resize

from src.model import GeneratorRRDB, Discriminator, FeatureExtractor
from src.ssim import ssim
from src.datamodule import denormalize

from icecream import ic


class LightningESRGAN(pl.LightningModule):
    def __init__(self,
                #  Basic parameters
                 batch_size=4,
                 num_workers=12,
                 lr=2e-4,
                 min_lr=1e-6,
                 lr_check_interval=1000,
                 lr_decay_factor=0.25,
                 lr_decay_patience=6,
                 warmup_steps=1000,
                #  Model parameters
                 channels=3,
                 filers=64,
                 upscale_factor=4,
                 hr_height=256,
                 hr_width=256,
                 num_residual_blocks=16,
                 lam_adv=5e-3,
                 lam_pixel=1e-2,
                 loss_g_scale=1,
                 loss_d_scale=1,
                 **kwargs) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.min_lr = min_lr
        self.lr_check_interval = lr_check_interval
        self.lr_decay_factor = lr_decay_factor
        self.lr_decay_patience = lr_decay_patience
        self.warmup_steps = warmup_steps
        
        self.channels = channels
        self.filters = filers
        self.upscale_factor = upscale_factor
        self.hr_height = hr_height
        self.hr_width = hr_width
        self.num_residual_blocks = num_residual_blocks
        self.lam_adv = lam_adv
        self.lam_pixel = lam_pixel
        self.loss_g_scale = loss_g_scale
        self.loss_d_scale = loss_d_scale
        self.save_hyperparameters()
        
        hr_shape = (self.hr_height, self.hr_width)
        self.generator = GeneratorRRDB(self.channels, self.filters, self.num_residual_blocks, int(math.log(self.upscale_factor, 2)))
        self.discriminator = Discriminator((self.channels, *hr_shape))
        self.feature_extractor = FeatureExtractor().eval()
        
        self.criterion_GAN = torch.nn.BCEWithLogitsLoss()
        self.criterion_content = torch.nn.L1Loss()
        self.criterion_pixel = torch.nn.L1Loss()
        
    def configure_optimizers(self):
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.9, 0.999))
        optimzier_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.9, 0.999))
        
        # scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=self.lr_check_interval, gamma=self.lr_decay_factor)
        # scheduler_G = torch.optim.lr_scheduler.StepLR(optimzier_G, step_size=self.lr_check_interval, gamma=self.lr_decay_factor)
        scheduler_D = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_D,
            factor=self.lr_decay_factor,
            patience=self.lr_decay_patience,
            threshold=1e-6,
            min_lr=self.min_lr,
            verbose=True,
        )
        scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimzier_G,
            factor=self.lr_decay_factor,
            patience=self.lr_decay_patience,
            threshold=1e-6,
            min_lr=self.min_lr,
            verbose=True,
        )
        
        lr_dict_D = {
            "scheduler": scheduler_D,
            "name": "lr_D",
            "monitor": "val_loss",
            "frequency": self.lr_check_interval,
        }
        lr_dict_G = {
            "scheduler": scheduler_G,
            "name": "lr_G",
            "monitor": "val_loss",
            "frequency": self.lr_check_interval,
        }
        
        return [
            {
                "optimizer": optimizer_D,
                "lr_scheduler": lr_dict_D,
            },
            {
                "optimizer": optimzier_G,
                "lr_scheduler": lr_dict_G,
            }
        ]
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        img_lr = batch["lr"].to(self.device)
        img_hr = batch["hr"].to(self.device)
        
        img_sr = self.generator(img_lr)
        
        if optimizer_idx == 0:
            pred_real = self.discriminator(img_hr)
            pred_fake = self.discriminator(img_sr.detach())
            
            loss_real = self.criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), torch.ones_like(pred_real))
            loss_fake = self.criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), torch.zeros_like(pred_fake))
            
            d_loss = (loss_real + loss_fake) / 2
            d_loss = self.loss_d_scale * d_loss
            
            dis_fake_dist = torch.sigmoid(pred_fake).detach().cpu().numpy().flatten().tolist()
            dis_real_dist = torch.sigmoid(pred_real).detach().cpu().numpy().flatten().tolist()
            
            if (self.global_step // 2) % self.trainer.log_every_n_steps == 0:                    
                wandb.log({
                    "discriminator/dis_real_dist": wandb.Histogram(dis_real_dist),
                    "discriminator/dis_fake_dist": wandb.Histogram(dis_fake_dist),
                }, step=self.global_step)
            
            self.log_dict({
                "discriminator/loss": d_loss,
                "discriminator/dis_real_loss": loss_real,
                "discriminator/dis_fake_loss": loss_fake,
            })
            
            return d_loss
        
        if optimizer_idx == 1:
            loss_pixel = self.criterion_pixel(img_sr, img_hr)
            loss_GAN = 0
            loss_content = 0
            gen_fake_dist = []
            if self.global_step > self.warmup_steps:
                pred_real = self.discriminator(img_hr).detach()
                pred_fake = self.discriminator(img_sr)
                
                loss_GAN = self.criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), torch.ones_like(pred_fake))
                
                gen_features = self.feature_extractor(img_sr)
                real_features = self.feature_extractor(img_hr).detach()
                loss_content = self.criterion_content(gen_features, real_features)
                
                gen_fake_dist = torch.sigmoid(pred_fake).detach().cpu().numpy().flatten().tolist()
            
            loss_g = self.lam_pixel * loss_pixel + self.lam_adv * loss_GAN + loss_content
            loss_g = self.loss_g_scale * loss_g
            
            if ((self.global_step - 1) // 2) % self.trainer.log_every_n_steps ==0:
                log_img = [img_lr[0].detach().cpu(), img_hr[0].detach().cpu(), img_sr[0].detach().cpu()]
                log_img = [Resize(256)(denormalize(img)) for img in log_img]
                log_img = torch.stack(log_img)
                wandb.log({
                    'img/train_img': wandb.Image(log_img),
                    "generator/gen_dist": wandb.Histogram(gen_fake_dist),
                }, step=self.global_step)
                
            self.log_dict({
                "generator/loss": loss_g,
                "generator/loss_pixel": loss_pixel,
                "generator/loss_GAN": loss_GAN,
                "generator/loss_content": loss_content,
            })
            
            return loss_g
    
    def validation_step(self, batch, batch_idx):
        img_lr = batch["lr"].to(self.device)
        img_hr = batch["hr"].to(self.device)
        
        img_sr = self.generator(img_lr)
        
        if batch_idx == 0:
            log_img = [img_lr[0].detach().cpu(), img_hr[0].detach().cpu(), img_sr[0].detach().cpu()]
            log_img = [Resize(256)(denormalize(img)) for img in log_img]
            log_img = torch.stack(log_img)
            wandb.log({'img/val_img': wandb.Image(log_img)}, step=self.global_step)
            
        loss_GAN = 0
        loss_content = 0
        loss_pixel = self.criterion_pixel(img_sr, img_hr)
        if self.global_step > self.warmup_steps:
            pred_real = self.discriminator(img_hr).detach()
            pred_fake = self.discriminator(img_sr)
            
            loss_GAN = self.criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), torch.ones_like(pred_fake))
            
            gen_features = self.feature_extractor(img_sr)
            real_features = self.feature_extractor(img_hr).detach()
            loss_content = self.criterion_content(gen_features, real_features)
        loss = self.lam_pixel * loss_pixel + self.lam_adv * loss_GAN + loss_content
        
        mse_loss = ((img_sr - img_hr) ** 2).data.mean()
        ssim_ = ssim(img_sr, img_hr).item()
        psnr = 10 * log10((img_hr.max()**2) / mse_loss)  
        
        self.log_dict({
            "val/loss_pixel": loss_pixel,
            "val/loss_content": loss_content,
            "val/loss_GAN": loss_GAN,
            "val/loss": loss,
            "val/ssim": ssim_,
            "val/psnr": psnr,
        }) 
        self.log_dict({
            "val_loss": loss,
            "val_loss_pixel": loss_pixel,
        }, logger=False)
        return loss
    
    def configure_gradient_clipping(
        self, optimizer, optimizer_idx, gradient_clip_val, gradient_clip_algorithm
    ):
        grad_norm = torch.nn.utils.clip_grad_norm_(
            sum([p["params"] for p in optimizer.param_groups], []), gradient_clip_val
        )
        self.log("grad_norm", grad_norm)
        
        
        