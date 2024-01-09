from training.trainers import DiffusionTrainer
from dataset import CraterDataset
from torchvision.transforms import (
    RandomVerticalFlip,
    RandomHorizontalFlip,
    RandomInvert,
    Normalize,
    Grayscale,
    ToTensor,
    RandomCrop
)
from diffusers import UNet2DModel, DDPMScheduler, get_cosine_schedule_with_warmup
from adabelief_pytorch import AdaBelief
from training.utils import Config
import os

def main(device):
    # Get initial configuration for the diffusion model
    config: Config = DiffusionTrainer.get_default_config()

    # Initialize Dataset
    dataset_size = 5000
    dataset = CraterDataset(dataset_size, "./data/")
    dataset.add_transforms([
        ToTensor(),
        RandomCrop((256,256)),
        RandomVerticalFlip(),
        RandomHorizontalFlip(),
        Grayscale(num_output_channels=3),
        RandomInvert(),
        Normalize([0.5], [0.5])
    ])

    # Initialize Model Weights
    unet = UNet2DModel(
        sample_size=256,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(64,64,128,128,256,256,512),
        down_block_types=(
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D"
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D"
        )
    )
    unet.to(device=device)

    # Initialize Optimizer & lr scheduler
    optimizer = AdaBelief(unet.parameters(), lr=config.learning_rate, print_change_log=False)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.learning_rate_warmup_steps,
        num_training_steps=(int(dataset_size / config.batch_size) * config.epochs),
    )

    # Initialize Scheduler
    scheduler = DDPMScheduler(num_train_timesteps=config.max_timesteps)

    # Initialize Trainer Class (Contains Training Loop)
    trainer = DiffusionTrainer(
        unet,
        optimizer,
        dataset,
        scheduler,
        lr_scheduler,
        device,
        DiffusionTrainer.get_default_config()
    )
    trainer.train()
    trainer.save_results("./final_model/")


if __name__ == "__main__":
    device = "cpu"
    main(device)