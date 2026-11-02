"""Run receipts, scalar logging and full-state checkpoints for long jobs."""
import json
import math
import os
import time
import warnings
from pathlib import Path
import torch


class RunRecorder:
    def __init__(self, config, accelerator, directory):
        self.config, self.accelerator = config, accelerator
        self.directory = Path(directory)
        self.run = None
        if accelerator.is_main_process and config.get("wandb_mode", "disabled") != "disabled":
            import wandb
            self.run = wandb.init(
                entity=config["wandb_entity"], project=config["wandb_project"],
                name=config["wandb_name"], config=config, id=config.get("wandb_run_id"),
                group=config.get("wandb_group"), job_type=config.get("job_type", "train"),
                dir=str(self.directory), mode=config["wandb_mode"],
                resume="must" if config.get("resume_checkpoint") else "never",
            )

            if self.run.offline:
                raise RuntimeError("Online W&B is required")
            (self.directory / "wandb_run.json").write_text(json.dumps({"id":self.run.id,"url":self.run.url,"mode":"online"},indent=2))

    def log(self, metrics, iteration, status="running"):
        if not self.accelerator.is_main_process:
            return
        row = {"time": time.time(), "iteration": iteration, "status": status, **metrics}
        invalid = [k for k,v in row.items() if isinstance(v, (int, float)) and not math.isfinite(v)]
        # Training loss and gradients are checked on every rank in the train loop.
        if invalid:
            warnings.warn(f"Non-finite auxiliary/logging metrics omitted: {invalid}")
            row = {k: (None if k in invalid else v) for k,v in row.items()}
            row["invalid_metrics"] = invalid
        metrics = {k:v for k,v in metrics.items() if k not in invalid}
        metrics.update({f"metric_valid/{k}": 0 for k in invalid})
        with (self.directory / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        tmp = self.directory / "progress.json.tmp"
        tmp.write_text(json.dumps(row, indent=2) + "\n")
        tmp.replace(self.directory / "progress.json")
        if self.run:
            self.run.log(metrics, step=iteration)

    def checkpoint(self, model, optimizer, scheduler, epoch, batch, iteration, **stats):
        if not self.accelerator.is_main_process:
            return
        state = {"model_state": self.accelerator.unwrap_model(model).state_dict(),
                 "optimizer_state": optimizer.state_dict(),
                 "scheduler_state": scheduler.state_dict() if scheduler else None,
                 "scaler_state": self.accelerator.scaler.state_dict() if self.accelerator.scaler else None,
                 "epoch": epoch, "next_batch": batch, "iteration": iteration,
                 "config": self.config, "stats": stats}
        dest = self.directory / "saves" / "last_full.pt"
        tmp = dest.with_suffix(".tmp")
        torch.save(state, tmp)
        os.replace(tmp, dest)

    def log_media(self, epoch, iteration):
        if not self.run:
            return
        import imageio.v3 as iio
        import wandb
        media = {}
        image = self.directory / "figures" / f"image_valid_{epoch}.jpg"
        animation = self.directory / "figures" / f"e{epoch}_traj_anim_0.gif"
        if image.exists():
            media["vis/validation_reconstruction"] = wandb.Image(str(image))
        if animation.exists():
            frames = iio.imread(animation)[..., :3].transpose(0, 3, 1, 2)
            media["vis/rollout"] = wandb.Video(frames, fps=self.config["animation_fps"], format="mp4")
        if media:
            self.run.log(media, step=iteration)

    def finish(self):
        if self.run:
            self.run.finish()
