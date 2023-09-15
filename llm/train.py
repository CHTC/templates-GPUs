import argparse
import logging
import os
from pathlib import Path

import wandb
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint


def train(run_name: str, use_wandb: bool = False):
    """Test training script with basic wandb logging."""

    STAGING_DIR = Path(os.getenv("STAGING_DIR"))
    RESULTS_DIR = STAGING_DIR / "results" / run_name

    if use_wandb:
        print("Using wandb for logging.")
        wandb.init(name=run_name, id=run_name, resume="allow")
    else:
        print("Not using wandb for logging.")

    # Main training section
    dataset = load_dataset("yelp_review_full")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    dataset = dataset.map(tokenize_function, batched=True)

    train_dataset = dataset["train"].shuffle(seed=42)
    eval_dataset = dataset["test"].shuffle(seed=42)

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased", num_labels=5
    )

    training_args = TrainingArguments(
        output_dir=RESULTS_DIR,
        evaluation_strategy="steps",
        num_train_epochs=1,
        report_to="wandb" if use_wandb else "none",
        save_strategy="steps",
        save_total_limit=3,
        # deepspeed="deepspeed_config.json",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    trainer.train(resume_from_checkpoint=last_checkpoint)

    if use_wandb:
        wandb.finish()


def main():
    """Run training script."""

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    # Get arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name", type=str, help="Name of run.")
    parser.add_argument(
        "-w", "--use_wandb", action="store_true", help="Use wandb for logging."
    )
    args = parser.parse_args()

    train(args.run_name, args.use_wandb)


if __name__ == "__main__":
    main()
