# Example Cat/Dog Classifier Training and Inference in CHTC
## Introduction

ChatGPT:

## Introduction
This repository demonstrates how to train and deploy a convolutional neural network (CNN) for classifying images of cats and dogs using the Center for High Throughput Computing (CHTC) resources. This example was created for the 2025 Data Science Research Bazaar. Inspiration, data, and the original code are taken from [the original Kaggle competition](https://www.kaggle.com/c/dogs-vs-cats) and [this blog post by Dr. Juan Zamora-Mora](https://www.doczamora.com/cats-vs-dogs-binary-classifier-with-pytorch-cnn).

The structure is split into a few different stages that I (Ian Ross) find myself visiting frequently when creating and deploying similar ML workflows within CHTC:
- Development - thinking about the problem and locally creating a minimal solution, while creating a foundation for the next stages (e.g. thinking about or creating a container)
- Running in CHTC - expanding the local development to run within CHTC
- Improving the CHTC runs - being more efficient with CHTC resources or, if applicable, going from 1 to many training runs

This development/running/improving cycle is revisited in both training and inference aspects of this example.

## Research Problem

Image classification is a fundamental computer vision task with many scientific applications. In this example, we're building a binary classifier to distinguish between cat and dog images, utilizing annotated data provided from the [Kaggle Dogs vs. Cats competition](https://www.kaggle.com/c/dogs-vs-cats). We are focused on the training and inference processes and will not be too concerned about the actual model architecture or dataset preparation. These are obviously critical components of the project, but are left as an exercise to the reader 😎.

## Development

### Objective
Get a minimally viable workflow running that can be scaled to CHTC resources. During this development phase, we'll focus on creating a foundation that works on a local machine while preparing for distributed execution. Key considerations include:
- Working with a subset of data to determine time, space, memory, and GPU requirements
- Ensuring data pipelines function correctly (reading from specified locations, transforming as needed, writing to specified directories)
- Developing and containerizing the software environment

### Approach
We've implemented a CNN model using PyTorch with the following components:
- A convolutional neural network architecture defined in `train.py`
- Custom datasets for training and inference
- Containerization with Docker for consistent environments across machines
- Scripts for both training and inference workflows

### Key Components of `train.py`

#### CNN Architecture
The `CatAndDogConvNet` class defines our model architecture:
```python
class CatAndDogConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolutional layers (3,16,32)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5, 5), stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=64 * 6 * 6, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=50)
        self.fc3 = nn.Linear(in_features=50, out_features=2)  # Binary classification (cat/dog)
```
- The model takes RGB images (3 channels) as input
- Uses three convolutional layers with increasing filter counts (16→32→64)
- Includes max pooling to reduce spatial dimensions
- Ends with three fully connected layers that reduce to a 2-class output
- The final output represents the logits for cat (0) and dog (1) classes

#### Dataset Handling
The `CatDogDataset` class manages data loading and preprocessing:
```python
class CatDogDataset(Dataset):
    def __init__(self, image_paths, transform):
        super().__init__()
        self.paths = image_paths
        self.len = len(self.paths)
        self.transform = transform

    def __len__(self): return self.len

    def __getitem__(self, index): 
        path = self.paths[index]
        image = Image.open(path).convert('RGB')
        image = self.transform(image)
        label = 0 if 'cat' in path else 1  # Label determination from filename
        return (image, label)
```
- Implements PyTorch's `Dataset` class for efficient data loading
- Label determination is handled by checking if 'cat' is in the filename (0 for cat, 1 for dog)
- The transform parameter applies standardized preprocessing to each image:
  ```python
  transform = transforms.Compose([
      transforms.Resize((224, 224)),  # Resize all images to consistent dimensions
      transforms.ToTensor(),           # Convert to PyTorch tensors
      transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values
  ])
  ```

#### Command Line Arguments
The script uses `typer` for clean command-line argument handling:
```python
def main(
    data_dir: Path = typer.Option("./data", "--data-dir", "-d", help="Directory containing the training data"),
    checkpoint_dir: Path = typer.Option("./checkpoints", "--checkpoint-dir", "-c", help="Directory to save model checkpoints"),
    epochs: int = typer.Option(10, "--epochs", "-e", help="Number of training epochs"),
):
```
- Allows flexible specification of data and output directories
- Sets reasonable defaults while enabling overrides from command line

#### Device Management
The script automatically selects the best available device:
```python
device = torch.device("cuda" if torch.cuda.is_available() else 
                     "mps" if torch.backends.mps.is_available() else 
                     "cpu")
```
- First checks for CUDA (NVIDIA GPUs)
- Then checks for MPS (Apple Silicon GPUs), included just for personal experiments
- Falls back to CPU if no accelerators are available

#### Model and Tensor Movement
All models and tensors are moved to the selected device:
```python
# Move model to device
model = CatAndDogConvNet().to(device)

# Within training loop:
X, y = X.to(device), y.to(device)  # Move batch data to device
```
- `.to(device)` transfers tensors to the appropriate hardware
- This is critical for GPU acceleration - operations happen on the device where the data resides
- Both model parameters and input data must be on the same device

This design allows the same code to run efficiently on various hardware configurations, from laptops to CHTC's GPU resources, without code changes.

## Deployment

### Objective
- Effectively utilize the resources available in CHTC to train my model (or many variants of my model)

### Approach

1. **Environment Setup**:
   - The provided `Dockerfile` creates a container with CUDA, Python 3.12, and necessary packages
   - Build and push the Docker image:
     ```bash
     docker build -t your-username/catdog .
     docker push your-username/catdog
     ```

2. **Training Job Submission**:
   - Use the provided `train.sub` file to submit jobs to CHTC
   - The submit file contains:
     - GPU requests with capability and memory specifications
     - Container setup with the Docker image
     - Input/output file transfer configuration
   
3. **Submit the Training Job**:
   ```bash
   condor_submit train.sub
   ```

4. **Monitor Job Progress**:
   ```bash
   condor_watch_q
   ```

5. **Retrieve Results**:
   - After completion, model weights will be in the `output` directory


## Improving deployment 

### Objective
Get even more out of the available resources by implementing techniques that allow us to "do more with less." In this phase, we're focused on:
- Implementing [checkpointing](https://chtc.cs.wisc.edu/uw-research-computing/checkpointing
) to resume training from where it left off
    - Building resilience against job eviction and machine issues
    - Enabling access to "short" job slots (which have more availability) for longer training runs
    - Utilizing backfill on prioritized nodes to effectively double available resources
And, beyond the scope of this work, this is where we'd also visit:
- Automated workflows via [DAGMan](https://chtc.cs.wisc.edu/uw-research-computing/htc/dagman-simple-example)
- Hyperparameter sweeps and ensembles of models (high throughput machine learning)
- Weights and Biases (or similar tools) for monitoring and tracking experiments

### Approach

1. **Checkpoint Implementation**:
   - The `train_with_checkpoint.py` script extends the base training with checkpoint functionality
   - Key additions:
     - Saving model state, optimizer state, and training metrics at regular intervals, and using special exit codes (85) to signal that a job should be resumed
        ```python
        if (epoch+1) % checkpoint_every == 0 or (epoch+1) == epochs:
            # Save checkpoint after each epoch
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "losses": losses,
                "accuracies": accuracies,
            }
            torch.save(checkpoint, model_path)
            print(f"Saving checkpoint to {model_path}. Epochs trained: {epoch + 1} of {epochs}")
            if epoch+1 == epochs:
                sys.exit(0)  # Normal completion
            else:
                sys.exit(85)  # Checkpoint signal
        ```
     - Loading from existing checkpoints when available
        ```python
        if os.path.exists(model_path):
            print(f"Loading checkpoint from {model_path}")
            checkpoint = torch.load(model_path, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            losses = checkpoint['losses']
            accuracies = checkpoint['accuracies']
            print(f"Resuming training at epoch {start_epoch}")
        ```

2. **Submit File Changes**:
   - Update the submit file with checkpoint configuration:
     ```
     checkpoint_exit_code = 85
     transfer_checkpoint_files = output/model.pth
     +is_resumable = true
     ```
   - Exit code 85 is a special signal to HTCondor that the job is not complete but has reached a checkpoint
   - In `train_with_checkpoint.py`, this happens with the line `sys.exit(85)` after saving a checkpoint
   - When the job exits with code 85:
    - HTCondor interprets exit code 85 as a "soft termination" rather than a job failure
    - The job is placed back in the queue to be resumed when resources are available


3. **Benefits**:
   - Resilience against job eviction and machine failures
   - Ability to use "short" job slots (12-hour max) for long training runs
   - Opportunity to run as backfill on prioritized nodes, effectively doubling available resources

4. **Submitting Checkpointed Jobs**:
   - Update the shell script to use the checkpointing-enabled training script and re-emit the python script's exit code:
    ```bash
    python train.py \
        --data-dir $wd/data/ \ 
        --checkpoint-dir $wd/output/ \
        --checkpoint-every 5
        --epochs 20
    exit $?
    ```
   - Use the same submission command: `condor_submit train.sub`

## Inference
The same development/deployment/improvement approach can be applied to our inference task. For brevity, we will highlight only the differences from the inference script and workflow.

### Objective
Effectively deploy our trained model for making predictions on new data. During this inference phase, key considerations include:
- Leveraging existing code and structure from the training phase
- Reusing the same container image with minimal modifications
- Evaluating whether GPUs or CPUs provide better throughput for inference tasks
- Setting up proper resource requirements for efficient execution

### Approach
1. **Model Preparation**:
   - Ensure your trained model file (model.pth) is available

2. **Test Data Preparation**:
   - Place test images in a directory
   - Compress the test data: `zip -r images.zip data/test/`

3. **Inference Script**:
   - The `infer.py` script:
     - Leverages the model and image transformations as the training script
     - Loads the trained model
     - Processes input images
     - Outputs probability scores for each image
     - Creates a CSV file with results

4. **Container Update**:
   - Either:
     - Add the inference script to your existing container (personal preference for reproducibility)
     - Transfer the script as part of the job

5. **Submit File Configuration**:
   - Use the provided `infer.sub` file, which:
     - Updates the container tag used
     - Transfers the model weights and test images

6. **Submit the Inference Job**:
   ```bash
   condor_submit infer.sub
   ```

7. **Interpret Results**:
   - The output CSV file `dog_probs.csv` contains each image ID and its probability of being a dog
   - Probability > 0.5 indicates dog, < 0.5 indicates cat

## Optimizations and Considerations

- **CPU vs GPU for Inference**: For large-scale inference, consider using CPUs which may provide higher throughput.
- **GPU Requirements**: The default configuration requests GPUs with compute capability 7.5+. Adjust based on your model needs.
- **Data Management**: For large datasets, consider using OSDF or other storage solutions instead of transferring files. See our [data management guide](https://chtc.cs.wisc.edu/uw-research-computing/htc-job-file-transfer).
