# Configuration for FL+DAVS Project 

# Dataset Configuration
DATASET = "pathmnist"  # Options: pathmnist, chestmnist, dermamnist
NUM_CLIENTS = 10  # Number of hospital nodes (simulated)
DATA_SPLIT = "non_iid"  # Options: "iid" or "non_iid"

# Model Configuration
MODEL_NAME = "SimpleCNN"
INPUT_CHANNELS = 3  # RGB images for MedMNIST
NUM_CLASSES = 9  # PathMNIST has 9 classes

# Training Configuration
LOCAL_EPOCHS = 2  # Number of epochs each client trains per round
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_ROUNDS = 20  # Total federated learning rounds

# Device Configuration
DEVICE = "cpu"  # Change to "cuda" if GPU is available

# Paths
DATA_DIR = "./data"  # MedMNIST will create subdirectories here
MODEL_SAVE_DIR = "./saved_models"
RESULTS_DIR = "./results"
