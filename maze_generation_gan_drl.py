# Cleaned/refactored code exported from the portfolio notebook.

# ============================================================================
# 1. Imports and configuration
# ============================================================================

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Deque, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import label

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# -------------------------------
# Project paths
# -------------------------------

PROJECT_ROOT = Path.cwd()

# Recommended GitHub layout:
# project/
#   data/
#     training_images/
#     binary_mazes.npy
#   outputs/
#   ECE651_Maze_Generation_GAN_DRL_Cleaned.ipynb

IMAGE_DIR = PROJECT_ROOT / "data" / "training_images"
BINARY_MAZES_PATH = PROJECT_ROOT / "data" / "binary_mazes.npy"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------
# Reproducibility and hardware
# -------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


def set_seed(seed: int) -> None:
    """Set common random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic mode improves repeatability but may reduce speed.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------------
# Training controls
# -------------------------------

RUN_FC_GAN = False
RUN_DCGAN = False
RUN_DQN = False

LATENT_DIM = 100
BATCH_SIZE = 32

# Original project-scale training lengths.
FC_GAN_EPOCHS = 3100
DCGAN_EPOCHS = 2100
DQN_EPISODES = 500

# For a quick smoke test, temporarily reduce these values.


# ============================================================================
# 2. Dataset preparation utilities
# ============================================================================

def png_to_binary_maze(image_path: Path, threshold: int = 128) -> np.ndarray:
    """
    Convert a maze PNG into a 2D binary matrix.

    Parameters
    ----------
    image_path:
        Path to a PNG maze image.
    threshold:
        Grayscale threshold. Pixels below the threshold are treated as walls.

    Returns
    -------
    np.ndarray
        2D uint8 matrix where 1 = wall and 0 = walkable space.
    """
    with Image.open(image_path) as img:
        grayscale = img.convert("L")
        arr = np.asarray(grayscale)
        binary = (arr < threshold).astype(np.uint8)
    return binary


def build_binary_dataset(
    image_dir: Path,
    output_path: Path,
    expected_count: Optional[int] = 1000,
) -> np.ndarray:
    """
    Read PNG files, convert them to binary maze images, validate common shape,
    and save the dataset as a NumPy array.
    """
    image_dir = Path(image_dir)
    output_path = Path(output_path)

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Training image directory was not found: {image_dir}"
        )

    if expected_count is not None:
        image_paths = [image_dir / f"{i}.png" for i in range(expected_count)]
        missing = [p.name for p in image_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} expected maze images are missing. "
                f"First missing files: {missing[:10]}"
            )
    else:
        image_paths = sorted(image_dir.glob("*.png"))

    if not image_paths:
        raise ValueError(f"No PNG files found in {image_dir}")

    mazes = [png_to_binary_maze(path) for path in image_paths]

    shapes = {maze.shape for maze in mazes}
    if len(shapes) != 1:
        raise ValueError(f"All maze images must have one common shape; found {shapes}")

    binary_mazes = np.stack(mazes).astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, binary_mazes)

    print(f"Saved {len(binary_mazes)} mazes to {output_path}")
    print(f"Rendered maze image shape: {binary_mazes.shape[1:]}")

    return binary_mazes


def load_binary_dataset(path: Path) -> np.ndarray:
    """Load and validate a saved binary-maze dataset."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"{path} was not found. Build the dataset from PNGs first."
        )

    mazes = np.load(path, allow_pickle=False)

    if mazes.ndim != 3:
        raise ValueError(
            f"Expected array shape (n_mazes, height, width); got {mazes.shape}"
        )

    values = np.unique(mazes)
    if not np.all(np.isin(values, [0, 1])):
        raise ValueError(
            f"Expected binary values 0/1, but found {values[:20]}"
        )

    return mazes.astype(np.uint8)


# Build the dataset only if it does not already exist.
if BINARY_MAZES_PATH.exists():
    binary_mazes = load_binary_dataset(BINARY_MAZES_PATH)
    print(f"Loaded {len(binary_mazes)} mazes from {BINARY_MAZES_PATH}")
    print(f"Rendered maze image shape: {binary_mazes.shape[1:]}")
elif IMAGE_DIR.exists():
    binary_mazes = build_binary_dataset(
        IMAGE_DIR,
        BINARY_MAZES_PATH,
        expected_count=1000,
    )
else:
    binary_mazes = None
    print(
        "Dataset not found yet. Place maze PNGs in data/training_images/ "
        "or place binary_mazes.npy in data/."
    )


# ============================================================================
# 3. Dataset visualization and connectivity checks
# ============================================================================

def count_walkable_components(maze: np.ndarray) -> int:
    """
    Count connected components in the walkable region of a binary maze.

    This is a pixel-level connectivity check:
      0 = walkable
      1 = wall

    IMPORTANT:
    A single connected walkable component does NOT by itself prove that a
    logical maze has exactly one solution path.
    """
    walkable = (maze == 0).astype(np.uint8)
    _, n_components = label(walkable)
    return int(n_components)


def dataset_connectivity_summary(mazes: np.ndarray) -> dict:
    """Summarize pixel-level walkable connectivity across the dataset."""
    component_counts = np.array(
        [count_walkable_components(maze) for maze in mazes],
        dtype=int,
    )

    return {
        "n_mazes": int(len(mazes)),
        "single_component": int(np.sum(component_counts == 1)),
        "multiple_components": int(np.sum(component_counts > 1)),
        "min_components": int(component_counts.min()),
        "max_components": int(component_counts.max()),
    }


def show_mazes(
    mazes: np.ndarray,
    indices: Sequence[int] = (0, 1, 2, 3),
    title: str = "Example Training Mazes",
) -> None:
    """Display selected binary mazes."""
    n = len(indices)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))

    if n == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        ax.imshow(mazes[idx], cmap="binary")
        ax.set_title(f"Maze {idx}")
        ax.axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


if binary_mazes is not None:
    show_mazes(binary_mazes)
    print(dataset_connectivity_summary(binary_mazes))


# ============================================================================
# 4. Shared GAN utilities
# ============================================================================

def make_gan_dataloader(
    mazes: np.ndarray,
    batch_size: int = 32,
) -> DataLoader:
    """
    Scale binary mazes from {0, 1} to {-1, 1} for Tanh generator output.
    """
    scaled = mazes.astype(np.float32) * 2.0 - 1.0
    tensor = torch.from_numpy(scaled).unsqueeze(1)
    dataset = TensorDataset(tensor)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )


@dataclass
class GANTrainingResult:
    generator: nn.Module
    discriminator: nn.Module
    generator_losses: list
    discriminator_losses: list


def save_generated_sample(
    generator: nn.Module,
    latent_dim: int,
    output_path: Path,
    title: str,
) -> None:
    """Generate one maze image and save/display it."""
    generator.eval()

    with torch.no_grad():
        z = torch.randn(1, latent_dim, device=DEVICE)
        generated = generator(z).squeeze().cpu().numpy()

    display_img = (generated + 1.0) / 2.0

    plt.figure(figsize=(5, 5))
    plt.imshow(display_img, cmap="gray", vmin=0, vmax=1)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.show()

    generator.train()


def train_gan(
    generator: nn.Module,
    discriminator: nn.Module,
    dataloader: DataLoader,
    *,
    latent_dim: int,
    epochs: int,
    learning_rate: float,
    betas: Tuple[float, float] = (0.9, 0.999),
    sample_every: int = 500,
    checkpoint_prefix: str = "gan",
) -> GANTrainingResult:
    """
    Train a standard GAN using binary cross-entropy loss.

    The discriminator includes a Sigmoid output, matching the original project,
    so BCELoss is used here.
    """
    generator = generator.to(DEVICE)
    discriminator = discriminator.to(DEVICE)

    criterion = nn.BCELoss()

    optimizer_g = optim.Adam(
        generator.parameters(),
        lr=learning_rate,
        betas=betas,
    )
    optimizer_d = optim.Adam(
        discriminator.parameters(),
        lr=learning_rate,
        betas=betas,
    )

    g_history = []
    d_history = []

    for epoch in range(1, epochs + 1):
        epoch_g = []
        epoch_d = []

        for (real_images,) in dataloader:
            real_images = real_images.to(DEVICE)
            batch_n = real_images.size(0)

            real_labels = torch.ones(batch_n, 1, device=DEVICE)
            fake_labels = torch.zeros(batch_n, 1, device=DEVICE)

            # Train generator.
            optimizer_g.zero_grad()

            z = torch.randn(batch_n, latent_dim, device=DEVICE)
            generated_images = generator(z)

            g_loss = criterion(
                discriminator(generated_images),
                real_labels,
            )
            g_loss.backward()
            optimizer_g.step()

            # Train discriminator.
            optimizer_d.zero_grad()

            real_loss = criterion(
                discriminator(real_images),
                real_labels,
            )
            fake_loss = criterion(
                discriminator(generated_images.detach()),
                fake_labels,
            )

            d_loss = 0.5 * (real_loss + fake_loss)
            d_loss.backward()
            optimizer_d.step()

            epoch_g.append(g_loss.item())
            epoch_d.append(d_loss.item())

        mean_g = float(np.mean(epoch_g))
        mean_d = float(np.mean(epoch_d))

        g_history.append(mean_g)
        d_history.append(mean_d)

        if epoch == 1 or epoch % 100 == 0:
            print(
                f"Epoch {epoch:4d}/{epochs} | "
                f"D loss: {mean_d:.4f} | G loss: {mean_g:.4f}"
            )

        if sample_every and (epoch == 1 or epoch % sample_every == 0):
            sample_path = OUTPUT_DIR / f"{checkpoint_prefix}_epoch_{epoch}.png"
            save_generated_sample(
                generator,
                latent_dim,
                sample_path,
                title=f"{checkpoint_prefix.upper()} — epoch {epoch}",
            )

            torch.save(
                generator.state_dict(),
                OUTPUT_DIR / f"{checkpoint_prefix}_generator_epoch_{epoch}.pth",
            )

    torch.save(
        generator.state_dict(),
        OUTPUT_DIR / f"{checkpoint_prefix}_generator_final.pth",
    )

    return GANTrainingResult(
        generator=generator,
        discriminator=discriminator,
        generator_losses=g_history,
        discriminator_losses=d_history,
    )


def plot_gan_losses(
    result: GANTrainingResult,
    title: str,
) -> None:
    """Plot generator and discriminator training losses."""
    epochs = np.arange(1, len(result.generator_losses) + 1)

    plt.figure(figsize=(9, 4.5))
    plt.plot(epochs, result.generator_losses, label="Generator loss")
    plt.plot(epochs, result.discriminator_losses, label="Discriminator loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================================
# 5. Fully Connected GAN
# ============================================================================

class FullyConnectedGenerator(nn.Module):
    def __init__(self, latent_dim: int, height: int, width: int):
        super().__init__()

        self.height = height
        self.width = width

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, height * width),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.model(z)
        return x.view(z.size(0), 1, self.height, self.width)


class FullyConnectedDiscriminator(nn.Module):
    def __init__(self, height: int, width: int):
        super().__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(height * width, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)


fc_gan_result = None

if RUN_FC_GAN:
    if binary_mazes is None:
        raise RuntimeError("Load/build the maze dataset before training.")

    set_seed(12)

    height, width = binary_mazes.shape[1:]
    dataloader = make_gan_dataloader(binary_mazes, BATCH_SIZE)

    fc_generator = FullyConnectedGenerator(
        LATENT_DIM,
        height,
        width,
    )
    fc_discriminator = FullyConnectedDiscriminator(
        height,
        width,
    )

    fc_gan_result = train_gan(
        fc_generator,
        fc_discriminator,
        dataloader,
        latent_dim=LATENT_DIM,
        epochs=FC_GAN_EPOCHS,
        learning_rate=1e-4,
        betas=(0.9, 0.999),
        sample_every=300,
        checkpoint_prefix="fc_gan",
    )

    plot_gan_losses(
        fc_gan_result,
        "Fully Connected GAN Training Loss",
    )
else:
    print("FC GAN training skipped. Set RUN_FC_GAN = True to train.")


# ============================================================================
# 6. DCGAN-style model
# ============================================================================

class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim: int, height: int, width: int):
        super().__init__()

        if height % 4 != 0 or width % 4 != 0:
            raise ValueError(
                "This DCGAN architecture requires height and width divisible by 4. "
                f"Received {(height, width)}."
            )

        self.init_h = height // 4
        self.init_w = width // 4

        self.fc = nn.Linear(
            latent_dim,
            128 * self.init_h * self.init_w,
        )

        self.model = nn.Sequential(
            nn.ConvTranspose2d(
                128, 128,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.ConvTranspose2d(
                128, 64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Conv2d(
                64, 1,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(
            z.size(0),
            128,
            self.init_h,
            self.init_w,
        )
        return self.model(x)


class DCGANDiscriminator(nn.Module):
    def __init__(self, height: int, width: int):
        super().__init__()

        if height % 4 != 0 or width % 4 != 0:
            raise ValueError(
                "This DCGAN discriminator requires height and width divisible by 4."
            )

        self.features = nn.Sequential(
            nn.Conv2d(
                1, 64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(0.2),

            nn.Conv2d(
                64, 128,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )

        feature_size = 128 * (height // 4) * (width // 4)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.features(image)
        return self.classifier(x)


dcgan_result = None

if RUN_DCGAN:
    if binary_mazes is None:
        raise RuntimeError("Load/build the maze dataset before training.")

    set_seed(23)

    height, width = binary_mazes.shape[1:]
    dataloader = make_gan_dataloader(binary_mazes, BATCH_SIZE)

    dc_generator = DCGANGenerator(
        LATENT_DIM,
        height,
        width,
    )
    dc_discriminator = DCGANDiscriminator(
        height,
        width,
    )

    dcgan_result = train_gan(
        dc_generator,
        dc_discriminator,
        dataloader,
        latent_dim=LATENT_DIM,
        epochs=DCGAN_EPOCHS,
        learning_rate=1e-4,
        betas=(0.5, 0.999),
        sample_every=500,
        checkpoint_prefix="dcgan",
    )

    plot_gan_losses(
        dcgan_result,
        "DCGAN Training Loss",
    )
else:
    print("DCGAN training skipped. Set RUN_DCGAN = True to train.")


# ============================================================================
# 7. DQN maze-generation environment and utilities
# ============================================================================

GridPosition = Tuple[int, int]


def nearest_walkable_to_corner(
    maze: np.ndarray,
    corner: str,
) -> GridPosition:
    """Find a walkable cell nearest a requested corner."""
    walkable = np.argwhere(maze == 0)

    if len(walkable) == 0:
        raise ValueError("Maze has no walkable cells.")

    h, w = maze.shape

    if corner == "top_left":
        distances = walkable[:, 0] + walkable[:, 1]
    elif corner == "bottom_right":
        distances = (
            (h - 1 - walkable[:, 0]) +
            (w - 1 - walkable[:, 1])
        )
    else:
        raise ValueError("corner must be 'top_left' or 'bottom_right'")

    pos = walkable[np.argmin(distances)]
    return int(pos[0]), int(pos[1])


def shortest_path_length(
    maze: np.ndarray,
    start: GridPosition,
    goal: GridPosition,
) -> Optional[int]:
    """
    Compute shortest 4-neighbor path length using BFS.

    Returns None when no valid path exists.
    """
    h, w = maze.shape

    if maze[start] == 1 or maze[goal] == 1:
        return None

    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        (row, col), distance = queue.popleft()

        if (row, col) == goal:
            return distance

        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = row + dr, col + dc

            if not (0 <= nr < h and 0 <= nc < w):
                continue

            next_pos = (nr, nc)

            if next_pos in visited:
                continue

            if maze[next_pos] == 1:
                continue

            visited.add(next_pos)
            queue.append((next_pos, distance + 1))

    return None


@dataclass
class MazeStep:
    state: np.ndarray
    reward: float
    done: bool
    accepted: bool
    path_length: Optional[int]


class MazeGenerationEnv:
    """
    Custom environment for DQN-based maze refinement.

    The state remains binary (0 path / 1 wall). Start and goal are stored
    separately rather than encoded as 2 and 3 inside the network state.
    """

    def __init__(
        self,
        maze: np.ndarray,
        max_steps: int = 25,
        invalid_action_penalty: float = -10.0,
    ):
        self.initial_maze = maze.astype(np.uint8).copy()
        self.h, self.w = maze.shape
        self.max_steps = max_steps
        self.invalid_action_penalty = invalid_action_penalty

        self.start = nearest_walkable_to_corner(
            self.initial_maze,
            "top_left",
        )
        self.goal = nearest_walkable_to_corner(
            self.initial_maze,
            "bottom_right",
        )

        initial_length = shortest_path_length(
            self.initial_maze,
            self.start,
            self.goal,
        )

        if initial_length is None:
            raise ValueError(
                "Selected start and goal are not connected in the initial maze."
            )

        self.maze = self.initial_maze.copy()
        self.steps = 0

    @property
    def state_dim(self) -> int:
        return self.h * self.w

    @property
    def action_dim(self) -> int:
        return self.h * self.w

    def reset(self) -> np.ndarray:
        self.maze = self.initial_maze.copy()
        self.steps = 0
        return self.maze.flatten().astype(np.float32)

    def step(self, action: int) -> MazeStep:
        self.steps += 1

        row, col = divmod(int(action), self.w)
        position = (row, col)

        # Start and goal must remain walkable.
        if position in (self.start, self.goal):
            done = self.steps >= self.max_steps
            return MazeStep(
                state=self.maze.flatten().astype(np.float32),
                reward=self.invalid_action_penalty,
                done=done,
                accepted=False,
                path_length=shortest_path_length(
                    self.maze,
                    self.start,
                    self.goal,
                ),
            )

        candidate = self.maze.copy()
        candidate[position] = 1 - candidate[position]

        path_length = shortest_path_length(
            candidate,
            self.start,
            self.goal,
        )

        if path_length is None:
            reward = self.invalid_action_penalty
            accepted = False
        else:
            self.maze = candidate
            reward = path_length / float(self.h * self.w)
            accepted = True

        done = self.steps >= self.max_steps

        return MazeStep(
            state=self.maze.flatten().astype(np.float32),
            reward=float(reward),
            done=done,
            accepted=accepted,
            path_length=path_length,
        )


def plot_rl_maze(
    maze: np.ndarray,
    start: GridPosition,
    goal: GridPosition,
    title: str,
) -> None:
    """Visualize a binary maze with explicit start/goal markers."""
    plt.figure(figsize=(6, 6))
    plt.imshow(maze, cmap="binary", vmin=0, vmax=1)

    plt.scatter(
        start[1], start[0],
        marker="o",
        s=70,
        label="Start",
    )
    plt.scatter(
        goal[1], goal[0],
        marker="*",
        s=110,
        label="Goal",
    )

    plt.title(title)
    plt.axis("off")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================================
# 8. DQN model, replay buffer, and training loop
# ============================================================================

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: Deque = deque(maxlen=capacity)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append(
            (state, action, reward, next_state, done)
        )

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class EpsilonGreedyPolicy:
    def __init__(
        self,
        start: float = 1.0,
        end: float = 0.01,
        decay: float = 0.995,
    ):
        self.epsilon = start
        self.end = end
        self.decay = decay

    def select_action(
        self,
        model: nn.Module,
        state: np.ndarray,
        action_dim: int,
    ) -> int:
        if random.random() < self.epsilon:
            return random.randrange(action_dim)

        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = model(state_tensor)

        return int(torch.argmax(q_values, dim=1).item())

    def update(self) -> None:
        self.epsilon = max(
            self.end,
            self.epsilon * self.decay,
        )


@dataclass
class DQNTrainingResult:
    model: nn.Module
    episode_rewards: list
    losses: list
    accepted_action_rates: list
    final_maze: np.ndarray
    start: GridPosition
    goal: GridPosition


def train_dqn_maze_generator(
    mazes: np.ndarray,
    *,
    episodes: int = 500,
    max_steps_per_episode: int = 25,
    buffer_capacity: int = 10_000,
    batch_size: int = 64,
    gamma: float = 0.99,
    learning_rate: float = 1e-3,
    target_update_every: int = 10,
) -> DQNTrainingResult:
    """
    Train a DQN that refines binary maze images while preserving solvability.
    """
    prototype = None

    for maze in mazes:
        try:
            env = MazeGenerationEnv(
                maze,
                max_steps=max_steps_per_episode,
            )
            prototype = maze
            break
        except ValueError:
            continue

    if prototype is None:
        raise ValueError(
            "No maze with connected start/goal could be found."
        )

    env = MazeGenerationEnv(
        prototype,
        max_steps=max_steps_per_episode,
    )

    state_dim = env.state_dim
    action_dim = env.action_dim

    print(f"DQN state dimension:  {state_dim}")
    print(f"DQN action dimension: {action_dim}")
    print(f"Maze pixel shape:      {(env.h, env.w)}")

    model = DQN(state_dim, action_dim).to(DEVICE)
    target_model = DQN(state_dim, action_dim).to(DEVICE)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )
    loss_fn = nn.MSELoss()

    replay = ReplayBuffer(buffer_capacity)
    policy = EpsilonGreedyPolicy(
        start=1.0,
        end=0.01,
        decay=0.995,
    )

    episode_rewards = []
    losses = []
    accepted_rates = []

    last_env = env

    for episode in range(1, episodes + 1):
        for _ in range(100):
            candidate = mazes[random.randrange(len(mazes))]
            try:
                env = MazeGenerationEnv(
                    candidate,
                    max_steps=max_steps_per_episode,
                )
                break
            except ValueError:
                continue
        else:
            raise RuntimeError(
                "Unable to sample a solvable maze after 100 attempts."
            )

        state = env.reset()
        total_reward = 0.0
        accepted_count = 0

        while True:
            action = policy.select_action(
                model,
                state,
                env.action_dim,
            )

            transition = env.step(action)
            next_state = transition.state

            replay.add(
                state,
                action,
                transition.reward,
                next_state,
                transition.done,
            )

            total_reward += transition.reward
            accepted_count += int(transition.accepted)

            if len(replay) >= batch_size:
                (
                    states,
                    actions,
                    rewards,
                    next_states,
                    dones,
                ) = replay.sample(batch_size)

                states_t = torch.as_tensor(
                    states,
                    dtype=torch.float32,
                    device=DEVICE,
                )
                actions_t = torch.as_tensor(
                    actions,
                    dtype=torch.long,
                    device=DEVICE,
                )
                rewards_t = torch.as_tensor(
                    rewards,
                    dtype=torch.float32,
                    device=DEVICE,
                )
                next_states_t = torch.as_tensor(
                    next_states,
                    dtype=torch.float32,
                    device=DEVICE,
                )
                dones_t = torch.as_tensor(
                    dones,
                    dtype=torch.float32,
                    device=DEVICE,
                )

                q_values = model(states_t).gather(
                    1,
                    actions_t.unsqueeze(1),
                ).squeeze(1)

                with torch.no_grad():
                    next_q = target_model(next_states_t).max(dim=1).values
                    targets = rewards_t + (
                        1.0 - dones_t
                    ) * gamma * next_q

                loss = loss_fn(q_values, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(float(loss.item()))

            state = next_state

            if transition.done:
                break

        policy.update()

        if episode % target_update_every == 0:
            target_model.load_state_dict(
                model.state_dict()
            )

        episode_rewards.append(total_reward)
        accepted_rates.append(
            accepted_count / max_steps_per_episode
        )

        last_env = env

        if episode == 1 or episode % 25 == 0:
            recent_reward = np.mean(
                episode_rewards[-25:]
            )
            recent_acceptance = np.mean(
                accepted_rates[-25:]
            )

            print(
                f"Episode {episode:4d}/{episodes} | "
                f"mean recent reward: {recent_reward:.3f} | "
                f"accepted actions: {recent_acceptance:.2%} | "
                f"epsilon: {policy.epsilon:.3f}"
            )

    torch.save(
        model.state_dict(),
        OUTPUT_DIR / "dqn_maze_generator_final.pth",
    )

    return DQNTrainingResult(
        model=model,
        episode_rewards=episode_rewards,
        losses=losses,
        accepted_action_rates=accepted_rates,
        final_maze=last_env.maze.copy(),
        start=last_env.start,
        goal=last_env.goal,
    )


dqn_result = None

if RUN_DQN:
    if binary_mazes is None:
        raise RuntimeError("Load/build the maze dataset before training.")

    set_seed(42)

    dqn_result = train_dqn_maze_generator(
        binary_mazes,
        episodes=DQN_EPISODES,
        max_steps_per_episode=25,
        buffer_capacity=10_000,
        batch_size=64,
        gamma=0.99,
        learning_rate=1e-3,
        target_update_every=10,
    )

    plot_rl_maze(
        dqn_result.final_maze,
        dqn_result.start,
        dqn_result.goal,
        "DQN-refined Maze",
    )
else:
    print("DQN training skipped. Set RUN_DQN = True to train.")


# ============================================================================
# 9. DQN training diagnostics
# ============================================================================

def plot_dqn_training(result: DQNTrainingResult) -> None:
    """Plot episode reward and accepted-action rate."""
    episodes = np.arange(
        1,
        len(result.episode_rewards) + 1,
    )

    plt.figure(figsize=(9, 4.5))
    plt.plot(
        episodes,
        result.episode_rewards,
    )
    plt.xlabel("Episode")
    plt.ylabel("Total episode reward")
    plt.title("DQN Maze Generator — Episode Reward")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 4.5))
    plt.plot(
        episodes,
        result.accepted_action_rates,
    )
    plt.xlabel("Episode")
    plt.ylabel("Accepted action rate")
    plt.title("DQN Maze Generator — Solvability-preserving Actions")
    plt.tight_layout()
    plt.show()

    if result.losses:
        plt.figure(figsize=(9, 4.5))
        plt.plot(result.losses)
        plt.xlabel("Gradient update")
        plt.ylabel("MSE loss")
        plt.title("DQN Training Loss")
        plt.tight_layout()
        plt.show()


if dqn_result is not None:
    plot_dqn_training(dqn_result)
