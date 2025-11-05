"""
Example script demonstrating causal reasoning with EEG data
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from braindecode.datasets import TUHAbnormal
from braindecode.preprocessing import create_fixed_length_windows, preprocess
import configs.preprocess_config as preprocess_config
from EEGClip.causal_reasoning import CausalReasoningEngine
from EEGClip.clip_models import EEGEncoder


def visualize_causal_matrix(causal_matrix, channel_names=None):
    """
    Visualize the causal matrix as a heatmap
    
    Args:
        causal_matrix: [num_nodes, num_nodes] - Causal relationship matrix
        channel_names: list - Names of EEG channels
    """
    if channel_names is None:
        channel_names = [f"Channel {i}" for i in range(causal_matrix.shape[0])]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        causal_matrix.cpu().detach().numpy(),
        xticklabels=channel_names,
        yticklabels=channel_names,
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".2f"
    )
    plt.title("Causal Relationships Between EEG Channels")
    plt.xlabel("Effect")
    plt.ylabel("Cause")
    plt.tight_layout()
    plt.savefig("causal_matrix.png", dpi=300, bbox_inches='tight')
    plt.show()


def main():
    # Configuration
    n_recordings_to_load = 10
    num_workers = 4
    
    # Load TUH dataset
    print("Loading TUH dataset...")
    dataset = TUHAbnormal(
        path=preprocess_config.tuh_data_dir,
        recording_ids=range(n_recordings_to_load),
        target_name="report",
        preload=False,
        add_physician_reports=True,
        n_jobs=num_workers,
    )
    
    # Preprocess the data
    print("Preprocessing data...")
    preprocess(dataset, preprocess_config.preprocessors)
    
    # Use validation set
    valid_set = dataset.split("train")["False"]
    
    # Create windowed dataset
    n_max_minutes = 3
    sfreq = 100
    n_minutes = 2
    input_window_samples = 1200
    n_preds_per_input = 519
    
    window_valid_set = create_fixed_length_windows(
        valid_set,
        start_offset_samples=60 * sfreq,
        stop_offset_samples=60 * sfreq + n_minutes * 60 * sfreq,
        preload=True,
        window_size_samples=input_window_samples,
        window_stride_samples=n_preds_per_input,
        drop_last_window=False,
    )
    
    # Create data loader
    valid_loader = torch.utils.data.DataLoader(
        window_valid_set,
        batch_size=4,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    
    # Initialize EEG encoder (same as used in EEG-CLIP)
    eeg_encoder = EEGEncoder(
        eeg_model_emb_dim=128,
        n_chans=21,
        eeg_model_pretrained=False,
        eeg_model_trainable=True,
    )
    
    # Initialize causal reasoning engine
    causal_engine = CausalReasoningEngine(
        eeg_model_emb_dim=128,
        projected_emb_dim=64,
        num_eeg_channels=21
    )
    
    # Get a sample batch
    print("Analyzing causal relationships...")
    sample_batch = next(iter(valid_loader))
    eeg_data, text_data, ids = sample_batch
    
    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eeg_data = eeg_data.to(device)
    eeg_encoder = eeg_encoder.to(device)
    causal_engine = causal_engine.to(device)
    
    # Encode EEG data
    with torch.no_grad():
        eeg_features = eeg_encoder(eeg_data)  # [B, N_pred, Enc_size]
        print(f"EEG features shape: {eeg_features.shape}")
    
    # Apply causal reasoning
    causal_matrix, counterfactual_features, interventions, causal_effects = causal_engine(eeg_features)
    
    print(f"Causal matrix shape: {causal_matrix.shape}")
    print(f"Counterfactual features shape: {counterfactual_features.shape}")
    print(f"Interventions shape: {interventions.shape}")
    print(f"Causal effects shape: {causal_effects.shape}")
    
    # Print some statistics
    print(f"Average causal relationship strength: {causal_matrix.mean().item():.4f}")
    print(f"Max causal relationship strength: {causal_matrix.max().item():.4f}")
    print(f"Min causal relationship strength: {causal_matrix.min().item():.4f}")
    print(f"Average predicted intervention: {interventions.mean().item():.4f}")
    print(f"Average causal effect: {causal_effects.mean().item():.4f}")
    
    # Visualize causal matrix for the first sample
    print("Generating causal relationship visualization...")
    channel_names = [
        "A1", "A2", "FP1", "FP2", "F3", "F4", "C3", "C4", 
        "P3", "P4", "O1", "O2", "F7", "F8", "T3", "T4", 
        "T5", "T6", "FZ", "CZ", "PZ"
    ]
    
    # Visualize the causal matrix for the first sample
    visualize_causal_matrix(causal_matrix[0], channel_names)
    
    # Show interventions for the first sample
    print("Top interventions:")
    intervention_values = interventions[0].cpu().detach().numpy()
    top_indices = np.argsort(intervention_values)[-5:][::-1]  # Top 5
    for i in top_indices:
        print(f"  Channel {channel_names[i]}: {intervention_values[i]:.4f}")
    
    print("Causal reasoning demonstration complete!")
    print("Causal matrix visualization saved as 'causal_matrix.png'")


if __name__ == "__main__":
    main()