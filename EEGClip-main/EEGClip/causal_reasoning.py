import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalGraphInference(nn.Module):
    """
    Infers causal relationships from EEG data
    """
    def __init__(self, input_dim, hidden_dim, num_nodes):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        
        # Graph neural network for modeling relationships between EEG channels
        self.gnn_layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_nodes * num_nodes)  # Adjacency matrix
        ])
        
        # Attention mechanism for temporal dependencies
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=4,
            batch_first=True
        )
        
        # Causal discovery network
        self.causal_discovery = nn.Sequential(
            nn.Linear(num_nodes * num_nodes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_nodes * num_nodes),
            nn.Sigmoid()
        )

    def forward(self, eeg_features):
        """
        Infer causal relationships from EEG features
        
        Args:
            eeg_features: [B, N_pred, Enc_size] - EEG features
            
        Returns:
            causal_matrix: [B, num_nodes, num_nodes] - Causal relationship matrix
        """
        batch_size = eeg_features.shape[0]
        
        # Apply attention to capture temporal dependencies
        attended_features, _ = self.temporal_attention(eeg_features, eeg_features, eeg_features)
        
        # Global average pooling to get a single representation
        pooled_features = torch.mean(attended_features, dim=1)  # [B, Enc_size]
        
        # Pass through GNN layers
        x = pooled_features
        for layer in self.gnn_layers[:-1]:
            x = layer(x)
        
        # Reshape to adjacency matrix
        adj_vector = self.gnn_layers[-1](x)  # [B, num_nodes * num_nodes]
        adj_matrix = adj_vector.view(batch_size, self.num_nodes, self.num_nodes)
        
        # Apply causal discovery network
        causal_vector = self.causal_discovery(adj_vector)
        causal_matrix = causal_vector.view(batch_size, self.num_nodes, self.num_nodes)
        
        return causal_matrix


class CounterfactualGenerator(nn.Module):
    """
    Generates counterfactual explanations for EEG-based predictions
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Encoder for EEG features
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Decoder for counterfactual generation
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Intervention predictor
        self.intervention_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, output_dim),
            nn.Sigmoid()
        )

    def forward(self, eeg_features, intervention_targets=None):
        """
        Generate counterfactual EEG features
        
        Args:
            eeg_features: [B, N_pred, Enc_size] - Original EEG features
            intervention_targets: [B, output_dim] - Optional intervention targets
            
        Returns:
            counterfactual_features: [B, N_pred, output_dim] - Counterfactual EEG features
            interventions: [B, output_dim] - Predicted interventions
        """
        # Global average pooling
        pooled_features = torch.mean(eeg_features, dim=1)  # [B, Enc_size]
        
        # Encode features
        encoded = self.encoder(pooled_features)  # [B, hidden_dim // 2]
        
        # Decode to counterfactual features
        counterfactual_features = self.decoder(encoded)  # [B, output_dim]
        
        # Predict interventions
        interventions = self.intervention_predictor(encoded)  # [B, output_dim]
        
        # If intervention targets are provided, apply them
        if intervention_targets is not None:
            counterfactual_features = counterfactual_features * intervention_targets
            
        # Reshape to match input dimensions
        counterfactual_features = counterfactual_features.unsqueeze(1).expand(-1, eeg_features.shape[1], -1)
        
        return counterfactual_features, interventions


class CausalReasoningEngine(nn.Module):
    """
    Main causal reasoning engine that combines graph inference and counterfactual generation
    """
    def __init__(self, eeg_model_emb_dim, projected_emb_dim, num_eeg_channels=21):
        super().__init__()
        self.eeg_model_emb_dim = eeg_model_emb_dim
        self.projected_emb_dim = projected_emb_dim
        self.num_eeg_channels = num_eeg_channels
        
        # Causal graph inference module
        self.causal_graph = CausalGraphInference(
            input_dim=eeg_model_emb_dim,
            hidden_dim=projected_emb_dim,
            num_nodes=num_eeg_channels
        )
        
        # Counterfactual generator
        self.counterfactual_gen = CounterfactualGenerator(
            input_dim=eeg_model_emb_dim,
            hidden_dim=projected_emb_dim,
            output_dim=eeg_model_emb_dim
        )
        
        # Causal effect predictor
        self.effect_predictor = nn.Sequential(
            nn.Linear(projected_emb_dim, projected_emb_dim),
            nn.ReLU(),
            nn.Linear(projected_emb_dim, projected_emb_dim // 2),
            nn.ReLU(),
            nn.Linear(projected_emb_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, eeg_features):
        """
        Perform comprehensive causal reasoning on EEG features
        
        Args:
            eeg_features: [B, N_pred, Enc_size] - EEG features from encoder
            
        Returns:
            causal_matrix: [B, num_nodes, num_nodes] - Causal relationships
            counterfactual_features: [B, N_pred, Enc_size] - Counterfactual features
            interventions: [B, Enc_size] - Predicted interventions
            causal_effects: [B, 1] - Predicted causal effects
        """
        # Infer causal relationships
        causal_matrix = self.causal_graph(eeg_features)
        
        # Generate counterfactuals
        counterfactual_features, interventions = self.counterfactual_gen(eeg_features)
        
        # Predict causal effects
        pooled_features = torch.mean(eeg_features, dim=1)  # [B, Enc_size]
        causal_effects = self.effect_predictor(pooled_features)  # [B, 1]
        
        return causal_matrix, counterfactual_features, interventions, causal_effects


def compute_causal_loss(causal_matrix, interventions, causal_effects, lambda_reg=0.1):
    """
    Compute loss for causal reasoning
    
    Args:
        causal_matrix: [B, num_nodes, num_nodes] - Predicted causal relationships
        interventions: [B, output_dim] - Predicted interventions
        causal_effects: [B, 1] - Predicted causal effects
        lambda_reg: float - Regularization weight
        
    Returns:
        loss: scalar - Total causal loss
    """
    # Encourage sparse causal relationships (regularization)
    sparsity_loss = torch.mean(torch.abs(causal_matrix))
    
    # Encourage meaningful interventions
    intervention_loss = torch.mean(interventions * (1 - interventions))  # Entropy regularization
    
    # Encourage non-trivial causal effects
    effect_loss = torch.mean((causal_effects - 0.5) ** 2)
    
    # Combine losses
    total_loss = sparsity_loss + intervention_loss + effect_loss + lambda_reg * (
        torch.mean(causal_matrix ** 2) + 
        torch.mean(interventions ** 2) + 
        torch.mean(causal_effects ** 2)
    )
    
    return total_loss