import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

# -----------------------------------------------------------------------------
# Part 1: Modules
# -----------------------------------------------------------------------------

class ResBlock(nn.Module):
    """
    Standard Residual Block
    Structure: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> (+) -> ReLU
    """
    def __init__(self, num_channels):
        super(ResBlock, self).__init__() 
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttentionBlock, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key   = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        N = H * W
        proj_query = self.query(x).view(B, -1, N).permute(0, 2, 1) 
        proj_key   = self.key(x).view(B, -1, N)                    
        energy     = torch.bmm(proj_query, proj_key)               
        attention  = F.softmax(energy, dim=-1)
        proj_value = self.value(x).view(B, -1, N)                  
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        return self.gamma * out + x

class DynamicFeatureUpdateHead(nn.Module):
    """
    [Botzone Compatible Version]
    1. No LayerNorm/BatchNorm (Safe for PyTorch 0.2.6 inference)
    2. Kaiming Initialization (Fast convergence without BN)
    """
    def __init__(self, in_channels, board_size=8):
        super(DynamicFeatureUpdateHead, self).__init__()
        self.N = board_size * board_size
        head_dim = 128 
        
        self.conv_src = nn.Conv2d(in_channels, head_dim, kernel_size=1) 
        self.conv_dst = nn.Conv2d(in_channels, head_dim, kernel_size=1) 
        self.conv_arr = nn.Conv2d(in_channels, head_dim, kernel_size=1) 
        
        # Deep Fusion MLP (No BN/LN)
        self.move_fusion = nn.Sequential(
            nn.Linear(head_dim * 2, head_dim * 2),
            nn.ReLU(),
            nn.Linear(head_dim * 2, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, head_dim) 
        )
        self.head_dim = head_dim
        
        # === Explicit Kaiming Initialization ===
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, specific_src=None, specific_dst=None):
        B, C, H, W = x.size()
        N = self.N
        head_dim = self.head_dim
        
        f_src = self.conv_src(x).view(B, -1, N) 
        f_dst = self.conv_dst(x).view(B, -1, N)
        f_arr = self.conv_arr(x).view(B, -1, N)
        
        q_src = f_src.permute(0, 2, 1) # (B, N, 128)
        k_dst = f_dst                  # (B, 128, N)
        
        move_logits = torch.bmm(q_src, k_dst) / (head_dim ** 0.5) 
        k_arr = f_arr 

        if specific_src is not None and specific_dst is not None:
            # [Training Mode] Sparse Computation
            batch_indices = torch.arange(B, device=x.device)
            v_src = q_src[batch_indices, specific_src].unsqueeze(1) 
            v_dst = f_dst.permute(0, 2, 1)[batch_indices, specific_dst].unsqueeze(1)
            pair_features = torch.cat([v_src, v_dst], dim=-1) 
            arrow_query = self.move_fusion(pair_features)     
            arrow_logits = torch.bmm(arrow_query, k_arr) / (head_dim ** 0.5)
        else:
            # [Inference Mode] Full Expansion
            v_src = q_src.unsqueeze(2).expand(B, N, N, head_dim) 
            v_dst = f_dst.permute(0, 2, 1).unsqueeze(1).expand(B, N, N, head_dim) 
            pair_features = torch.cat([v_src, v_dst], dim=-1)
            pair_features_flat = pair_features.reshape(B, -1, pair_features.size(-1))
            arrow_queries = self.move_fusion(pair_features_flat)
            arrow_logits = torch.bmm(arrow_queries, k_arr) / (head_dim ** 0.5)
        
        return move_logits, arrow_logits

# -----------------------------------------------------------------------------
# Part 2: Main Network
# -----------------------------------------------------------------------------

class AmazonsNNet(nn.Module):
    def __init__(self, game, args):
        super(AmazonsNNet, self).__init__()
        self.board_x, self.board_y = game.getBoardSize()
        self.args = args
        self.num_res_blocks = getattr(args, 'num_res_blocks', 20) 
        self.num_channels = args.num_channels 
        self.input_channels = getattr(args, 'input_channels', 7)

        self.conv1 = nn.Conv2d(self.input_channels, self.num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.num_channels)
        self.res_layers = nn.ModuleList([ResBlock(self.num_channels) for _ in range(self.num_res_blocks)])
        self.global_attn = SelfAttentionBlock(self.num_channels)
        
        self.policy_head = DynamicFeatureUpdateHead(self.num_channels, self.board_x)
        
        self.val_conv = nn.Conv2d(self.num_channels, 32, kernel_size=1)
        self.val_bn = nn.BatchNorm2d(32)
        self.val_fc1 = nn.Linear(32 * self.board_x * self.board_y, 256)
        self.val_fc2 = nn.Linear(256, 1)

    # === [Optimization Interface] Feature Extraction ===
    def extract_features(self, s):
        """
        Run Backbone only. Returns feature map (B, C, H, W).
        Used to cache features for micro-batch training.
        """
        s = F.relu(self.bn1(self.conv1(s)))
        for layer in self.res_layers:
            s = layer(s)
        s = self.global_attn(s)
        return s

    # === [Optimization Interface] Heads Only ===
    def forward_heads(self, s_features, specific_src, specific_dst, batch_idxs):
        """
        Run Heads using pre-computed features.
        s_features: (Batch_Size, C, H, W) - The global batch features
        batch_idxs: Indices to slice from s_features for the current micro-batch
        """
        # 1. Expand Features for Micro-Batch
        s_expanded = s_features[batch_idxs]
        
        # 2. Value Head
        v = F.relu(self.val_bn(self.val_conv(s_expanded)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.val_fc1(v))
        v_out = torch.tanh(self.val_fc2(v))

        # 3. Policy Head
        move_logits, arrow_logits = self.policy_head(s_expanded, specific_src, specific_dst)
        
        # 4. Format Output
        K = s_expanded.size(0)
        move_logits_flat = move_logits.view(K, -1)
        log_p_move = F.log_softmax(move_logits_flat, dim=1) 
        log_p_arrow = F.log_softmax(arrow_logits, dim=-1)

        return log_p_move, log_p_arrow, v_out

    # === [Standard Interface] Full Forward ===
    def forward(self, s, specific_src=None, specific_dst=None, batch_idxs=None):
        """ Standard forward pass for inference or non-optimized training """
        s = self.extract_features(s)
        
        if batch_idxs is not None:
            # Training mode with batch indices
            return self.forward_heads(s, specific_src, specific_dst, batch_idxs)
        else:
            # Inference mode (usually batch_idxs is None)
            # Use range if not provided to simulate 1-to-1 mapping
            B = s.size(0)
            dummy_idxs = torch.arange(B, device=s.device)
            return self.forward_heads(s, specific_src, specific_dst, dummy_idxs)

    def get_complex_distribution(self, s):
        self.eval() 
        # Note: Botzone compat handled by DynamicFeatureUpdateHead internal structure
        # but here we rely on standard pytorch eval behavior
        with torch.no_grad():
            s = self.extract_features(s)
            move_logits, arrow_logits = self.policy_head(s)
            B = s.size(0)
            p_move = F.softmax(move_logits.view(B, -1), dim=1).view(B, 64, 64)
            p_arrow = F.softmax(arrow_logits, dim=2)
            
            v = F.relu(self.val_bn(self.val_conv(s)))
            v = v.view(-1, 32 * self.board_x * self.board_y)
            v = F.relu(self.val_fc1(v))
            v = torch.tanh(self.val_fc2(v))
        return p_move, p_arrow, v