import torch 
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class FunctionalMap_Attention_Structured_Mesh_2D(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0., basis_num=64, H=101, W=31, kernel=3, alpha_init=0, scale_q=False, solve_half=False):
        super().__init__()
        self.dim_head = embed_dim // num_heads
        self.heads = num_heads
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, num_heads, 1, 1]) * 0.5)
        self.freq_keep = basis_num
        self.H = H
        self.W = W
        self.in_project_x = nn.Conv2d(embed_dim, embed_dim, kernel, 1, kernel // 2)
        self.in_project_fx = nn.Conv2d(embed_dim, embed_dim, kernel, 1, kernel // 2)
        self.in_project_basis = nn.Linear(self.dim_head, basis_num)
        for l in [self.in_project_basis]:
            torch.nn.init.orthogonal_(l.weight)
        self.to_q = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_k = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_v = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.sigmoid = nn.Sigmoid()
        self.register_buffer('I_d', torch.eye(self.dim_head).unsqueeze(0))

        self.scale_q = scale_q 
        self.solve_half = solve_half
    
    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).contiguous().permute(0, 3, 1, 2).contiguous()
        
        ### (1) To basis
        fx_mid = self.in_project_fx(x).permute(0, 2, 3, 1).contiguous().reshape(B, N, self.heads, self.dim_head) \
                .permute(0, 2, 1, 3).contiguous()
        x_mid = self.in_project_x(x).permute(0, 2, 3, 1).contiguous().reshape(B, N, self.heads, self.dim_head) \
                .permute(0, 2, 1, 3).contiguous()
        
        slice_weights = self.softmax(
            self.in_project_basis(x_mid) / torch.clamp(self.temperature, min=0.1, max=5))
        slice_norm = slice_weights.sum(2)
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))
        
        ### (2) Functional Attention among basis coefficients - DUAL FORM
        q_slice_token = self.to_q(slice_token)
        if self.scale_q:
            q_slice_token = q_slice_token * self.dim_head ** 0.5
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)

        kH = k_slice_token.transpose(2, 3)                               # [B, H, D, G]
        kTk = torch.matmul(kH, k_slice_token)                            # [B, H, D, D]
        alpha = self.sigmoid(self.alpha)
        reg_dual = (1 - alpha) * kTk + alpha * self.I_d                  # [B, H, D, D]
        Z = torch.linalg.solve(reg_dual, kH)                             # [B, H, D, G]
        if self.solve_half:
            Z = 0.5 * Z
        C = torch.matmul(q_slice_token, Z)                               # [B, H, G, G]
        
        out_slice_token = torch.matmul(C, v_slice_token)
        
        ### (3) Inverse projection
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)
    
class FunctionalMap_Attention_Structured_Mesh_2D_Shared(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0., basis_num=64,  alpha_init=0):
        super().__init__()
        self.dim_head = embed_dim // num_heads
        self.heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.freq_keep = basis_num


        self.to_q = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_k = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_v = nn.Linear(self.dim_head, self.dim_head, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout)
        )

        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.sigmoid = nn.Sigmoid()
        self.register_buffer('I_d', torch.eye(self.dim_head).unsqueeze(0))


    def forward(self, slice_token, slice_weights):
        B, H, G, C = slice_token.shape

        ### (2) Functional Attention among basis coefficients - DUAL FORM
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)

        kH = k_slice_token.transpose(2, 3)                    # [B, H, D, G]
        kTk = torch.matmul(kH, k_slice_token)                 # [B, H, D, D]
        alpha = self.sigmoid(self.alpha)
        reg_dual = (1 - alpha) * kTk + alpha * self.I_d       # [B, H, D, D]
        Z = torch.linalg.solve(reg_dual, kH)                  # [B, H, D, G]
        C = torch.matmul(q_slice_token, Z)                    # [B, H, G, G]
        
        out_slice_token = torch.matmul(C, v_slice_token)

        ### (3) Inverse projection
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        
        return self.to_out(out_x), slice_token, out_slice_token
    
class FunctionalMap_Attention_Irregular_Mesh_Shared(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0., basis_num=64, alpha_init=1e-3):
        super().__init__()
        self.dim_head = embed_dim // num_heads
        self.heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.basis_num = basis_num

        self.to_q = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_k = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_v = nn.Linear(self.dim_head, self.dim_head, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.register_buffer('I_d', torch.eye(self.dim_head).unsqueeze(0))
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.sigmoid = nn.Sigmoid()

    def forward(self, slice_token, slice_weights):
        B, H, G, C = slice_token.shape

        ### (2) Functional Attention among basis coefficients - DUAL FORM
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)

        kH = k_slice_token.transpose(2, 3)                             # [B, H, D, G]
        kTk = torch.matmul(kH, k_slice_token)                          # [B, H, D, D]      
        alpha = self.sigmoid(self.alpha)
        reg_dual = (1 - alpha) * kTk + alpha * self.I_d                # [B, H, D, D]
        Z = torch.linalg.solve(reg_dual, kH)                           # [B, H, D, G]
        C = torch.matmul(q_slice_token, Z)                             # [B, H, G, G]
        
        out_slice_token = torch.matmul(C, v_slice_token)
        
        ### (3) Inverse projection
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x), slice_token, out_slice_token