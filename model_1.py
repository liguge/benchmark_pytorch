import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 配置参数
num_classes = 3
signal_size = [5120, 1]
patch_size = [512, 1]
conv_size = 512
rate = 2
stride = int(conv_size / rate)
num_patches = int((signal_size[0] - patch_size[0]) / stride) + 1
print(num_patches)
projection_dim = 64
num_heads = 8
transformer_units = [
    projection_dim,
    projection_dim*2,
    projection_dim,
]
k = 10
weight_layers = 1
transformer_layers = 5


def mlp(hidden_units, dropout_rate=0.1):
    layers = []
    for i in range(len(hidden_units) - 1):
        in_dim = hidden_units[i]
        out_dim = hidden_units[i + 1]
        layers.append(nn.Linear(in_features=in_dim, out_features=out_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout_rate))
    return nn.Sequential(*layers)
    # layers = []
    # for units in hidden_units:
    #     layers.append(nn.Linear(units, units))
    #     layers.append(nn.ReLU())
    #     layers.append(nn.Dropout(dropout_rate))
    # return nn.Sequential(*layers)


class ConcatClassTokenAddPosEmbed(nn.Module):
    def __init__(self, embed_dim=64, num_patches=9, name=None):
        super(ConcatClassTokenAddPosEmbed, self).__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, embed_dim, 1))
        self.pos_embed = nn.Parameter(torch.randn(1, embed_dim, num_patches + 1) * 0.02)

    def forward(self, inputs):
        batch_size = inputs.size(0)

        # 扩展cls_token到批次大小
        cls_token = self.cls_token.expand(batch_size, -1, -1)

        # 连接cls_token和输入
        x = torch.cat([cls_token, inputs], dim=-1)

        # 添加位置嵌入
        x = x + self.pos_embed

        return x


class MPI(nn.Module):
    def __init__(self, filters=projection_dim, kernel_size=3, strides=1, conv_padding='same'):
        super(MPI, self).__init__()
        if conv_padding == 'same':
            padding = (kernel_size - 1) // 2
        else:
            padding = 0

        self.conv1 = nn.Conv1d(in_channels=filters, out_channels=filters,
                               kernel_size=kernel_size, stride=strides,
                               padding='same', dilation=1)
        self.conv2 = nn.Conv1d(in_channels=filters, out_channels=filters,
                               kernel_size=kernel_size, stride=strides,
                               padding='same', dilation=2)
        self.conv3 = nn.Conv1d(in_channels=filters, out_channels=filters,
                               kernel_size=kernel_size, stride=strides,
                               padding='same', dilation=3)

    def forward(self, x):
        # x shape: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)

        x1 = F.relu(self.conv1(x).unsqueeze(-1))
        x2 = F.relu(self.conv2(x).unsqueeze(-1))
        x3 = F.relu(self.conv3(x).unsqueeze(-1))

        # 拼接并在最后一个维度上求平均
        combined = torch.stack([x1, x2, x3], dim=-1)
        combined = torch.mean(combined, dim=-1).squeeze(-1)

        # 转换回原始形状
        combined = combined.transpose(1, 2)

        return combined


class PatchEncoderCUM(nn.Module):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoderCUM, self).__init__()

        self.projection_dim = projection_dim

        # 卷积层
        self.cum = nn.Conv1d(in_channels=1, out_channels=projection_dim,
                             kernel_size=conv_size, stride=stride,
                             padding=0, dilation=1)

        # 位置嵌入
        self.position_embedding = nn.Embedding(projection_dim, num_patches)

    def forward(self, patch):
        encoded = F.gelu(self.cum(patch))
        positions = torch.arange(0, self.projection_dim, dtype=torch.long, device=patch.device)
        pos_embed = self.position_embedding(positions)
        encoded = encoded + pos_embed.unsqueeze(0)
        return encoded


class CTL(nn.Module):
    def __init__(self, k=10, embed_dim=64, num_patches=19, num_heads=8, weight_layers=1):
        super(CTL, self).__init__()
        self.k = k
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.num_heads = num_heads
        self.weight_layers = weight_layers
        self.concat_class_token = ConcatClassTokenAddPosEmbed(embed_dim=embed_dim, num_patches=num_patches)
        self.layernorm1 = nn.LayerNorm(num_patches+1, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(num_patches+1, eps=1e-6)
        self.attention_layers = nn.MultiheadAttention(embed_dim=projection_dim, num_heads=num_heads, dropout=0.1, batch_first=True)
        self.tanh = nn.Tanh()

    def forward(self, inputs):
        crop = self.concat_class_token(inputs)
        temporary = crop
        for i in range(self.weight_layers):
            x1 = self.layernorm1(crop)
            x1 = x1.permute(0, 2, 1)
            attention_output, _ = self.attention_layers(x1, x1, x1)  # q,k,v均为x1（自注意力）
            x2 = attention_output.permute(0, 2, 1) + crop
            x3 = self.layernorm2(x2)
            crop = self.tanh(x3)
        weights = torch.sum(crop, dim=1)
        # top_values, top_indices = torch.topk(weights, k=self.k, dim=1, sorted=True)  # sorted=True对应原代码默认排序
        # selected_values = torch.gather(temporary, dim=1, index=top_indices.unsqueeze(1))
        top_values, top_indices = torch.topk(weights, k=k, dim=1, sorted=True)  # 对应tf.math.top_k
        batch_size = temporary.shape[0]  # 获取批量大小
        feature_dim = temporary.shape[1]  # 获取每个patch的特征维度
        top_indices_expanded = top_indices.unsqueeze(-1).expand(batch_size, k, feature_dim)
        selected_values = torch.gather(temporary, dim=1, index=top_indices_expanded.permute(0, 2, 1))
        return selected_values, crop


# CFSPT函数转换（PyTorch版本，对应原模型构建）
class CFSPT(nn.Module):
    def __init__(self):
        super(CFSPT, self).__init__()
        self.patch_encoder = PatchEncoderCUM(num_patches=num_patches, projection_dim=projection_dim)
        self.mpi = MPI(filters=projection_dim, kernel_size=3, strides=1, conv_padding='same')
        self.dropout = nn.Dropout(p=0.1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # 对应GlobalAveragePooling1D
        self.fc = nn.Linear(projection_dim, num_classes)  # 最终分类层
        self.clt = CTL(k)
        self.mlp  = mlp(transformer_units, dropout_rate=0.01)
        # 定义Transformer编码块（堆叠transformer_layers层）
        self.transformer_blocks = nn.ModuleList([])
        for _ in range(transformer_layers):
            # 自注意力层
            attn_layer = nn.MultiheadAttention(embed_dim=projection_dim, num_heads=num_heads, dropout=0.1,
                                               batch_first=True)
            # LayerNorm层（PyTorch中单独定义，方便残差连接）
            ln1 = nn.LayerNorm(projection_dim, eps=1e-6)
            ln2 = nn.LayerNorm(projection_dim, eps=1e-6)
            # MLP层（封装为nn.Sequential，或使用上述mlp函数）
            self.transformer_blocks.append(nn.ModuleList([attn_layer, ln1, ln2]))

    def forward(self, inputs):
        # 对应原代码PatchEncoderCUM
        add_pos_embed = self.patch_encoder(inputs)
        # 对应原代码MPI
        # 注意：MPI输入需适配PyTorch Conv1d格式，先permute再还原
        add_pos_embed_perm = add_pos_embed.permute(0, 2, 1)
        add_pos_embed_mpi = self.mpi(add_pos_embed_perm)
        add_pos_embed = add_pos_embed_mpi.permute(0, 2, 1)

        # 对应原代码CTL
        add_pos_embed, ALL = self.clt(add_pos_embed)
        add_pos_embed = add_pos_embed.permute(0, 2, 1)
        # Transformer编码块循环
        for attn_layer, ln1, ln2 in self.transformer_blocks:
            # 第一层LayerNorm + 自注意力 + 残差连接
            x1 = ln1(add_pos_embed)
            attention_output, _ = attn_layer(x1, x1, x1)
            x2 = attention_output + add_pos_embed  # 残差连接

            # 第二层LayerNorm + MLP + 残差连接
            x3 = ln2(x2)
            x3 = self.mlp(x3)
            add_pos_embed = x3 + x2  # 残差连接

        # 对应原代码Dropout + GlobalAveragePooling1D
        features = self.dropout(add_pos_embed)
        # GlobalAveragePooling1D：先permute适配，再池化，最后还原维度
        features_perm = features.permute(0, 2, 1)
        features = self.global_avg_pool(features_perm).squeeze(-1)

        # 对应原代码Dense + softmax
        logits = self.fc(features)
        logits = F.softmax(logits, dim=-1)

        return logits


# 模型初始化与使用示例
if __name__ == "__main__":
    # 定义输入形状（对应原代码input_shape）
    input_shape = torch.rand(1, 1, 5120)
    # 初始化模型
    model = CFSPT()
    output = model(input_shape)
    # 打印输出形状
    print(f"模型输出形状: {output.shape}")  # 输出：torch.Size([2, num_classes])