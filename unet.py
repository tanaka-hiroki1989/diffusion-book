import torch.nn as nn
import torch
import math
import torch.nn.functional as F

class TimeEmbedding(nn.Module):
    """
    どの時間ステップの画像なのかを区別するため,
    Positional Encodingをデータに施す.
    """
    def __init__(
        self, 
        T: int, 
        d_model: int, 
        temb_dim: int
    ):
        """
        T: int, 
        d_model: int, 
        temb_dim: int
        """
        super().__init__()
        assert d_model % 2 == 0
        position = torch.arange(T).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(T, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe
        self.emb = nn.Embedding.from_pretrained(
            pe, freeze=True)

        self.dense0 = nn.Linear(d_model, temb_dim)
        self.dense1 = nn.Linear(temb_dim, temb_dim)

    def forward(self, t):
        temb = self.emb(t)
        temb = F.silu(self.dense0(temb))
        temb = self.dense1(temb)
        return temb
        

class ResidualBlock(nn.Module):
    """
    畳み込みを行う.
    Group Normalizationを使用.
    活性化関数にはSwish関数を使用.
    入力画像を畳み込んだ後, Positional Encodingする.
    Attentionを組み込んである.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        temb_dim,
        shortcut=True,
        attention=False,
        dropout=0.0
    ):
        super().__init__()
        # Group Normalization
        self.norm1 = nn.GroupNorm(
            num_groups=32, 
            num_channels=in_channels)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=3,
            stride=1,
            padding=1)
        
        self.temb_proj = nn.Linear(
            in_features=temb_dim,
            out_features=out_channels
        )
        # Group Normalization
        self.norm2 = nn.GroupNorm(
            num_groups=32, 
            num_channels=out_channels)
        self.dropout = nn.Dropout2d(p=dropout)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=3,
            stride=1,
            padding=1)

        if in_channels != out_channels:
            if shortcut:
                self.shortcut = nn.Conv2d(
                    in_channels=in_channels, 
                    out_channels=out_channels, 
                    kernel_size=3,
                    stride=1,
                    padding=1)
            else:
                raise NotImplementedError()

        if attention:
            self.attn = Attention(d_model=out_channels)
        else:
            self.attn = nn.Identity()
        
    def forward(self, x, temb):
        """
        x: 入力画像
        temb:
        """
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        
        # add in timestep embedding
        h = h + self.temb_proj(F.silu(temb))[:, :, None, None]
        h = F.silu(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        
        if hasattr(self, 'shortcut'):
            x = self.shortcut(x)
        assert x.shape == h.shape, (x.shape, h.shape)

        h = h + x
        h = self.attn(h)
        return h


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, with_conv=True):
        super().__init__()
        """
        width_convで畳み込みをかけるかどうかを指定.
        """
        self.with_conv = with_conv
        self.conv = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=3,
            stride=1,
            padding=1)

    def forward(self, x, *dummy):
        B, C, H, W = x.shape
        # 最近傍補間により拡大
        x = F.interpolate(
            x, 
            scale_factor=2, 
            mode='nearest')
        if self.with_conv:
            x = self.conv(x)
        assert list(x.shape) == [B, C, H * 2, W * 2]
        return x


class DownSample(nn.Module):
    def __init__(self,  in_channels, out_channels, with_conv=True):
        """
        with_conv により, 畳み込みかプーリングかを指定.
        """
        super().__init__()
        self.with_conv = with_conv
        self.conv = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=3,
            stride=2,
            padding=1)
    
    def forward(self, x, *dummy):
        B, C, H, W = x.shape
        if self.with_conv:
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2)
        assert list(x.shape) == [B, C, H // 2, W // 2], (x.shape, [B, C, H // 2, W // 2])
        return x


class Attention(nn.Module):
    def __init__(
        self, 
        d_model
    ):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=d_model)

        bias = True
        self.q = nn.Conv2d(d_model, d_model, kernel_size=1, stride=1, padding=0, bias=bias)
        self.k = nn.Conv2d(d_model, d_model, kernel_size=1, stride=1, padding=0, bias=bias)
        self.v = nn.Conv2d(d_model, d_model, kernel_size=1, stride=1, padding=0, bias=bias)
        self.proj_out = nn.Conv2d(d_model, d_model, kernel_size=1, stride=1, padding=0, bias=bias)


    def forward(self, x, *dummy):
        B, C, H, W = x.shape
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj_out(h)
        h = h + x
        return h


class Unet(nn.Module):
    def __init__(
        self,
        T: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        temb_dim: int,
        num_res_blocks,
        ch_mult=(1, 2, 4, 8),
        dropout=0.0,
        attn_block_indices=(1,),
        resamp_with_conv=True
    ):
        """
        Params
        ---
        T: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        temb_dim: int, Positional Encodingの特徴次元数
        num_res_blocks, 各解像度のレイヤーのResidualBlockの数
        ch_mult=(1, 2, 4, 8),
        dropout=0.0,
        attn_block_indices=(1,), Attentionを使用する解像度のレイヤーのインデックス
        resamp_with_conv=True

        - hidden channelsとch_multの積が, 各解像度のレイヤーのチャンネル次元になる.
        """
        super().__init__()
        num_resolutions = len(ch_mult)

        self.time_embedding = TimeEmbedding(
            T=T, 
            d_model=hidden_channels,
            temb_dim=temb_dim
        )

        # Downsampling
        self.conv_in = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=hidden_channels, 
            kernel_size=3,
            stride=1,
            padding=1)
        self.down_blocks = nn.ModuleList()
        channels_list = [hidden_channels]
        out_channels_ = hidden_channels
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                in_channels_ = out_channels_
                out_channels_ = hidden_channels * ch_mult[i_level]
                self.down_blocks.append(
                    ResidualBlock(
                        in_channels=in_channels_, 
                        out_channels=out_channels_, 
                        temb_dim=temb_dim,
                        attention=i_level in attn_block_indices,
                        dropout=dropout
                    )
                )
                channels_list.append(out_channels_)
            if i_level != num_resolutions - 1:
                self.down_blocks.append(
                    DownSample(
                        in_channels=out_channels_,
                        out_channels=out_channels_,
                        with_conv=resamp_with_conv
                    )
                )
                channels_list.append(out_channels_)

        # Middle
        self.mid_blocks = nn.ModuleList([
            ResidualBlock(
                in_channels=out_channels_, 
                out_channels=out_channels_, 
                temb_dim=temb_dim,
                attention=True,
                dropout=dropout
            ),
            ResidualBlock(
                in_channels=out_channels_, 
                out_channels=out_channels_, 
                temb_dim=temb_dim,
                attention=False,
                dropout=dropout
            )
        ])

        # Upsampling
        self.up_blocks = nn.ModuleList()
        for i_level in reversed(range(num_resolutions)):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks + 1):
                in_channels_ = channels_list.pop() + out_channels_
                out_channels_ = hidden_channels * ch_mult[i_level]
                self.up_blocks.append(
                    ResidualBlock(
                        in_channels=in_channels_, 
                        out_channels=out_channels_,
                        temb_dim=temb_dim,
                        attention=i_level in attn_block_indices,
                        dropout=dropout
                    )
                )
            if i_level != 0:
                self.up_blocks.append(
                    UpSample(
                        in_channels=out_channels_,
                        out_channels=out_channels_,
                        with_conv=resamp_with_conv
                    )
                )
        assert len(channels_list)==0, channels_list

        self.conv_out = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=out_channels_),
            nn.SiLU(),
            nn.Conv2d(
                in_channels=out_channels_, 
                out_channels=out_channels, 
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

    def forward(self, x, t):
        temb = self.time_embedding(t)
        h = self.conv_in(x)
        hs = [h]
        for layer in self.down_blocks:
            h = layer(h, temb)
            hs.append(h)
        for layer in self.mid_blocks:
            h = layer(h, temb)
        for layer in self.up_blocks:
            if isinstance(layer, ResidualBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)
        h = self.conv_out(h)

        assert len(hs) == 0
        return h


def integration_test():
    batch_size = 2
    model = Unet(
        T=T, 
        in_channels=3, 
        hidden_channels=32,
        out_channels=3, 
        temb_dim=32 * 4,
        ch_mult=(1, 2, 3, 4),
        num_res_blocks=2, 
        dropout=0.1,
        attn_block_indices=(2,),
        resamp_with_conv=True
    )
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(1000, (batch_size, ))
    y = model(x, t)