import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY


class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


def get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    if type(kernel_size) is int:
        use_large_impl = kernel_size > 5
    else:
        assert len(kernel_size) == 2 and kernel_size[0] == kernel_size[1]
        use_large_impl = kernel_size[0] > 5
    has_large_impl = 'LARGE_KERNEL_CONV_IMPL' in os.environ
    # if has_large_impl and in_channels == out_channels and out_channels == groups and use_large_impl and stride == 1 and padding == kernel_size // 2 and dilation == 1:
    #     sys.path.append(os.environ['LARGE_KERNEL_CONV_IMPL'])
    #     #   Please follow the instructions https://github.com/DingXiaoH/RepLKNet-pytorch/blob/main/README.md
    #     #   export LARGE_KERNEL_CONV_IMPL=absolute_path_to_where_you_cloned_the_example (i.e., depthwise_conv2d_implicit_gemm.py)
    #     # TODO more efficient PyTorch implementations of large-kernel convolutions. Pull requests are welcomed.
    #     # Or you may try MegEngine. We have integrated an efficient implementation into MegEngine and it will automatically use it.
    #     return DWConv(in_channels, kernel_size, bias=bias)
    # else:
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=bias)


class LargeKernelConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, groups):
        super(LargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        padding = kernel_size // 2

        self.lkb_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, dilation=1, groups=groups, bias=True)

    def forward(self, inputs):
        out = self.lkb_reparam(inputs)
        return out


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()

        self.out_channels = dim * mlp_ratio

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")

        self.fc1 = nn.Conv2d(dim, dim * mlp_ratio, 1)
        self.pos = nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio,
                             3, padding=1, groups=dim * mlp_ratio)
        self.fc2 = nn.Conv2d(dim * mlp_ratio, dim, 1)
        self.act = nn.GELU()

        scale_sobel_x = torch.randn(size=(dim * mlp_ratio, 1, 1, 1)) * 1e-3
        self.scale_sobel_x = nn.Parameter(torch.FloatTensor(scale_sobel_x))
        sobel_x_bias = torch.randn(dim * mlp_ratio) * 1e-3
        sobel_x_bias = torch.reshape(sobel_x_bias, (dim * mlp_ratio,))
        self.sobel_x_bias = nn.Parameter(torch.FloatTensor(sobel_x_bias))
        self.mask_sobel_x = torch.zeros(
            (dim * mlp_ratio, 1, 3, 3), dtype=torch.float32)
        for i in range(dim * mlp_ratio):
            self.mask_sobel_x[i, 0, 0, 1] = 1.0
            self.mask_sobel_x[i, 0, 1, 0] = 2.0
            self.mask_sobel_x[i, 0, 2, 0] = 1.0
            self.mask_sobel_x[i, 0, 0, 2] = -1.0
            self.mask_sobel_x[i, 0, 1, 2] = -2.0
            self.mask_sobel_x[i, 0, 2, 2] = -1.0
        self.mask_sobel_x = nn.Parameter(
            data=self.mask_sobel_x, requires_grad=False)

        scale_sobel_y = torch.randn(size=(dim * mlp_ratio, 1, 1, 1)) * 1e-3
        self.scale_sobel_y = nn.Parameter(torch.FloatTensor(scale_sobel_y))
        sobel_y_bias = torch.randn(dim * mlp_ratio) * 1e-3
        sobel_y_bias = torch.reshape(sobel_y_bias, (dim * mlp_ratio,))
        self.sobel_y_bias = nn.Parameter(torch.FloatTensor(sobel_y_bias))
        self.mask_sobel_y = torch.zeros(
            (dim * mlp_ratio, 1, 3, 3), dtype=torch.float32)
        for i in range(dim * mlp_ratio):
            self.mask_sobel_y[i, 0, 0, 0] = 1.0
            self.mask_sobel_y[i, 0, 0, 1] = 2.0
            self.mask_sobel_y[i, 0, 0, 2] = 1.0
            self.mask_sobel_y[i, 0, 2, 0] = -1.0
            self.mask_sobel_y[i, 0, 2, 1] = -2.0
            self.mask_sobel_y[i, 0, 2, 2] = -1.0
        self.mask_sobel_y = nn.Parameter(
            data=self.mask_sobel_y, requires_grad=False)

        scale_laplacian = torch.randn(size=(dim * mlp_ratio, 1, 1, 1)) * 1e-3
        self.scale_laplacian = nn.Parameter(torch.FloatTensor(scale_laplacian))
        laplacian_bias = torch.randn(dim * mlp_ratio) * 1e-3
        laplacian_bias = torch.reshape(laplacian_bias, (dim * mlp_ratio,))
        self.laplacian_bias = nn.Parameter(torch.FloatTensor(laplacian_bias))
        self.mask_laplacian = torch.zeros(
            (dim * mlp_ratio, 1, 3, 3), dtype=torch.float32)
        for i in range(dim * mlp_ratio):
            self.mask_laplacian[i, 0, 0, 0] = 1.0
            self.mask_laplacian[i, 0, 1, 0] = 1.0
            self.mask_laplacian[i, 0, 1, 2] = 1.0
            self.mask_laplacian[i, 0, 2, 1] = 1.0
            self.mask_laplacian[i, 0, 1, 1] = -4.0
        self.mask_laplacian = nn.Parameter(
            data=self.mask_laplacian, requires_grad=False)

    def forward(self, x):

        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        # x = x + self.act(self.pos(x))
        out = self.pos(x)
        if not self.merge_kernel:
            out += F.conv2d(input=x, weight=self.scale_sobel_x * self.mask_sobel_x,
                            bias=self.sobel_x_bias, stride=1, padding=1, groups=self.out_channels)
            out += F.conv2d(input=x, weight=self.scale_sobel_y * self.mask_sobel_y,
                            bias=self.sobel_y_bias, stride=1, padding=1, groups=self.out_channels)
            out += F.conv2d(input=x, weight=self.scale_laplacian * self.mask_laplacian,
                            bias=self.laplacian_bias, stride=1, padding=1, groups=self.out_channels)
        x = x + self.act(out)
        x = self.fc2(x)

        return x

    def merge_mlp(self,):
        inf_kernel = self.scale_sobel_x * self.mask_sobel_x + \
            self.scale_sobel_y * self.mask_sobel_y + \
            self.scale_laplacian * self.mask_laplacian + self.pos.weight
        inf_bias = self.sobel_x_bias+self.sobel_y_bias + \
            self.laplacian_bias+self.pos.bias
        self.merge_kernel = True
        self.pos.weight.data = inf_kernel
        self.pos.bias.data = inf_bias
        self.__delattr__('scale_sobel_x')
        self.__delattr__('mask_sobel_x')
        self.__delattr__('sobel_x_bias')
        self.__delattr__('sobel_y_bias')
        self.__delattr__('mask_sobel_y')
        self.__delattr__('scale_sobel_y')
        self.__delattr__('laplacian_bias')
        self.__delattr__('scale_laplacian')
        self.__delattr__('mask_laplacian')


class ConvMod(nn.Module):
    def __init__(self, dim, dw_size):
        super().__init__()

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.a = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            LargeKernelConv(dim, dim, dw_size, stride=1, groups=dim)
        )
        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):

        x = self.norm(x)
        a = self.a(x)
        x = a * self.v(x)
        x = self.proj(x)

        return x


class Block(nn.Module):
    def __init__(self, dim, dw_size, mlp_ratio=4.):
        super().__init__()

        self.attn = ConvMod(dim, dw_size)
        self.mlp = MLP(dim, mlp_ratio)
        layer_scale_init_value = 1e-6
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(x)
        x = x + self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(x)
        return x


class Layer(nn.Module):
    def __init__(self, dim, depth, dw_size, mlp_ratio=4.):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim, dw_size, mlp_ratio) for i in range(depth)
        ])
        self.conv = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.conv(x) + x


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        h, w = self.input_resolution
        flops = h * w * self.num_feat * 3 * 9
        return flops


@ARCH_REGISTRY.register()
class CFSR(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, depths=(6, 6, 6, 6), dw_size=3,
                 mlp_ratio=4., upscale=2, img_range=1., upsampler=''):
        super(CFSR, self).__init__()
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.num_layers = len(depths)

        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = Layer(
                embed_dim, depths[i_layer], dw_size, mlp_ratio)
            self.layers.append(layer)
        self.norm = LayerNorm(embed_dim, eps=1e-6,
                              data_format="channels_first")
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        if self.upsampler == 'pixelshuffledirect':
            self.upsample = UpsampleOneStep(upscale, embed_dim, in_chans)
        else:
            self.conv_last = nn.Conv2d(embed_dim, in_chans, 3, 1, 1)
        # self.merged_inf = False
        self.merge_all()

    def forward_features(self, x):
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return x

    def forward(self, x):
        # if not self.merged_inf:
        #     self.merge_all()

        h, w = x.shape[2:]
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffledirect':
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)
        else:
            x_first = self.conv_first(x)
            res = self.conv_after_body(
                self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean

        return x

    def merge_all(self,):
        named_modules = list(self.named_modules())
        for name, module in named_modules:
            if isinstance(module, MLP):
                print(f"Merging MLP layer: {name}")
                module.merge_mlp()
        self.merged_inf = True
