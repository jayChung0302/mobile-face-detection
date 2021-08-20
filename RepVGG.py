import torch
import torch.nn as nn
import numpy as np
import copy

from torch.nn.functional import pad
from torch.profiler.profiler import ProfilerAction
from torch.serialization import DEFAULT_PROTOCOL
from SE_block import SEBlock

__all__ = ['RepVGG_A0', 'RepVGG_A1', 'RepVGG_A2', 'RepVGG_B0', 'RepVGG_B1', 'RepVGG_B1g2',\
    'RepVGG_B1g4', 'RepVGG_B2', 'RepVGG_B2g2', 'RepVGG_B2g4', 'RepVGG_B3', 'RepVGG_B3g2',\
        'RepVGG_B3g4', 'RepVGG_D2se']

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, \
        out_channels=out_channels, kernel_size=kernel_size, stride=stride, \
            padding=padding, groups=groups))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,\
        stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.se = SEBlock(out_channels, out_channels//16) if use_se else nn.Identity()
        self.nonlinearity = nn.ReLU()
        
        padding_l1 = padding - kernel_size // 2

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=\
                out_channels, kernel_size=kernel_size, stride=stride, padding=padding,\
                     dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) \
                if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels,\
                 kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, \
                kernel_size=1, stride=stride, padding=padding_l1, groups=groups)
            print(f'RepVGG Block, identity= {self.rbr_identity}')
    
    def forward(self, input):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam))
        
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(input)
        
        return self.nonlinearity(self.se(self.rbr_dense(input) + self.rbr_1x1(input) + id_out))
    
    def get_custom_L2(self):
        K3 = self.rbr_dense.conv.weight
        K1 = self.rbr_1x1.conv.weight
        t3 = (self.rbr_dense.bn.weight / ((self.rbr_dense.bn.running_var + \
            self.rbr_dense.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()
        t1 = (self.rbr_1x1.bn.weight / ((self.rbr_1x1.bn.running_var + \
            self.rbr_1x1.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()
        
        l2_loss_circle = (K3 ** 2).sum() - (K3[:, :, 1:2, 1:2] ** 2).sum()
        eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1
        l2_loss_eq_kernel = (eq_kernel ** 2 / (t3 ** 2 + t1 ** 2)).sum()
        return l2_loss_circle + l2_loss_eq_kernel
                
    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid + bias3x3 + bias1x1 + biasid
    
    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])


    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return

        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels,\
            out_channels=self.rbr_dense.conv.out_channels, kernel_size=self.rbr_dense.kernel_size,\
                padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation,\
                    groups=self.rbr_dense.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for param in self.parameters():
            param.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self,'rbr_identity'):
            self.__delattr__('rbr_identity')

class RepVGG(nn.Module):
    def __init__(self, num_blocks, num_classes=1000, width_multiplier=None, \
        override_groups_map=None, deploy=False, use_se=False):
        super(RepVGG, self).__init__()

        assert len(width_multiplier) == 4

        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()
        self.use_se = use_se

        assert 0 not in self.override_groups_map
        

        self.in_planes = min(64, int(64 * width_multiplier[0]))
        self.stage0 = RepVGGBlock(in_channels=3, out_channels=self.in_planes, \
            kernel_size=3, stride=2, padding=1, deploy=self.deploy, use_se=self.use_se)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0],\
             stride=2)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1],\
             stride=2)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2],\
             stride=2)
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3],\
             stride=2)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes, \
                kernel_size=3, stride=stride, padding=1, groups=cur_groups, deploy=self.deploy,\
                    use_se=self.use_se))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

optional_groupwise_layers = [2 * i for i in range(1, 14)]
g2_map = {l: 2 for l in optional_groupwise_layers}
g4_map = {l: 4 for l in optional_groupwise_layers}

def RepVGG_A0(deploy=False, num_classes=1000):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=num_classes,\
        width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, \
            deploy=deploy)

def RepVGG_A1(deploy=False, num_classes=1000):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=num_classes,\
        width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, \
            deploy=deploy)

def RepVGG_A2(deploy=False, num_classes=1000):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=num_classes,\
        width_multiplier=[1.5, 1.5, 1.5, 2.75], override_groups_map=None, \
            deploy=deploy)

def RepVGG_B0(deploy=False, num_classes=1000):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,\
        width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, \
            deploy=deploy)

def RepVGG_B1(deploy=False, num_classes=1000):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,\
        width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, \
            deploy=deploy)

def RepVGG_B2g2(deploy=False, num_classes=1000):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g2_map, deploy=deploy)

def RepVGG_B2g4(deploy=False, num_classes=1000):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g4_map, deploy=deploy)


def RepVGG_B3(deploy=False, num_classes=1000):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=None, deploy=deploy)

def RepVGG_B3g2(deploy=False, num_classes=1000):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=g2_map, deploy=deploy)

def RepVGG_B3g4(deploy=False, num_classes=1000):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=g4_map, deploy=deploy)

def RepVGG_D2se(deploy=False, num_classes=1000):
    return RepVGG(num_blocks=[8, 14, 24, 1], num_classes=num_classes,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None, deploy=deploy, use_se=True)

def repvgg_model_convert(net: torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        net = copy.deepcopy(net)
    
    for module in net.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    
    if save_path is not None:
        torch.save(net.state_dict(), save_path)
    return net

if __name__ == '__main__':
    from torch.profiler import profile, ProfilerActivity, record_function
    x = torch.randn(1, 3, 224, 224)
    net = RepVGG_A0()
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True, \
        record_shapes=True) as prof:
        net(x)
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
    '''measured in colab CPU. (vs mobilenet_v3: 82.496ms)
    --------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                            Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls  
    --------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                     aten::empty         0.98%       1.664ms         0.98%       1.664ms       4.823us      27.47 Mb      27.47 Mb           345  
                       aten::add         2.21%       3.759ms         2.26%       3.850ms      36.667us      13.69 Mb      13.69 Mb           105  
                 aten::clamp_min         0.68%       1.165ms         1.30%       2.210ms      50.227us      13.69 Mb       6.84 Mb            44  
                   aten::resize_         0.11%     180.000us         0.11%     180.000us      10.588us       3.30 Mb       3.30 Mb            17  
                     aten::addmm         1.90%       3.230ms         1.90%       3.240ms       3.240ms       3.91 Kb       3.91 Kb             1  
             aten::empty_strided         0.21%     366.000us         0.21%     366.000us       5.463us         268 b         268 b            67  
                    aten::conv2d         0.15%     262.000us        48.58%      82.745ms       1.881ms      13.69 Mb           0 b            44  
               aten::convolution         0.16%     264.000us        48.43%      82.483ms       1.875ms      13.69 Mb           0 b            44  
              aten::_convolution         0.31%     535.000us        48.27%      82.219ms       1.869ms      13.69 Mb           0 b            44  
        aten::mkldnn_convolution        35.97%      61.268ms        36.24%      61.725ms       2.286ms      10.38 Mb           0 b            27  
    --------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
    Self CPU time total: 170.315ms
    '''
    from torchinfo import summary
    summary(net, input_size=(1, 3, 224, 224))
    '''
    ==========================================================================================
    Total params: 9,117,960
    Trainable params: 9,117,960
    Non-trainable params: 0
    Total mult-adds (G): 1.52
    ==========================================================================================
    Input size (MB): 0.60
    Forward/backward pass size (MB): 64.33
    Params size (MB): 36.47
    Estimated Total Size (MB): 101.41
    ==========================================================================================

    '''
