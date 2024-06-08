from torch import nn
from networks.utils import mean_variance_norm,calc_mean_std

loss =nn.MSELoss()

def calc_content_loss(input, target, norm=False):

    if (norm == False):

        content_loss =loss(input[0], target[0]) +loss(input[1], target[1]) +loss(input[2], target[2]) +loss(input[3], target[3]) + loss(input[4], target[4])

        return content_loss

    else:

        content_loss = loss(mean_variance_norm(input[3]), mean_variance_norm(target[3])) + \
                       loss(mean_variance_norm(input[4]), mean_variance_norm(target[4]))

        return content_loss


def calc_style_loss(input, target):

    style_loss = 0

    for i in range(len(input)):

        input_mean, input_std = calc_mean_std(input[i])
        target_mean, target_std = calc_mean_std(target[i])

        style_loss += (loss(input_mean, target_mean) + loss(input_std, target_std))

    return style_loss

def calc_perceptual_loss(input, target):

    perceptual_loss = 0

    weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    for i in range(len(input)):

        perceptual_loss += loss(input[i], target[i]) * weights[i]

    return perceptual_loss