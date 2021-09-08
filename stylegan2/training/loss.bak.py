# losses
vgg_model = Vgg16()


def perceptual_loss(img1, img2):
    weight = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]  # TODO: to be verified
    weight.reverse()
    img1_features = vgg_model(img1.float())
    img2_features = vgg_model(img2.float())
    loss = nn.L1Loss().cuda()
    weighted_dist = 0
    for i in range(4):
        weighted_dist = weighted_dist + weight[i] * loss(img1_features[i], img2_features[i])
    return weighted_dist


def l1_loss(image1, image2):
    loss = nn.L1Loss()
    return loss(image1, image2)


def reconstruction_loss(image1, image2):
    # equation 3
    loss_rec = l1_loss(image1, image2) + perceptual_loss(image1, image2)
               # + stylegan2loss
    return loss_rec


def patch_loss(image1, image2):
    pass


# L_total is minimised w. r. t. the parameters of
# PNet, ANet and GNet, while maximised w. r. t. D and DPatch
def compute_total_loss(I_s, I_t, I_s_prime, I_s_t_prime):
    # use equation 8 in case of unpaired data
    L_t = reconstruction_loss(I_s_prime, I_s) + lamda_patch * patch_loss(I_s_t_prime, I_t)
    return L_t