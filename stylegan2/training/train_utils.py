def get_pose(image):
    pose = None
    return pose  # H x W x 3


def get_appearance(image):
    appearance = None
    return appearance  # H_a x W_a x 3


def encode_pose(pose):
    pnet_model = PNet()
    encoded_pose = pnet_model(pose)
    return encoded_pose  # H/16 x W/16 x 512


def encode_appearance(appearance):
    anet_model = ANet()
    encoded_appearance = pnet_model(pose)
    return encoded_appearance  # 2048


def reconstruct_image(E, z):
    reconstructed_image = None
    GNet = StyleGAN2()
    reconstructed_image = GNet(E, z)
    return reconstructed_image  #HxWx3


def get_generated_pairs(I_s, I_t):
    # extract pose from source and target
    P_s = get_pose(I_s)
    P_t = get_pose(I_t)

    # extract appearance from source and target
    A_s = get_appearance(I_s)
    A_t = get_appearance(I_t)

    # encode pose for source and target
    E_s = encode_pose(P_s)
    E_t = encode_pose(P_t)

    # encode appearance for source and target
    z_s = encode_pose(P_s)
    z_t = encode_pose(P_t)

    # reconstruct I_s as I_s_prime
    I_s_prime = reconstruct_image(E_s, z_s)  #HxWx3

    # reconstruct I_s_t as I_s_t_prime
    I_s_t_prime = reconstruct_image(E_s, z_t)  # TODO E_t is used for supervision

    return I_s_prime, I_s_t_prime

