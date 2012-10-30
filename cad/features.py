
def load_image(image_file, log_parts, log_invparts):
    edges, img = ag.features.bedges_from_image(image_file, k=5, radius=1, minimum_contrast=0.05, contrast_insensitive=True, return_original=True, lastaxis=True)

    ret = ag.features.code_parts(edges, log_parts, log_invparts, threshold)
    ret2 = ret.argmax(axis=2)

