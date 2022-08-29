from torch.nn import functional as F

def pad(img, fac, mode='replicate'):
    """
    pad img such that height and width are divisible by fac
    """
    _, _, h, w = img.shape
    padH = fac - (h % fac)
    padW = fac - (w % fac)
    if padH == fac and padW == fac:
        # return img, ft.identity
        return img, (0, 0, 0, 0)
    if padH == fac:
        padTop = 0
        padBottom = 0
    else:
        padTop = padH // 2
        padBottom = padH - padTop
    if padW == fac:
        padLeft = 0
        padRight = 0
    else:
        padLeft = padW // 2
        padRight = padW - padLeft
    assert (padTop + padBottom + h) % fac == 0
    assert (padLeft + padRight + w) % fac == 0

    padding_tuple = (padLeft, padRight, padTop, padBottom)

    return F.pad(img, padding_tuple, mode), padding_tuple


def undo_pad(img, padLeft, padRight, padTop, padBottom, target_shape=None):
    # the 'or None' makes sure that we don't get 0:0
    img_out = img[..., padTop:(-padBottom or None), padLeft:(-padRight or None)]
    if target_shape:
        h, w = target_shape
        assert img_out.shape[-2:] == (h, w), (img_out.shape[-2:], (h, w), img_out.shape,
                                              (padLeft, padRight, padTop, padBottom))
    return img_out
