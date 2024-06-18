# webknossos_utils/__init__.py

import numpy as np
import webknossos as wk
from webknossos import BoundingBox
from collections import namedtuple

Annotation = namedtuple("Annotation", ["ID", "Dataset", "Name"])
Annotation.__doc__ = """
A namedtuple representing an annotation layer in WebKnossos.

Fields:
    ID (int): The unique identifier of the annotation layer.
    Dataset (str): The identifier of the dataset the annotation layer belongs to.
    Name (str): The name of the annotation layer.
"""

Pixel_size = namedtuple("Pixel_size", ["x", "y", "z", "MAG", "unit"])
Pixel_size.__doc__ = """
A namedtuple representing the pixel size information for WebKnossos data.

Fields:
    x (float): The pixel size along the x-axis.
    y (float): The pixel size along the y-axis.
    z (float): The pixel size along the z-axis.
    MAG (Vector): A Vector object representing the magnification factors.
    unit (str): The unit of measurement for the pixel sizes.
"""


def skibbox2wkbbox(ski_bbox, pSize):
    """
    Convert a bounding box from scikit-image format to WebKnossos format. Assumes at the moment a singe z plane

    Args:
        ski_bbox (dict): A dictionary containing the bounding box coordinates in scikit-image format.
            The dictionary should have keys 'bbox-0', 'bbox-1', 'bbox-2', and 'bbox-3' representing
            the minimum x, minimum y, maximum x, and maximum y coordinates, respectively.
        pSize (namedtuple): A namedtuple containing the 'MAG' field with the pixel size information
            from WebKnossos.

    Returns:
        BoundingBox: A BoundingBox object representing the bounding box in WebKnossos format.

    Raises:
        AssertionError: If ski_bbox is not a dictionary, if the 'bbox-0', 'bbox-1', 'bbox-2', and 'bbox-3'
            keys are not present in ski_bbox, or if pSize is not a namedtuple with the 'MAG' field.

    Example:
        >>> ski_bbox = {'bbox-0': 10, 'bbox-1': 20, 'bbox-2': 30, 'bbox-3': 40}
        >>> pSize = Pixel_size(MAG=Vector(1.0, 2.0, 3.0))
        >>> wk_bbox = skibbox2wkbbox(ski_bbox, pSize)
        >>> print(wk_bbox)
        BoundingBox(corner=array([20., 10.,  0.]), size=array([20., 40.,  3.]))
    """
    assert isinstance(ski_bbox, dict), "ski_bbox must be a dict"
    assert all(key in ski_bbox for key in ('bbox-0', 'bbox-1', 'bbox-2', 'bbox-3')), "bbox is ill defined"
    assert isinstance(pSize, Pixel_size), "pSize is a namedtuple and must contain a MAG from webknossos"
    assert "MAG" in pSize._fields, "issues with pixel size info, mag missing"
    MAG = pSize.MAG
    corner = np.array([ski_bbox['bbox-1'], ski_bbox['bbox-0'], 0]) * np.array([MAG.x, MAG.y, MAG.z])
    size = np.array([ski_bbox['bbox-3'] - ski_bbox['bbox-1'], ski_bbox['bbox-2'] - ski_bbox['bbox-0'], 1]) * np.array([MAG.x, MAG.y, MAG.z])
    wk_bbox = BoundingBox(corner, size).align_with_mag(MAG, True)

    return wk_bbox

def load_3d_lbl(lbl_layer, wk_bbox, wk_mag, delta, AUTH_TOKEN, WK_TIMEOUT="3600"):
    """
    Load a 3D labeled volume from a Webknossos dataset, with an optional buffer around the bounding box.

    Args:
        lbl_layer (webknossos.dataset.layer.SegmentationLayer): The segmentation layer containing the labeled data.
        wk_bbox (webknossos.geometry.bounding_box.BoundingBox): The bounding box defining the region of interest.
        wk_mag (wk.geometry.mag.Mag): The MAG from webknossos to use
        delta (int): The size of the buffer (in voxels) to add around the bounding box.
        AUTH_TOKEN (str): The authentication token required for accessing the Webknossos dataset.
        WK_TIMEOUT (str, optional): The timeout value (in seconds) for the Webknossos server connection. Default is "3600".

    Returns:
        numpy.ndarray: A 3D NumPy array containing the labeled data within the specified bounding box and buffer region.

    Raises:
        TypeError: If `lbl_layer` is not a `webknossos.dataset.layer.SegmentationLayer` instance.
        TypeError: If `wk_bbox` is not a `webknossos.geometry.bounding_box.BoundingBox` instance.
        TypeError: If `wk_mag` is not a `wk.geometry.mag.Mag` instance.
        TypeError: If `delta` is not an integer.
        TypeError: If `AUTH_TOKEN` is not a string.
        TypeError: If `WK_TIMEOUT` is not a string.
        ValueError: If `delta` is negative.
    """
    # Input testing
    if not isinstance(lbl_layer, wk.SegmentationLayer):
        raise TypeError("lbl_layer must be an instance of webknossos.dataset.layer.SegmentationLayer")
    if not isinstance(wk_bbox, wk.geometry.bounding_box.BoundingBox):
        raise TypeError("wk_bbox must be an instance of webknossos.geometry.bounding_box.BoundingBox")
    if not isinstance(wk_mag, wk.geometry.mag.Mag):
        raise TypeError("wk_mag must be an instance of wk.geometry.mag.Mag")
    if not isinstance(delta, int):
        raise TypeError("delta must be an integer")
    if not isinstance(AUTH_TOKEN, str):
        raise TypeError("AUTH_TOKEN must be a string")
    if not isinstance(WK_TIMEOUT, str):
        raise TypeError("WK_TIMEOUT must be a string")
    if delta < 0:
        raise ValueError("delta must be non-negative")
    
    z_size = lbl_layer.bounding_box.size.to_np()[2]

    size_factor_x = 1
    size_factor_y = 1
    new_topleft = wk_bbox.topleft
    if wk_bbox.topleft[0] - delta > 0:
        new_topleft = new_topleft - (delta, 0, 0)
        size_factor_x = 2
    if wk_bbox.topleft[1] - delta > 0:
        new_topleft = new_topleft - (0, delta, 0)
        size_factor_y = 2

    new_box_size = wk_bbox.size + (size_factor_x*delta, size_factor_y*delta, z_size-1)

    new_bbox = BoundingBox(new_topleft, new_box_size).align_with_mag(wk_mag, True)


    with wk.webknossos_context(token=AUTH_TOKEN, timeout=WK_TIMEOUT):
        if wk_mag.__str__ == '1':
            lbl_data = lbl_layer.get_finest_mag().read(absolute_offset=new_bbox.topleft,
                                                            size=new_bbox.size)
        else:
            lbl_data = lbl_layer.get_mag(wk_mag).read(absolute_offset=new_bbox.topleft,
                                                            size=new_bbox.size)
    
    return lbl_data