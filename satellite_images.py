# -*- coding: utf-8 -*-
from typing import Tuple
import io
import math
import requests
import PIL
import numpy as np

Coordinate = Tuple[float, float]
TileIndex = Tuple[int, int]


def GPS_to_mercator(lat: float, lon: float) -> Coordinate:
    """
    Converts a WGS84 GPS coordinate (lat, lon) to mercator coordinates (x, y)

    Parameters
    ----------
    lat : float
        latitude (between -90° and 90°)
    lon : float
        longitude (between -180° and 180°)

    Returns
    -------
    tuple of floats :
        the (x, y) coordinate in mercator projection.
        x is between 0 and 1
        y is bewteen 0 and 1 for latitudes between -85.05° and 85.05°
    """
    assert (-90. <= lat <= 90.), "latitude must be between -90. and 90."
    assert (-180. <= lon <= 180.), "longitude must be between -180. and 180."
    x = (lon+180)/360
    y = 0.5 + math.log(math.tan(math.radians((lat+90)/2))) / (2*math.pi)
    return x, y


def mercator_to_GPS(x: float, y: float) -> Coordinate:
    """
    Converts a mercator coordinate (x, y) to WGS84 GPS coordinate (lat, lon)

    Parameters
    ----------
    x : float
        x coordinate
    y : float
        y coordinate

    Returns
    -------
    tuple of float :
        the (lat, lon) GPS coordinates.
        lat is between -90° and 90°
        lon is between -180° and 180°
    """
    lon = x*360 - 180
    lat = math.degrees(math.atan(math.exp((y - 0.5)*2*math.pi)))*2 - 90
    return lat, lon


def mercator_to_image(position: Coordinate, image: np.ndarray,
                      bottom_left: Coordinate,
                      top_right: Coordinate) -> Coordinate:
    """
    Converts a mercator coordinate to a (py, px) image coordinate in
    floating point pixels.

    The image's top left corner is at (0., 0.) and the image's bottom right
    corner is at (height, width).
    The top left pixel's center is at (0.5, 0.5) and the bottom right pixel's
    center is at (height - 0.5, width - 0.5).

    Parameters
    ----------
    position : tuple of float
        (x, y) position in mercator coordinates
    image : np.ndarray
        satellite image of shape (height, width, ...)
    bottom_left : tuple of float
        (x, y) coordinate of the image's bottom left corner
    top_right : tuple of float
        (x, y) coordinate of the image's top_right corner

    Returns
    -------
    tuple of float :
        the (py, px) coordinates of the position in the image
    """
    # Shape of the image in pixels
    S = np.flip(image.shape[:2])  # flipping to have a (width, height) vector
    # Mercator coordinate of the image's top left corner
    TL = np.array([bottom_left[0], top_right[1]])
    # Mercator coordinate of the image's bottom right corner
    BR = np.array([top_right[0], bottom_left[1]])
    # Mercator coordinate of the position
    P = np.array(position)
    # Calculate piwel coordinates in the image
    px, py = (P - TL)/(BR - TL)*S
    return py, px


def GPS_to_image(gps: Coordinate, image: np.ndarray,
                 bottom_left: Coordinate,
                 top_right: Coordinate) -> Coordinate:
    """
    Converts a WGS84 GPS coordinate to a (py, px) image coordinate in
    floating point pixels.

    This is a wrapper around 'mercator_to_image'.

    Parameters
    ----------
    gps : tuple of float
        a (lat, lon) coordinate
    image : np.ndarray
        satellite image of shape (height, width, ...)
    bottom_left : tuple of float
        (x, y) coordinate of the image's bottom left corner
    top_right : tuple of float
        (x, y) coordinate of the image's top_right corner

    Returns
    -------
    tuple of float :
        the (py, px) coordinates of the position in the image
    """
    position = GPS_to_mercator(*gps)
    bottom_left = GPS_to_mercator(*bottom_left)
    top_right = GPS_to_mercator(*top_right)
    return mercator_to_image(position, image, bottom_left, top_right)


def _tile_file(i: int, j: int, zoom: int, key: str) -> bytes:
    """
    Query a MapTiler tile.
    The returned bytes are the content of a 512x512 RGB jpg file.

    The mercator projection is cut into a grid of n by n tiles,
    where n=2**zoom. The top left tile is at (i=0, j=0)
    and the bottom right tile is at (i=n-1, j=n-1).

    Parameters
    ----------
    i : int
        the y index of the tile in the grid (between 0 and 2**zoom - 1)
    j : int
        the x index of the tile in the grid (between 0 and 2**zoom - 1)
    zoom : int
        the zoom parameter (between 0 and 17)
    key : str
        the MapTiler API key

    Returns
    -------
    bytes :
        The binary content of the 512x512 jpg image file
    """
    assert (0 <= zoom), "zoom level must be positive"
    assert (0 <= i < 2**zoom) and (0 <= j < 2**zoom), "Tile index out of grid bounds"
    url = f"https://api.maptiler.com/tiles/satellite/{zoom}/{j}/{i}@2x.jpg?key={key}"
    print("Querying tile ...", end="", flush=True)
    r = requests.get(url)
    print("\tDone", flush=True)
    code = r.status_code
    if code != 200:
        raise ConnectionError(f"https request failed with code: {code}")
    return r.content


def tile(*args) -> np.ndarray:
    """
    Query a tile as an np.ndarray representing an image.

    This is a wrapper around _tile_file(*args).

    Returns
    -------
    np.ndarray :
        the image as a numpy array
    """
    data = _tile_file(*args)
    return np.array(PIL.Image.open(io.BytesIO(data)))


def _tile_index(x: float, y: float, zoom: int) -> TileIndex:
    """
    Converts a mercator coordinate to a tile index that contains it,
    for a given zoom.

    Coordinate outside of mercator projection are not supported:
    (0 <= x <= 1) and (0 <= y <= 1) must be verified

    Parameters
    ----------
    merc : Coordinate
        a (x, y) mercator coordinate
    zoom : int
        a zoom level

    Returns
    -------
    tuple of int :
        the (i, j) coordinates in the grid of tiles
    """
    assert (0. <= x <= 1.) and (0. <= y <= 1.), "Coordinate out of mercator map"
    L = 2**-zoom  # length in the mercator coordinates of a tile side
    i, j = int((1-y) / L), int(x / L)
    return i, j


def _tile_corner(i: int, j: int, zoom: int) -> Coordinate:
    """
    For a tile index (i, j) and a zoom level,
    returns the mercator coordinates of it's bottom left corner.

    Parameters
    ----------
    i : int
        the y index of the tile in the grid (between 0 and 2**zoom - 1)
    j : int
        the x index of the tile in the grid (between 0 and 2**zoom - 1)
    zoom : int
        a zoom level
    """
    assert (0 <= i < 2**zoom) and (0 <= j < 2**zoom), "Tile index out of grid bounds"
    L = 2**-zoom  # length in the mercator coordinates of a tile side
    x = j*L
    y = 1. - (i+1)*L
    return x, y


def query(bottom_left: Coordinate, top_right: Coordinate,
          key: str = "wcbri0AocqWcoVy0wNe3",
          resolution: [float, str] = "auto") -> np.ndarray:
    """
    Returns a satellite image of the rectangular area
    specified by the two input corners.

    The returned image is in Mercator projection,
    so latitudes of the corners must be between -85.05° and 85.05°

    Parameters
    ----------
    bottom_left : tuple of float
        The tuple of (lat, lon) WGS84 GPS coordinates
        of the bottom left corner
    top_right : tuple of float
        The tuple of (lat, lon) WGS84 GPS coordinates
        of the top right corner
    key : str
        A free API key from MapTiler cloud
        https://www.maptiler.com/cloud/
    resolution : float or str
        the number of ° of longitude per pixel,
        or one of:
            "max" (~5.36E-6 °/pixel)
            "min" (~0.703 °/pixel)
            "auto" (the output image's width is 512 pixels)
    """
    assert bottom_left[0] < top_right[0], "'bottom_left' can't have a latitude superior to 'top_right'"
    # Calculate the zoom
    lat_delta = top_right[0] - bottom_left[0]  # delta of latitude in °
    if isinstance(resolution, str):
        if resolution == "max":
            zoom = 17
        elif resolution == "min":
            zoom = 0
        elif resolution == "auto":
            resolution = lat_delta/512
        else:
            raise ValueError(f"str values for 'resolution' must be one of ['min', 'max'], got '{resolution}'")
    if not isinstance(resolution, str):
        zoom = math.ceil(math.log2(360/512/resolution))
        zoom = max(0, min(zoom, 17))
    n = 2**zoom  # The mercator projection is cut into an n by n grid of tiles
    L = 1/n  # Length of a tile in mercator coordinates
    # convert to mercator coordinates
    bottom_left = GPS_to_mercator(*bottom_left)
    top_right = GPS_to_mercator(*top_right)
    # calculate the ranges of tile indexes containing the rectangle
    i_bot, j_left = _tile_index(*bottom_left, zoom)
    i_top, j_right = _tile_index(*top_right, zoom)
    if j_left <= j_right:
        columns = [j for j in range(j_left, j_right+1)]
    else:  # To be cyclic along x axis
        rows = [i % n for i in range(j_left, j_right+n+1)]
    rows = [i for i in range(i_top, i_bot+1)]
    # Stitch the tiles together
    stitch = np.concatenate([np.concatenate([tile(i, j, zoom, key)
                             for j in columns], axis=1)
                             for i in rows], axis=0)
    # Crop the the rectangle of interest in the stitched tiles
    BOTTOM_LEFT = np.array(_tile_corner(i_bot, j_left, zoom))  # Mercator coordinate of the image's bottom left corner
    TOP_RIGHT = BOTTOM_LEFT + [L*len(columns), L*len(rows)]  # Mercator coordinate of the image's top right corner
    top_left = bottom_left[0], top_right[1]  # Mercator coordinate of the GPS window's top left corner
    bottom_right = top_right[0], bottom_left[1]  # Mercator coordinates of the GPS window's bottom right corner
    i0, j0 = mercator_to_image(top_left, stitch, BOTTOM_LEFT, TOP_RIGHT)
    i1, j1 = mercator_to_image(bottom_right, stitch, BOTTOM_LEFT, TOP_RIGHT)
    image = stitch[int(i0):int(i1), int(j0):int(j1), ...]
    # downsample image to target resolution
    if not isinstance(resolution, str):
        scale = (lat_delta/resolution)/image.shape[1]
        new_size = np.rint(np.flip(image.shape[:2]) * scale).astype(int)
        im = PIL.Image.fromarray(image)
        image = np.array(im.resize(new_size))
    # Return the resulting image
    return image


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    image = query((48.848, 2.295), (48.852, 2.305), resolution="auto")
    plt.imshow(image)
    plt.show()
