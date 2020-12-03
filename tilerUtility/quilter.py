"""
Description:
    class to manage conversion of a single larger geoTiff to 
    a collection of "tiles", smaller images.
"""


import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from affine import Affine
from pyproj import Proj
from pyproj import transform


class MosiacBuilder():
    """ A class to manage building mosaics/tiles/quilts from a geotiff images. 
    """
    __slots__ = ['filepath', 'src', 'pad_val', 'tile_size']
    def __init__(self, filepath, side_len):
        """ Initialize the builder. Note, this step is not very memory intensive. 
        
        Args:
            - filepath (str): the full filepath to the tiled geotiff file
            - side_len (int): the pixel value of the side length of a mosiac tile.
                only squares are supported.
        """
        self.filepath = filepath
        self.src = rasterio.open(self.filepath)
        self.pad_val = 255 # for empty tiles on the edges of the frame
        self.tile_size = side_len


    def get_window(self, row, col, side_len, return_win=False):
        """ A method to get a subset of a given raster given 
        row/column offsets from (0,0). 

        Args:
            - row (int): offset along latitude, y axis offset'
            - col (int): offset along longitude, x axis offset
            - side_len (int): how large is the image?
            - return_win (bool): if True, return the window object

        Returns:
            - subset (np.array): a window from the original raster. NOTE: by
                default this array is channels first, RGB: (3, side_len, side_len)

            - window (affine.Affine): DEPENDS ON BOOL FLAG: `return_win` set to True. 
                transformation associated with top-left corner
                of `subset`. Used for translation to lat/lon
        """
        window = rasterio.windows.Window(col, row, side_len, side_len)
        subset = self.src.read(window=window)    

        if return_win:
            return subset, window
        else: 
            return subset


    def get_latlon_point(self, row, col):
        """ Get lat andf lon from a single point 
        
        Args:
            - col (int): offset along longitude, x axis offset
            - row (int): offset along latitude, y axis offset'

        Returns:
            - lat (float): latitude
            - lon (float): longitude
        """
        p1 = Proj(self.src.crs)
        window = rasterio.windows.Window(col, row, 1, 1)
        trnsfrm = self.src.window_transform(window)
        T1 = trnsfrm * Affine.translation(0.5, 0.5)
        p2 = Proj(proj='latlong', datum='WGS84')
        x, y = self.src.xy(row, col)
        lon, lat = transform(p1, p2, x, y)
        return lat, lon

    
    def get_sample(self, n):    
        """A function to get a random sample"""
        pass


    def build_mosaic(self, side_len):
        """ A method to build a set number of SQUARE
        images with a given side length from a raster.side_len.

        Args:
            - side_len (int): the pixel value of the side length of a mosiac tile.
                only squares are supported.

        Returns:
            - tiles (list of np.array): a list of subsets
            - center_coords (list of list): a list of [lat, lon] coordinates for the
                center of an image
        """
        tiles = []
        center_coords = []

        row_idx = 0
        clip_num = 0

        for i in range(int(self.src.shape[0] / side_len)+1):
            col_idx = 0
            for j in range(int(self.src.shape[1] / side_len)+1):

                clip_num += 1
                if clip_num % 500 == 0:

                # get clip
                clip = self.get_window(row_idx, col_idx, side_len)

                # handle non-square clips
                if clip.shape[1] != side_len or clip.shape[2] != side_len:
                    pad = np.full((3, side_len, side_len), self.pad_val)
                    pad[:, 0:clip.shape[1], 0:clip.shape[2]] = clip
                    clip = pad.copy()

                tiles.append(clip)

                # get center lat/lon
                lat, lon = self.get_latlon_point(row_idx + side_len // 2, col_idx + side_len // 2)

                center_coords.append([lat, lon])

                # increment counters
                col_idx += side_len
            row_idx += side_len

        return tiles, center_coords


   