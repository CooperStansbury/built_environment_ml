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
    __slots__ = ['path', 'src', 'pad_val', 'side_len']
    def __init__(self, path):
        """ Initialize the builder. Note, this step is not very memory intensive. 
        
        Args:
            - path (str): the full filepath to the tiled geotiff file
            - side_len (int): the pixel value of the side length of a mosiac tile.
                only squares are supported.
        """
        self.side_len = None
        self.path = path
        self.src = rasterio.open(self.path)
        self.pad_val = 255 # for empty tiles on the edges of the frame


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

    
    def get_sample(self, n, side_len):    
        """A function to get a random sample
        
        Args:
            - n (int): number of samples to return
            - side_len (int):

        Returns:
            - tiles (list of np.array): a list of subsets
            - center_coords (list of list): a list of [lat, lon] coordinates for the
                center of an image
        """
        self.side_len = side_len
        tiles = []
        center_coords = []

        # get random starting positions 
        rand_rows = np.random.randint(self.src.shape[0]-side_len , size=(n,))
        rand_cols = np.random.randint(self.src.shape[1]-side_len, size=(n,))

        for row_idx, col_idx in zip(rand_rows, rand_cols):
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

        return tiles, center_coords
    

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
        self.side_len = side_len
        tiles = []
        center_coords = []

        row_idx = 0
        clip_num = 0

        for i in range(int(self.src.shape[0] / side_len)+1):
            col_idx = 0
            for j in range(int(self.src.shape[1] / side_len)+1):

                clip_num += 1
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


    def build_fig(self, size=(5, 5), dpi=300):
        """A function to build a matplotlib figure. Primary
        goal is to sandardize the easy stuff.

        Args:
            - size (tuple): how big should the plot be?
            - dpi (int): the "quality" of the image
        Returns:
            fig (plt.figure)
        """
        fig = plt.figure(figsize=size, 
                        facecolor='w',
                        dpi=dpi)

        plt.axis('off')
        return fig

    
    def save_tiles(self, tiles, output_dir):
        """A function to save tiles.

        Args:
            - tiles (np.array) list of tiles
            - output_dir (str): path to save

        Returns:
            None (save only)
        """
        save_path = f"{output_dir}/tiles.npy"
        tiles_np = np.asarray(tiles)
        np.save(save_path, tiles_np)
        print("done saving .npy!")


    def save_coords(self, coords, output_dir):
        """A function to save the output coordinates.

        Args:
            - coords (list): list of tile corrdinates
            - output_dir (str): path to save

        Returns:
            None (save only)

        """
        new_rows = []
        for i, (lat, lon) in enumerate(coords):
            row = {
                'tile_id': i,
                'lat':lat,
                'long':lon,
                'side_length': self.side_len 
            }

            new_rows.append(row)

        coord_df = pd.DataFrame(new_rows)
        coord_df.to_csv(f"{output_dir}/coordinate_map.csv", index=False)
        print("done saving coordinates!")


    def save_to_png(self, tiles, output_dir, channel=None):
        """A function to convert 3 dim RGB matrices to .png
        files. Filename depends on order of 'tiles' input.

        Args:
            - tiles (np.array) list of tiles
            - output_dir (str): path to save
            - channel (int): the channel to save, or None

        Returns:
            None (save only)
        """
        plt.ioff()

        for idx, tile in enumerate(tiles):
            save_path = f"{output_dir}/tile_{idx}"
            fig = self.build_fig()
            img = np.moveaxis(tile, 0, 2)
            
            if channel is None:            
                plt.imshow(img)
                plt.savefig(save_path)
                plt.close()
            else:
                plt.imshow(img[:, :, channel])
                plt.savefig(save_path)
                plt.close()

        print("done converting to png!")






    