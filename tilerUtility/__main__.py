"""
Description:
    Entry point for tiling utility. 
"""
import os
import sys
import argparse

# local modules
import quilter


def resolve_path(filepath):
    """A function to return the absolute path given a 
    file path. No exception handling right now.

    Args:
        - filepath (str): a path
    
    Returns:
        - path (str): absolute path
    """
    return os.path.abspath(filepath)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A command line interface tiling utility.")

    parser.add_argument("-i", "--input_filepath", help="path to geotiff file for tiling.")

    parser.add_argument("-size", "--tile_size", 
                        help="number of pixels, or side length of each tile. tiles are sqaure.")

    parser.add_argument("-n", "--sample_size", nargs="?", default=None, 
                        help="number of images to randomly sample. If not specified, all tiles returned.")

    parser.add_argument("-o", "--output_dir", nargs="?", default='tilerUtility/outputs', 
                        help="output directory for saving images. no slash `/` at the end. ")

    args = parser.parse_args()
    input_filepath = resolve_path(args.input_filepath)
    tile_size = int(args.tile_size)
    output_dir = resolve_path(args.output_dir)

    # make builder
    builder = quilter.MosiacBuilder(path=input_filepath)

    if not args.sample_size is None:
        n = int(args.sample_size)

        tiles, coords = builder.get_sample(n, side_len=tile_size)

        builder.save_coords(coords, output_dir)
        builder.save_tiles(tiles, output_dir)
        builder.save_to_png(tiles, output_dir)
    
    # else logic is to run the entire geotiff



    

