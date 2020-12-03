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
    parser.add_argument("-s", "--tile_size", help="number of pixels, or side length of each tile. tiles are sqaure.")
    parser.add_argument("-o", "--output_dir", help="output directory for saving images.")

    # TODO: add option to return arrays or force conversion to png?

    args = parser.parse_args()

    input_filepath = resolve_path(args.input_filepath)
    tile_size = int(args.tile_size)
    output_dir = resolve_path(args.output_dir)


    # make builder
    # builder = quilter.MosiacBuilder(input_filepath, tile_size)

    # run builder 
    # tiles, center_coords = builder.build_mosaic()



    

