# tilerUtility

## Overview

This is a tool to convert a single large geoTIFF file into smaller sections. 

## Usage

```
usage:  [-h] [-i INPUT_FILEPATH] [-size TILE_SIZE] [-n [SAMPLE_SIZE]]
        [-o [OUTPUT_DIR]]

A command line interface tiling utility.

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_FILEPATH, --input_filepath INPUT_FILEPATH
                        path to geotiff file for tiling.
  -size TILE_SIZE, --tile_size TILE_SIZE
                        number of pixels, or side length of each tile. tiles
                        are sqaure.
  -n [SAMPLE_SIZE], --sample_size [SAMPLE_SIZE]
                        number of images to randomly sample. If not specified,
                        all tiles returned.
  -o [OUTPUT_DIR], --output_dir [OUTPUT_DIR]
                        output directory for saving images. no slash `/` at
                        the end.
```