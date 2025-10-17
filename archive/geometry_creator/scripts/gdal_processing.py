from typing import Union
from osgeo import gdal
from pathlib import Path


def reproject_and_clip(input_raster,
                       output_raster,
                       projection,
                       shapefile: str = '',
                       resolution: float = 0.):
    if resolution:
        if shapefile:
            options = gdal.WarpOptions(cutlineDSName=shapefile,
                                       cropToCutline=True,
                                       format='GTIFF',
                                       dstSRS=projection,
                                       xRes=resolution,
                                       yRes=resolution)
        else:
            options = gdal.WarpOptions(cropToCutline=True,
                                       format='GTIFF',
                                       dstSRS=projection,
                                       xRes=resolution,
                                       yRes=resolution)
    else:
        if shapefile:
            options = gdal.WarpOptions(cutlineDSName=shapefile,
                                       cropToCutline=True,
                                       format='GTIFF',
                                       dstSRS=projection)
        else:
            options = gdal.WarpOptions(cropToCutline=True,
                                       format='GTIFF',
                                       dstSRS=projection)

    gdal.Warp(srcDSOrSrcDSTab=input_raster,
              destNameOrDestDS=output_raster,
              options=options)

    return output_raster


def create_mosaic(file_path: Union[Path, str],
                  file_name: str,
                  tiles: list) -> str:
    file_target = f'{file_path}/{file_name}.vrt'
    mosaic = gdal.BuildVRT(destName=file_target,
                           srcDSOrSrcDSTab=tiles)
    mosaic.FlushCache()

    return file_target


def vrt_to_geotiff(vrt_path: str,
                   geotiff_path: str):
    src_ds = gdal.Open(vrt_path, 0)  # open the VRT in read-only mode
    gdal.Translate(geotiff_path, src_ds, format='GTiff',
                   creationOptions=['COMPRESS:DEFLATE', 'TILED:YES'],
                   callback=gdal.TermProgress_nocb)
    # Properly close the datasets to flush to disk
    src_ds = None
    return geotiff_path
