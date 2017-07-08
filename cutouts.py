"""Generate cutouts of all objects in a survey.

Matthew Alger <matthew.alger@anu.edu.au>
Research School of Astronomy and Astrophysics
The Australian National University
2017
"""

import argparse
import logging
import os.path

import astropy.io.fits
import scipy.misc

import tgss

def cutouts(objects, cutout_radius, output_path):
    """Generate cutouts of given objects."""
    for name, coord in objects:
        logging.info('Generating cutout for {}'.format(name))
        cutout = survey.cutout(coord, cutout_radius)
        fits = astropy.io.fits.PrimaryHDU(
            data=cutout.data,
            header=cutout.wcs.to_header())
        path = os.path.join(
            output_path, '{0}_{1}x{1}'.format(name, cutout_radius))
        fits.writeto(path + '.fits', overwrite=True)
        scipy.misc.imsave(path + '.png', cutout.data)

def cutouts_radius(
        survey, search_centre, search_radius, cutout_radius, output_path):
    """Generate cutouts of objects within a radius."""
    objects = survey.objects_radius(search_centre, search_radius)
    cutouts(objects, cutout_radius, output_path)


if __name__ == '__main__':
    logging.root.setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('survey', choices=['tgss'])
    parser.add_argument('--cutout-radius', '-c', type=float, default=0.05)
    parser.add_argument('--output', '-o')
    args = parser.parse_args()

    survey = tgss.TGSS('/home/alger/myrtle1/tgss',
                       '/home/alger/myrtle1/tgss/TGSSADR1_7sigma_catalog.tsv',
                       '/home/alger/myrtle1/tgss/grid_layout.rdb')
    cutouts(survey, args.cutout_radius, args.output)
