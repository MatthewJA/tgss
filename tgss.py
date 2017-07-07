"""TGSS-specific functions.

Matthew Alger <matthew.alger@anu.edu.au>
Research School of Astronomy and Astrophysics
The Australian National University
2017
"""

import os.path
import re

import astropy.io.fits
import astropy.wcs
import numpy
import scipy.spatial

import survey

class TGSS(survey.Survey):
    """TIFR GMRT Sky Survey Alternative Data Release 1."""

    name = 'TGSS'
    wavelength = 2.0325
    resolution = 25

    def __init__(self, image_dir_path, catalogue_path, pointings_path):
        self.image_dir_path = image_dir_path
        self.catalogue_path = catalogue_path
        self.pointings_path = pointings_path
        # Read the pointings in. We will have an array of IDs and
        # an array of corresponding pointing centres.
        self.pointing_ids = []
        self.pointing_centres = []
        split_re = re.compile('\s+')
        with open(self.pointings_path) as pointings_tsv:
            for row in pointings_tsv:
                row = split_re.split(row)
                if len(row) != 4:
                    continue
                id_, ra, dec, a = row
                assert not a
                self.pointing_ids.append(id_)
                self.pointing_centres.append((float(ra), float(dec)))
        self.pointing_ids = numpy.array(self.pointing_ids)
        self.pointing_centres = numpy.array(self.pointing_centres)
        self.pointing_centres_tree = scipy.spatial.KDTree(self.pointing_centres)

    def query_image_tile(self, coord):
        # First-pass: Nearest-neighbour search with a KDTree.
        _, q = self.pointing_centres_tree.query(coord)
        return self.pointing_ids[q]

    def cutout(self, coord, radius):
        # First-pass: Cut directly from any containing square.
        image_id = self.query_image_tile(coord)
        with astropy.io.fits.open(
                os.path.join(self.image_dir_path, image_id + '.FITS')) as fits:
            wcs = astropy.wcs.WCS(fits[0].header)
            min_ra = coord[0] - radius
            min_dec = coord[1] - radius
            max_ra = coord[0] + radius
            max_dec = coord[1] + radius
            min_x, min_y = wcs.all_world2pix((min_ra, min_dec), 1)
            max_x, max_y = wcs.all_world2pix((max_ra, max_dec), 1)
            assert max_x >= min_x
            assert max_y >= min_y
            cutout = fits[0].data[0, 0, min_x:max_x, min_y:max_y]


if __name__ == '__main__':
    tgss = TGSS('', '', '/Users/alger/data/TGSS/pointing_grid.rdb')
