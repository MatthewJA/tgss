"""NVSS-specific functions.

Matthew Alger <matthew.alger@anu.edu.au>
Research School of Astronomy and Astrophysics
The Australian National University
2017
"""

import csv
import os
import re

import astropy.coordinates
import astropy.io.fits
import astropy.nddata
import astropy.units
import astropy.wcs
import numpy
import scipy.spatial

import survey


class Cutout3D:
    """3D cutout with same API as Cutout2D."""

    def __init__(self, cutouts):
        self.wcs = cutouts[0].wcs
        self.data = numpy.stack([cutout.data for cutout in cutouts])
        self.bbox_cutout = cutouts[0].bbox_cutout
        self.bbox_original = cutouts[0].bbox_original
        self.center_cutout = cutouts[0].center_cutout
        self.center_original = cutouts[0].center_original
        self.origin_cutout = cutouts[0].origin_cutout
        self.origin_original = cutouts[0].origin_original
        self.position_cutout = cutouts[0].position_cutout
        self.position_original = cutouts[0].position_original


class NVSS(survey.Survey):
    """NRAO VLA Sky Survey."""

    name = 'NVSS'
    wavelength = 0.21429
    resolution = 45

    def __init__(self, image_dir_path, catalogue_path):
        self.image_dir_path = image_dir_path
        self.catalogue_path = catalogue_path
        # Read the pointings in. We will have an array of IDs and
        # an array of corresponding pointing centres.
        self.pointing_ids = []
        self.pointing_centres = []
        filename_re = re.compile(r'C(\d{2})(\d{2})([MP])(\d{2})\.gz')
        for filename in os.listdir(image_dir_path):
            # Filenames are of the form CHHMMSDD.
            match = filename_re.match(filename)
            if not match:
                continue

            h, m, s, d = match.groups()
            s = {'P': '+', 'M': '-'}[s]
            ra = '{} {} 00'.format(h, m)
            dec = '{}{} 00 00'.format(s, d)
            coord = astropy.coordinates.SkyCoord(
                ra=ra, dec=dec, unit=('hourangle', 'deg'))
            coord = (coord.ra.deg, coord.dec.deg)
            self.pointing_ids.append(filename[:-3])
            self.pointing_centres.append(coord)

        self.pointing_ids = numpy.array(self.pointing_ids)
        self.pointing_centres = numpy.array(self.pointing_centres)
        self.pointing_centres_tree = scipy.spatial.KDTree(self.pointing_centres)

        self._process_catalogue()

    def _process_catalogue(self):
        """Store catalogue data."""
        with astropy.io.fits.open(self.catalogue_path) as catalogue_file:
            n = catalogue_file[1].data.shape[0]
            ras = catalogue_file[1].data['RA(2000)']
            decs = catalogue_file[1].data['DEC(2000)']
            coords = numpy.stack([ras, decs]).T
            names = ['NVSS#{}'.format(i) for i in range(n)]
        self.catalogue_tree = scipy.spatial.KDTree(coords)
        self.catalogue_names = numpy.array(names)
        self.catalogue_coords = coords
        self.name_to_index = {
            str(name): index
            for index, name in enumerate(self.catalogue_names)}

    def query_image_tile(self, coord):
        # First-pass: Nearest-neighbour search with a KDTree.
        _, q = self.pointing_centres_tree.query(coord)
        return self.pointing_ids[q]

    def cutout(self, coord, radius):
        # First-pass: Cut directly from any containing square.
        image_id = self.query_image_tile(coord)
        with astropy.io.fits.open(
                os.path.join(self.image_dir_path, image_id + '.gz')) as fits:
            # WCS order: RA, DEC, STOKES, FREQ
            # Data order: FREQ, STOKES, DEC, RA
            # WCS that contains only RA/DEC.
            wcs = astropy.wcs.WCS(fits[0].header).dropaxis(3).dropaxis(2)
            coord = astropy.coordinates.SkyCoord(
                     ra=coord[0], dec=coord[1],
                     unit='degree')
            size = radius * astropy.units.degree
            # One channel for each Stokes parameter.
            cutouts = [astropy.nddata.Cutout2D(
                fits[0].data[0, s], coord, size,
                wcs=wcs, mode='partial') for s in range(3)]
            return Cutout3D(cutouts)

    def objects_radius(self, coord, radius):
        # First-pass: 2D KDTree, Euclidean approximation.
        nearby = self.catalogue_tree.query_ball_point(coord, radius)
        return ((self.catalogue_names[i], self.catalogue_coords[i])
                for i in nearby)

    def objects(self):
        return ((self.catalogue_names[i], self.catalogue_coords[i])
                for i in range(len(self.catalogue_names)))

    def is_compact(self, name):
        raise NotImplementedError()


if __name__ == '__main__':
    nvss = NVSS('/home/alger/myrtle1/nvss',
                '/home/alger/myrtle1/nvss/CATALOG.FIT')
    # cutout = nvss.cutout((173.496704, 49.062008), 0.05)
    # import matplotlib.pyplot as plt
    # plt.subplot(1, 1, 1, projection=cutout.wcs)
    # plt.imshow(cutout.data, origin='lower')
    # plt.show()
