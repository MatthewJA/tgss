"""TGSS-specific functions.

Matthew Alger <matthew.alger@anu.edu.au>
Research School of Astronomy and Astrophysics
The Australian National University
2017
"""

import csv
import os.path
import re

import astropy.io.fits
import astropy.nddata
import astropy.units
import astropy.wcs
import numpy
import scipy.spatial

import survey


def is_compact(s_int, s_peak, e_s_int, e_s_peak):
    """Check if an object is compact."""
    if not s_int:
        s_int = s_peak
        e_s_int = e_s_peak

    r = numpy.log(s_int / s_peak)
    e_r = numpy.sqrt(
        (e_s_int / s_int) ** 2 +
        (e_s_peak / s_peak) ** 2)

    return r < 2 * e_r


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

        self._process_catalogue()

    def _process_catalogue(self):
        """Store catalogue data."""
        with open(self.catalogue_path) as catalogue_file:
            reader = csv.DictReader(catalogue_file, delimiter='\t')
            n = sum(1 for _  in reader)
            catalogue_file.seek(0)
            next(catalogue_file)
            compact = numpy.zeros((n,), dtype=bool)
            coords = numpy.zeros((n, 2))
            names = []
            for i, row in enumerate(reader):
                coords[i, 0] = float(row['RA'])
                coords[i, 1] = float(row['DEC'])
                names.append(row['Source_name'])
                compact[i] = is_compact(
                    float(row['Total_flux']),
                    float(row['Peak_flux']),
                    float(row['E_Total_flux']),
                    float(row['E_Peak_flux']))
        self.catalogue_tree = scipy.spatial.KDTree(coords)
        self.catalogue_names = numpy.array(names)
        self.catalogue_coords = coords
        self.catalogue_compact = compact
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
                os.path.join(self.image_dir_path,
                             image_id + '_5x5.MOSAIC.FITS')) as fits:
            wcs = astropy.wcs.WCS(fits[0].header).dropaxis(3).dropaxis(2)
            coord = astropy.coordinates.SkyCoord(
                     ra=coord[0], dec=coord[1],
                     unit='degree')
            size = radius * astropy.units.degree
            cutout = astropy.nddata.Cutout2D(
                fits[0].data[0, 0], coord, size,
                wcs=wcs, mode='partial')
            return cutout

    def objects_radius(self, coord, radius):
        # First-pass: 2D KDTree, Euclidean approximation.
        nearby = self.catalogue_tree.query_ball_point(coord, radius)
        return ((self.catalogue_names[i], self.catalogue_coords[i])
                for i in nearby)

    def objects(self):
        return ((self.catalogue_names[i], self.catalogue_coords[i])
                for i in range(len(self.catalogue_names)))

    def is_compact(self, name):
        return self.catalogue_compact[self.name_to_index[name]]


if __name__ == '__main__':
    tgss = TGSS('/home/alger/myrtle1/tgss',
                '/home/alger/myrtle1/tgss/TGSSADR1_7sigma_catalog.tsv',
                '/home/alger/myrtle1/tgss/grid_layout.rdb')
    cutout = tgss.cutout((173.496704, 49.062008), 0.05)
    import matplotlib.pyplot as plt
    plt.subplot(1, 1, 1, projection=cutout.wcs)
    plt.imshow(cutout.data, origin='lower')
    plt.show()
