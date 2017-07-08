"""Represent a survey.

Matthew Alger <matthew.alger@anu.edu.au>
Research School of Astronomy and Astrophysics
The Australian National University
2017
"""

import abc

class Survey(abc.ABC):
    """Interface to represent a survey.

    This should be subclassed for each survey.
    """

    # Name of survey.
    name = None

    # Wavelength of survey (m).
    wavelength = None

    # Resolution of survey (arcsec^2).
    resolution = None

    @abc.abstractmethod
    def query_image_tile(self, coord):
        """Query an image containing a coord.

        Parameters
        ----------
        coord : (float, float)
            RA, dec

        Returns
        -------
        str path to image
        """

    @abc.abstractmethod
    def cutout(self, centre, radius):
        """Retrieve a cutout image.

        Parameters
        ----------
        centre : (float, float)
            RA, dec
        radius : float
            in degrees

        Returns
        -------
        NumPy array
        """

    @abc.abstractmethod
    def objects(self, centre, radius):
        """Get catalogue objects within a radius.
        
        Parameters
        ----------
        centre : (float, float)
            RA, dec
        radius : float
            in degrees

        Returns
        -------
        Iterable of (name, (ra, dec))
        """
