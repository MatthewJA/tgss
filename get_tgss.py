#!/usr/bin/env python3
"""Download TGSS ADR1.

Matthew Alger <matthew.alger@anu.edu.au>
"""

import argparse
import logging
import os
import re
import shutil

import requests

TGSS_BASE_URL = 'http://tgssadr.strw.leidenuniv.nl/mosaics/'


def main(tgss_dir: str):
    # Fetch a list of all TGSS images.
    r = requests.get(TGSS_BASE_URL)
    urls = re.findall(r'href="(.*?\.FITS)"', r.text)
    for i, url in enumerate(urls):
        progress = '({}/{})'.format(i + 1, len(urls))
        target_url = os.path.join(TGSS_BASE_URL, url)
        logging.info('Downloading {}'.format(target_url))
        target_path = os.path.join(tgss_dir, url)
        if os.path.exists(target_path):
            logging.info('{} {}: From cache'.format(progress, url))
            continue

        r = requests.get(target_url, stream=True)
        if r.status_code != 200:
            logging.warning('{} {}: Could not download ({})'.format(
                progress, url, r.status_code))
            continue

        with open(target_path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

        logging.info('{} {}: Downloaded'.format(progress, url))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', help='Download directory')
    args = parser.parse_args()
    logging.root.setLevel(logging.INFO)
    main(args.dir)
