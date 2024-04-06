import click
import pathlib
import csv
import json
import time
import math
import collections

# Hack to fix error in multi processing
# os.environ['OPENBLAS_NUM_THREADS'] = '1'

import multiprocessing as mp


import numpy as np
import brahe
from brahe import Epoch
import brahe.data_models as bdm
from brahe.access.tessellation import tessellate
from brahe.access.access import find_location_accesses

# Monkey Patch ECI -> ECEF transformation due to segfault in upstream dependency
# This changes the transformation to only consider the Earth's rotation and not the polar motion, precession, or nutation
# This is a simplification but is sufficient for this use case
def sECEFtoECIpatch(epc, x):
    """Transforms an Earth fixed state into an Inertial state

    The transformation is accomplished using the IAU 2006/2000A, CIO-based
    theory using classical angles. The method as described in section 5.5 of
    the SOFA C transformation cookbook.

    Args:
        epc (Epoch): Epoch of transformation
        x (np.ndarray): Earth-fixed state (position, velocity) [*m*; *m/s*]

    Returns:
        x_ecef (np.ndarray): Inertial state (position, velocity)
    """

    # Ensure input is array-like
    x = np.asarray(x)

    # Set state variable size
    dim_x = len(x)
    x_eci = np.zeros((dim_x,))

    # Extract State Components
    r_ecef = x[0:3]

    if dim_x >= 6:
        v_ecef = x[3:6]

    # Compute Sequential Transformation Matrices
    rot = brahe.earth_rotation(epc)

    # Create Earth's Angular Rotation Vector
    omega_vec = np.array([0, 0, brahe.constants.OMEGA_EARTH]) # Neglect LOD effect

    # Calculate ECEF State
    x_eci[0:3] = ( rot ).T @ r_ecef
    # x_eci[0:3] = (pm @ rot @ bpn).T @ r_ecef

    if dim_x >= 6:
        x_eci[3:6] = (rot ).T @ (v_ecef + brahe.utils.fcross(omega_vec, r_ecef))

    return x_eci

brahe.frames.sECEFtoECI = sECEFtoECIpatch

# Utility Functions
def compute_collects(spacecraft, request, t_start, t_end):
    # print(f'Computing collects: Spacecraft {spacecraft.id} - Request {request.description}')
    ts = time.time()
    try:
        tiles = tessellate(spacecraft, request)
    except Exception as e:
        return [], []

    cts = []
    for tile in tiles:
        cts.extend(find_location_accesses(spacecraft, tile, t_start, t_end, request=request))

    te = time.time()
    # print(f'Finished computing collects: Spacecraft {spacecraft.id} - Request {request.}. Took {te-ts:.2f} seconds.')

    return tiles, cts

# Input Path

@click.command()
@click.option('--days', required=True, default=1, type=int, help='Number of days to generate.')
@click.option('--sats', required=True, default=1, type=int, help='Number of satellites')
@click.option('--deck', required=True, type=str, help='Target deck to use')
@click.option('--limit', required=False, default=None, type=int, help='Limit the number of requests to process')
def cmd(days, sats, deck, limit):
    # Display
    print(f'Utilizing {mp.cpu_count()} cores for scenario generation')

    # Set Simulation Duration
    t_start = Epoch(2020, 1, 1, time_system='UTC')
    DAYS = days
    t_end = t_start + 86400 * DAYS

    SATS = sats

    COLLECT_DURATION = 30

    SPACECRAFT_FILE = f'./data/spacecraft/spacecraft_{SATS:d}.json'
    REQUESTS_FILE = f'./data/requests/{deck}.json'

    if not pathlib.Path(SPACECRAFT_FILE).exists():
        raise RuntimeError(f'Spacecraft file {SPACECRAFT_FILE} not found.')

    if not pathlib.Path(REQUESTS_FILE).exists():
        raise RuntimeError(f'Requests deck {REQUESTS_FILE} not found.')

    OUTPUT_DIR = f'outputs/{pathlib.Path(REQUESTS_FILE).stem}_{DAYS:d}_days_{SATS:d}_sats'
    if not pathlib.Path(OUTPUT_DIR).exists():
        pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    tscn_start = time.time()

    # Load Spacecraft
    spacecraft = []
    spacecraft_json = json.load(open(SPACECRAFT_FILE, 'r'))
    for scj in spacecraft_json:
        spacecraft.append(bdm.Spacecraft(**scj))

    # Load Requests
    requests = []
    requests_json = json.load(open(REQUESTS_FILE, 'r'))

    for req in requests_json:
        requests.append(bdm.Request(**req))

    if limit is not None:
        requests = requests[:limit]

    # Set Request Duration
    for req in requests:
        req.properties.collect_duration = COLLECT_DURATION

    print(f'Loaded {len(spacecraft):d} spacecraft and {len(requests):d} requests')

    # Tile Requests and compute collects
    tiles = []
    collects = []

    print(f'Tessellating requests and computing collects...')

    # Compute Collects
    mpctx = mp.get_context('forkserver')
    with mpctx.Pool(mp.cpu_count()) as pool:
        # Multi processing work
        work = []

        for request in requests:
            for sc in spacecraft:
                work.append((sc, request, t_start, t_end))

        results = pool.starmap(compute_collects, work)

        # Extend Collects
        for res in results:
            if len(res[0]) > 0:
                tiles.extend(res[0])
                collects.extend(res[1])

    collects.sort(key=lambda o: o.t_start)

    # Save Outputs
    spacecraft_out = []
    for sc in spacecraft:
        spacecraft_out.append(json.loads(sc.json()))
    json.dump(spacecraft_out, open(f'{OUTPUT_DIR}/spacecraft.json', 'w'), indent=4)

    requests_out = []
    for request in requests:
        requests_out.append(json.loads(request.json()))
    json.dump(requests_out, open(f'{OUTPUT_DIR}/requests.json', 'w'), indent=4)

    for sc in spacecraft:
        sc_tiles = filter(lambda t: sc.spacecraft_id in t.spacecraft_ids, tiles)
        tiles_out = []
        for tile in sc_tiles:
            tiles_out.append(json.loads(tile.json()))
        json.dump(tiles_out, open(f'{OUTPUT_DIR}/tiles_sc_{sc.id}.json', 'w'), indent=4)

    # Save Collects By Spacecraft
    for sc in spacecraft:
        sc_collects = filter(lambda c: c.spacecraft_id == sc.id, collects)
        collects_out = []
        for collect in sc_collects:
            collects_out.append(json.loads(collect.json()))
        json.dump(collects_out, open(f'{OUTPUT_DIR}/collects_sc_{sc.id}.json', 'w'), indent=4)

    tscn_end = time.time()

    run_minutes = int((tscn_end-tscn_start)//60)
    run_seconds = (tscn_end-tscn_start) - run_minutes*60

    print(f'Finished saving outputs')
    print(f'Found {len(collects):d} collects across {len(spacecraft):d} spacecraft')
    print(f'Scenario generation complete. Took {run_minutes:d} minutes {run_seconds:.02f} seconds')

if __name__ == '__main__':
    cmd()