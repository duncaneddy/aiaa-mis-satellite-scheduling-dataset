import click
import pathlib
import json
import time
import pickle
import typing
import math
import itertools

import numpy as np
from scipy import spatial as sps
import multiprocessing as mp

import brahe
import brahe.data_models as bdm


def spacecraft_max_slew(spacecraft: bdm.Spacecraft) -> float:
    '''Return maximum possible slew time required between any two orientations

    Args:
        spacecraft (:obj:`Spacecraft`): Spacecraft object

    Returns:
        float: Maximum slew time between two orientations
    '''

    # Spacecraft Agility Properties
    slew_rate = spacecraft.model.slew_rate
    settling_time = spacecraft.model.settling_time

    # Quick Exit if easily feasible
    max_slew_time = 180.0 / slew_rate + settling_time

    return max_slew_time


def los_vector(sat_ecef: np.ndarray, opp_ecef: np.ndarray):
    '''Compute line of site vector between satellite and opportunity

    Args:
        sat_ecef (:obj:`np.ndarray`): Satellite position in Earth-Fixed Frame
        opp_ecef (:obj:`np.ndarray`): Opportunity center position in Earth-Fixed Frame

    Returns:
        np.ndarray: Line-of-Site vector betweens satellite and target in Earth Fixed Frame
    '''

    # Ensure np-ness
    sat_ecef = np.asarray(sat_ecef)
    opp_ecef = np.asarray(opp_ecef)

    # Compute initial look angle
    z_los = opp_ecef - sat_ecef[0:3]  # Compute look angle

    # Normalize vector
    z_los = z_los / np.linalg.norm(z_los)

    return z_los


def feasible_slew(spacecraft: bdm.Spacecraft, opp1: bdm.Opportunity, opp2: bdm.Opportunity, fast: bool = True):
    '''Determine whetether a transition between imaging consecutive opportunities
    is feasible.

    Args:
        spacecraft (:obj:`Spacecraft`): Spacecraft performing slew
        opp1 (:obj:`Oppotrunity`): First opportunity
        opp2 (:obj:`Oppotrunity`): Second opportunity
        fast (:obj:`Bool`): Fast version allowing short-circuit termination

    Returns:
        bool: `True` if slew is feasible `False` otherwise
        float: Slew time required
        float: Slew time available
    '''

    # Compute Available transition time
    t_transition = (opp2.t_start - opp1.t_end).total_seconds()

    # Quick Exit if easily feasible
    max_slew_time = spacecraft_max_slew(spacecraft)
    if t_transition <= 0.0:
        return False, max_slew_time, t_transition, 180.0
    elif fast == True and t_transition > max_slew_time:
        return True, max_slew_time, t_transition, 180.0

    # Get Starting line of site vector
    z_start = opp1.access_properties.los_end
    z_end = opp2.access_properties.los_start

    # Compute slew angle
    slew_angle = math.acos(np.dot(z_start, z_end)) * brahe.RAD2DEG

    # Compute slew time
    slew_time = slew_angle / spacecraft.model.slew_rate + spacecraft.model.settling_time

    if slew_time > t_transition:
        # Required slew time greater than available transition time
        return False, slew_time, t_transition, slew_angle
    else:
        return True, slew_time, t_transition, slew_angle


def agility_constraint(spacecraft: typing.List[bdm.Spacecraft],
                       collects: typing.List[bdm.Contact],
                       cpus: int = mp.cpu_count()):
    '''Apply agility constraints to schedule problem

    Args:
        spacecraft (:obj:`List[Spacecraft]`): Spacecraft object
        collects (:obj:`Collect`): Collects to compute conflicts for

    Returns:
        List[Tuple[Opportunity, Opportunity]]: List of conflicting opportunity pairs
        int: Number of total possible conflicting opportunities
    '''

    # Resulting Conflicts Result
    sc_conflicts = {sc.id: [] for sc in spacecraft}

    # Number of CPUs for run
    cpus = min(cpus, mp.cpu_count())

    tsc = time.time()

    # Set Process context
    mpctx = mp.get_context('forkserver')
    with mpctx.Pool(cpus) as pool:
        print(f'Using {cpus:d}/{mp.cpu_count():d} CPUs to compute agility constraints')

        # Compute conflicts on per-spacecraft basis
        for sc in spacecraft:
            print(f'Computing agility conflicts for spacecraft {sc.id:d}...', flush=True)
            ts = time.time()

            # Get Opportunities for conflict
            opportunities = list(filter(lambda x: x.spacecraft_id == sc.id, collects))
            opportunities.sort(key=lambda x: x.t_start)
            num_sco = len(opportunities)

            # Create K-D Tree structure for quick lookup of possible conflicts

            # First create lookup of MJD -> Opportunity
            tlt_opportunities = {
                (opp.t_start - opportunities[0].t_start).total_seconds(): opp for opp in opportunities
            }

            # Create KD-Tree indexiting Opportunities
            kd_input = np.array(list(tlt_opportunities.keys())).reshape(-1, 1)
            kdtree = sps.cKDTree(kd_input)

            # Compute Spacecraft Max Slew
            MAX_SLEW_TIME = spacecraft_max_slew(sc)

            # Parallelized Work Vector
            work = []

            # Use KD tree to iterate over all opportunities and look up nearest
            # neighbors to see if any collect computation needs to be done
            # We only check opportunities within < MAX_SLEW_TIME of the
            # current opportunity since everything beyond that must be fine

            # for topp1 in tlt_opportunities.keys():
            for opp1 in opportunities:
                # Get possible indexes for conflicting opportunities
                # This query can return opportunities earlier in time than the
                # Current, so we need to catch that later.
                # Add additional t_duration to search window
                pidx = kdtree.query_ball_point([(opp1.t_start - opportunities[0].t_start).total_seconds()],
                                               MAX_SLEW_TIME + opp1.t_duration)
                for idx_opp2 in pidx:
                    opp2 = tlt_opportunities[kd_input[idx_opp2][0]]
                    if opp2.t_start > opp1.t_start:
                        work.append((sc, opp1, opp2))

            # Perform access computation in parallel for all work
            results = pool.starmap(feasible_slew, work)

            for idx, res in enumerate(results):
                if res[0] == False:
                    sc_conflicts[work[idx][0].id].append([work[idx][1].id, work[idx][2].id])

            te = time.time()
            print(
                f'Finished computing agility conflicts for spacecraft {sc.id}. Found {len(sc_conflicts[sc.id]):d}/{int(num_sco * (num_sco - 1) / 2):d} conflicts. A {(len(sc_conflicts[sc.id])) / int(num_sco * (num_sco - 1) / 2) * 100:.2f}% conflict rate. Took {te - ts:.2f} seconds.',
                flush=True)

    tec = time.time()
    print(f'Finished computing all agility constraints. Took {tec - tsc:.2f} seconds.', flush=True)

    return list(itertools.chain.from_iterable(sc_conflicts.values()))


def collect_contact_conflicts(spacecraft: typing.List[bdm.Spacecraft],
                              contacts: typing.List[bdm.Collect],
                              collects: typing.List[bdm.Contact],
                              constraints_file: str = None, cpus: int = mp.cpu_count()):
    '''Apply agility constraints to schedule problem

    Args:
        spacecraft (:obj:`List[Spacecraft]`): Spacecraft object
        contacts (:obj:`Contact`): Contacts to compute conflicts for
        collects (:obj:`Collect`): Collects to compute conflicts for
        constraints_file (:obj:`str`): Base filepath for constraints

    Returns:
        List[Tuple[Opportunity, Opportunity]]: List of conflicting opportunity pairs
        int: Number of total possible conflicting opportunities
    '''

    # Resulting Conflicts Result
    sc_conflicts = {sc.id: [] for sc in spacecraft}
    conflicting_collects = []

    # Number of CPUs for run
    cpus = min(cpus, mp.cpu_count())

    tsc = time.time()

    # Set Process context
    mpctx = mp.get_context('forkserver')
    with mpctx.Pool(cpus) as pool:
        print(f'Using {cpus:d}/{mp.cpu_count():d} CPUs to compute contact-collect constraints')

        # Compute conflicts on per-spacecraft basis
        for sc in spacecraft:
            print(f'Computing contact-collect conflicts for spacecraft {sc.id:d}...', flush=True)
            ts = time.time()

            # Get Opportunities for conflict
            sc_collects = list(filter(lambda x: x.spacecraft_id == sc.id, collects))
            sc_collects.sort(key=lambda x: x.t_start)
            num_sco = len(sc_collects)

            # Create K-D Tree structure for quick lookup of possible conflicts

            # First create lookup of MJD -> Opportunity
            tlt_opportunities = {
                (c.t_start - sc_collects[0].t_start).total_seconds(): c for c in sc_collects
            }

            # Create KD-Tree indexiting Opportunities
            kd_input = np.array(list(tlt_opportunities.keys())).reshape(-1, 1)
            kdtree = sps.cKDTree(kd_input)

            # Compute Spacecraft Max Slew
            MAX_SLEW_TIME = spacecraft_max_slew(sc)

            # Parallelized Work Vector
            work = []

            # Use KD tree to iterate over all opportunities and look up nearest
            # neighbors to see if any collect computation needs to be done
            # We only check opportunities within < MAX_SLEW_TIME of the
            # current opportunity since everything beyond that must be fine

            # Get all collects that may interfere with a contacts
            for contact in contacts:
                # Get all collects that may interfere with the contact before
                pidx = kdtree.query_ball_point([(contact.t_start - sc_collects[0].t_start).total_seconds()],
                                               MAX_SLEW_TIME + contact.t_duration)
                for idx_col in pidx:
                    collect = tlt_opportunities[kd_input[idx_col][0]]
                    work.append((sc, collect, contact))

                # Get all collects that may interfere with the contact after
                pidx = kdtree.query_ball_point([(contact.t_end - sc_collects[0].t_start).total_seconds()],
                                               MAX_SLEW_TIME + contact.t_duration)
                for idx_col in pidx:
                    collect = tlt_opportunities[kd_input[idx_col][0]]
                    work.append((sc, contact, collect))

            # Perform access computation in parallel for all work
            results = pool.starmap(feasible_slew, work)

            for idx, res in enumerate(results):
                if res[0] == False:
                    if type(work[idx][1]) == bdm.Collect:
                        sc_conflicts[work[idx][0].id].append(work[idx][1])
                        conflicting_collects.append(work[idx][1])
                    else:
                        sc_conflicts[work[idx][0].id].append(work[idx][2])
                        conflicting_collects.append(work[idx][2])

            te = time.time()
            print(
                f'Finished computing contact-collect for spacecraft {sc.id}. Found {len(sc_conflicts[sc.id]):d} conflicts. Took {te - ts:.2f} seconds.',
                flush=True)

    tec = time.time()
    print(f'Finished computing all contact-collect constraints. Took {tec - tsc:.2f} seconds.', flush=True)

    return conflicting_collects


def compute_repeat_constraint(request, collects):
    request_collects = list(filter(lambda c: c.request_id == request.id, collects))
    return [r.id for r in request_collects]


def request_repeat_constraint(requests: typing.List[bdm.Request],
                              collects: typing.List[bdm.Collect], cpus: int = mp.cpu_count()):
    '''Apply minimum contact frequency constraint to require that contacts be
    taken at least at a certain frequency.

    Args:
        requests (:obj:`List[Request]`): List of all requests
        collects (:obj:`List[Collect]`): List of all collects

    Returns:
˝
    '''

    conflicts = []

    mpctx = mp.get_context('forkserver')
    with mpctx.Pool(cpus) as pool:
        print(f'Using {cpus:d}/{mp.cpu_count():d} CPUs to compute repetition constraints')

        work = []
        for req in requests:
            work.append((req, collects))

        results = pool.starmap(compute_repeat_constraint, work)

        for r in results:
            if len(r) > 1:
                conflicts.append(r)

    return conflicts


def create_metis_file(filename: str, collects, constraints):
    '''Create Metis file

    Args:
        filename (str): Metis graph file to create.
        collects (:obj:`List[Collects]`): All scheduling collects
        constraints (:obj:`Dict[str, List]`): Set of constraints to encode in graph.
    '''

    # Get all opportunities
    opportunities = collects
    opportunities.sort(key=lambda o: o.t_start)

    # Number of nodes
    N = len(opportunities)

    # Create forward and reverse mappings
    map_id_to_node = {}
    map_node_to_id = {}

    for idx, opp in enumerate(opportunities):
        map_node_to_id[idx + 1] = opp.id
        map_id_to_node[opp.id] = idx + 1

    # Create METIS graph object
    graph = {}
    for opp in opportunities:
        graph[map_id_to_node[opp.id]] = set()

    # Iterate over all conflict sets
    for constraint_set, cset in constraints.items():
        for constraint in cset:
            # Apply each constraint as an individual edge
            for opp_p_id in constraint:
                for opp_s_id in constraint:
                    if opp_p_id != opp_s_id:
                        graph[map_id_to_node[opp_p_id]].add(map_id_to_node[opp_s_id])
                        graph[map_id_to_node[opp_s_id]].add(map_id_to_node[opp_p_id])

    # Count Edges
    E = 0
    for node, edges in graph.items():
        E += len(edges)

    # METIS format double-lists edges so true number is E/2
    E = int(E / 2)

    # Convert edges to sorted lists
    for node in graph.keys():
        graph[node] = list(graph[node])
        graph[node].sort()

    # Save Object to file
    filename = pathlib.Path(filename)

    # Create Directory
    if not filename.parent.exists():
        filename.parent.mkdir(parents=True, exist_ok=True)

    # Ensure filename is propertly formatted
    if pathlib.Path(filename).suffix != '.metis':
        filename = filename.with_suffix('.metis')

    # Save to file
    with open(filename, 'w') as fp:
        # Write Header
        fp.write(f'{N} {E}\n')

        # Write Nodes/Edges
        for node, edges in graph.items():
            fp.write(" ".join(str(e) for e in edges) + "\n")

    # Return forward and reverse mappings
    return map_id_to_node, map_node_to_id

@click.command()
@click.option('--days', required=True, default=1, type=int, help='Number of days to generate.')
@click.option('--sats', required=True, default=1, type=int, help='Number of satellites')
@click.option('--look-angle-min', required=False, default=0.0, type=float, help='Minimum look angle')
@click.option('--look-angle-max', required=False, default=90.0, type=float, help='Maximum look angle')
def cmd(days, sats, look_angle_min, look_angle_max):

    t_start = time.time()

    # Load Data
    INPUT_DIR = f'outputs/worldcities_{days:d}_days_{sats:d}_sats'

    collects = []

    for scid in range(sats):
        with open(f'{INPUT_DIR}/collects_sc_{scid + 1}.json', 'r') as fp:
            collects.extend(json.load(fp))

    print(f'Loaded {len(collects)} collects')

    filtered_collects = list(filter(
        lambda x: x['access_properties']['look_angle_max'] <= look_angle_max and look_angle_min <=
                  x['access_properties']['look_angle_min'], collects))

    t_end = time.time()

    print(f'{len(filtered_collects)} collects after filtering. Took {t_end - t_start:.2f} seconds to filter.')

    # Load Spacecraft
    spacecraft_json = json.load(open(f'{INPUT_DIR}/spacecraft.json', 'r'))
    spacecraft = [bdm.Spacecraft(**scj) for scj in spacecraft_json]

    # Load Requests
    requests_json = json.load(open(f'{INPUT_DIR}/requests.json', 'r'))
    requests = [bdm.Request(**req) for req in requests_json]

    # Create Collect Objects
    collects = [bdm.Collect(**c) for c in filtered_collects]

    # Define Constraints File
    constraints_file = pathlib.Path(f'{INPUT_DIR}/constraints.pickle')

    # Compute Agility Constraints
    print(f'Computing agility constraints...', flush=True)
    ts = time.time()
    agility_conflicts = agility_constraint(spacecraft, collects, cpus=mp.cpu_count())
    te = time.time()

    print(f'Finished computing agility constraints. Took {te - ts:.2f} seconds.', flush=True)

    # Collect Repetition Constraints
    print(f'Computing repeition constraints...', flush=True)
    ts = time.time()
    repeat_conflicts = request_repeat_constraint(requests, collects, cpus=mp.cpu_count())
    te = time.time()

    print(
        f'Finished computing repetition constraints. Found {len(repeat_conflicts):d}/{len(requests):d} conflicts. A {len(repeat_conflicts) / len(requests) * 100:.2f}% conflict rate. Took {te - ts:.2f} seconds.',
        flush=True)

    # Save data File

    # Data structure for conlficts:
    conflicts = {
        'agility_conflicts': agility_conflicts,
        'repeat_conflicts': repeat_conflicts,
    }

    if not constraints_file.suffix == '.pickle':
        constraints_file = constraints_file.with_name(f'{constraints_file.stem}.pickle')

    print(f'Saving repeat and station conflicts to file "{constraints_file.resolve()}"...', flush=True)
    ts = time.time()
    pickle.dump(conflicts, open(constraints_file, 'wb'))
    te = time.time()
    print(f'Save complete. Took {te - ts:.2f} seconds.', flush=True)

    # Create Metis File

    print(f'Creating Metis File...', flush=True)
    ts = time.time()
    metis_file = f'{INPUT_DIR}/metis_graph.metis'
    _, _ = create_metis_file(metis_file, collects, conflicts)
    te = time.time()
    print(f'Finished creating Metis File. Took {te - ts:.2f} seconds.', flush=True)

    t_end = time.time()
    print(f'Completed. Took {t_end - t_start:.2f} seconds.')

if __name__ == '__main__':
    cmd()
