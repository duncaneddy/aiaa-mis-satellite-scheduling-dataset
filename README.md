# Satellite Scheduling World Cities Data Set
The Satellite Scheduling World Cities Data Set is the a set of cities treated as point locations used to simulate a 
set of image collection tasking requests for AIAA paper "A Maximum Independent Set Method for Scheduling Earth-Observing Satellite Constellations".
It provides an open reference and benchmark for the satellite task scheduling problem. This could also be considered as
a sparse Maximum Independent Set problem for a generic graph. The requests represent point collects, from which we can compute
multiple distinct collection opportunities. The tasking problem is then to select a subset of these collects that it is
possible for the spacecraft to feasibly collect in a given time period, subject to constraints on the spacecraft's
agility and constraints on only collecting a single collect per request (no duplication of effort).

The data set is hosted on both [Github](https://github.com/duncaneddy/aiaa-mis-satellite-scheduling-dataset) and 
[Zenodo](https://zenodo.org/records/10934666). The Github repository contains the original source data, the associated requests generated
from the source data, and scripts to reproduce the scenario files. Zenodo (DOI 10.5281/zenodo) hosts copies of the output
Metis graph files and collect data files. Due to the large size of produced files these are not included in the Github 
repository.

Information on the Metis graph file format can be found [here](https://people.sc.fsu.edu/~jburkardt/data/metis_graph/metis_graph.html).

## Reproduction

The following steps can be used to reproduce the data set used in the paper. They were tested on MacOS 14.4.1 with Python 3.12.2
on April 5, 2024.

1. Install python requirements
```bash
pip install -r requirements.txt
```

2. Generate requests from world cities data
```bash
python ./scripts/generate_worldcities_requests.py
```

3. Compute collects from requests
```bash
python ./scripts/generate_scenario.py --days 1 --sats 4 --deck worldcities --limit 10000
```

4. Generate Metis graph files from collects
```bash
python ./scripts/create_metis_file.py --sats 6 --days 1 --look-angle-max 55.0
```

## Notes

Please note that while the source data and generation methods are identical to the satellite
task planning paper it was created for. The specific generated problems do not exactly reproduce the
scenario in the paper. Since the original reproduction, updates in upstream software dependencies have changed
the output of the generation process (specifically, Earth orientaiton parameter handling libraries). This can be 
determined by considering the cardinality of the generated collect set.
However, these differences are generally small and since the constriant rate is similar, the results should be
comparable.

| Spacecraft Count | Original Scenario Collects | Repository Scenario Collects |
|------------------|----------------------------|-------------------------------|
| 4                | 59356                         | 59624                            |
| 6                | 90777                         | 91204                            |
| 12               | 180008                         | 180939                            |
| 24               | 359170                         | 361519                            |

This repository also adds additional scenarios for 1, 2, and 36 satellites. Note, the 
provided scenarios represent the largest 10,000 request data set. Should a smaller request set
be desired, the requests should be filtered to the top `x` request based on city population and any
collects not associated with those requests should be discarded.

## Acknowledgement

If this data set is used in your research, please cite the following paper:

[A Maximum Independent Set Method for Scheduling Earth-Observing Satellite Constellations](https://arc.aiaa.org/doi/abs/10.2514/1.A34931)

```bibtex
@article{eddy2021maximum,
  title={A Maximum Independent Set Method for Scheduling Earth-Observing Satellite Constellations},
  author={Eddy, Duncan and Kochenderfer, Mykel J},
  journal={Journal of Spacecraft and Rockets},
  volume={58},
  number={5},
  pages={1416--1429},
  year={2021},
  publisher={American Institute of Aeronautics and Astronautics}
}
```

## Licensing

The source of the world cities data is from the [simplemaps.com](https://simplemaps.com/data/world-cities) website,
licensed under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/)
with the specific license found at `./data/worldcities_license.txt`.