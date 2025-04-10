import sys
import util
import data_analysis
from glob import glob


def info(path):
    n_sample, sign, density = \
        util.load(path, "meas_eqlt/n_sample", "meas_eqlt/sign",
                        "meas_eqlt/density", "meas_uneqlt/pair_bb")
    if n_sample.max() == 0:
        print("no data")
        return
    mask = (n_sample == n_sample.max())
    sign, density = sign[mask], density[mask]
    data_analysis.jackknife(pair_bb, sign)


def main(argv):
    #wildcard path expansion on Windows
    for path in argv[1:]:
        paths = sorted(glob(path))
        if len(paths) == 0:
            print("No paths matching:"+ path)
        else:
            for p in paths:
                print(p)
                info(p)

if __name__ == "__main__":
    main(sys.argv)