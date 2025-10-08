import numpy as np
import matplotlib.pyplot as plt

from ..profile_data import ProfileData, gen_data_from_config

def main():
    import argparse
    import toml
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.toml', help='Configuration file (.toml)')
    parser.add_argument('-p', '--plot', action='store_true', help='Display a plot of the data')
    parser.add_argument('-f', '--force-write', action='store_true', help='Overwrite datafile if it exists')
    parser.add_argument('datafile', type=str, help='Data file (.npz) to output or plot')
    args = parser.parse_args()
    config = toml.load(args.config)
    
    try:
        data = ProfileData.from_npz(args.datafile)
    except FileNotFoundError:
        args.force_write = True
    
    if args.force_write:
        print(f'Writing output to {args.datafile}...')
        data = gen_data_from_config(config)
        data.save_npz(args.datafile)
    
    if args.plot:
        for profile in data.profiles[:8]:
            plt.plot(data.phase, profile)
        plt.xlabel('Phase (cycles)')
        plt.ylabel('Intensity (rel. peak)')
        plt.show()

        pc = data.plot()
        plt.colorbar(pc)
        plt.show()

if __name__ == '__main__':
    main()
