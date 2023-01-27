# We still need to support Python 3.6, so stick to modules in its standard lib
import argparse
import configparser
from pathlib import Path

init_parser = argparse.ArgumentParser(add_help=False)
parser = argparse.ArgumentParser(
    description='MRSNet',
    epilog='MRSNet homepage: https://github.com/Shawn-S-ong/MRS-Script/'
)
# Add sections of config file to query below
def readConfig(*config_sections):
    init_parser.add_argument(
        '-c', '--config',
        default=['MRSNet.conf', 'MRSNet.conf.local'],
        nargs='+',
        help='Path to configuration file [default: MRSNet.conf, MRSNet.conf.local]'
    )
    configfile_arg, argv = init_parser.parse_known_args()
    if configfile_arg.config:
        config = configparser.SafeConfigParser()
        config.read(configfile_arg.config)
        print(configfile_arg.config)
        config_dict = dict()
        for config_section in config_sections:
            config_dict.update(dict(config.items(config_section)))
        parser.add_argument(
            '-c', '--config',
            default='MRSNet.conf',
            nargs='+',
            help='Specify a custom configuration file [default: MRSNet.conf, MRSNet.conf.local]'
        )
        parser.set_defaults(**config_dict)
	
    #return(argv)

# Currently, we need to add a lot of boilerplate code to each file.
# TODO: Explore options to simplify
#       (such as a config section per script/module, instantiated here.)
