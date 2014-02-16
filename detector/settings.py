
import ConfigParser
import re
import os

def load_settings(fp):
    conf = ConfigParser.ConfigParser(dict_type=dict)
    conf.readfp(fp)
    d = {}
    for section in conf.sections():
        d[section] = {}
        for k, v in conf.items(section):
            match = re.match(r"^\s*\[\s*([A-Za-z0-9_]+)\s*\]\s*$", v.split('#')[0])
            if match is not None:
                d[section][k] = d[match.group(1)]
            else:
                ev = eval(v)
                if isinstance(ev, str):
                    ev = os.path.expandvars(ev)
                d[section][k] = ev
    return d

def change_settings(settings, settings_change_string):
    # Change settings
    for opt in filter(len, map(str.strip, settings_change_string.split(';'))):
        vv = opt.split('=')
        assert len(vv) == 2
        k, v = map(str.strip, vv)
        if k in ('train_dir_seed',):
            settings['detector'][k] = eval(v)
    return settings

def argparse_settings(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('settings', metavar='<settings file>', type=argparse.FileType('r'), help='Filename of settings file')

    args = parser.parse_args()
    settings_file = args.settings
    return load_settings(settings_file)
