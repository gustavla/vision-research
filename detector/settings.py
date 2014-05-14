from __future__ import division, print_function, absolute_import
import ConfigParser
import re
import os

def load_settings(fp):
    conf = ConfigParser.ConfigParser(dict_type=dict)
    conf.readfp(fp)
    d = {}
    class Group(object):
        def __init__(self, name):
            self.name = name

    for section in conf.sections():
        d[section] = {}
        for k, v in conf.items(section):
            match = re.match(r"^\s*\[\s*([A-Za-z0-9_]+)\s*\]\s*$", v.split('#')[0])
            if match is not None:
                #print(section)
                #print(k)
                #print(match.group(1))
                #d[section][k] = Group d[match.group(1)]
                d[section][k] = Group(match.group(1))
            else:
                ev = eval(v)
                if isinstance(ev, str):
                    ev = os.path.expandvars(ev)
                d[section][k] = ev

    # Replace the Group objects with the referenced info
    for section, info in d.items():
        for k, v in info.items():
            if isinstance(v, Group):
                info[k] = d[v.name]
        
    return d

def change_settings(settings, settings_change_string):
    # Change settings
    for opt in filter(len, map(str.strip, settings_change_string.split(';'))):
        vv = opt.split('=')
        assert len(vv) == 2, vv
        k, v = map(str.strip, vv)
        if k in ('train_dir_seed', 'file', 'train_limit', 'inflate_bounding_box', 'inflate_feature_frame', 'scale_prior', 'indices_suppress_radius', 'spread_radii', 'scale_factor', 'scale_suppress_radius', 'selective_bkg', 'image_size', 'min_probability_mult_avg', 'penalty_parameter', 'num_mixtures', 'train_em_seed'):
            settings['detector'][k] = eval(v)
        elif k == 'scale_factor_invexp': 
            settings['detector']['scale_factor'] = 2**(1./float(v))
        elif k == 'seed':
            settings['oriented-parts']['seed'] = eval(v)
        elif k == 'oriented_parts_file':
            settings['oriented-parts']['file'] = eval(v)
        else:
            print('ERROR: Unhandled settings {}'.format(k))
    return settings

def argparse_settings(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('settings', metavar='<settings file>', type=argparse.FileType('r'), help='Filename of settings file')
    parser.add_argument('--modify-settings', type=str, default='', help='Overwrite settings')

    args = parser.parse_args()
    settings_file = args.settings
    d = load_settings(settings_file)
    d = change_settings(d, args.modify_settings)
    return d
