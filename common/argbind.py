"""
Utilities for binding function arguments to 
the command line under different scopes as needed.
Read on to see how to bind a function.

REQUIREMENTS
------------
numpydoc
pyyaml-

HOW TO BIND A FUNCTION
----------------------
Decorate the function with bind_to_parser,
adding it to PARSE_FUNCS. The argument
parser inspects each function in PARSE_FUNCS
and adds it to the argument flags. For example:

    @bind_to_parser('train', 'val')
    def autoclip(percentile : float = 10.0):
        print(f'Called autoclip with percentile={percentile}')

This functions arguments are available at:

    python example.py --autoclip.percentile=N

The function arguments must be annotated with
their type. Only keyword arguments are included
in the ArgumentParser.

You can optionally define additional patterns to match
for different scopes. This will use the arguments
given on that pattern when the scope is set to that
pattern. The argument is available on command line at
--pattern/func.kwarg. The patterns used were 'train'
and 'val' so the additional arguments are also
available for binding:

    python example.py \ 
        --autoclip.percentile=100 
        --train/autoclip.percentile=1
        --val/autoclip.percentile=5

Use with the corresponding code:
    >>> # above this, parse the args
    >>> with scope(args):
    >>>     autoclip() # prints 100
    >>> with scope(args, 'train'):
    >>>     autoclip() # prints 1
    >>> with scope(args, 'val'):
    >>>     autoclip() # prints 5

SAVING AND LOADING ARGUMENTS VIA .YML FILES
-------------------------------------------
Instead of memorizing complex command line arguments for 
different experiments or configurations, one can save 
and load args via .yml files. To use this, use the 
--args.save and --args.load arguments to your script.
The first will save the arguments that were used in
the run (including all default values for all functions) 
to a .yml file, for example:

    stages.run:
    - TRAIN
    - EVALUATE
    - ANALYZE

Then, when you use --args.load with the path to the saved
file, when the stages function is called, it will run 
TRAIN, then EVALUATE, then ANALYZE. If you edit it to 
look like this:

    stages.run:
    - TRAIN
    - EVALUATE

then only the first two stages will be run. The .yml files are
saved with a flat structure (no nesting). If you provide command line
arguments, then only the parameters that are on the command line
override those in the .yml file. For example:

    python -m src.exp --args.load args.yml --stages.run TRAIN

will only run the TRAIN stage, even if args.yml file looks like
above. 

NOTE: If a boolean is flipped to True in the .yml file, there's no
way to override it from the command line. If you want a flag to
be flippable, make the argument an int instead of a bool and use
0 and 1 for True and False. Then you can override from command
line like --func.arg 0 or --func.arg 1.
"""

import inspect
from contextlib import contextmanager
import argparse
from typing import List, Dict
from numpydoc.docscrape import FunctionDoc
import textwrap
import yaml
import sys
import os
from pathlib import Path
import ast

PARSE_FUNCS = {}
ARGS = {}
USED_ARGS = {}
PATTERN = None
DEBUG = False
HELP_WIDTH = 60

@contextmanager
def scope(parsed_args, pattern=''):
    """
    Context manager to put parsed arguments into 
    a state.
    """
    parsed_args = parsed_args.copy()
    remove_keys = []
    matched = {}

    global ARGS
    global PATTERN

    if parsed_args is None:
        parsed_args = ARGS

    old_args = ARGS
    old_pattern = PATTERN

    for key in parsed_args:
        if '/' in key:
            if key.split('/')[0] == pattern:
                matched[key.split('/')[-1]] = parsed_args[key]
            remove_keys.append(key)
    
    parsed_args.update(matched)
    for key in remove_keys:
        parsed_args.pop(key)
    ARGS = parsed_args
    PATTERN = pattern
    yield

    ARGS = old_args
    PATTERN = old_pattern

def copy_doc(doc_func):
    def decorator(func):
        desc = func.__doc__
        new_doc = doc_func.__doc__
        if desc is not None:
            new_doc = new_doc.replace(
                "[DESCRIPTION]",
                desc
            )
        func.__doc__ = new_doc
        return func
    return decorator

def bind_to_parser(*patterns, no_global=False):
    """
    Wrap the function so it looks in ARGS (managed 
    by the scope context manager) for keyword 
    arguments.
    """

    def decorator(func):
        PARSE_FUNCS[func.__name__] = (func, patterns, no_global)
        def cmd_func(*args, **kwargs):
            prefix = func.__name__
            sig = inspect.signature(func)
            cmd_kwargs = {}

            for key, val in sig.parameters.items():
                arg_type = val.annotation
                arg_val = val.default
                if arg_val is not inspect.Parameter.empty:
                    arg_name = f'{prefix}.{key}'
                    if arg_name in ARGS:
                        cmd_kwargs[key] = ARGS[arg_name]
                        use_key = arg_name
                        if PATTERN:
                            use_key = f'{PATTERN}/{use_key}'
                        USED_ARGS[use_key] = ARGS[arg_name]

            kwargs.update(cmd_kwargs)
            if 'args.debug' not in ARGS: ARGS['args.debug'] = False
            if ARGS['args.debug'] or DEBUG:
                _prefix = f"{PATTERN}/{prefix}" if PATTERN else prefix
                print(f"{_prefix} <- {parse_dict_to_str(kwargs)}")

            return func(*args, **kwargs)
        return cmd_func
    
    return decorator

def parse_dict_to_str(x):
    return ', '.join([f'{k}={v}' for k, v in x.items()])

def get_used_args():
    """
    Gets the args that have been used so far
    by the script (e.g. their function they target
    was actually called).
    """
    return USED_ARGS

def dump_args(args, output_path):
    """
    Dumps the provided arguments to a
    file.
    """
    path = Path(output_path)
    os.makedirs(path.parent, exist_ok=True)
    with open(path, 'w') as f:
        yaml.Dumper.ignore_aliases = lambda *args : True
        x = yaml.dump(args, Dumper=yaml.Dumper)
        prev_line = None
        output = []
        for line in x.split('\n'):
            cur_line = line.split('.')[0].strip()
            if not cur_line.startswith('-'):
                if cur_line != prev_line and prev_line:
                    line = f'\n{line}'
                prev_line = line.split('.')[0].strip()
            output.append(line)
        f.write('\n'.join(output))

def load_args(input_path):
    """
    Loads arguments from a given input path. If $include key is in
    the args, you can include other y
    """
    with open(input_path, 'r') as f:
        data = yaml.load(f, Loader=yaml.Loader)
    
    if '$include' in data:
        include_args = {}
        for include_file in data['$include']:
            with open(include_file, 'r') as f:
                _include_args = yaml.load(f, Loader=yaml.Loader)
            include_args.update(_include_args)
        include_args.update(data)
        data = include_args
        
    if 'args.debug' not in data:
        data['args.debug'] = DEBUG
    return data

class str_to_list():
    def __init__(self, _type):
        self._type = _type
    def __call__(self, values):
        _values = values.split(' ')
        _values = [self._type(v) for v in _values]
        return _values

class str_to_dict():
    def __init__(self):
        pass

    def _guess_type(self, s):
        try:
            value = ast.literal_eval(s)
        except ValueError:
            return s
        else:
            return value

    def __call__(self, values):
        values = values.split(' ')
        _values = {}

        for elem in values:
            key, val = elem.split('=', 1)
            key = self._guess_type(key)
            val = self._guess_type(val)
            _values[key] = val

        return _values

def parse_args():
    """
    Goes through all detected functions that are
    bound and adds them to the argument parser,
    along with their scopes. Then parses the
    command line and returns a dictionary.

    Parameters
    ----------
    output_path : str, optional
        Saves the args that are parsed to the given 
        file for running the command again, by default None
    """
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter
    )

    p.add_argument('--args.save', type=str, required=False, 
        help="Path to save all arguments used to run script to.")
    p.add_argument('--args.load', type=str, required=False,
        help="Path to load arguments from, stored as a .yml file.")
    p.add_argument('--args.debug', type=int, required=False, default=0, 
        help="Print arguments as they are passed to each function.")

    # Add kwargs from function to parser
    for func_name in PARSE_FUNCS:
        func, patterns, no_global = PARSE_FUNCS[func_name]
        sig = inspect.signature(func)
        prefix = func.__name__

        docstring = FunctionDoc(func)
        parameter_help = docstring['Parameters']
        parameter_help = {
            x.name: ' '.join(x.desc) for x in parameter_help
        }

        f = p.add_argument_group(
            title=f"Generated arguments for function {prefix}",
        )

        for key, val in sig.parameters.items():
            arg_type = val.annotation
            arg_val = val.default

            if arg_val is not inspect.Parameter.empty:
                arg_names = []
                arg_help = {}
                help_text = ''
                if key in parameter_help:
                    help_text = textwrap.fill(parameter_help[key], width=HELP_WIDTH)
                if not no_global:
                    arg_names.append(f'--{prefix}.{key}')
                    arg_help[arg_names[-1]] = help_text
                for pattern in patterns:
                    arg_names.append(f'--{pattern}/{prefix}.{key}')
                    arg_help[arg_names[-1]] = argparse.SUPPRESS
                for arg_name in arg_names:
                    inner_types = [str, int, float, bool]
                    list_types = [List[x] for x in inner_types]

                    if arg_type is bool:
                        f.add_argument(arg_name, action='store_true', 
                            help=arg_help[arg_name])
                    elif arg_type in list_types:
                        _type = inner_types[list_types.index(arg_type)]
                        f.add_argument(arg_name, type=str_to_list(_type), 
                            default=arg_val, help=arg_help[arg_name])
                    elif arg_type is Dict:
                        f.add_argument(arg_name, type=str_to_dict(), 
                            default=arg_val, help=arg_help[arg_name])
                    else:
                        f.add_argument(arg_name, type=arg_type, 
                            default=arg_val, help=arg_help[arg_name])
            
        desc = docstring['Summary']
        desc = ' '.join(desc)

        if patterns:
            desc += (
                f" Additional scope patterns: {', '.join(list(patterns))}. "
                "Use these by prefacing any of the args below with one "
                "of these patterns. For example: "
                f"--{patterns[0]}/{prefix}.{key} VALUE."
            )

        desc = textwrap.fill(desc, width=HELP_WIDTH)
        f.description = desc
    
    used_args = [x.replace('--', '') for x in sys.argv if x.startswith('--')]
    used_args.extend(['args.save', 'args.load'])

    args = vars(p.parse_args())
    load_args_path = args.pop('args.load')
    save_args_path = args.pop('args.save')
    debug_args = args.pop('args.debug')
    
    pattern_keys = [key for key in args if '/' in key]
    top_level_args =[key for key in args if '/' not in key]

    for key in pattern_keys:
        # If the top-level arguments were altered but the ones
        # in patterns were not, change the scoped ones to
        # match the top-level (inherit arguments from top-level).
        pattern, arg_name = key.split('/')
        if key not in used_args:
            args[key] = args[arg_name]
    
    if load_args_path:
        loaded_args = load_args(load_args_path)
        # Overwrite defaults with things in loaded arguments.
        # except for things that came from the command line.
        for key in loaded_args:
            if key not in used_args:
                args[key] = loaded_args[key]
        for key in pattern_keys:
            pattern, arg_name = key.split('/')
            if key not in loaded_args and key not in used_args:
                if arg_name in loaded_args:
                    args[key] = args[arg_name]
                
    for key in top_level_args:
        if key in used_args:
            for pattern_key in pattern_keys:
                pattern, arg_name = pattern_key.split('/')
                if key == arg_name and pattern_key not in used_args:
                    args[pattern_key] = args[key]

    if save_args_path:
        dump_args(args, save_args_path)

    # Put them back in case the script wants to use them
    args['args.load'] = load_args_path
    args['args.save'] = save_args_path
    args['args.debug'] = debug_args
    
    return args
