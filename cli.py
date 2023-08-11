import argparse
import glob
import json
import os

'''Code for argument parsing when running the script (on a server)'''
def arguments():
    """
    Parse Arguments, display help, and exit, if invalid options are passed.

    Returns
    -------
    Namespace
        All found and parsed arguments.

    The majority of this function is intellectual property of Christian Gruhl and Jörn Schmeißing.
    Cf. for more information:
    https://stackoverflow.com/questions/11540854/file-as-command-line-argument-for-argparse-error-message-if-argument-is-not-va
    https://docs.python.org/3/library/argparse.html
    """
    desc = """Topic Modelling Bachelor Thesis
           © 2020 - 2023
             Christian Gruhl,
             Jörn Schmeißing,
             Klara Gutekunst
    """
    #========== C O M M O N ==========#
    parser = argparse.ArgumentParser(description=desc, add_help=True, formatter_class=argparse.RawTextHelpFormatter)

    output_help = """path to directory to write the result to. 
    If not set, the result is printed to stdout."""
    common_options = parser.add_argument_group("Common Options", description="Common options for subcommands")
    common_options.add_argument("-o", "--output", help=output_help, dest="output_path", required=False, nargs='*', metavar="PATH",
                    type=lambda x: is_valid_file(parser, x))
    #common_options.add_argument("-j", "--n_jobs", type=int, default=1, help="number of jobs for parallel execution.")
    common_options.add_argument("-i", "--infile", dest="filename", required=False, nargs='+',
                    help="input file(s).", metavar="FILE",
                    type=lambda x: is_valid_file(parser, x))
    common_options.add_argument("-d", "--directory", dest="directory", required=False,
                                help="directory, which contains input files.", metavar="DIRECTORY")
    common_options.add_argument("-c", "--cluster", dest="cluster", required=False, default=False,
                                help="whether the script runs on the IES cluster. Necessary to configure path to poppler for run_pdf2image.",)
    common_options.add_argument("-n", "--number", dest="number", required=False, type=int,
                                help="custom number which may influence programm. For instance, if set for pdf_matrix, it determines number of dimensions.",)


    return parser.parse_args()

def is_valid_file(parser, arg):
    if '*' in arg:  # wildcard
        #print('1' + arg)
        #print('2' + arg.split('*')[0])
        return os.path.exists(arg.split('*')[0])
    elif not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg

def write_results(result, path, args):
    """
    Write results to screen or to file. Depends on args.

    If args.outpath is set the output is written to path.
    If args.outfile is set the output is written to outfile.
    If neither is set, the result is printed.

    Parameters
    ----------
    result : dict
        Object to write or print.
    path : str
        Target file to write to.
    args : Namespace
        The programm arguments.
     """
    if args.outpath and path is not None:
        with open(path, 'w') as f:
            json.dump(result, f)
    elif args.outfile is not None:
        with open(args.outfile, 'w') as f:
            json.dump(result, f)
    else:
        print(result)

def get_output_filepath(args: argparse.Namespace) -> str:
    """
    Get the output filepath from the arguments.
    """
    if args.output_path:
        if (not (args.output_path[0]).endswith('/')) and (not '.' in args.output_path[0]):
            return args.output_path[0] + '/'
        return args.output_path[0]
    else:
        return None

def get_input_filepath(args: argparse.Namespace) -> list:
    """
    Get a list of input filepaths from the arguments.
    """
    if args.directory:
        file_paths = glob.glob(args.directory)
    elif args.filename:
        file_paths = args.filename
    else:
        print('Please specify a directory or a filename.')
        exit()
    return file_paths