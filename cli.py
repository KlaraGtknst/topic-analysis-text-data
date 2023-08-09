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

    outfile_help = """file to write result to in json format.
        Supports token replacement for the current date and search_result token replacement.
        Date string: {%%Y-%%m-%%H-%%M}
        Token string: {$.key1.key2} 
        Random id: {+RRR}
        If multiple input files are present (depending on the subcommand) multiple outfiles are written,
        therefore the token support.
        """
    common_options = parser.add_argument_group("Common Options", description="Common options for subcommands")
    common_options.add_argument("-o", "--outfile", help=outfile_help)
    #common_options.add_argument("-j", "--n_jobs", type=int, default=1, help="number of jobs for parallel execution.")
    common_options.add_argument("-u", "--update", action='store_true', default=False, help="Update files/ inplace from path if the subcommand supports it.")
    common_options.add_argument("-i", "--infile", dest="filename", required=False, nargs='+',
                    help="input file(s).", metavar="FILE",
                    type=lambda x: is_valid_file(parser, x))
    common_options.add_argument("-d", "--directory", dest="directory", required=False,
                                help="directory, which contains input files.", metavar="DIRECTORY")

    return parser.parse_args()

def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg

def write_results(result, path, args):
    """
    Write results to screen or to file. Depends on args.

    If args.update is set the output is written to path.
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
    if args.update and path is not None:
        with open(path, 'w') as f:
            json.dump(result, f)
    elif args.outfile is not None:
        with open(args.outfile, 'w') as f:
            json.dump(result, f)
    else:
        print(result)

def get_filepath(args):
    if args.directory:
        file_paths = glob.glob(args.directory)
    elif args.filename:
        file_paths = args.filename
    else:
        print('Please specify a directory or a filename.')
        exit()
    return file_paths