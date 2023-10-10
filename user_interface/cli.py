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

    parser.add_argument('file_to_run', nargs=1, help='filename with .py ending, whose main method should be run.', metavar="FILE",
                        type=lambda x: is_valid_py_file(parser, x))

    output_help = """path to directory to write the result to. 
    If not set, the result is printed to stdout."""
    common_options = parser.add_argument_group("Common Options", description="Common options for subcommands")
    common_options.add_argument("-o", "--output", help=output_help, dest="output_path", required=False, metavar="PATH",
                    type=lambda x: is_valid_file(parser, x))
    #common_options.add_argument("-j", "--n_jobs", type=int, default=1, help="number of jobs for parallel execution.")
    common_options.add_argument("-i", "--infile", dest="filename", required=False, nargs='+',
                    help="input file(s).", metavar="FILE",
                    type=lambda x: is_valid_file(parser, x))
    common_options.add_argument("-d", "--directory", dest="directory", required=False,
                                help="directory, which contains input files.", metavar="DIRECTORY")
    common_options.add_argument("-D", "--imgDir", dest="input_image_directory", required=False,
                                help="directory, which contains input iamge files. Used for, e.g., db_elasticsearch.py", metavar="DIRECTORY")
    common_options.add_argument("-n", "--number", dest="number", required=False, type=int,
                                help="custom number which may influence programm. For instance, if set for pdf_matrix.py, it determines number of dimensions.",)
    common_options.add_argument("-a", "--clientAddr", dest="client_addr", required=False,
                                help="address of elasticsearch client on server.")
    common_options.add_argument("-p", "--pools", dest="n_pools", required=False, type=int,
                                help="Number of Pools, i.e. how many CPUs present to parallize.")
    common_options.add_argument("-m", "--model_names", dest="model_names", required=False, nargs='+',
                    help="name of models to use. If not given, all are used", metavar="NAME", #action='append',
                    type=lambda x: is_valid_model_name(parser, x))

    return parser.parse_args()

def is_valid_py_file(parser, arg):
    if not arg.endswith('.py'):
        parser.error("The file %s is not a python file!" % arg)
    else:
        return arg
    
def is_valid_model_name(parser, arg):
    if not arg in ['universal', 'doc2vec', 'hugging', 'infer', 'ae', 'tfidf', 'none']:
            parser.error("The model name %s is not valid!" % arg)
    return arg

def get_model_names(args):
    if args.model_names:
        return args.model_names
    else:
        return ['universal', 'doc2vec', 'hugging', 'infer', 'ae', 'tfidf']

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

def get_filepath(args: argparse.Namespace, option: str = 'output') -> str:
    """
    :param args: The programm arguments.
    :param option: 'output' or 'image'

    Get the filepath to the output or image input from the arguments.
    """
    arg = args.output_path if option=='output' else args.input_image_directory
    if arg:
        if (not (arg).endswith('/')) and (not '.' in arg):
            return arg + '/'
        return arg
    else:
        return ''

def get_input_filepath(args: argparse.Namespace) -> list:
    """
    Get a list of input filepaths from the arguments.
    """
    if args.directory:
        if args.directory.endswith('*.pdf'):    # one directory
            file_paths = glob.glob(args.directory)
        else:   # recursive search through multiple directories
            file_paths = [os.path.join(r,file) for r,d,f in os.walk(args.directory) for file in f if file.endswith('.pdf')]
            
    elif args.filename:
        file_paths = args.filename
    else:
        print('Please specify a directory or a filename.')
        exit()
    return file_paths