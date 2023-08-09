""" expresso command line interface """
import sys
import argparse
import numpy as np
import signal
import logging

from expresso.json import read_json, write_json
from expresso.util import Search, Dataset, Predictor, Evaluation
from expresso.param_dist import ParameterDistribution
from expresso.aggregate import merge_search
from expresso.model_selection import compute_scores, best_parameters

import json

def arguments():
    """
    Parse Arguments, display help, and exit, if invalid options are passed.

    Returns
    -------
    Namespace
        All found and parsed arguments.
    """
    desc = """ Expresso - a novelty detection experimental evaluation framework.
    
            _____
            \ E X) P R E S S O
             \_/´     N O V E L T Y   D E T E C T I O N 
           =======

           © 2020
             Christian Gruhl,
             Jörn Schmeißing
    """
    #========== C O M M O N ==========#
    parser = argparse.ArgumentParser(description=desc, add_help=True, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("-v", dest="verbosity", help="Verbosity, e.g., for debugging.", action="count")

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
    common_options.add_argument("-j", "--n_jobs", type=int, default=1, help="number of jobs for parallel execution.")
    common_options.add_argument("-u", "--update", action='store_true', default=False, help="Update files inplace if the subcommand supports it.")

    subparsers = parser.add_subparsers(title="Commands", dest="command", description="Valid actions to perform:")
    subparsers.required = True

    #========== S E A R C H ==========#
    search_description = """Perform a parameter search. 
"""
    search_parser = subparsers.add_parser("search", help="parameter search", description=search_description)
    search_parser.add_argument("-i", "--n_iter", type=int, default=10, help="number of search iterations to perform")
    search_parser.add_argument("-s", "--search", nargs='?', default=None, help="Specification of a search as json file (search_result), might be omitted if all overrides are set.")
    search_parser.add_argument("-n", "--nested_id", type=int, default=None, help="Use nested evaluation if data set supports it. The given parameter performs the search for the n-th split.")


    s_overrides = search_parser.add_argument_group("Search Overrides", description="Override the corresponding fields in the given search_result (-s).")
    s_overrides.add_argument("-m", "--method", choices=["RandomSearch"], help="Enforce given search method.")
    s_overrides.add_argument("-d", "--dataset", help="Use data set information from json file (dataset).")
    s_overrides.add_argument("-p", "--predictor", help="Use predictor information from json file (predictor).")
    s_overrides.add_argument("-b", "--params", help="Use parameter distribution from json file (params_dist).")

    #========== A G G R E G A T E ==========#
    agg_description = "Merge multiple search_results (in json format) into a single search_result."
    agg_parser = subparsers.add_parser("aggregate", help="aggregate search_results", description=agg_description)
    agg_parser.add_argument("search_result", nargs="+", help="search_results in json format to be merged.")

    #========== S C O R E ==========#
    score_description = ""
    score_parser = subparsers.add_parser("score", help="score search_results", description=score_description)
    score_parser.add_argument("-s","--scoring", help="json file with scoring information.")

    score_subparser = score_parser.add_subparsers(title="Score Commands", dest="score_action", description="Scoring actions to perform.")

    # C O M P U T E
    scomp_parser = score_subparser.add_parser("compute", help="compute new scores for search_results")
    scomp_parser.add_argument("-r","--recompute", action="store_true", default=False, help="recompute scores even if they already exist.")
    scomp_parser.add_argument("-f", "--scores", help="Additional scoring functions to compute as json list, e.g., '[\"f1\",\"accuracy\"]'.")
    scomp_parser.add_argument("search_result", nargs="+", help="search_results in json format to be scored.")
    # FIXME I'm not yet sure if this is possible to implement (in an easy fashion...).

    # B E S T
    sbest_parser = score_subparser.add_parser("best", help="Find best scores present in search_results")
    sbest_parser.add_argument("-f", "--best", help="Scoring to use to find the best result, as json list, e.g., '[\"f1\",\"accuracy\"]'. Precedes -s.")
    sbest_parser.add_argument("search_result", nargs="+", help="search_results in json format to be scored.")

    #========== V A L I D A T E ==========#
    valid_parser = subparsers.add_parser("evaluate", help="evaluate an experiment")
    valid_parser.add_argument("search_result", nargs="+", help="search_results in json format to be evaluated.")

    valid_parser.add_argument("-s", "--evaluation", help="evaluation information in json format (search_result).")
    valid_parser.add_argument("-n", "--nested_id", type=int, default=None, help="Use nested evaluation if data set supports it. The given parameter evaluates the n-th split.")
    valid_parser.add_argument("-e", "--force_eval",  action="store_true", default=False, help="Override the data sets .fold_params.search flag to False to enforce evaluation behaviour.")

    v_overrides = valid_parser.add_argument_group("Evaluation Overrides", description="Override the corresponding fields in the given search_results and evaluation (-s).")
    v_overrides.add_argument("-d", "--dataset", help="Use data set information from json file (dataset).")
    v_overrides.add_argument("-p", "--predictor", help="Use predictor information from json file (predictor).")

    return parser.parse_args()


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
        write_json(result, path)
    elif args.outfile is not None:
        write_json(result, args.outfile, True)
    else:
        print(result)

def override_components(target, args):
    """
    Handle -d, -p, and -b options and also -n.
    Loads, creates and overrides Dataset, Predictor, and PrameterDistribution.
    Also adds `nested_id` to Dataset (also when overriden) and handles the -e option for validation.

    Parameters
    ----------
    target : Search or Evaluation or other?
        The wrapper where the fields are overridden.
    args : Namespace
        Program arguments
     """
    if hasattr(args,'dataset') and args.dataset is not None:
        ds_dict = read_json(args.dataset)
        target.dataset = Dataset(**ds_dict)
    if hasattr(args, 'predictor') and args.predictor is not None:
        pd_dict = read_json(args.predictor)
        target.predictor = Predictor(**pd_dict)
    if hasattr(args, 'params') and args.params is not None:
        pr_dict = read_json(args.params)
        target.params_dist = ParameterDistribution(**pr_dict)
    if hasattr(args, 'nested_id') and args.nested_id is not None:
        target.dataset.args["nested_id"] = args.nested_id
    if hasattr(args, 'force_eval') and args.force_eval:
        target.dataset.args["fold_params"]["search"] = False

def parse_scores(score_str):
    """ 
    Parse score arguments, that is, -f for score compute and score best.

    Parameters
    ----------
    score_str : str
        JSON representation of a list to be parsed into strings.

    Raises
    ------
    ValueError
        If the string is malformed
    """
    try:
        return json.loads(score_str)
    except json.JSONDecodeError as e:
        raise ValueError("The format of the scorers is invalid! %s" % score_str) from e

def cmd_search(args):
    search_dict = {} if args.search is None else read_json(args.search)
    search = Search(**search_dict)

    if args.method is not None:
        # TODO this is not really nice
        search.kwargs["method"] = args.method
    
    override_components(search, args)

    search.runtime_params["n_iter"] = args.n_iter
    search.runtime_params["n_jobs"] = args.n_jobs

    args.shutdown_ctx.add(search)

    # allows the generation of search.json without fold results
    if args.n_iter == 0:
        # TODO still not nice, but does some weird internal loading - sorry.
        search.search
    else:
        search.fit()

    write_results(search, args.search, args)

def cmd_aggregate(args):
    sr = [read_json(f) for f in args.search_result]

    # merge in n*log(n)
    while len(sr) != 1:
        rem = [sr[-1]] if len(sr) % 2 == 1 else []
        # Add parallel execution? Probably to much overhead ...
        sr = [merge_search(sr[i-1], sr[i]) for i in np.arange(1,len(sr),2)] + rem

    write_results(sr[0], None, args)

def cmd_score(args):
    scoring_dict = {} if args.scoring is None else read_json(args.scoring)

    if args.score_action == "best":
        select_best = scoring_dict.get("select_best", []) if args.best is None else parse_scores(args.best)
        
        for f in args.search_result:
            search_result = read_json(f)
            search_result["best_params"] = best_parameters(search_result["results"], select_best)

            write_results(search_result, f, args)

    elif args.score_action == "compute":
        additonal_scores = set(parse_scores(args.scores)) if args.scores is not None else set()
        c_scores = set(scoring_dict.get("scoring", []))
        scoring = list(c_scores | additonal_scores)

        for f in args.search_result:
            search_result = read_json(f)
            updated_result = compute_scores(search_result, scorers=scoring, recompute=args.recompute, n_jobs=args.n_jobs)

            write_results(updated_result, f, args)

def cmd_evaluate(args):
    evaluation_dict = {} if args.evaluation is None else read_json(args.evaluation)

    for f in args.search_result:
        search_result = read_json(f)

        # set evaluation specific information
        search_result.update(evaluation_dict)

        evaluation = Evaluation(**search_result)
        override_components(evaluation, args)

        evaluation.runtime_params["n_jobs"] = args.n_jobs

        evaluation.fit()

        write_results(evaluation, f, args)

class ShutdownHandler:
    """ Handles signals and calls shutdown on registered objects."""

    def __init__(self):
        self.call_count = 0
        self.ctx = set()

    def __call__(self, signum, frame):
        logging.warn("Received signal %i" % signum)
        if self.call_count == 0:
            logging.warn("Initiating graceful shutdown. Next signal will cause termination.")
            self.call_count += 1
            print(self.ctx)
            for mod in self.ctx:
                if hasattr(mod, "shutdown") and callable(mod.shutdown):
                    logging.info("%s.shutdown()" % mod)
                    mod.shutdown()
                else:
                    logging.error("Module without shutdown method registered: %s" % mod)

        elif self.call_count > 0:
            logging.warn("TERMINATING")
            sys.exit()

def main():
    args = arguments()

    handler = ShutdownHandler()
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
    args.shutdown_ctx = handler.ctx

    {
    "search": cmd_search,
    "aggregate": cmd_aggregate,
    "score": cmd_score,
    "evaluate": cmd_evaluate
    }[args.command](args)
        



