# This example is a proof of principle for how NetLogo models can be controlled using pyNetLogo and the ema_workbench.
# Note that this example uses the NetLogo 6 version of the predator prey model that comes with NetLogo.
# If you are using NetLogo 5, replace the model file with the one that comes with NetLogo.

from __future__ import unicode_literals, absolute_import, division, print_function
import sys
sys.path.append('/home/kzx0010/EMAworkbench')
from ema_workbench.connectors.netlogo import NetLogoModel

from ema_workbench import (RealParameter, IntegerParameter, ema_logging, TimeSeriesOutcome, MultiprocessingEvaluator,
                           CategoricalParameter)

import pandas as pd
import numpy as np
from ema_workbench import save_results, ema_logging, load_results


def run_model(number_experiments=0, run_length=0, replications=0, processes=1):
    # run experiment
    # number_experiments is how many samplings are done
    # run_length is number of ticks
    # replications is repetitions of same model instance
    # processes is how many processes your computer can support -- paralleling

    # perform experiments
    n = number_experiments

    # turn on logging
    ema_logging.log_to_stderr(ema_logging.INFO)

    model = NetLogoModel('plume',
                         wd="/home/kzx0010/Plume-Model",
                         model_file="plume_original.nlogo", gui=False, netlogo_version='6')

    # number of ticks
    model.run_length = run_length

    # repeat model instance 'replications' times
    model.replications = replications

    # TODO: fix params
    model.uncertainties = [IntegerParameter("population", 50, 100),
                           RealParameter("vision", 50, 100),
                           RealParameter("minimum-separation", 0, 5),
                           RealParameter("max-align-turn", 0, 20),
                           RealParameter("max-cohere-turn", 0, 10),
                           RealParameter("max-separate-turn", 0, 20),
                           RealParameter("plume-spread", 0, 1),
                           RealParameter("coverage-data-decay", 1, 60)]
                           # ,
                           # CategoricalParameter("search-algorithm", ["default", "random", "symmetric"],
                           #                      default="symmetric")]

    model.outcomes = [TimeSeriesOutcome('coverage-std')]
                      # TimeSeriesOutcome('TIME'),
                      # TimeSeriesOutcome('first-detection'),
                      # TimeSeriesOutcome('last-detection'),
                      # TimeSeriesOutcome('total-coverage')]

    with MultiprocessingEvaluator(model, n_processes=processes) as evaluator:
        return evaluator.perform_experiments(n)


def save_to_1d_csv(results_output):
    experiments, outcomes = results_output

    for out in outcomes:
        for experiment in outcomes[out]:
            for replication in range(len(experiment)):
                data = pd.DataFrame(experiment[replication])
                data.to_csv('./models/plumeNetlogo/csv/' + out + 'rep' + str(replication) + '.csv', sep=',')


# ** Advanced Analysis **

def pairs_plotting(results_output, group_by=None, legend=False):
    from ema_workbench.analysis import pairs_plotting
    import matplotlib.pyplot as plt
    fig, axes = pairs_plotting.pairs_scatter(results_output, group_by, legend)
    fig.set_size_inches(8, 8)
    plt.show()


# custom plotting of an outcome
def plot_outcome(results_output):
    import matplotlib.pyplot as plt
    experiments, outcomes = results_output

    for an_outcome in results_output[1]:
        real_outcome = outcomes[an_outcome]
        ticks = real_outcome.shape[2]
        x = np.linspace(0, ticks, num=ticks)
        plt.plot(x, real_outcome[0][0])
        plt.xlabel("Ticks")
        plt.ylabel(an_outcome)
        plt.show()


def drop_fields(_experiments, fields):
    from numpy.lib import recfunctions as rf
    t = _experiments
    for field in fields:
        t = rf.drop_fields(t, field, True)  # asrecarray=
    return t


def feature_scoring(results_output):
    from ema_workbench.analysis import feature_scoring
    import matplotlib.pyplot as plt
    import seaborn as sns

    experiments, outcomes = results_output
    x = drop_fields(experiments, ['model', 'policy'])
    end_states = {k: v[:, 1, -1] for k, v in outcomes.items()}
    print(end_states)
    fs = feature_scoring.get_feature_scores_all(x, end_states)
    sns.heatmap(fs, cmap='viridis', annot=True)
    plt.show()


def regional_sensitivity_analysis(results_output, performance_key):
    from ema_workbench.analysis import regional_sa
    import seaborn as sns
    import matplotlib.pyplot as plt

    experiments, outcomes = results_output
    x = drop_fields(experiments, ['model', 'policy'])
    y = outcomes[performance_key][:, -1, -1] < 1000
    regional_sa.plot_cdfs(x, y)
    sns.despine()
    sns.set_style('white')
    plt.show()


def prim_analysis(results_output, performance_key):
    from ema_workbench.analysis import prim
    import matplotlib.pyplot as plt

    experiments, outcomes = results_output
    #x = drop_fields(experiments, ['model', 'policy'])
    #y = outcomes[performance_key][:, -1, -1] < 0.8

    #print(y)

    prim_alg = prim.Prim([1,2,3,4,5,6,7,8], [9,8,7,6,5,4,3,2,1], threshold=0.8)
    box1 = prim_alg.find_box()
    box1.show_tradeoff()
    plt.show()


def save(results_output):
    # added by bcr
    #save_to_1d_csv(results_output)
    # ** SAVING TO CSV **
    # EMA Workbench function
    save_results(results_output, '1000_plume_model.tar.gz')


def do_prim():
    import matplotlib.pyplot as plt
    import ema_workbench.analysis.prim as prim

    def classify(data):
        # get the output for deceased population
        result = data['deceased population region 1']
        # print(result.shape)
        # print(result)
        # make an empty array of length equal to number of cases
        classes = np.zeros(result.shape[0])
        # if deceased population is higher then 1.000.000 people, classify as 1
        classes[result[:, -1] > 1000000] = 1

        print(sum(classes))

        print(result.shape)
        print(classes.shape)

        return classes

    def classify2(data):
        # result = data['total-coverage']
        result = data['total-coverage']

        th = 3434

        # print(result)
        # print(result.shape)
        # print(result)
        classes = np.zeros(result.shape[0])
        classes[result[4,1,:] > th] = 1

        print(sum(classes))
        print(result.shape)
        print(classes.shape)

        print(classes)

        #print(result.shape)
        #print(result[4,1,:])
        #print(np.max(result[4,1,:]))
        #print(result[4, 1, :].shape)
        return classes

    # load data
    #fn = r'./1000 scenarios 5 policies.tar.gz'
    fn = r'./data/1000 flu cases no policy.tar.gz'
    results = load_results(fn)

    # perform prim on modified results tuple
    prim_obj = prim.setup_prim(results, classify, threshold=0.0, threshold_type=1)

    box_1 = prim_obj.find_box()
    box_1.show_ppt()
    box_1.show_tradeoff()
    box_1.inspect(5, style='graph', boxlim_formatter="{: .2f}")
    box_1.inspect(5)
    box_1.select(5)
    box_1.write_ppt_to_stdout()
    box_1.show_pairs_scatter()
    prim_obj.display_boxes()
    plt.show()


if __name__ == '__main__':

    results = run_model(number_experiments=1, run_length=200, replications=1, processes=6)

    #results = ([],[])
    #prim_analysis(results, 'total-coverage')

    #do_prim()





    # ** Working **
    save(results)
    # regional_sensitivity_analysis(results, 'total-coverage')
    # feature_scoring(results)
    # pairs_plotting(results, group_by=None, legend=False)
    # plot_outcome(results)


    # ** NOT WORKING **
    # prim_analysis(results2)
    # dimensional_stacking(results)
    # dimensionalStacking(results, 'total-coverage')





# def dimensional_stacking(results_output):
#     from ema_workbench.analysis import dimensional_stacking
#     import matplotlib.pylab as plt
#
#     experiments, outcomes = results_output
#
#     x = experiments
#     y = outcomes['coverage'] < 0.8
#     dimensional_stacking.create_pivot_plot(x, y, 2, nbins=3)
#     plt.show()
#
#
# def dimensionalStacking(results, performanceKey):
#     from ema_workbench.analysis import dimensional_stacking
#     from numpy.lib import recfunctions as rf
#     import matplotlib.pylab as plt
#
#     experiments, outcomes = results
#     temp = rf.drop_fields(experiments, 'model', asrecarray=True)
#     x = rf.drop_fields(temp, 'policy', asrecarray=True)
#     outcome = outcomes[performanceKey]
#     endState = outcome[:, -1, -1]
#     y = endState < 50
#     dimensional_stacking.create_pivot_plot(x, y, 2, nbins=3)
#     plt.show()
