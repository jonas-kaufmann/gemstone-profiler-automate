#!/usr/bin/env python

# Matthew J. Walker
# Created: 5 June 2017

import os
import run_experiment
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import pandas as pd
import numpy as np

REG_MAX = 2**32


def get_pmc_rate(current, last, time_period_s):
    if current >= last:
        return (current - last) / time_period_s
    else:
        return ((REG_MAX - last) + current) / time_period_s


# This function is to deal with the uneven sample period of temperature
def mean_temperature_interpolated(milli_list, t_list):
    # 1. Interpolate
    import numpy as np

    temp_for_each_milli = []
    milli_list_f = [float(x) for x in milli_list]
    eval_at = [float(i) for i in range(milli_list[0], milli_list[-1], 1)]
    temp_for_each_milli = np.interp(eval_at, milli_list_f, t_list)
    return np.mean(temp_for_each_milli)


# This method derives basic stats from post-processed data
# E.g. adding the sum of the counts and the average of the rates
def add_cluster_sum_and_average_counts(input_df):
    # assumption: clusters have 4 CPUs each: BAD ASSUMPTION! Fix this later
    # (proposed solution: get experimental platform software to dervice this
    # and encode into headers). overhead only in pmc-get-header
    # pmc-setup dervices which cores need to report frequency

    # this bit of code assumes that CPUs of the same type (e.g. A7) are in the
    # same cluster (not necessarily true, depending on the device)

    # 1. Find all CPUs of the same type (e.g. A7 or A15)
    # 2. Find the PMC events (assumption that CPUs of the same type measure the
    # same counters). This assumption is valid for the experimental platform sw
    # 3. Create columns 'Total Cortex-A17 Counts' and 'Average Cortex-A7 Rate'
    # for each event and cycle count
    pass


def combine_event_and_log_vals_float(
    start_row, end_row, continuous_df, col_header
):
    vals = []
    vals.append(float(start_row[col_header]))
    for i in range(0, len(continuous_df.index)):
        vals.append(float(continuous_df[col_header].iloc[i]))
    vals.append(float(end_row[col_header]))
    return vals


# This method accepts a list of the raw PMC observations
# and finds the different between the first value and the last
# (checking for overflows)
def get_pmc_diff_from_list(pmc_vals):
    overflows = 0
    for i in range(1, len(pmc_vals)):
        if pmc_vals[i] < pmc_vals[i - 1]:
            # overflow!
            overflows += 1
    if not overflows:
        return pmc_vals[-1] - pmc_vals[0]
    else:
        return REG_MAX - pmc_vals[0] + pmc_vals[-1] + REG_MAX * (overflows - 1)


def postprocess_experiment(
    experiment_dir, output_filepath, temperature_file=None
):
    import numpy as np

    pmc_events_log_df = pd.read_csv(
        os.path.join(experiment_dir, run_experiment.FILENAME_PMC_EVENTS_LOG),
        sep="\t",
    )
    pmc_continuous_log_df = pd.read_csv(
        os.path.join(
            experiment_dir, run_experiment.FILENAME_PMC_CONTINUOUS_LOG
        ),
        sep="\t",
    )

    # get core mask:
    # TODO improve this
    core_mask = ""
    if os.path.isfile(
        os.path.join(
            experiment_dir, "../", run_experiment.FILENAME_CORE_MASK_OUT
        )
    ):
        with open(
            os.path.join(
                experiment_dir, "..", run_experiment.FILENAME_CORE_MASK_OUT
            ),
            "r",
        ) as f:
            core_mask = f.read().strip()
        f.closed
    else:
        with open(
            os.path.join(experiment_dir, run_experiment.FILENAME_ARGS), "r"
        ) as f:
            text = f.read()
            if text.find("0,1,2,3") > -1:
                core_mask = "0,1,2,3"
            elif text.find("4,5,6,7") > -1:
                core_mask = "4,5,6,7"
            else:
                raise ValueError("Can't find core mask!")
        f.closed
    # count number of overflows
    # need the workload names!
    workloads_temp_df = pmc_events_log_df[
        pmc_events_log_df["label"].str.contains(" start")
    ]
    # identify PMC columns
    all_cols = [i for i in pmc_events_log_df.columns.values]
    pmc_cols = [
        i
        for i in pmc_events_log_df.columns.values
        if i.find("cntr") > -1 or i.find("count") > -1
    ]
    new_df_cols = [
        "workload name",
        "core mask",
        "duration (s)",
        "no. samples",
        "start time (ms)",
        "end time (ms)",
        "start date",
        "end date",
    ]
    freq_cols = [
        i for i in pmc_events_log_df.columns.values if i.find("Freq (MHz)") > -1
    ]
    power_cols = [
        i for i in pmc_events_log_df.columns.values if i.find("Power") > -1
    ]
    temperature_cols = [
        x
        for x in pmc_events_log_df.columns.values
        if x.find("Temperature") > -1 or x.find("temperature") > -1
    ]
    for freq in freq_cols:
        new_df_cols.append(freq)
    for pmc in pmc_cols:
        new_df_cols.append(pmc + " diff")
    for pmc in pmc_cols:
        new_df_cols.append(pmc + " rate")
    for temperature in temperature_cols:
        new_df_cols.append(temperature + " mean (interp)")
    # if temperature_file:
    # new_df_cols.append('Ambient Temperature')
    new_df = pd.DataFrame(columns=new_df_cols)
    # time_series_pmc_cols = [x+' rate' for x in pmc_cols]
    # time_series_df = pd.DataFrame(columns=['workload name','milliseconds', \
    #        'datetime']+freq_cols+time_series_pmc_cols+temperature_cols)
    time_series_df = pd.DataFrame(columns=["temp"])
    for i in range(0, len(workloads_temp_df.index)):
        current_workload = workloads_temp_df["label"].iloc[i].split()[0]
        # get the start time stamp and end time stamp (milli)
        start_row = pmc_events_log_df[
            pmc_events_log_df["label"] == current_workload + " start"
        ]
        end_row = pmc_events_log_df[
            pmc_events_log_df["label"] == current_workload + " end"
        ]
        if len(end_row) < 1:
            print("Error: could not find end row. Skipping this workload")
            print("Press enter to continue")
            input()
            continue
        elif len(end_row) > 1:
            print(
                "Error: more than one end entry for the same workload! Skipping"
                " this workload"
            )
            print("Press enter to continue")
            input()
            continue
        elif len(start_row) > 1:
            print("Error multiple start rows for same workload. Skipping")
            print("Press enter to continue")
            input()
            continue
        start_time = int(start_row["milliseconds"])
        end_time = int(end_row["milliseconds"])
        delta_time = float(end_time - start_time) / 1000.0
        # now use the continuous log to get the in between times
        continuous_df = pmc_continuous_log_df[
            pmc_continuous_log_df["milliseconds"] > start_time
        ]
        continuous_df = continuous_df[continuous_df["milliseconds"] < end_time]
        # now get pmcs
        num_samples = 0
        row_dict = {}
        for pmc in pmc_cols:
            pmc_vals = []
            pmc_vals.append(int(start_row[pmc]))
            for j in range(0, len(continuous_df.index)):
                pmc_vals.append(int(continuous_df[pmc].iloc[j]))
            pmc_vals.append(int(end_row[pmc]))
            pmc_diff = get_pmc_diff_from_list(pmc_vals)
            row_dict[pmc + " diff"] = pmc_diff
            row_dict[pmc + " rate"] = pmc_diff / delta_time
        num_samples = len(pmc_vals)
        # process temperature samples
        for t_col in temperature_cols:
            t_vals = []
            milli_vals = []
            t_vals.append(float(start_row[t_col]))
            milli_vals.append(int(start_row["milliseconds"]))
            for j in range(0, len(continuous_df.index)):
                t_vals.append(float(continuous_df[t_col].iloc[j]))
                milli_vals.append(int(continuous_df["milliseconds"].iloc[j]))
            t_vals.append(float(end_row[t_col]))
            milli_vals.append(int(end_row["milliseconds"]))
            mean_interp = mean_temperature_interpolated(milli_vals, t_vals)
            row_dict[t_col + " mean (interp)"] = mean_interp
        # process power samples
        for p_col in power_cols:
            p_vals = []
            milli_vals = []
            p_vals.append(float(start_row[p_col]))
            milli_vals.append(int(start_row["milliseconds"]))
            for j in range(0, len(continuous_df.index)):
                p_vals.append(float(continuous_df[p_col].iloc[j]))
                milli_vals.append(int(continuous_df["milliseconds"].iloc[j]))
            p_vals.append(float(end_row[p_col]))
            milli_vals.append(int(end_row["milliseconds"]))
            mean_interp = mean_temperature_interpolated(milli_vals, p_vals)
            row_dict[p_col + " mean (interp)"] = mean_interp
        # process ambient temperature - incl. time series
        ambi_in_range = None
        if temperature_file:
            ambi_temp_df = pd.read_csv(temperature_file, header=None, sep="\t")
            ambi_temp_df.columns = [
                "up count",
                "milliseconds",
                "datetime",
                "Ambient Temperature",
            ]
            # try and find in range:
            ambi_in_range_df = ambi_temp_df[
                (ambi_temp_df["milliseconds"] > start_time)
                & (ambi_temp_df["milliseconds"] < end_time)
            ]
            if len(ambi_in_range_df.index) < 1:
                raise NotImplementedError(
                    "Cannot yet deal with the situation "
                    + "where there is no temperature value within the sample"
                )
            row_dict["Ambient Temperature mean (no interp)"] = np.mean(
                ambi_in_range_df["Ambient Temperature"]
            )
            row_dict["Ambient Temperature mean (interp)"] = (
                mean_temperature_interpolated(
                    ambi_in_range_df["milliseconds"].tolist(),
                    [
                        float(x)
                        for x in ambi_in_range_df[
                            "Ambient Temperature"
                        ].tolist()
                    ],
                )
            )
        row_dict["workload name"] = current_workload
        row_dict["core mask"] = core_mask
        row_dict["duration (s)"] = delta_time
        row_dict["no. samples"] = num_samples
        row_dict["start time (ms)"] = start_time
        row_dict["end time (ms)"] = end_time
        row_dict["start date"] = start_row["datetime"].iloc[0]
        row_dict["end date"] = end_row["datetime"].iloc[0]
        # freq_cols = [i for i in pmc_events_log_df.columns.values if i.find('Freq (MHz)') > -1] ???
        for f in freq_cols:
            freq_vals = combine_event_and_log_vals_float(
                start_row, end_row, continuous_df, f
            )
            first_freq = freq_vals[0]
            for j in range(1, len(freq_vals)):
                if freq_vals[j] != first_freq:
                    raise ValueError(
                        "Frequency changes in middle of workload! ("
                        + current_workload
                        + ")"
                    )
            row_dict[f] = first_freq
        new_df = new_df.append(row_dict, ignore_index=True)
        # now do time series stuff
        # working on splice (continuous_df) with start and end
        ambient_temp_col = []
        time_series_cols = (
            ["label", "milliseconds", "datetime"]
            + freq_cols
            + temperature_cols
            + pmc_cols
        )
        temp_t_series_df = start_row[time_series_cols]
        temp_t_series_df = temp_t_series_df.append(
            continuous_df[time_series_cols], ignore_index=True
        )
        temp_t_series_df = temp_t_series_df.append(
            end_row[time_series_cols], ignore_index=True
        )
        temp_t_series_df["workload specific"] = temp_t_series_df["label"].apply(
            lambda x: temp_t_series_df["label"].iloc[0].replace(" start", "")
        )
        temp_t_series_df["workload group"] = temp_t_series_df["label"].apply(
            lambda x: temp_t_series_df["label"]
            .iloc[0]
            .replace(" start", "")
            .replace("prelseep-", "")
            .replace("postsleep-", "")
            .replace("postslseep-", "")
        )
        # pmcs in t_series:
        if temperature_file:
            temp_t_series_df["Ambient Temperature mean"] = temp_t_series_df[
                "milliseconds"
            ].apply(
                lambda x: np.mean(ambi_in_range_df["Ambient Temperature"])
            )  # doesn't use x on purpose
            temp_t_series_df["Ambient Temperature Lerp"] = temp_t_series_df[
                "milliseconds"
            ].apply(
                lambda x: np.interp(
                    float(x),
                    [
                        float(y)
                        for y in ambi_in_range_df[
                            "Ambient Temperature"
                        ].tolist()
                    ],
                    [
                        float(y)
                        for y in ambi_in_range_df[
                            "Ambient Temperature"
                        ].tolist()
                    ],
                )
            )
        mid_millis = [0]
        for r in range(1, len(temp_t_series_df.index)):
            mid_millis.append(
                int(temp_t_series_df["milliseconds"].iloc[r - 1])
                + (
                    (
                        int(temp_t_series_df["milliseconds"].iloc[r])
                        - int(temp_t_series_df["milliseconds"].iloc[r - 1])
                    )
                    / int(2)
                )
            )
        temp_t_series_df["pmc mid millis"] = mid_millis
        for pmc in pmc_cols:
            # temp_t_series_df[pmc+' rate test'] = temp_t_series_df.apply(lambda row: \
            #        get_pmc_rate(\
            #        row[pmc], \
            #        row[pmc].shift(-1), \
            #        (float(row['milliseconds'])-float(row['milliseconds'].shift(-1)))/1000 \
            #        ),axis=1)
            pmc_rates = [0]
            for r in range(1, len(temp_t_series_df.index)):
                pmc_rates.append(
                    get_pmc_rate(
                        float(temp_t_series_df[pmc].iloc[r]),
                        float(temp_t_series_df[pmc].iloc[r - 1]),
                        (
                            float(temp_t_series_df["milliseconds"].iloc[r])
                            - float(
                                temp_t_series_df["milliseconds"].iloc[r - 1]
                            )
                        )
                        / 1000.0,
                    )
                )
            temp_t_series_df[pmc + " rate last interval"] = pmc_rates
            # lerp pmcs
            temp_t_series_df[pmc + " rate lerp"] = temp_t_series_df[
                "milliseconds"
            ].apply(
                lambda x: np.interp(float(x), mid_millis[1:], pmc_rates[1:])
            )
        if len(time_series_df.columns.values) < 2:
            # not yet initialised
            time_series_df = temp_t_series_df
        else:
            # has been initialise, append to what's already there
            time_series_df = time_series_df.append(
                temp_t_series_df, ignore_index=True
            )
        # pmcs take a little more work
        # new_pmc_df = pd.DataFrame(columns=['milliseconds']+[x+' rate' for x in pmc_cols])
        # for r in range(1, len(temp_t_series_df.index)):
        #    pmc_dict = {}
        #    pmc_dict['milliseconds'] = temp_t_series_df['milliseconds'].iloc[r] - temp_t_series_df['milliseconds'].iloc[r-1]
        #    for pmc in pmc_cols:
        #        pmc_dict[pmc+' rate'] = get_pmc_rate(temp_t_series_df[pmc].iloc[r],temp_t_series_df[pmc].iloc[r-1], \
        #                (float(temp_t_series_df['milliseconds'].iloc[r])-float(temp_t_series_df['milliseconds'].iloc[r-1]))/1000.0)
        #    new_pmc_df.append(row_dict,ignore_index=True)

        # for pmc in pmc_cols:
        #    temp_t_series_df[pmc+' rate interp'] = temp_t_series_df
    new_df.to_csv(os.path.join(output_filepath), sep="\t")
    time_series_df.to_csv(output_filepath + "-time-series.csv", sep="\t")


def consolidate_iterations(files_list):
    # This method takes the postprocessed iteration files,
    # goes through each workload in turn, comparing between
    # the iterations, and chooses one of them to include in
    # the single end file.
    import pandas as pd

    iteration_dfs = [pd.read_csv(x, sep="\t") for x in files_list]
    # check workloads and execution
    # go through workload by workload:
    # assumption: workloads should be identical in all dfs
    workloads = iteration_dfs[0]["workload name"].tolist()
    columns: list = iteration_dfs[0].columns.values.tolist()
    mean_std_columns = ["duration mean [s]", "duration std [s]"]
    columns.extend(mean_std_columns)
    combined_df = pd.DataFrame(columns=columns)
    chosen_iteration_index_list = []
    for wl_i in range(0, len(workloads)):
        execution_times = [
            df["duration (s)"].iloc[wl_i] for df in iteration_dfs
        ]
        ordered_execution_times = sorted(execution_times)
        chosen_time = ordered_execution_times[len(execution_times) // 2]
        chosen_index = execution_times.index(chosen_time)
        df_row = iteration_dfs[chosen_index].iloc[wl_i]
        mean_std = pd.Series(
            data=[np.mean(execution_times), np.std(execution_times)],
            index=mean_std_columns,
        )
        df_row = df_row.append(mean_std)
        combined_df = combined_df.append(df_row, ignore_index=True)
        chosen_iteration_index_list.append(chosen_index)
    combined_df.insert(2, "iteration index", chosen_iteration_index_list)
    return combined_df


def combine_pmc_runs(pmc_files):
    import os
    import pandas as pd
    import math

    combined_df = None
    is_df_created = False
    execution_times = []
    for pmc_i in range(0, len(pmc_files)):
        pmc_filename = os.path.basename(pmc_files[pmc_i])
        pmc_dirname = os.path.basename(
            os.path.normpath(os.path.dirname(pmc_files[pmc_i]))
        )
        if not is_df_created:
            combined_df = pd.read_csv(pmc_files[pmc_i], sep="\t")
            execution_times.append(combined_df["duration (s)"].tolist())
            is_df_created = True
            continue
        temp_df = pd.read_csv(pmc_files[pmc_i], sep="\t")
        execution_times.append(temp_df["duration (s)"].tolist())
        # check data is the same
        # find the deviation
        new_cntr_cols = [
            x
            for x in temp_df.columns.values
            if (x.find("cntr") > -1 and x not in combined_df.columns.values)
        ]
        # just check that one of the new columns is not named the same as in combined_df
        for col in new_cntr_cols:
            if col in combined_df.columns.values:
                raise ValueError(
                    "Error: "
                    + col
                    + " is already in the DF (adding: "
                    + pmc_dirname
                    + ")"
                )
        # check workloads
        combined_workloads = combined_df["workload name"].tolist()
        temp_workloads = temp_df["workload name"].tolist()
        if not combined_workloads == temp_workloads:
            print("Combined: " + combined_workloads)
            print("Temp: " + temp_workloads)
            raise ValueError("Workloads are not the same!!!")
        # add to df
        combined_df = pd.concat([combined_df, temp_df[new_cntr_cols]], axis=1)
    # print("Analysing the S.D. between exeuction times between PMC runs:")
    mean_list = []
    sd_list = []
    for wl_i in range(0, len(combined_df.index)):
        total = 0
        row_string = combined_df["workload name"].iloc[wl_i] + ":  "
        for run_i in range(0, len(execution_times)):
            row_string += str(execution_times[run_i][wl_i]) + "   "
            total += execution_times[run_i][wl_i]
        mean = total / len(execution_times)
        row_string += " mean: " + str(mean) + "  "
        accum = 0
        for run_i in range(0, len(execution_times)):
            accum += (execution_times[run_i][wl_i] - mean) ** 2
        variance = accum / len(execution_times)
        row_string += "var: " + str(variance) + "  "
        sd = math.sqrt(variance)
        row_string += "SD:  " + str(sd)
        mean_list.append(mean)
        sd_list.append(sd)
        # print(row_string)
    workload_duration_col_index = (
        (combined_df.columns.values).tolist().index("duration (s)")
    )
    combined_df.insert(
        workload_duration_col_index, "duration mean (s)", mean_list
    )
    return combined_df


def postprocess_new_sytle_experiments(
    experiment_top_dir, temperature_file=None
):
    import os
    import pandas as pd

    pmc_dirs = [
        x
        for x in os.listdir(experiment_top_dir)
        if (
            os.path.isdir(os.path.join(experiment_top_dir))
            and x.startswith("pmc-run-")
        )
    ]
    pmc_files_to_combine = []
    for pmc_run_i in range(0, len(pmc_dirs)):
        # NOTE: pmc_dirs are not in correct order!
        current_pmc_dir = None
        for pmc_dir in pmc_dirs:
            if pmc_dir.startswith("pmc-run-{0:0>2}".format(pmc_run_i)):
                current_pmc_dir = pmc_dir
                break
        if not current_pmc_dir:
            raise IOError(
                "Could not find directory for pmc-run-{0:0>2}".format(pmc_run_i)
            )
        current_pmc_dir = os.path.join(experiment_top_dir, current_pmc_dir)
        iteration_dirs = [
            x
            for x in os.listdir(current_pmc_dir)
            if (
                os.path.isdir(os.path.join(current_pmc_dir, x))
                and x.find("iteration-") > -1
            )
        ]
        iteration_files_to_consolidate = []
        for iter_i in range(0, len(iteration_dirs)):
            current_iter_dir = None
            for iter_dir in iteration_dirs:
                if iter_dir.startswith("iteration-{0:0>2}".format(iter_i)):
                    current_iter_dir = iter_dir
                    break
            if not current_iter_dir:
                raise IOError(
                    "Could not find directory for iteration-{0:0>2}".format(
                        iter_i
                    )
                )
            current_iter_dir = os.path.join(current_pmc_dir, current_iter_dir)
            # first process each of the iterations, and then combine them
            iteration_postprocessed_filename = os.path.join(
                current_iter_dir, "postprocessed.csv"
            )
            postprocess_experiment(
                current_iter_dir,
                iteration_postprocessed_filename,
                temperature_file=temperature_file,
            )
            iteration_files_to_consolidate.append(
                iteration_postprocessed_filename
            )
        combined_iterations_df = consolidate_iterations(
            iteration_files_to_consolidate
        )
        combined_iterations_df.to_csv(
            os.path.join(current_pmc_dir, "consolidated-iterations.csv"),
            sep="\t",
        )
        pmc_files_to_combine.append(
            os.path.join(current_pmc_dir, "consolidated-iterations.csv")
        )
    combined_pmcs_df = combine_pmc_runs(pmc_files_to_combine)
    print(combined_pmcs_df)
    combined_pmcs_df.to_csv(
        os.path.join(experiment_top_dir, "consolidated-pmc-runs.csv"), sep="\t"
    )


# Three stages:
# 1) post processing (i.e. convert raw output files to df (including calculating pmc rate)
# 2) (for new experiments with multiple pmc runs and iterations) consolidating
# 3) elaborating/enriching - add key stats to existing data (e.g. average cluster PMCs, quick utilisating view)
if __name__ == "__main__":
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--experiment-dir",
        dest="experiment_dir",
        required=True,
        help="The path to the experiment directory to analyse, "
        + 'e.g. "../powmon-experiment-000/"',
    )
    parser.add_argument(
        "-t",
        "--temperature-file",
        dest="temperature_file",
        required=False,
        help="Path to the file of ambient temperature logs",
    )
    parser.add_argument(
        "--elaborate-only",
        dest="elaborate_only",
        required=False,
        action="store_true",
        help="Only runs the final elaboration stage without re-running "
        + "the post-processing and consolidating",
    )
    args = parser.parse_args()

    # Works in two cases:
    # 1) it is an 'old-style', single-run experiment with the files under the top experiment directory
    # 2) it has the top directory, pmc-run directories inside, with iteration directories inside each of these
    # (it doesn't to different combinations).

    subdirs = [
        x
        for x in os.listdir(args.experiment_dir)
        if os.path.isdir(os.path.join(args.experiment_dir, x))
    ]

    is_new_style_experiment = False
    for dir in subdirs:
        if dir.find("pmc-run") > -1:
            is_new_style_experiment = True
            break
    if is_new_style_experiment:
        # go through discovering directories, consolidating etc.
        if not args.elaborate_only:
            postprocess_new_sytle_experiments(
                args.experiment_dir, temperature_file=args.temperature_file
            )
        # elaborate(args.experiment_dir)
    else:
        if not args.elaborate_only:
            postprocess_experiment(
                args.experiment_dir,
                os.path.join(
                    args.experiment_dir,
                    run_experiment.FILENAME_PMC_EVENTS_LOG + "-analysed.csv",
                ),
            )
        # elaborate(args.experiment_dir)
