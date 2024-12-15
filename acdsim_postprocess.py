import sys
import pandas as pd
import os
import glob
import numpy as np
import matplotlib.pyplot as plt


def get_acdsim_start_stop_ms(program_out_filepath):
    with open(program_out_filepath, "r") as file:
        lines = file.readlines()

    start_ms = []
    stop_ms = []
    for line in lines:
        if line.startswith("AC/DSim START TS"):
            split = line.split()
            start_ns = int(split[3])
            start_ms.append(start_ns // (10**6))
        if line.startswith("AC/DSim STOP TS"):
            split = line.split()
            stop_ns = int(split[3])
            stop_ms.append(stop_ns // (10**6))

    return list(zip(start_ms, stop_ms))


def main():
    assert len(sys.argv) == 3
    experiment_dir = sys.argv[1]
    out_dir = sys.argv[2]
    consolidated_pmc_runs = pd.read_csv(
        os.path.join(experiment_dir, "consolidated-pmc-runs.csv"),
        sep="\t",
    )
    workloads = consolidated_pmc_runs["workload name"]
    median_iterations = consolidated_pmc_runs["iteration index"]

    csv_time_series_files = sorted(
        glob.glob(
            os.path.join(
                experiment_dir,
                "pmc-run-00-0x11/iteration-0*/postprocessed.csv-time-series.csv",
            )
        )
    )
    # Discard first two iterations
    csv_time_series_files = csv_time_series_files[2:]
    assert csv_time_series_files
    iterations_df = []
    for csv_time_series_file in csv_time_series_files:
        iterations_df.append(
            pd.read_csv(
                csv_time_series_file,
                sep="\t",
            )
        )

    for i in range(len(workloads)):
        workload = workloads[i]
        median_iter = median_iterations[i]
        print("----------------")
        print(workload, median_iter)
        cpu_energy_estimates = []
        fpga_energy_estimates = []
        durations = []
        cpu_power_series = []
        fpga_power_series = []
        milliseconds_series = []

        for j in range(len(iterations_df)):
            acdsim_start_stop_ms = get_acdsim_start_stop_ms(
                os.path.join(
                    experiment_dir,
                    f"pmc-run-00-0x11/iteration-0{j+2}/program-output.log",
                )
            )
            assert acdsim_start_stop_ms
            iteration_df = iterations_df[j]
            start, stop = acdsim_start_stop_ms[i]
            duration = (stop - start) / 1000.0
            durations.append(duration)
            evaluate_at = list(range(start, stop, 1))
            milliseconds_series.append(evaluate_at)

            # estimate CPU energy
            measured_cpu_samples = iteration_df["Power CPU (mW)"].to_list()
            measured_milliseconds = iteration_df["milliseconds"].to_list()
            cpu_power_time_series_acdsim = np.interp(
                evaluate_at,
                measured_milliseconds,
                measured_cpu_samples,
                left=-1,
                right=-1,
            )
            cpu_power_series.append(cpu_power_time_series_acdsim)
            if -1 in cpu_power_time_series_acdsim:
                raise RuntimeError(
                    "AC/DSim window out of range of measurements"
                )
            cpu_power_mean = np.mean(cpu_power_time_series_acdsim)
            cpu_energy_estimates.append(cpu_power_mean * duration)

            # estimate FPGA energy
            measured_fpga_samples = iteration_df["Power FPGA (mW)"].to_list()
            measured_milliseconds = iteration_df["milliseconds"].to_list()
            fpga_power_time_series_acdsim = np.interp(
                evaluate_at,
                measured_milliseconds,
                measured_fpga_samples,
                left=-1,
                right=-1,
            )
            fpga_power_series.append(fpga_power_time_series_acdsim)
            if -1 in fpga_power_time_series_acdsim:
                raise RuntimeError(
                    "AC/DSim window out of range of measurements"
                )
            fpga_power_mean = np.mean(fpga_power_time_series_acdsim)
            fpga_energy_estimates.append(fpga_power_mean * duration)

        # produce new output file
        milliseconds_0 = np.array(milliseconds_series[median_iter]) - milliseconds_series[median_iter][0]
        df_dict = {
            "milliseconds": milliseconds_0,
            "Power CPU (mW)": cpu_power_series[median_iter],
            "Power FPGA (mW)": fpga_power_series[median_iter],
        }
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        pd.DataFrame(df_dict).to_csv(
            os.path.join(out_dir, f"{workload}-time_series.csv"), "\t"
        )

        print("CPU")
        print("mean energy", round(np.mean(cpu_energy_estimates), 2))
        print("median energy", round(np.median(cpu_energy_estimates), 2))
        print("stddev energy", round(np.std(cpu_energy_estimates), 2))

        print("FPGA")
        print("mean energy", round(np.mean(fpga_energy_estimates), 2))
        print("median energy", round(np.median(fpga_energy_estimates), 2))
        print("stddev energy", round(np.std(fpga_energy_estimates), 2))

        print("Duration")
        print("mean", round(np.mean(durations), 2))
        print("median", round(np.median(durations), 2))
        print("stddev", round(np.std(durations), 2))


if __name__ == "__main__":
    main()
