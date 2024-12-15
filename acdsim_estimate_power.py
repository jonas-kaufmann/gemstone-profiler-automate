import os
import sys
import re
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas
import glob

sampling_interval = 10


def write_tcl_script(
    tcl_path: str, xpr_path: str, saif_files: list[str], output_dir: str
) -> list[str]:
    out_files = []
    lines = ["open_run impl_1"]
    for saif_file in saif_files:
        lines.append("reset_switching_activity -all")
        lines.append(f"read_saif -no_strip {saif_file}")

        out_name = os.path.basename(saif_file)
        out_name = out_name.removesuffix(".saif")
        out_path = f"{output_dir}/{out_name}-estimation.txt"
        out_files.append(out_path)
        lines.append(f"report_power -no_propagation -file {out_path}")

    lines = [f"{line}\n" for line in lines]

    with open(tcl_path, "w", encoding="utf-8") as file:
        file.writelines(lines)

    return out_files


def modify_saif_files(top_module: str, saif_files: list[str], outdir) -> None:
    modified_saif_files = []
    for saif_file in saif_files:
        lines = []
        with open(saif_file, mode="r", encoding="utf-8") as file:
            for line in file.readlines():
                line = re.sub(r"vta_sim.*$", top_module, line)
                lines.append(line)

        modified_saif_file = os.path.basename(saif_file)
        modified_saif_file = modified_saif_file.removesuffix(".saif")
        modified_saif_file += "-modified.saif"
        modified_saif_file = os.path.join(outdir, modified_saif_file)
        modified_saif_files.append(modified_saif_file)
        with open(modified_saif_file, mode="w", encoding="utf-8") as file:
            file.writelines(lines)
    return modified_saif_files


def read_power_estimates(out_files: list[str]) -> list[float]:
    power_nrs = []
    for out_file in out_files:
        if not os.path.exists(out_file):
            return []
        with open(out_file, mode="r", encoding="utf-8") as file:
            lines = file.readlines()

        ps8 = None
        ps_static = None
        total = None
        for line in lines:
            if re.match(r"^\| *PS8 *\|.+\|", line):
                split = line.split()
                ps8 = float(split[3])
                continue
            if re.match(r"^\| *PS Static *\|.+\|", line):
                split = line.split()
                ps_static = float(split[4])
                continue
            if re.match(r"^\| *Total *\|.+\|", line):
                split = line.split()
                total = float(split[3])
                break
        power_nrs.append(total - ps8 - ps_static)

    return power_nrs


def plot_power_estimates(power_nrs: list[float]) -> None:
    plt.plot(power_nrs)
    plt.show()


def sort_filenames(files: list[str]) -> list[str]:
    file_tuples = []
    for file in files:
        dirname = os.path.dirname(file)
        basename = os.path.basename(file)
        basename = basename.removesuffix(".saif")
        split = basename.split("-")
        basename = "-".join(split[:-1])
        file_tuples.append((f"{dirname}/{basename}", int(split[-1])))

    file_tuples.sort()
    return [f"{name}-{index}.saif" for name, index in file_tuples]


def estimate_cpu_power(gemstone_dir, workdir, simbricks_experiment_dir):
    # preprocess gem5 stats
    gem5_combine_experiments = os.path.join(
        gemstone_dir, "gemstone-gem5-validate/gem5_combine_experiments.py"
    )
    gem5_out = os.path.join(simbricks_experiment_dir, "1/gem5-out./")
    preprocessed_gem5_stats = os.path.join(
        workdir, "preprocessed_gem5_stats.csv"
    )
    cmd = (
        f"python {gem5_combine_experiments} --results-dir {gem5_out} -o"
        f" {preprocessed_gem5_stats}"
    )
    exit_code = os.system(cmd)
    if exit_code != 0:
        raise RuntimeError(f"Executing the following command failed: {cmd}")

    # invoke power estimation
    gemstone_apply_power = os.path.join(
        gemstone_dir, "gemstone-applypower/gemstone_apply_power.py"
    )
    power_model_params = os.path.join(
        gemstone_dir, "gemstone-applypower/models/gs-A15.params"
    )
    power_model_vlookup = os.path.join(
        gemstone_dir, "gemstone-applypower/maps/gem5-A15-vlookup.map"
    )
    power_model_vflookup = os.path.join(
        gemstone_dir, "gemstone-applypower/vf-lookup/vf-A57.csv"
    )
    estimation_out = os.path.join(workdir, "cpu_estimation_out.csv")
    cmd = (
        f"python {gemstone_apply_power} --params"
        f" {power_model_params} --input-data"
        f" {preprocessed_gem5_stats} --var-map"
        f" {power_model_vlookup} --vf-lookup"
        f" {power_model_vflookup} --output-file {estimation_out}"
    )
    exit_code = os.system(cmd)
    if exit_code != 0:
        raise RuntimeError(f"Executing the following command failed: {cmd}")

    # read in power time series
    estimation_out_df = pandas.read_csv(estimation_out, sep="\t")
    power_time_series = estimation_out_df["power model total power"].to_list()
    timestamps = []
    for ts in estimation_out_df["finalTick"]:
        timestamps.append(int(ts / 10**9))

    assert len(power_time_series) == len(timestamps)
    return timestamps, power_time_series


def align_fpga_timestamps(experiment_dir: str, timestamps_fpga: list[int]):
    exp_out_json = experiment_dir.removesuffix("/") + "-1.json"
    with open(exp_out_json, "r") as file:
        exp_out = json.load(file)

    gem5_restore_ts = None
    for line in exp_out["sims"]["host."]["stderr"]:
        line: str
        if line.startswith(
            "src/sim/simulate.cc:199: info: Entering event queue @"
        ):
            gem5_restore_ts = int(line.split()[6].removesuffix("."))
            gem5_restore_ts //= 10**9
            break
    if gem5_restore_ts is None:
        print(
            "Couldn't find timestamp when gem5 entered simulation after"
            " checkpoint."
        )
        sys.exit(1)

    xsim_start_ts = None
    for line in exp_out["sims"]["dev..vta_xsim"]["stdout"]:
        if line.startswith("Disabling pseudo-synchronization at"):
            xsim_start_ts = int(line.split()[3].removeprefix("simbricks_time="))
            xsim_start_ts //= 10**9
            break
    if xsim_start_ts is None:
        print("Couldn't find xsim start timestamp")
        sys.exit(1)

    offset = gem5_restore_ts + xsim_start_ts
    aligned_timestamps = (np.array(timestamps_fpga) + offset).tolist()
    return aligned_timestamps


def piecewice_constant(
    timestamps: list[int], power_series: list[float], eval_at, right, left
):
    assert len(timestamps) == len(power_series)
    timestamps_interp = []
    power_series_interp = []
    for i in range(len(timestamps)):
        if i != 0:
            start = timestamps[i - 1]
            duration = timestamps[i] - timestamps[i - 1]
        else:
            start = timestamps[0] - sampling_interval
            duration = sampling_interval
        timestamps_interp.extend(start + i for i in range(duration))
        power_series_interp.extend([power_series[i]] * duration)

    power_series_interp = np.interp(
        eval_at, timestamps_interp, power_series_interp, left, right
    )
    return power_series_interp


def write_estimation_result(
    workdir,
    timestamps_cpu: list[int],
    power_series_cpu: list[float],
    timestamps_fpga: list[int],
    power_series_fpga: list[float],
    fpga_idle_power: float,
):
    eval_at = list(range(timestamps_cpu[0], timestamps_cpu[-1]))
    power_series_cpu_interp = piecewice_constant(
        timestamps_cpu, power_series_cpu, eval_at, -1, -1
    )
    assert -1 not in power_series_cpu
    if power_series_fpga:
        power_series_fpga_interp = piecewice_constant(
            timestamps_fpga,
            power_series_fpga,
            eval_at,
            fpga_idle_power,
            fpga_idle_power,
        )
    else:
        power_series_fpga_interp = [0] * len(eval_at)

    df_dict = {
        "milliseconds": eval_at,
        "Power CPU (W)": power_series_cpu_interp,
        "Power FPGA (W)": power_series_fpga_interp,
    }
    if timestamps_fpga:
        df_dict["FPGA Start"] = [timestamps_fpga[0] - sampling_interval] * len(
            eval_at
        )
    acdsim_estimation = os.path.join(workdir, "acdsim_estimation.csv")
    pandas.DataFrame(df_dict).to_csv(acdsim_estimation, sep="\t")
    return eval_at, (power_series_cpu_interp + power_series_fpga_interp)


def read_saif_durations(saif_files):
    durations = []
    for saif_file in saif_files:
        with open(saif_file, "r") as file:
            lines = file.readlines()

        for line in lines:
            split = line.split()
            if split[0] == "(DURATION":
                duration_ps = int(split[1][:-1])
                durations.append(duration_ps // 10**9)
                break

    return durations


def estimate_fpga_power(
    vivado_dir, vivado_xpr, top_level_name, experiment_dir, workdir
):
    # prepare saif files
    saif_files_glob = os.path.join(
        experiment_dir, "1/dev..vta_xsim-xsim/*.saif"
    )
    saif_files = sort_filenames(glob.glob(saif_files_glob))
    saif_files = modify_saif_files(top_level_name, saif_files, workdir)

    tcl_script = os.path.join(workdir, "acdsim_estimate_power.tcl")
    out_files = write_tcl_script(tcl_script, vivado_xpr, saif_files, workdir)

    # run vivado
    cmds = [
        f"source '{vivado_dir}/settings64.sh'",
        f"vivado -mode batch -source '{tcl_script}' '{vivado_xpr}'",
    ]
    cmd = " && ".join(cmds)

    # try reading without invoking estimation first
    fpga_power_series = read_power_estimates(out_files)
    if out_files and not fpga_power_series:
        rc = os.system(f'bash -c "{cmd}"')
        if rc != 0:
            print("The following command failed:", cmd)
            sys.exit(1)
        fpga_power_series = read_power_estimates(out_files)
    durations = read_saif_durations(saif_files)
    assert len(fpga_power_series) == len(durations)

    timestamps = []
    total_millis = 0
    for duration in durations:
        total_millis += duration
        timestamps.append(total_millis)

    return timestamps, fpga_power_series


def main():
    if len(sys.argv) != 8:
        print(
            "Usage: acdsim_estimate_power.py <vivado installation directory>"
            " <vivado project .xpr> <top-level module> <SimBricks experiment"
            " directory> <output directory> <gemstone directory> <FPGA idle"
            " power>"
        )
        sys.exit(1)

    vivado_dir = sys.argv[1]
    vivado_xpr = sys.argv[2]
    top_level_name = sys.argv[3]
    experiment_dir = sys.argv[4]
    workdir = sys.argv[5]
    gemstone_dir = sys.argv[6]
    fpga_idle_power = float(sys.argv[7])

    os.makedirs(workdir, exist_ok=True)

    cpu_millis, cpu_power_series = estimate_cpu_power(
        gemstone_dir, workdir, experiment_dir
    )
    fpga_millis, fpga_power_series = estimate_fpga_power(
        vivado_dir, vivado_xpr, top_level_name, experiment_dir, workdir
    )
    print("FPGA power consumption samples:")
    print(fpga_power_series)
    if fpga_millis:
        fpga_millis = align_fpga_timestamps(experiment_dir, fpga_millis)
    write_estimation_result(
        workdir,
        cpu_millis,
        cpu_power_series,
        fpga_millis,
        fpga_power_series,
        fpga_idle_power,
    )


if __name__ == "__main__":
    main()
