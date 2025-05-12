import os

# Attempt to import user's functions; provide mocks if not found
from autoflatten.freesurfer import run_mris_flatten, setup_freesurfer

# --- Configuration ---
FS6_PATH = "/data2/freesurfer-6.0"
FS8_PATH = "/data2/freesurfer-8.0.0"
SUBJECTS_DIR = "/data2/freesurfer_subjects/all-subjects"

N_THREADS = 32
SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_PATCH_FILE_NAME = "lh.MVauto_autoflatten.patch.3d"
INPUT_PATCH_FULL_PATH = os.path.join(HERE, INPUT_PATCH_FILE_NAME)
HEMI = "lh"

# All outputs will go into this single directory
SINGLE_OUTPUT_DIR = os.path.join(HERE, "20250510_test_params")

# --- Recommended Baseline Parameter Values (closer to mris_flatten internal defaults) ---
REC_DEFAULT_N = 40
REC_DEFAULT_A = 1024  # -a: n_averages
REC_DEFAULT_DT = 0.1  # -dt: initial time step
REC_DEFAULT_TOL = 0.2  # -tol: convergence tolerance
REC_DEFAULT_MOMENTUM = 0.9  # -momentum
REC_DEFAULT_L_DIST = 1.0  # -dist (spring constant)
REC_DEFAULT_L_NLAREA = 1.0  # -nlarea (area preservation constant)
REC_DEFAULT_DISTANCES_TUPLE = (7, 12)  # -distances: nbhd_size, max_nbrs
REC_DEFAULT_DILATE = 0  # -dilate: mris_flatten default if flag is not present

# --- User's Baseline Parameters (from their original script for their reference) ---
USER_DT_BASELINE = 0.5
USER_TOL_BASELINE = 0.1
USER_DILATE_BASELINE = 1
USER_DISTANCES_BASELINE = (15, 80)
USER_N_BASELINE = 40
USER_A_BASELINE = REC_DEFAULT_A  # Assuming user wants mris_flatten's default for 'a' if not set otherwise

# --- Test Scenario Definitions ---
test_scenarios = []

# # Scenario Set 1: Baselines
# test_scenarios.append(
#     {
#         "name": "FS6_RecommendedDefaults",
#         "fs_version_path": FS6_PATH,
#         "params": {
#             "distances_tuple": REC_DEFAULT_DISTANCES_TUPLE,
#             "dilate": REC_DEFAULT_DILATE,
#             "n": REC_DEFAULT_N,
#             "tol": REC_DEFAULT_TOL,
#             "dt": REC_DEFAULT_DT,
#             "a": REC_DEFAULT_A,
#             "momentum": REC_DEFAULT_MOMENTUM,
#             "dist": REC_DEFAULT_L_DIST,
#             "nlarea": REC_DEFAULT_L_NLAREA,
#         },
#     }
# )
# test_scenarios.append(
#     {
#         "name": "FS8_RecommendedDefaults",
#         "fs_version_path": FS8_PATH,
#         "params": {
#             "distances_tuple": REC_DEFAULT_DISTANCES_TUPLE,
#             "dilate": REC_DEFAULT_DILATE,
#             "n": REC_DEFAULT_N,
#             "tol": REC_DEFAULT_TOL,
#             "dt": REC_DEFAULT_DT,
#             "a": REC_DEFAULT_A,
#             "momentum": REC_DEFAULT_MOMENTUM,
#             "dist": REC_DEFAULT_L_DIST,
#             "nlarea": REC_DEFAULT_L_NLAREA,
#         },
#     }
# )

# # Scenario Set 2: Investigating dt Behavior (FS8, based on RecommendedDefaults)
# test_scenarios.append(
#     {
#         "name": "FS8_RecDefaults_SmallDt_HighIters",
#         "fs_version_path": FS8_PATH,
#         "params": {
#             "distances_tuple": REC_DEFAULT_DISTANCES_TUPLE,
#             "dilate": REC_DEFAULT_DILATE,
#             "dt": 0.01,
#             "n": 200,
#             "a": 4096,  # Key changes
#             "tol": REC_DEFAULT_TOL,
#             "momentum": REC_DEFAULT_MOMENTUM,
#             "dist": REC_DEFAULT_L_DIST,
#             "nlarea": REC_DEFAULT_L_NLAREA,
#         },
#     }
# )

# # Scenario Set 3: Investigating Dilation (based on RecommendedDefaults)
# dilate_values_to_test = [0, 1, 2]
# for dilate_val in dilate_values_to_test:
#     for fs_ver_label, fs_path_val in [("FS6", FS6_PATH), ("FS8", FS8_PATH)]:
#         test_scenarios.append(
#             {
#                 "name": f"{fs_ver_label}_RecDefaults_Dilate{dilate_val}",
#                 "fs_version_path": fs_path_val,
#                 "params": {
#                     "distances_tuple": REC_DEFAULT_DISTANCES_TUPLE,
#                     "dilate": dilate_val,  # Key change
#                     "n": REC_DEFAULT_N,
#                     "tol": REC_DEFAULT_TOL,
#                     "dt": REC_DEFAULT_DT,
#                     "a": REC_DEFAULT_A,
#                     "momentum": REC_DEFAULT_MOMENTUM,
#                     "dist": REC_DEFAULT_L_DIST,
#                     "nlarea": REC_DEFAULT_L_NLAREA,
#                 },
#             }
#         )

# # Scenario Set 4: Optional General Exploration for FS8 (based on RecommendedDefaults)
# dist_nlarea_weights = [(1.0, 1.2), (1.2, 1.0)]
# for d_coeff, nl_coeff in dist_nlarea_weights:
#     test_scenarios.append(
#         {
#             "name": f"FS8_RecDefaults_Weights_d{str(d_coeff).replace('.', 'p')}_nl{str(nl_coeff).replace('.', 'p')}",
#             "fs_version_path": FS8_PATH,
#             "params": {
#                 "distances_tuple": REC_DEFAULT_DISTANCES_TUPLE,
#                 "dilate": REC_DEFAULT_DILATE,
#                 "n": REC_DEFAULT_N,
#                 "tol": REC_DEFAULT_TOL,
#                 "dt": REC_DEFAULT_DT,
#                 "a": REC_DEFAULT_A,
#                 "momentum": REC_DEFAULT_MOMENTUM,
#                 "dist": d_coeff,
#                 "nlarea": nl_coeff,
#             },  # Key changes
#         }
#     )

# momentum_values_to_test = [0.8, 0.95]
# for mom_val in momentum_values_to_test:
#     test_scenarios.append(
#         {
#             "name": f"FS8_RecDefaults_Momentum{str(mom_val).replace('.', 'p')}",
#             "fs_version_path": FS8_PATH,
#             "params": {
#                 "distances_tuple": REC_DEFAULT_DISTANCES_TUPLE,
#                 "dilate": REC_DEFAULT_DILATE,
#                 "n": REC_DEFAULT_N,
#                 "tol": REC_DEFAULT_TOL,
#                 "dt": REC_DEFAULT_DT,
#                 "a": REC_DEFAULT_A,
#                 "momentum": mom_val,  # Key change
#                 "dist": REC_DEFAULT_L_DIST,
#                 "nlarea": REC_DEFAULT_L_NLAREA,
#             },
#         }
#     )

# # User's original distances_param exploration (e.g., for FS8, based on RecommendedDefaults)
# user_distances_param_original = [(15, 80), (15, 100), (7, 12), (20, 20), (30, 30)]
# for dist_tuple in user_distances_param_original:
#     test_scenarios.append(
#         {
#             "name": f"FS8_RecDefaults_Dist{dist_tuple[0]}_{dist_tuple[1]}",
#             "fs_version_path": FS8_PATH,
#             "params": {
#                 "distances_tuple": dist_tuple,
#                 "dilate": REC_DEFAULT_DILATE,  # Key change
#                 "n": REC_DEFAULT_N,
#                 "tol": REC_DEFAULT_TOL,
#                 "dt": REC_DEFAULT_DT,
#                 "a": REC_DEFAULT_A,
#                 "momentum": REC_DEFAULT_MOMENTUM,
#                 "dist": REC_DEFAULT_L_DIST,
#                 "nlarea": REC_DEFAULT_L_NLAREA,
#             },
#         }
#     )

# Some parameters that I think could work
# Refactored: Use a for loop for similar scenarios
tol_dt_pairs = [
    # (0.2, 0.1),
    # (0.2, 0.05),
    # (0.1, 0.1),
    # (0.1, 0.05),
    # (0.05, 0.1),
    # (0.01, 0.1),
    (0.005, 0.1),
    (0.001, 0.1),
]
for tol, dt in tol_dt_pairs:
    scenario_name = f"FS8_dist2080_dilate1_n200_tol{str(tol).replace('.', 'p')}_dt{str(dt).replace('.', 'p')}_p1"
    test_scenarios.append(
        {
            "name": scenario_name,
            "fs_version_path": FS8_PATH,
            "params": {
                "distances_tuple": (20, 80),
                "dilate": 1,
                "n": 200,
                "tol": tol,
                "dt": dt,
                "p": 1,
            },
        }
    )

# --- Main Execution Loop ---
if __name__ == "__main__":
    if not os.path.exists(INPUT_PATCH_FULL_PATH):
        print(f"ERROR: Input patch file not found: {INPUT_PATCH_FULL_PATH}")
        exit(1)

    os.makedirs(SINGLE_OUTPUT_DIR, exist_ok=True)
    print(f"All outputs will be saved in: {SINGLE_OUTPUT_DIR}")

    # Store original environment to restore PATH correctly for each setup_freesurfer call
    original_env_path = os.environ.get("PATH", "")
    original_env_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    original_env_dyld_library_path = os.environ.get("DYLD_LIBRARY_PATH", "")

    for i, scenario in enumerate(test_scenarios):
        print(
            f"\n--- Running Scenario {i + 1}/{len(test_scenarios)}: {scenario['name']} ---"
        )

        # Reset PATH to original before calling setup_freesurfer to avoid concatenation issues
        os.environ["PATH"] = original_env_path
        if original_env_ld_library_path:
            os.environ["LD_LIBRARY_PATH"] = original_env_ld_library_path
        else:
            os.environ.pop(
                "LD_LIBRARY_PATH", None
            )  # Ensure it's removed if it wasn't there originally
        if original_env_dyld_library_path:
            os.environ["DYLD_LIBRARY_PATH"] = original_env_dyld_library_path
        else:
            os.environ.pop("DYLD_LIBRARY_PATH", None)

        # Use user's setup_freesurfer function
        setup_freesurfer(scenario["fs_version_path"], SUBJECTS_DIR)
        print(f"Using FreeSurfer from: {os.environ.get('FREESURFER_HOME')}")

        # Define a unique output name prefix for mris_flatten, placed in the single output directory
        input_file_basename_for_output = INPUT_PATCH_FILE_NAME.split(".")[
            0
        ]  # E.g., "lh" from "lh.some.patch.3d"
        if INPUT_PATCH_FILE_NAME.endswith(
            ".patch.3d"
        ):  # A more specific base if it's a patch file
            input_file_basename_for_output = INPUT_PATCH_FILE_NAME[: -len(".patch.3d")]

        scenario_name_slug = (
            scenario["name"]
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace(",", "")
        )

        # output_name_prefix for run_mris_flatten should not have extensions
        output_name_prefix = os.path.join(
            SINGLE_OUTPUT_DIR,
            f"{input_file_basename_for_output}_{scenario_name_slug}",  # e.g., .../lh.MVauto_autoflatten_FS6_UserTypical
        )

        # Prepare arguments for user's run_mris_flatten function
        current_params = scenario["params"].copy()

        distances_str_arg = None
        if "distances_tuple" in current_params and current_params["distances_tuple"]:
            d_t = current_params.pop("distances_tuple")
            distances_str_arg = f"{d_t[0]},{d_t[1]}"

        # Set n_threads depending on FS version
        if scenario["fs_version_path"] == FS8_PATH:
            n_threads_to_use = N_THREADS
        else:
            n_threads_to_use = None

        print(f"Output files will start with prefix: {output_name_prefix}")
        print(
            f"Parameters for mris_flatten call: {current_params}, distances='{distances_str_arg}'"
        )

        log_file_path = f"{output_name_prefix}.run_log.txt"  # Log for this specific run

        try:
            with open(log_file_path, "w") as log_file:
                log_file.write(f"Scenario: {scenario['name']}\n")
                log_file.write(f"FS Version: {scenario['fs_version_path']}\n")
                log_file.write(f"Parameters: {scenario['params']}\n")
                log_file.write(f"Distances arg: {distances_str_arg}\n")
                log_file.write(f"Output prefix: {output_name_prefix}\n\n")
                log_file.flush()  # Ensure header is written

                # Call user's run_mris_flatten function
                # Assuming it handles stdout/stderr redirection or returns them
                result = run_mris_flatten(
                    subject="MVauto",  # or set appropriately if subject is known
                    hemi=HEMI,
                    patch_file=INPUT_PATCH_FULL_PATH,
                    output_dir=SINGLE_OUTPUT_DIR,
                    output_name=os.path.basename(output_name_prefix) + ".flat.patch.3d",
                    norand=False,
                    seed=SEED,
                    threads=n_threads_to_use,
                    distances=tuple(map(int, distances_str_arg.split(",")))
                    if distances_str_arg
                    else None,
                    n=current_params.get("n"),
                    dilate=current_params.get("dilate"),
                    extra_params={
                        k: v
                        for k, v in current_params.items()
                        if k not in ("n", "dilate")
                    },
                    overwrite=True,
                )

                # Process result (highly dependent on what run_mris_flatten returns)
                # Example: if it returns a dict with 'returncode', 'stdout', 'stderr'
                if isinstance(result, dict):
                    if result.get("returncode") == 0:
                        print(f"Successfully completed: {scenario['name']}")
                    else:
                        print(
                            f"run_mris_flatten for {scenario['name']} reported non-zero return code: {result.get('returncode')}. Log: {log_file_path}"
                        )

                    # Append stdout/stderr from result to the log file if available
                    with open(log_file_path, "a") as log_file_append:
                        if result.get("stdout"):
                            log_file_append.write("\n--- STDOUT ---\n")
                            log_file_append.write(str(result.get("stdout")))
                        if result.get("stderr"):
                            log_file_append.write("\n--- STDERR ---\n")
                            log_file_append.write(str(result.get("stderr")))
                else:  # If result is not a dict, or some other success indicator
                    print(
                        f"Completed scenario (unknown status): {scenario['name']}. Log: {log_file_path}"
                    )

        except Exception as e:
            print(
                f"An exception occurred while running scenario '{scenario['name']}': {e}. Log: {log_file_path}"
            )
            import traceback

            with open(log_file_path, "a") as log_file_append:
                log_file_append.write("\n--- PYTHON EXCEPTION ---\n")
                traceback.print_exc(file=log_file_append)

    # Restore original environment at the very end
    os.environ["PATH"] = original_env_path
    if original_env_ld_library_path:
        os.environ["LD_LIBRARY_PATH"] = original_env_ld_library_path
    else:
        os.environ.pop("LD_LIBRARY_PATH", None)
    if original_env_dyld_library_path:
        os.environ["DYLD_LIBRARY_PATH"] = original_env_dyld_library_path
    else:
        os.environ.pop("DYLD_LIBRARY_PATH", None)

    print(f"\nAll {len(test_scenarios)} test scenarios processed.")
    print(f"Outputs are in: {SINGLE_OUTPUT_DIR}")
