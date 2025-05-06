import os
import subprocess
import glob

def run_fastsurfer_single(sif_path, t1_path, subject_id, output_dir):
    t1_dir = os.path.dirname(t1_path)
    t1_filename = os.path.basename(t1_path)
    subj_output_dir = os.path.join(output_dir, subject_id, "mri")

    if os.path.exists(subj_output_dir):
        print(f"Skipping {subject_id}: already processed")
        return "skipped"

    command = [
        "singularity", "run", "--nv",
        "-B", f"{t1_dir}:/input",
        "-B", f"{output_dir}:/output",
        sif_path,
        "--t1", f"/input/{t1_filename}",
        "--sid", subject_id,
        "--sd", f"/output/{subject_id}",
        "--parallel",
        "--seg_only"
    ]

    try:
        print(f"Running FastSurfer for {subject_id}")
        print("Command:", " ".join(command))

        log_path = os.path.join(output_dir, subject_id, f"{subject_id}_log.txt")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        with open(log_path, "w") as log_file:
            subprocess.run(command, check=True, stdout=log_file, stderr=log_file)

        print(f"Output saved to: {os.path.join(output_dir, subject_id)}")
        return "success"

    except subprocess.CalledProcessError as e:
        print(f"ERROR: FastSurfer failed for {subject_id} with return code {e.returncode}")
        return "error"
    except Exception as e:
        print(f"ERROR: Unexpected failure for {subject_id}: {str(e)}")
        return "error"


## File Paths
sif_path = "/projectnb/ec523/projects/proj_GS_LQ_EPB/containers/fastsurfer.sif"
adni_root = "/projectnb/ec523/projects/proj_GS_LQ_EPB/data/T1w/ADNI/ADNI"
adni_out = "/projectnb/ec523/projects/proj_GS_LQ_EPB/data/T1w_segmented/ADNI_processed"
os.makedirs(adni_out, exist_ok=True)

summary = {
    "ADNI": {"success": [], "skipped": [], "error": []}
}

## Process Files
adni_subjects = [d for d in os.listdir(adni_root) if os.path.isdir(os.path.join(adni_root, d))]

for subject_id in adni_subjects:
    subject_path = os.path.join(adni_root, subject_id)

    try:
        preferred_folder = os.path.join(subject_path, "MPR____N3__Scaled")
        if not os.path.isdir(preferred_folder):
            all_subfolders = [os.path.join(subject_path, f) for f in os.listdir(subject_path)
                              if os.path.isdir(os.path.join(subject_path, f))]
            if not all_subfolders:
                print(f"Skipping {subject_id}")
                summary["ADNI"]["error"].append(subject_id)
                continue
            preferred_folder = sorted(all_subfolders)[0]

        date_folders = [d for d in os.listdir(preferred_folder)
                        if os.path.isdir(os.path.join(preferred_folder, d))]
        if not date_folders:
            print(f"Skipping {subject_id}")
            summary["ADNI"]["error"].append(subject_id)
            continue

        earliest_date = sorted(date_folders)[0]
        date_folder_path = os.path.join(preferred_folder, earliest_date)

        nii_candidates = glob.glob(os.path.join(date_folder_path, "**", "*.nii"), recursive=True)
        if not nii_candidates:
            print(f"No .nii file found for {subject_id}")
            summary["ADNI"]["error"].append(subject_id)
            continue

        t1_path = nii_candidates[0]
        result = run_fastsurfer_single(sif_path, t1_path, subject_id, adni_out)
        summary["ADNI"][result].append(subject_id)

    except Exception as e:
        print(f"ERROR: Failed while processing {subject_id}: {str(e)}")
        summary["ADNI"]["error"].append(subject_id)


## Get output summary
for dataset in summary:
    print(f"{dataset}:")
    for status, subjects in summary[dataset].items():
        print(f"  {status.capitalize()}: {len(subjects)}")
