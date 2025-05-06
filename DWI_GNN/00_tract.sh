#!/bin/bash -l
#$ -l h_rt=80:00:00       # Time limit
#$ -P ec523               # Project Name
#$ -N tract_HCP           # Job name
#$ -pe omp 8              # Number of CPUs
#$ -j y                    # Merge the error and output streams into a single file
#$ -m ea                   # Email when job ends
#$ -M gsummers@bu.edu      # Email address used to send the job report

module load dsi_studio

# Define the source directory where subject folders are located
SOURCE_DIR="/projectnb/nsa-aphasia/gsummers/ec523Data/ADNI"

    cd "${SOURCE_DIR}"
    # Iterate over each subject folder in the subdirectory
    for SUBJECT_FOLDER in *; do
	subject_name=$(basename "$SUBJECT_FOLDER")
        # Define the file path
        FILE_PATH="${SOURCE_DIR}/${SUBJECT_FOLDER}"
	cd "${FILE_PATH}"
        # create src file
	dsi_studio --action=src --source=${subject_name}.nii
	#run reconstruction
	dsi_studio --action=rec --source=*.src.gz --method=4 --param0=1.25
	#Generate Structural Connectivity Matrix
	dsi_studio --action=trk --source=*.fib.gz --fiber_count=1000000 --output=no_file --connectivity=Brainnectome --connectivity_value=count,qa,trk
	#remove .tt.gz files
	rm -rf *.gz



    done

echo "All NIFTI files processed."
