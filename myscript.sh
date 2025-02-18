#!/bin/bash

python_script="/home/ywang/predict_ht/SPRinT.py"

# Define parameter values
agg_list=("avg" "pct" "sum" "var")

fname_list=("icd9_eICU" "icd9_mimic" "Health_and_Household" "Electronics" "night-street")
algo_list=("SPRinT" "PQA-PT" "PQA-RT" "SUPG-PT" "SUPG-RT" "TopK")
version_name="test0217"

# Define attributes, IDs, beta values, and cost settings for each dataset
declare -A attr_map
declare -A attr_id_map
declare -A beta_map
declare -A initial_cost_map
declare -A total_cost_map

# Attribute mappings
attr_map["icd9_eICU"]="age"
attr_map["icd9_mimic"]="HeartRate"
attr_map["Health_and_Household"]="ratings"
attr_map["Electronics"]="ratings"
attr_map["night-street"]="carSpeed"

# Attribute ID mappings
attr_id_map["icd9_eICU"]=1
attr_id_map["icd9_mimic"]=0
attr_id_map["Health_and_Household"]=0
attr_id_map["Electronics"]=0
attr_id_map["night-street"]=0

# Beta mappings based on agg function
beta_map["avg"]=0.5
beta_map["var"]=2
beta_map["pct"]=1
beta_map["sum"]=1

# Cost mappings based on dataset
initial_cost_map["icd9_mimic"]=150
total_cost_map["icd9_mimic"]=500

initial_cost_map["Electronics"]=1200
total_cost_map["Electronics"]=2000

initial_cost_map["Health_and_Household"]=2000
total_cost_map["Health_and_Household"]=3000

# Default cost values for other datasets
for fname in "${fname_list[@]}"; do
    if [[ -z "${initial_cost_map[$fname]}" ]]; then
        initial_cost_map[$fname]=600
        total_cost_map[$fname]=1000
    fi
done

# Loop through all combinations
for algo in "${algo_list[@]}"; do
    for agg in "${agg_list[@]}"; do
        # Determine hypothesis_type based on agg value
        if [[ "$agg" == "pct" ]]; then
            hypothesis_type="P-NNH"
        else
            hypothesis_type="NNH"
        fi

        # Get the correct beta value
        beta=${beta_map[$agg]}

        for fname in "${fname_list[@]}"; do
            # Get the correct attr, attr_id, initial_cost, and total_cost
            attr=${attr_map[$fname]}
            attr_id=${attr_id_map[$fname]}
            initial_cost=${initial_cost_map[$fname]}
            total_cost=${total_cost_map[$fname]}

            # Define log file name
            log_file="./results/${algo}/${agg}_${fname}_${version_name}.log"

            # Create results directory if not exists
            mkdir -p "./results/${algo}"

            # Run the Python script with nohup
            echo "Running: algo=$algo, agg=$agg, hypothesis_type=$hypothesis_type, Fname=$fname, attr=$attr, attr_id=$attr_id, beta=$beta, initial_cost=$initial_cost, total_cost=$total_cost"
            nohup ./venv/bin/python3 "$python_script" \
                --algo "$algo" \
                --agg "$agg" \
                --hypothesis_type "$hypothesis_type" \
                --Fname "$fname" \
                --attr "$attr" \
                --attr_id "$attr_id" \
                --beta "$beta" \
                --version "$version" \
                --initial_cost "$initial_cost" \
                --total_cost "$total_cost" > "$log_file" 2>&1 &
        done
    done
done

echo "All experiments are running in the background. Check logs in ./results/{algo}/"
