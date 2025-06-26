#!/bin/bash

python_script="SPRinT.py"

# Define parameter values
agg_list=("avg" "var" "pct" "sum" "min" "max" "median" "count") 

fname_list=("eICU" "MIMIC-III" "Jigsaw" "yelp" "Electronics") 
algo_list=("SPRinT-C" "SPRinT-V" "PQA-PT" "PQA-RT" "SUPG-PT" "SUPG-RT" "TopK")
file_suffix="test"
scalability_factor=0

# Define attributes, IDs, and cost settings for each dataset
declare -A attr_map
declare -A attr_id_map
declare -A s_p_map
declare -A s_map

# Attribute mappings
attr_map["eICU"]="age"
attr_map["MIMIC-III"]="HeartRate"
attr_map["Jigsaw"]="downvote"
attr_map["yelp"]="rating"
attr_map["Electronics"]="rating"

# Attribute ID mappings
attr_id_map["eICU"]=1
attr_id_map["MIMIC-III"]=0
attr_id_map["Jigsaw"]=0
attr_id_map["yelp"]=0
attr_id_map["Electronics"]=0

# Cost mappings based on dataset
s_p_map["MIMIC-III"]=150
s_map["MIMIC-III"]=500

s_p_map["Electronics"]=1200
s_map["Electronics"]=2000

# Default cost values for other datasets
for fname in "${fname_list[@]}"; do
    if [[ -z "${s_p_map[$fname]}" ]]; then
        s_p_map[$fname]=600
        s_map[$fname]=1000
    fi
done

# Loop through all combinations
for algo in "${algo_list[@]}"; do
    for agg in "${agg_list[@]}"; do
        # Determine hypothesis_type based on agg value
        if [[ "$agg" == "pct" || "$agg" == "count" ]]; then
            hypothesis_type="P-NNH"
        else
            hypothesis_type="NNH"
        fi

        # Get the correct beta value
        # beta=${beta_map[$agg]}

        for fname in "${fname_list[@]}"; do
            # Dist_t=0.8
            # Set Dist_t based on the dataset name
            if [[ "$fname" == "Jigsaw" ]]; then
                Dist_t=0.5
            elif [[ "$fname" == "eICU" ]]; then
                Dist_t=0.8
            elif [[ "$fname" == "MIMIC-III" ]]; then
                Dist_t=0.9
            elif [[ "$fname" == "Electronics" ]]; then
                Dist_t=0.6
            elif [[ "$fname" == "yelp" ]]; then
                Dist_t=0.7
            else
                Dist_t=0.8
            fi

            # Get the correct attr, attr_id, s_p, and s
            attr=${attr_map[$fname]}
            attr_id=${attr_id_map[$fname]}
            s_p=${s_p_map[$fname]}
            s=${s_map[$fname]}

            # Define log file name
            log_file="./results/${algo}/${agg}_${fname}_${file_suffix}.log"

            # Create results directory if not exists
            mkdir -p "./results/${algo}"

            # Run the Python script with nohup
            echo "Running: algo=$algo, agg=$agg, hypothesis_type=$hypothesis_type, Fname=$fname, attr=$attr, attr_id=$attr_id, scalability_factor=$scalability_factor, s_p=$s_p, s=$s"
            nohup ./venv/bin/python3 "$python_script" \
                --algo "$algo" \
                --agg "$agg" \
                --hypothesis_type "$hypothesis_type" \
                --Fname "$fname" \
                --attr "$attr" \
                --attr_id "$attr_id" \
                --Dist_t "$Dist_t" \
                --scalability_factor "$scalability_factor" \
                --file_suffix "$file_suffix" \
                --s_p "$s_p" \
                --s "$s" > "$log_file" 2>&1 &
        done
    done
done

echo "All experiments are running in the background. Check logs in ./results/{algo}/"
