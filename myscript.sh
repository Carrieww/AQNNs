#!/bin/bash

python_script="./SPRinT.py"

# Define parameter values
agg_list=("pct" "avg" "sum" "var")

fname_list=("Jigsaw") # "eICU" "MIMIC-III" "Jackson" "Jigsaw"
algo_list=("SPRinT")  # "SUPG-PT" "SUPG-RT" "PQA-PT" "PQA-RT" "TopK"
file_suffix="test"
scalability_factor=0

# Define attributes, IDs, beta values, and cost settings for each dataset
declare -A attr_map
declare -A attr_id_map
declare -A beta_map
declare -A s_p_map
declare -A s_map

# Attribute mappings
attr_map["eICU"]="age"
attr_map["MIMIC-III"]="HeartRate"
attr_map["Amazon-HH"]="ratings"
attr_map["Amazon-E"]="ratings"
attr_map["Jackson"]="carSpeed"
attr_map["Jigsaw"]="downvote"

# Attribute ID mappings
attr_id_map["eICU"]=1
attr_id_map["MIMIC-III"]=0
attr_id_map["Amazon-HH"]=0
attr_id_map["Amazon-E"]=0
attr_id_map["Jackson"]=0
attr_id_map["Jigsaw"]=0

# Beta mappings based on agg function
beta_map["avg"]=0.5
beta_map["var"]=2
beta_map["pct"]=1
beta_map["sum"]=1

# Cost mappings based on dataset
s_p_map["MIMIC-III"]=150
s_map["MIMIC-III"]=500

s_p_map["Amazon-E"]=1200
s_map["Amazon-E"]=2000

s_p_map["Amazon-HH"]=2000
s_map["Amazon-HH"]=3000


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
        if [[ "$agg" == "pct" ]]; then
            hypothesis_type="P-NNH"
        else
            hypothesis_type="NNH"
        fi

        # Get the correct beta value
        beta=${beta_map[$agg]}

        for fname in "${fname_list[@]}"; do
            # Set Dist_t based on the dataset name
            if [[ "$fname" == "Jigsaw" ]]; then
                Dist_t=0.3
            elif [[ "$fname" == "MIMIC-III" ]]; then
                Dist_t=0.9
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
            echo "Running: algo=$algo, agg=$agg, hypothesis_type=$hypothesis_type, Fname=$fname, attr=$attr, attr_id=$attr_id, beta=$beta, scalability_factor=$scalability_factor, s_p=$s_p, s=$s"
            nohup ./venv/bin/python3 "$python_script" \
                --algo "$algo" \
                --agg "$agg" \
                --hypothesis_type "$hypothesis_type" \
                --Fname "$fname" \
                --attr "$attr" \
                --attr_id "$attr_id" \
                --beta "$beta" \
                --Dist_t "$Dist_t" \
                --scalability_factor "$scalability_factor" \
                --file_suffix "$file_suffix" \
                --s_p "$s_p" \
                --s "$s" > "$log_file" 2>&1 &
        done
    done
done

echo "All experiments are running in the background. Check logs in ./results/{algo}/"
