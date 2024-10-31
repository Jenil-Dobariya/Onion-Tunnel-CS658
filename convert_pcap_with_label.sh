#!/bin/bash

# Check if an input directory path is provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <path_to_directory_with_pcap_files> <label>"
    exit 1
fi

input_dir="$1"
label="$2"

# Check if the provided path is a directory
if [ ! -d "$input_dir" ]; then
    echo "Please enter a valid directory containing .pcap files"
    exit 1
fi

# Create the main data directory and necessary subdirectories
mkdir -p ./data/in
mkdir -p ./data/out

for pcap_file in "$input_dir"/*.pcap; do
    # Clear the ./data/in directory
    rm -f ./data/in/*

    # Copy the current .pcap file to ./data/in
    cp "$pcap_file" ./data/in/

    # Execute the CICFlowMeter for the current .pcap file
    sudo ./pcap_to_csv_convertor/bin/CICFlowMeter
done

# Check if any CSV files were generated
if [ -z "$(ls -A ./data/out/*.csv 2>/dev/null)" ]; then
    echo "CSV files not generated. Please check CICFlowMeter execution."
    rm -rf ./data
    exit 1
fi

output_csv="./output.csv"
header_saved=true

for csv_file in ./data/out/*.csv; do
    if [ "$header_saved" = false ]; then
        # Copy header and content if it's the first file
        cat "$csv_file" > "$output_csv"
        header_saved=true
    else
        # Append only the content (skip header) for subsequent files
        tail -n +2 "$csv_file" >> "$output_csv"
    fi
done

sed -i "s/No Label/$label/g" "$output_csv"

# Clean up the ./data directory and logs
rm -rf ./data
sudo rm -rf ./logs