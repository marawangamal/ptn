#!/bin/bash

# Check if a node prefix is provided, otherwise default to all nodes
node_prefix="${1:-}"

if [[ -z "$node_prefix" ]]; then
    echo "Usage: $0 <node_prefix>"
    echo "Example: $0 haha"
    exit 1
fi

# Get the unique list of nodes matching the prefix
nodes=$(sinfo --Node | grep "^${node_prefix}" | awk '{print $1}' | sort | uniq)

# Check if any nodes were found
if [[ -z "$nodes" ]]; then
    echo "No nodes found matching prefix '${node_prefix}'"
    exit 1
fi

# Get the longest node name for alignment
max_node_length=$(echo "$nodes" | awk '{print length}' | sort -nr | head -n 1)

# Collect all rows for sorting
temp_file=$(mktemp)
printf "%-${max_node_length}s  GPU     CPU     Mem\n" "NODE" > "$temp_file"

# Iterate through each unique node and fetch required details
for node in $nodes; do
    # Get total GPUs
    total_gpus=$(scontrol show node $node | grep -oP 'Gres=gpu(:\w+)*:\K[0-9]+' || echo "0")
    
    # Get allocated GPUs from AllocTRES
    allocated_gpus=$(scontrol show node $node | grep -oP 'AllocTRES=.*gres/gpu=\K[0-9]+' || echo "0")
    
    # Calculate available GPUs
    available_gpus=$((total_gpus - allocated_gpus))
    
    # Get total CPUs
    total_cpus=$(scontrol show node $node | grep -oP 'CPUTot=\K[0-9]+' || echo "0")
    
    # Get allocated CPUs from AllocTRES
    allocated_cpus=$(scontrol show node $node | grep -oP 'AllocTRES=cpu=\K[0-9]+' || echo "0")
    
    # Calculate available CPUs
    available_cpus=$((total_cpus - allocated_cpus))
    
    # Get total memory in MB
    total_mem_mb=$(scontrol show node $node | grep -oP 'RealMemory=\K[0-9]+' || echo "0")
    total_mem_gb=$((total_mem_mb / 1024))  # Convert to GB
    
    # Get allocated memory in MB from AllocTRES
    allocated_mem_mb=$(scontrol show node $node | grep -oP 'AllocTRES=.*mem=\K[0-9]+[A-Z]*' | sed 's/G//;s/M//' || echo "0")
    
    # Ensure allocated memory is in GB for consistency
    if [[ $allocated_mem_mb =~ G$ ]]; then
        allocated_mem_gb=${allocated_mem_mb//G/}
    elif [[ $allocated_mem_mb =~ M$ ]]; then
        allocated_mem_gb=$((allocated_mem_mb / 1024))
    else
        allocated_mem_gb=$allocated_mem_mb
    fi

    # Calculate available memory in GB
    available_mem_gb=$((total_mem_gb - allocated_mem_gb))
    
    # Collect the row for sorting
    printf "%-${max_node_length}s  %d/%d   %d/%d   %dGB/%dGB\n" \
        "$node" "$available_gpus" "$total_gpus" "$available_cpus" "$total_cpus" "$available_mem_gb" "$total_mem_gb" >> "$temp_file"
done

# Sort by available GPUs (column 2) in descending order and print
tail -n +2 "$temp_file" | sort -k2,2nr | cat <(head -n 1 "$temp_file") - 

# Clean up temporary file
rm -f "$temp_file"