#!/bin/bash

# Default data directory
DATA_DIR="./"

DS_URL="https://physionet.org/static/published-projects/sleep-accel/motion-and-heart-rate-from-a-wrist-worn-wearable-and-labeled-sleep-from-polysomnography-1.0.0.zip"
# Define 
RAW_DS="$DATA_DIR/motion-and-heart-rate-from-a-wrist-worn-wearable-and-labeled-sleep-from-polysomnography-1.0.0"
DS_DIR="sleep-accel"
ZIP_FILE="sleep-accel-dataset.zip"


# Function to download files
download() {
    echo "Downloading files to $DATA_DIR..."
    
    wget "$DS_URL" -O "$DATA_DIR/$ZIP_FILE"
    
    echo "Download complete."
}

# Function to prepare files
prepare() {
    echo "Preparing files in $DATA_DIR..."

    # Unzip the dataset
    unzip "$DATA_DIR/$ZIP_FILE" -d "$DATA_DIR"

    # Iterate over the subdirectories (heart_rate, labels, motion, steps)
    for SUBDIR in heart_rate labels motion steps; do
        for FILE in "$RAW_DS/$SUBDIR"/*; do
            # Extract the user ID from the filename
            USER_ID=$(basename "$FILE" | cut -d'_' -f1)

            # Create the user directory if it doesn't exist
            USER_DIR="$DATA_DIR/$DS_DIR/$USER_ID"
            mkdir -p "$USER_DIR"

            # Define the destination filename based on the subdirectory
            case $SUBDIR in
                heart_rate)
                    DEST_FILE="$USER_DIR/heartrate.txt"
                    ;;
                labels)
                    DEST_FILE="$USER_DIR/labels.txt"
                    ;;
                motion)
                    DEST_FILE="$USER_DIR/acceleration.txt"
                    ;;
                steps)
                    DEST_FILE="$USER_DIR/steps.txt"
                    ;;
                *)
                    echo "Unknown subdirectory: $SUBDIR"
                    exit 1
                    ;;
            esac

            # Copy the file to the user's directory
            cp "$FILE" "$DEST_FILE"
        done
    done

    echo "Preparation complete."
}

# Function to cleanup the data directory.
cleanup() {
    echo "Removing temporary files..."

    # Remove the unzipped dataset
    rm -rf "$RAW_DS"
    rm -rf "$ZIP_FILE"

    echo "Cleanup complete."
}

# Function to display usage instructions
usage() {
    echo "Download and process the PhysioNet Sleep-Accel dataset"
    echo "Usage: $0 [--data-dir PATH] [COMMAND]"
    echo "Commands:"
    echo "  download   Download the necessary files."
    echo "  prepare    Prepare the downloaded files."
    echo "  cleanup    Remove temporary files."
    echo "  -h, --help Display this help message."
    echo "Options:"
    echo "  --data-dir PATH   Specify the data directory (default: ./data)."
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --data-dir)
            DATA_DIR="$2"
            shift # past argument
            shift # past value
            ;;
        download)
            COMMAND="download"
            shift
            ;;
        prepare)
            COMMAND="prepare"
            shift
            ;;
	cleanup)
	    COMMAND="cleanup"
	    shift
	    ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Error: Unknown command or option: $1"
            usage
            ;;
    esac
done

# Ensure the data directory exists
if [ ! -d "$DATA_DIR" ]; then
    mkdir -p "$DATA_DIR" || { echo "Failed to create directory $DATA_DIR"; exit 1; }
fi

# Execute the command
if [[ -z "$COMMAND" ]]; then
    download
    prepare
    cleanup
else
    case "$COMMAND" in
        download)
            download
            ;;
        prepare)
            prepare
            ;;
	cleanup)
	    cleanup
	    ;;
        *)
            echo "Error: Unknown command: $COMMAND"
            usage
            ;;
    esac
fi
