#!/bin/bash

# ==== Configuration ====
REMOTE_USER=""
REMOTE_HOST=""        
REMOTE_DIR=""
LOCAL_DIR=""           
SCRIPT_NAME=""                



# ==== Sync Local Project to Remote ====
echo "Syncing local files to remote..."
rsync -avz --exclude-from='utils/.rsyncignore'  "$LOCAL_DIR/" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR"

# ==== Launch Python script on Remote ====
echo "Running script on remote server..."
ssh -t $REMOTE_USER@$REMOTE_HOST "cd \"$REMOTE_DIR\" && uv run -m "$SCRIPT_NAME""

