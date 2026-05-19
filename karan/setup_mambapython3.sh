#!/bin/bash
# setup_mambapython3.sh
# Create a symbolic link to mambaforge python3.10 as mambapython3

# Target interpreter
TARGET="/z1-scratch/mambaforge/bin/python3.10"

# Where to place the symlink (user-local bin)
LINK_DIR="$HOME/bin"
LINK_NAME="mambapython3"
LINK_PATH="$LINK_DIR/$LINK_NAME"

# Ensure bin directory exists
mkdir -p "$LINK_DIR"

# Remove existing link if it exists
if [ -L "$LINK_PATH" ] || [ -e "$LINK_PATH" ]; then
    echo "Removing existing $LINK_PATH"
    rm -f "$LINK_PATH"
fi

# Create new symlink
ln -s "$TARGET" "$LINK_PATH"
echo "✅ Created symlink: $LINK_PATH -> $TARGET"

# Ensure ~/bin is in PATH
if [[ ":$PATH:" != *":$LINK_DIR:"* ]]; then
    echo "Adding $LINK_DIR to PATH in ~/.bashrc"
    echo "export PATH=\$HOME/bin:\$PATH" >> ~/.bashrc
    source ~/.bashrc
fi

# Show result
echo "Now you can run: $LINK_NAME --version"
