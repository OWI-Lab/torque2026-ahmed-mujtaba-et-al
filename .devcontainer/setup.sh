#!/bin/bash

# Devcontainer setup script for Torque Dynamic Table
# This script installs UV and syncs the Python environment

set -e

echo "🚀 Setting up Torque Dynamic Table development environment..."

# Install UV
echo "📦 Installing UV..."
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add UV to PATH for current session
export PATH="$HOME/.local/bin:$PATH"

# Add UV to PATH permanently
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# Verify UV installation
echo "✅ UV installed successfully:"
~/.local/bin/uv --version

# Sync the environment
echo "🔄 Syncing Python environment..."
~/.local/bin/uv sync --all-packages --all-extras --all-groups 

echo "🎉 Development environment setup complete!"
echo "💡 You can now run 'uv run jupyter lab' to start the analysis dashboard"
