#!/bin/bash

# Usage: ./export_mermaid.sh simulator_flow.mmd

INPUT="$1"
BASENAME=$(basename "$INPUT" .mmd)

# Export PNG (high resolution)
mmdc -i "$INPUT" -o "${BASENAME}.png" --width 2000

# Export SVG (perfect for notebooks and PDF export)
mmdc -i "$INPUT" -o "${BASENAME}.svg"

echo "✅ Exported: ${BASENAME}.png and ${BASENAME}.svg"
