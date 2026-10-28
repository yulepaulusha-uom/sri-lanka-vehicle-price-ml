#!/bin/sh
set -eu
exec python -m streamlit run app.py --server.port 8000 --server.address 0.0.0.0 --server.headless true
