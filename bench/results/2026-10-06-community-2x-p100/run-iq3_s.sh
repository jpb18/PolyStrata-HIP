#!/bin/sh
cd "/home/user/Strata"
exec "/home/user/Strata/.venv/bin/python" "/home/user/Strata/serve/server.py" "--engine" "strata" "--config" "/home/user/Strata/strata-iq3_s.json" "--port" "8080" "--open"
