@echo off
copy /Y vercel\pyproject.toml pyproject.toml >nul
copy /Y vercel\vercel.json vercel.json >nul
echo Synced vercel config to repo root.
