@echo off
rem Starts the server so that your other devices on your Tailscale network can reach it too, on port 8765, without opening a browser.
rem There is no login: anyone on your tailnet can use the app and see your runs and datasets.
call "%~dp0serve.bat" --tailscale --no-browser --port 8765 %*
