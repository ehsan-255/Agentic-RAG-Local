@echo off
echo Running cleanup of temporary files...
echo.
call postgresql_network_setup\cleanup.bat
echo.
echo Cleanup process completed.
pause 