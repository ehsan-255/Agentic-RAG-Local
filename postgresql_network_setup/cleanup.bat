@echo off
echo Cleaning up temporary files...

REM Delete temporary files
del firewall_check.txt 2>nul
del port_check.txt 2>nul
del local_version.txt 2>nul
del ip_version.txt 2>nul
del vector_test.txt 2>nul
del version.txt 2>nul
del data_dir.txt 2>nul
del listen_addresses.txt 2>nul

echo Cleanup completed!
echo.
pause 