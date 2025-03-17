@echo off
echo Checking PostgreSQL network configuration...

REM Create a backup if not already done
if not exist "C:\PostgreSQL\data\postgresql.conf.bak" (
    copy "C:\PostgreSQL\data\postgresql.conf" "C:\PostgreSQL\data\postgresql.conf.bak"
)

REM Check if listen_addresses is already set to '*'
findstr /C:"listen_addresses = '*'" "C:\PostgreSQL\data\postgresql.conf" > nul
if %errorlevel% equ 0 (
    echo PostgreSQL is already configured to listen on all interfaces.
) else (
    echo Updating PostgreSQL to listen on all interfaces...
    powershell -Command "(Get-Content 'C:\PostgreSQL\data\postgresql.conf') -replace '#listen_addresses = ''localhost''', 'listen_addresses = ''*''' | Set-Content 'C:\PostgreSQL\data\postgresql.conf'"
)

REM Check if port is set to 5432
findstr /C:"port = 5432" "C:\PostgreSQL\data\postgresql.conf" > nul
if %errorlevel% equ 0 (
    echo PostgreSQL port is already set to 5432.
) else (
    echo Setting PostgreSQL port to 5432...
    powershell -Command "(Get-Content 'C:\PostgreSQL\data\postgresql.conf') -replace '#port = 5432', 'port = 5432' | Set-Content 'C:\PostgreSQL\data\postgresql.conf'"
)

REM Increase max_connections for better performance with RAG system
echo Increasing max_connections for better performance...
powershell -Command "(Get-Content 'C:\PostgreSQL\data\postgresql.conf') -replace 'max_connections = 100', 'max_connections = 200' | Set-Content 'C:\PostgreSQL\data\postgresql.conf'"

echo Configuration updated successfully!
echo.
echo Please restart PostgreSQL for the changes to take effect:
echo pg_ctl -D "C:/PostgreSQL/data" restart
echo.
pause 