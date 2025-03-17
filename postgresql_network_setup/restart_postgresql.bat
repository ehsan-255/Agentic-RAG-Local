@echo off
echo Restarting PostgreSQL service...

REM Stop the PostgreSQL service
net stop postgresql-x64-17

REM Wait a moment
timeout /t 5

REM Start the PostgreSQL service
net start postgresql-x64-17

echo PostgreSQL service has been restarted.
echo.
echo Verifying PostgreSQL is running and listening...
"C:\Program Files\PostgreSQL\17\bin\pg_isready.exe"

echo.
echo Testing connection...
set PGPASSWORD=9340
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres -c "SELECT version();" > version.txt
type version.txt

echo.
echo Network configuration is complete!
pause 