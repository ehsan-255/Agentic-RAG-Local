@echo off
echo Updating .env file with PostgreSQL connection details...

REM Create a backup of the .env file
copy .env .env.bak

REM Get the computer's IP address
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4 Address"') do (
    set IP_ADDRESS=%%a
    goto :found_ip
)
:found_ip
set IP_ADDRESS=%IP_ADDRESS:~1%

REM Update the .env file with PostgreSQL connection details
echo.>> .env
echo # PostgreSQL Connection Details>> .env
echo POSTGRES_HOST=%IP_ADDRESS%>> .env
echo POSTGRES_PORT=5432>> .env
echo POSTGRES_USER=postgres>> .env
echo POSTGRES_PASSWORD=9340>> .env
echo POSTGRES_DB=postgres>> .env
echo POSTGRES_CONNECTION_STRING=postgresql://postgres:9340@%IP_ADDRESS%:5432/postgres>> .env

echo .env file updated successfully!
echo.
echo Your PostgreSQL connection details:
echo Host: %IP_ADDRESS%
echo Port: 5432
echo User: postgres
echo Password: 9340
echo Database: postgres
echo Connection String: postgresql://postgres:9340@%IP_ADDRESS%:5432/postgres
echo.
pause 