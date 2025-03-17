@echo off
echo Testing PostgreSQL Connection...
echo.

REM Get the computer's IP address
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4 Address"') do (
    set IP_ADDRESS=%%a
    goto :found_ip
)
:found_ip
set IP_ADDRESS=%IP_ADDRESS:~1%

echo Your PostgreSQL connection details:
echo Host: %IP_ADDRESS%
echo Port: 5432
echo User: postgres
echo Password: 9340
echo Database: postgres
echo.

REM Test connection using psql
set PGPASSWORD=9340
echo Testing local connection...
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -h localhost -U postgres -c "SELECT version();" > local_version.txt
type local_version.txt
echo.

echo Testing connection using IP address...
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -h %IP_ADDRESS% -U postgres -c "SELECT version();" > ip_version.txt
type ip_version.txt
echo.

echo Testing vector extension...
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -h %IP_ADDRESS% -U postgres -c "CREATE EXTENSION IF NOT EXISTS vector; SELECT * FROM pg_extension WHERE extname = 'vector';" > vector_test.txt
type vector_test.txt
echo.

echo Connection tests completed!
pause 