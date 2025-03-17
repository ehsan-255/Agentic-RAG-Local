@echo off
echo Checking Windows Firewall Configuration for PostgreSQL...
echo.

REM Check if PostgreSQL firewall rule exists
netsh advfirewall firewall show rule name="PostgreSQL" > firewall_check.txt
findstr /c:"PostgreSQL" firewall_check.txt > nul
if %errorlevel% equ 0 (
    echo PostgreSQL firewall rule already exists.
    type firewall_check.txt
) else (
    echo PostgreSQL firewall rule does not exist.
    echo.
    echo Would you like to create a firewall rule for PostgreSQL? (Y/N)
    set /p create_rule=
    if /i "%create_rule%"=="Y" (
        echo Creating firewall rule for PostgreSQL...
        netsh advfirewall firewall add rule name="PostgreSQL" dir=in action=allow protocol=TCP localport=5432 > nul
        echo Firewall rule created successfully!
    ) else (
        echo Firewall rule creation skipped.
    )
)

echo.
echo Checking if port 5432 is open...
netstat -an | findstr ":5432" > port_check.txt
type port_check.txt

echo.
echo Firewall check completed!
pause 