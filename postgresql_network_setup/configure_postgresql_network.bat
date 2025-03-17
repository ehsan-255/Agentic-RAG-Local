@echo off
echo PostgreSQL Network Configuration for Agentic RAG System
echo =====================================================
echo.
echo This script will configure PostgreSQL to:
echo 1. Listen on all network interfaces
echo 2. Allow connections from your local network
echo 3. Increase max_connections for better performance
echo 4. Configure Windows Firewall
echo 5. Restart PostgreSQL to apply changes
echo.
echo IMPORTANT: This script requires administrator privileges
echo.
pause

REM Run the postgresql.conf update script
call %~dp0update_postgresql_conf.bat

REM Run the pg_hba.conf update script
call %~dp0update_pg_hba.bat

REM Check and configure Windows Firewall
call %~dp0check_firewall.bat

REM Restart PostgreSQL
call %~dp0restart_postgresql.bat

REM Update .env file with PostgreSQL connection details
call %~dp0update_env_file.bat

REM Test the PostgreSQL connection
call %~dp0test_postgresql_connection.bat

echo.
echo Configuration complete!
echo Your PostgreSQL server is now configured for the Agentic RAG system.
echo.
echo Next steps:
echo 1. Create the necessary database schema for your Agentic RAG system
echo 2. Configure your application to connect to PostgreSQL
echo 3. Test the full system functionality
echo.
pause 