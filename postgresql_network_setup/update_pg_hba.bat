@echo off
echo Updating PostgreSQL access control configuration...

REM Create a backup if not already done
if not exist "C:\PostgreSQL\data\pg_hba.conf.bak" (
    copy "C:\PostgreSQL\data\pg_hba.conf" "C:\PostgreSQL\data\pg_hba.conf.bak"
)

REM Add a new line to allow connections from the local network (192.168.0.0/16)
echo. >> "C:\PostgreSQL\data\pg_hba.conf"
echo # Allow connections from local network for Agentic RAG application >> "C:\PostgreSQL\data\pg_hba.conf"
echo host    all             all             192.168.0.0/16           scram-sha-256 >> "C:\PostgreSQL\data\pg_hba.conf"

REM Add a new line to allow connections from localhost with password authentication
echo host    all             all             0.0.0.0/0                scram-sha-256 >> "C:\PostgreSQL\data\pg_hba.conf"

echo Configuration updated successfully!
echo.
echo Please restart PostgreSQL for the changes to take effect:
echo pg_ctl -D "C:/PostgreSQL/data" restart
echo.
pause 