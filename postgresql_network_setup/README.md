# PostgreSQL Network Configuration for Agentic RAG System

This directory contains scripts to configure PostgreSQL for network access, which is required for the Agentic RAG system.

## Prerequisites

- PostgreSQL 17 installed
- pgvector extension installed
- Administrator privileges on your computer

## Configuration Steps

1. **Run the Master Configuration Script**

   Right-click on `configure_postgresql_network.bat` and select "Run as administrator". This script will:
   
   - Configure PostgreSQL to listen on all network interfaces
   - Allow connections from your local network
   - Increase max_connections for better performance
   - Restart PostgreSQL to apply changes

2. **Update Environment Variables**

   Run `update_env_file.bat` to update your `.env` file with the PostgreSQL connection details.

3. **Test the Connection**

   Run `test_postgresql_connection.bat` to verify that PostgreSQL is properly configured and accessible.

## Individual Scripts

If you prefer to run the steps individually:

- `update_postgresql_conf.bat` - Updates the PostgreSQL configuration file
- `update_pg_hba.conf.bat` - Updates the client authentication configuration
- `check_firewall.bat` - Checks and configures Windows Firewall for PostgreSQL
- `restart_postgresql.bat` - Restarts the PostgreSQL service
- `update_env_file.bat` - Updates the .env file with connection details
- `test_postgresql_connection.bat` - Tests the PostgreSQL connection
- `cleanup.bat` - Removes temporary files created during the configuration process

## Troubleshooting

If you encounter issues:

1. Check that PostgreSQL service is running
2. Verify that Windows Firewall allows connections on port 5432
3. Ensure you're running the scripts with administrator privileges
4. Check the PostgreSQL logs at `C:\PostgreSQL\data\log`

## Reverting Changes

If you need to revert the changes:

1. Copy `C:\PostgreSQL\data\postgresql.conf.bak` to `C:\PostgreSQL\data\postgresql.conf`
2. Copy `C:\PostgreSQL\data\pg_hba.conf.bak` to `C:\PostgreSQL\data\pg_hba.conf`
3. Restart PostgreSQL

## Next Steps

After configuring PostgreSQL for network access:

1. Create the necessary database schema for your Agentic RAG system
2. Configure your application to connect to PostgreSQL
3. Test the full system functionality 