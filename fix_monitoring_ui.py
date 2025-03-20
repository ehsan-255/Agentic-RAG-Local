"""
Script to fix indentation issues in monitoring_ui.py
"""

with open('src/ui/monitoring_ui.py', 'r') as f:
    content = f.read()

# Fix indentation issues
content = content.replace("            with col1:", "    with col1:")
content = content.replace("            \"Total API Calls\", ", "            \"Total API Calls\",")
content = content.replace("        display_task_monitoring()", "            display_task_monitoring()")

# Write fixed content back to the file
with open('src/ui/monitoring_ui.py', 'w') as f:
    f.write(content)

print("Fixed indentation issues in monitoring_ui.py") 