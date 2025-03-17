# GitHub Repository Setup Guide

This guide explains how to set up a GitHub repository for your Agentic RAG project, following best practices for Python projects.

## Creating a New GitHub Repository

1. **Log in to GitHub** and navigate to your profile.

2. **Click on "New repository"** or go to https://github.com/new.

3. **Fill in the repository details**:
   - Repository name: `agentic-rag-local` (or your preferred name)
   - Description: "An intelligent documentation crawler and RAG agent"
   - Visibility: Public or Private based on your preference
   - Initialize with:
     - ✅ Add a README file
     - ✅ Add .gitignore (select Python template)
     - ✅ Choose a license (MIT is a good choice for open-source projects)

4. **Click "Create repository"**.

## Setting Up Your Local Project

1. **Clone the repository** to your local machine:
   ```bash
   git clone https://github.com/yourusername/agentic-rag-local.git
   cd agentic-rag-local
   ```

2. **Copy your project files** into the repository folder, following the structure:
   ```
   agentic-rag-local/
   ├── .env.example
   ├── .gitignore
   ├── LICENSE
   ├── README.md
   ├── requirements.txt
   ├── setup.py
   ├── data/
   │   └── site_pages.sql
   ├── docs/
   │   ├── installation.md
   │   └── usage.md
   ├── src/
   │   ├── __init__.py
   │   ├── crawling/
   │   │   ├── __init__.py
   │   │   └── docs_crawler.py
   │   ├── db/
   │   │   ├── __init__.py
   │   │   └── schema.py
   │   ├── rag/
   │   │   ├── __init__.py
   │   │   └── rag_expert.py
   │   └── ui/
   │       ├── __init__.py
   │       └── streamlit_app.py
   └── tests/
       ├── __init__.py
       └── test_rag.py
   ```

3. **Add your files** to the repository:
   ```bash
   git add .
   ```

4. **Create your first commit**:
   ```bash
   git commit -m "Initial project structure"
   ```

5. **Push to GitHub**:
   ```bash
   git push origin main
   ```

## Best Practices for GitHub Projects

### 1. Branch Management

- Use the **main** branch for stable, production-ready code
- Create **feature branches** for new features:
  ```bash
  git checkout -b feature/new-feature-name
  ```
- Create **bugfix branches** for bug fixes:
  ```bash
  git checkout -b fix/bug-description
  ```

### 2. Pull Requests

- Always create a pull request for merging changes
- Request code reviews from team members
- Use descriptive titles and detailed descriptions
- Link related issues in the description

### 3. Issues

- Use GitHub Issues to track bugs, features, and tasks
- Label issues appropriately (bug, enhancement, documentation, etc.)
- Assign issues to team members
- Use milestones to group related issues

### 4. Documentation

- Keep the README.md updated with:
  - Project description
  - Installation instructions
  - Usage examples
  - Contribution guidelines
- Maintain separate documentation in the docs/ folder

### 5. CI/CD (Optional)

Consider setting up GitHub Actions for:
- Running tests automatically
- Checking code style
- Building and deploying the application

Example GitHub Actions workflow file (`.github/workflows/python-app.yml`):

```yaml
name: Python application

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python 3.11
      uses: actions/setup-python@v2
      with:
        python-version: 3.11
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
        pip install pytest
    - name: Test with pytest
      run: |
        pytest
```

### 6. Security

- Never commit sensitive information like API keys
- Use environment variables for secrets
- Consider using GitHub Secrets for CI/CD workflows

By following these best practices, you'll create a well-structured, maintainable GitHub repository that follows industry standards for Python projects. 