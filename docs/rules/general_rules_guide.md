# Comprehensive Guide to Cursor Rules: Best Practices and Implementation

This guide presents an optimized approach to Cursor Rules based on current best practices as of March 2025. It incorporates your existing rules while enhancing them with structural improvements and additional guidelines from the developer community.

## 1. Understanding Cursor Rules

### Types of Cursor Rules

1. **Global Rules**
   - Set in Cursor Settings under "General" > "Rules for AI"
   - Apply across all projects and act as a system prompt
   - Function as personal coding preferences and consistent behavior guidelines

2. **Project Rules**
   - Stored as `.mdc` files in the `.cursor/rules` directory
   - Project-specific instructions that help AI understand codebase conventions
   - Added via Cursor Settings > General > Project Rules or directly through the editor
   - `.cursorrules` is applied project-wide and edited directly in the editor

3. **Rule Activation Mechanisms**
   - **alwaysApply**: Ensures rules are injected into the context unconditionally
   - **globs**: Matches files based on specific patterns, activating rules when relevant
   - **description**: Defines scenarios where rules should apply

---

## 2. General Communication Rules

### Tone and Approach
- Be casual unless otherwise specified
- Be terse and concise
- Suggest solutions I didn't think about—anticipate my needs
- Treat me as an expert
- Be accurate and thorough

### Response Structure
- Give the answer immediately, then provide explanations if necessary
- Value good arguments over authorities; the source is irrelevant
- You may use speculation or prediction, just flag it clearly
- No moral lectures
- Discuss safety only when crucial and non-obvious

### Content Handling
- If content policy is an issue, provide the closest acceptable response
- Cite sources when possible at the end, not inline
- No need to mention knowledge cutoff
- No need to disclose you're an AI
- Split into multiple responses if one response isn't enough

---

## 3. Code Quality Standards

### Core Principles
- Focus on readability over performance
- Fully implement all requested functionality
- Leave NO todos, placeholders, or missing pieces
- Write well-explained comments throughout new or changed code
- Always prefer simple solutions
- Avoid duplication of code, check for existing similar functionality
- Keep the codebase clean and organized
- Avoid files over 200-300 lines; refactor at that point

### Architecture and Design
- Use functional and declarative programming patterns where appropriate
- Enforce modular design by specifying guidelines for separating concerns
- Implement proper documentation through comments and references
- Maintain consistent naming conventions across the project
- Only modify code directly relevant to the specific request

### Type Safety and Validation
- Require type hints for all function signatures (especially in Python)
- Validate inputs using appropriate validation techniques
- Use defensive programming principles for external data

---

## 4. Development Workflow

### Task Approach
- Focus only on areas of code relevant to the task
- Do not touch code unrelated to the task being performed
- Complete one task fully before moving to the next
- Break down problems into smaller components and analyze each step first
- Present complete reasoning based on code and logs before making changes

### Iterative Refinement
- Use TDD (Test-Driven Development) when appropriate
- Break large tasks into smaller, testable components
- Include regular validation against requirements
- Integrate human oversight at each development stage
- Begin with high-level design before implementation

### Change Management
- Avoid making major architecture changes to working features unless explicitly instructed
- Consider what other methods and areas of code might be affected by changes
- Take into account separate Dev, Test, and Prod environments
- When fixing issues, exhaust options with existing implementation before introducing new patterns
- Remove old implementations if new approaches are used

---

## 5. File Structure and Organization

### Structure Guidelines
- Organize project files logically with clear hierarchical patterns
- Store shared utilities in centralized folders
- Use environment-specific configuration approach
- Place `.cursor/rules/` in root directories for version control integration

### Reference Patterns
- Use the `@` syntax to reference files (e.g., `@filepath/file.ts`)
- Implement glob patterns to define which files rules apply to
- Use exclusion patterns to prevent irrelevant files from cluttering workspace

---

## 6. Testing and Quality Assurance

### Testing Standards
- Write thorough tests for all major functionality
- End-to-end testing works best for comprehensive coverage
- Implement test coverage minimums where appropriate
- Use appropriate testing frameworks for the technology stack

### Quality Controls
- Enforce linting and style guide compliance
- Prioritize test coverage for critical functionality
- Run automated validation where possible
- Do not use mock data for Dev or Prod environments; only use in test environments
- Never add stubbing or fake data patterns affecting Dev or Prod

---

## 7. Project Rules Best Practices

### Implementation Strategy
- Keep rules simple and focused; don't dump everything into one big file
- Generate a new rule whenever the AI makes a repeated error
- Implement visibility rules to track what's being applied during AI actions
- For complex projects, create domain-specific rules (frontend/backend)

### Rule Content
- Define framework and architectural standards
- Specify technology preferences to prevent unwanted technology switches
- Include error handling requirements
- Document API standards and conventions
- Define security requirements for sensitive operations

### Rule Organization
- Organize rules by domain or functional area
- Consider language-specific rules for polyglot projects
- Reference architecture documentation using `@` syntax for clarity
- Use strategic glob patterns for more targeted rule application

---

## 8. Example Rule Implementations

### Global Rule Example

# Core Principles
- Use TypeScript for all new code
- Follow clean code principles
- Prefer async/await over callbacks
- Write comprehensive error handling

# Code Style
- Consistent indentation (2 spaces)
- Prefer const over let when variables don't change
- Use descriptive variable names with proper casing
- Limit line length to 100 characters


### Project Rule Example (TypeScript)

Description: TypeScript coding standards
Globs: *.ts, *.tsx
alwaysApply: true
---
# TypeScript Standards
- Use strict type checking
- Avoid 'any' type when possible
- Use interfaces for object definitions
- Implement proper error handling

# React Components (for .tsx files)
- Use functional components
- Implement proper prop types
- Follow React best practices

@file ../tsconfig.json


### Python Rule Example

Description: Python development standards
Globs: *.py
---
# Python Coding Standards
- Follow PEP 8 style guide
- Use type hints for all function signatures
- Prefer dataclasses for data modeling
- Use pytest for testing

# Error Handling
- Use early returns for invalid states
- Log errors consistently
- Implement proper exception handling


---