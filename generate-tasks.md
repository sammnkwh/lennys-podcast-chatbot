# Rule: Generating a Task List from User Requirements

## Goal

To guide an AI assistant in creating a detailed, step-by-step task list in Markdown format based on user requirements, feature requests, or existing documentation. The task list should guide a developer through implementation.

## Output

- **Format:** Markdown (`.md`)
- **Location:** `/tasks/`
- **Filename:** `tasks-[feature-name].md` (e.g., `tasks-user-profile-editing.md`)

## Process

1.  **Receive Requirements:** The user provides a feature request, task description, or points to existing documentation
2.  **Analyze Requirements:** The AI analyzes the functional requirements, user needs, and implementation scope from the provided information
3.  **Phase 0: MECE Component Breakdown:** Before generating tasks, break down the project into **Mutually Exclusive and Collectively Exhaustive (MECE)** components. Each component should:
    - Have a single, clear responsibility
    - Not overlap with other components
    - Together, cover 100% of what's needed to complete the project

    Present the components to the user in a table format showing:
    - Component name
    - Responsibility
    - Input
    - Output
    - File mapping (which file(s) will implement this component)

    Inform the user: "I have broken down the project into MECE components. Review and respond with 'Go' to proceed to task generation, or suggest changes."
4.  **Wait for Confirmation:** Pause and wait for the user to respond with "Go".
5.  **Phase 1: Generate Parent Tasks:** Based on the MECE components, create the file and generate the main, high-level tasks required to implement the feature.

    **Rules:**
    - **Task 0.0** is always "Create feature branch" (unless user requests otherwise)
    - **Each MECE component can have multiple parent tasks** - use as many as needed to fully implement that component (e.g., Component 1 might need tasks 1.0, 1.1, 1.2)
    - **Group parent tasks by component** - clearly label which component each task belongs to
    - **Final component is always "Integration"** - with its own parent task(s) to merge all components together and verify the complete system works

    Present these tasks to the user in the specified format (without sub-tasks yet). Inform the user: "I have generated the high-level tasks based on the MECE components. Ready to generate the sub-tasks? Respond with 'Go' to proceed."
6.  **Wait for Confirmation:** Pause and wait for the user to respond with "Go".
7.  **Phase 2: Generate Sub-Tasks:** Once the user confirms, break down each parent task into smaller, actionable sub-tasks necessary to complete the parent task. Ensure sub-tasks logically follow from the parent task and cover the implementation details implied by the requirements.
8.  **Identify Relevant Files:** Based on the MECE components and tasks, identify potential files that will need to be created or modified. List these under the `Relevant Files` section, grouped by component, including corresponding test files if applicable.
9.  **Generate Final Output:** Combine the MECE components, parent tasks, sub-tasks, relevant files, and notes into the final Markdown structure.
10. **Save Task List:** Save the generated document in the `/tasks/` directory with the filename `tasks-[feature-name].md`, where `[feature-name]` describes the main feature or task being implemented (e.g., if the request was about user profile editing, the output is `tasks-user-profile-editing.md`).

## Output Format

The generated task list _must_ follow this structure:

```markdown
## MECE Components

| # | Component | Responsibility | Input | Output | Files |
|---|-----------|----------------|-------|--------|-------|
| 1 | Component Name | What it does | What it receives | What it produces | `file.py` |
| 2 | Component Name | What it does | What it receives | What it produces | `file.py` |

## Relevant Files

### Component 1: [Component Name]
- `path/to/file1.py` - Brief description of why this file is relevant.
- `path/to/file1_test.py` - Unit tests for `file1.py`.

### Component 2: [Component Name]
- `path/to/file2.py` - Brief description.
- `path/to/file2_test.py` - Unit tests for `file2.py`.

### Notes

- Unit tests should typically be placed alongside the code files they are testing (e.g., `my_module.py` and `my_module_test.py` in the same directory).
- Use appropriate test runner commands for your framework (e.g., `pytest`, `npx jest`).

## Instructions for Completing Tasks

**IMPORTANT:** As you complete each task, you must check it off in this markdown file by changing `- [ ]` to `- [x]`. This helps track progress and ensures you don't skip any steps.

Example:
- `- [ ] 1.1 Read file` → `- [x] 1.1 Read file` (after completing)

Update the file after completing each sub-task, not just after completing an entire parent task.

## Tasks

- [ ] 0.0 Create feature branch
  - [ ] 0.0.1 Create and checkout a new branch for this feature (e.g., `git checkout -b feature/[feature-name]`)

### Component 1: [Component Name]
- [ ] 1.0 [First parent task for Component 1]
  - [ ] 1.0.1 [Sub-task]
  - [ ] 1.0.2 [Sub-task]
- [ ] 1.1 [Second parent task for Component 1]
  - [ ] 1.1.1 [Sub-task]
  - [ ] 1.1.2 [Sub-task]

### Component 2: [Component Name]
- [ ] 2.0 [First parent task for Component 2]
  - [ ] 2.0.1 [Sub-task]
- [ ] 2.1 [Second parent task for Component 2]
  - [ ] 2.1.1 [Sub-task]
  - [ ] 2.1.2 [Sub-task]

### Component N: Integration
- [ ] N.0 Wire all components together in main entry point
  - [ ] N.0.1 [Sub-task]
  - [ ] N.0.2 [Sub-task]
- [ ] N.1 End-to-end testing of complete workflow
  - [ ] N.1.1 [Sub-task]
- [ ] N.2 Final commit and push
  - [ ] N.2.1 [Sub-task]
```

## Interaction Model

The process explicitly requires **two pauses** to get user confirmation:
1. After **Phase 0 (MECE breakdown)** - ensures the architectural decomposition is correct before planning tasks
2. After **Phase 1 (Parent tasks)** - ensures the high-level plan aligns with user expectations before diving into details

This two-stage confirmation prevents wasted effort if the component breakdown or task structure needs adjustment.

## MECE Guidelines

When breaking down a project into MECE components:

- **Mutually Exclusive:** No two components should do the same thing. If you find overlap, merge them or redefine boundaries.
- **Collectively Exhaustive:** The components together must cover everything needed. Nothing should fall through the cracks.
- **Single Responsibility:** Each component should have one clear job. If a component does multiple unrelated things, split it.
- **Clear Interfaces:** Define what each component receives (input) and produces (output). This clarifies dependencies.
- **File Mapping:** Each component should map to specific files. This makes the architecture tangible.
- **Independent Development:** Each component should be buildable and testable on its own. The final Integration task merges them all together.

### Example MECE Breakdown

For a RAG chatbot project:

| # | Component | Responsibility | Input | Output | Files |
|---|-----------|----------------|-------|--------|-------|
| 1 | Document Processor | Load and chunk text files | File path | List of text chunks with metadata | `utils/document_processor.py` |
| 2 | Vector Store | Store embeddings, perform similarity search | Chunks or query | Confirmation or relevant chunks | `utils/vector_store.py` |
| 3 | LLM Chain | Generate answers and follow-ups | Query, context, history | Answer + follow-up questions | `utils/llm_chain.py` |
| 4 | UI | Render chat interface, handle interactions | User input | Rendered app | `app.py` |
| 5 | Logger | Send follow-ups to external service | Follow-up questions | Logged event | `utils/langfuse_logger.py` |

## Target Audience

Assume the primary reader of the task list is a **junior developer** who will implement the feature.
