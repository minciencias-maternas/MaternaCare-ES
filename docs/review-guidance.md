# Review and comment guidance

This guide provides reusable wording and expectations for issue discussions and pull request reviews. GitHub does not provide a repository-wide native template for arbitrary issue or review comments, so use these patterns as guidance rather than expecting a comment-template picker.

## Review principles

- Be respectful and specific. Discuss the change, not the contributor.
- Point to the affected behavior, file, or evidence and explain why it matters.
- Separate blocking requests from non-blocking suggestions.
- Ask a focused question when evidence is incomplete instead of assuming intent.
- Avoid requesting or posting credentials, private records, or identifiable health information.

## Blocking finding

Use when a change should not merge until an issue is addressed.

> **Blocking:** [Describe the concrete problem and its impact.] Could you [specific change or check]? [Relevant evidence or reproduction.] Please use synthetic or public data in any example.

## Non-blocking suggestion

Use for an improvement that does not prevent acceptance.

> **Suggestion (non-blocking):** [Describe the potential improvement and its benefit.] One possible approach is [specific suggestion].

## Clarification or evidence request

> Could you clarify [specific assumption or result]? The current evidence shows [observation], and [additional context] would help assess [impact]. Please omit sensitive data.

## Approval

> **Approved:** The change addresses [scope] and I found no blocking concerns. [Optional note about a remaining non-blocking follow-up.]

## Issue discussion

For bug reports, ask for the smallest missing reproduction detail and relevant sanitized environment information. For feature proposals, clarify the user need, scope, evidence, and assumptions before implementation. Keep the discussion aligned with the issue's approval status; maintainers apply `status:approved` when the proposal is ready to implement.
