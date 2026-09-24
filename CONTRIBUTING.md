# Contributing to MaternaCare-ES

Thank you for helping improve MaternaCare-ES. Contributions include code, research and implementation notes, dataset tooling, and corrections to existing documentation.

## Start with an issue

Open a bug report or feature request before starting a substantial change. Check existing issues for duplicates and describe the problem, evidence, and proposed scope. Keep blank issues available for reports that do not fit a form.

Wait until a maintainer applies the `status:approved` label before beginning implementation. This approval gate gives maintainers a chance to confirm scope and approach; contributors should not apply that label themselves. For small typo or documentation corrections, an issue is still welcome and helps link the change to its context.

## Set up the project

Use Python 3.11 and create a virtual environment. Choose the dependency file for your environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

- NVIDIA GPU: `pip install -r requirements-cuda.txt`
- CPU: `pip install -r requirements-cpu.txt`

`requirements.txt` contains shared dependencies and is not a complete standalone installation target; it does not install PyTorch. Avoid mixing CPU and CUDA PyTorch installations in one environment. Model downloads and some project workflows also require Hugging Face access, accepted model terms, or service credentials; never include credentials in an issue or change.

## Branches and commits

Create a branch from `main` after the issue is approved. Use the repository branch convention `type/description`, with a lowercase description, for example `docs/contribution-guide` or `fix/dataset-loader`.

Use conventional commit messages, such as `docs: clarify CPU setup` or `fix(data): handle missing metadata`. Do not include secrets, generated model weights, private data, or unrelated changes in a commit.

## Make and test changes

- Keep changes focused on the approved issue and explain relevant design or research assumptions.
- Add or update tests when behavior changes. Run the complete available test suite from the repository root:

  ```bash
  python -m unittest discover -s tests
  ```

- The repository currently has no CI workflow. This command is the documented local test suite, not a claim that automated checks run on pull requests. If a change cannot be tested locally, explain why and describe what was checked instead.
- Update documentation when behavior, setup, research methods, or interpretation changes. Prefer a concise technical note for substantial research or implementation context; see [the technical-note template](docs/templates/technical-note.md).

## Research, data, and privacy

State the source and version of datasets, models, and external references when relevant. Distinguish observed results from assumptions, and document limitations and reproducibility details. Do not present model output as clinical advice or as a substitute for professional judgment.

Do not submit identifiable health information, private records, credentials, or other sensitive data in issues, pull requests, logs, screenshots, examples, or test fixtures. Use synthetic or appropriately public examples, and remove identifying details while preserving enough context to reproduce a problem.

## Open a pull request

Use the [pull request template](.github/PULL_REQUEST_TEMPLATE.md). Every pull request must:

1. Link the issue with `Closes #N` and refer to an issue that has maintainer-applied `status:approved`.
2. Summarize the change and its scope, and identify the change type.
3. Report tests run and any tests that do not apply or could not be run.
4. Describe relevant data, model, privacy, and documentation impacts.
5. Meet the contributor checklist in the template.

A maintainer applies exactly one existing `type:*` label to the pull request. Contributors should not invent labels or claim the repository validates issue linkage, labels, or test results automatically; no GitHub Actions checks are currently configured.

## Review and discussion

Keep issue and review comments respectful, specific, and actionable. Explain the evidence behind concerns and distinguish blocking requests from suggestions. See [review and comment guidance](docs/review-guidance.md).
