# Security rules for agents

Every agent working in this repository follows these rules. They apply to code, docs, tests, fixtures, notebooks, and commit messages.

## Secrets

1. Never write a real key, token, password, connection string, or webhook URL into any file. Read secrets from environment variables.
2. `.env` files stay local and are ignored by git. Only `.env.example` is committed, and it holds placeholders such as `ANTHROPIC_API_KEY=your-key-here`.
3. Never print a secret in logs, error messages, test output, or a README example.
4. If a secret was ever committed, treat it as leaked: tell the user to rotate it first. Rewriting history comes second and does not make the key safe again.

## Data

1. Never commit real personal data: names, emails, phone numbers, addresses, account numbers, bank statements, messages, or health records.
2. Fixtures and sample data are synthetic and labeled as synthetic in the file name or a header.
3. Before adding any CSV, JSON, SQLite, PDF, or notebook with outputs, check it for real people's data. Clear notebook outputs that show private data.

## Dependencies

1. Ask before adding a dependency. Prefer the standard library or a package already in use.
2. Pin versions in lockfiles and commit the lockfile.
3. Pin GitHub Actions to a full commit SHA with the version in a comment.
4. Never pipe a remote script into a shell. Download, verify a checksum, then run.

## GitHub Actions

1. Every workflow sets `permissions: contents: read` at the top and widens it per job only when needed.
2. Never use `pull_request_target` with a checkout of the PR's code.
3. Never interpolate `${{ github.event.* }}` text straight into a `run:` step. Pass it through `env:`.
4. Checkout steps use `persist-credentials: false` unless the job pushes.
