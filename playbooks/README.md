# Approved operational catalog

This directory intentionally contains no operational procedures. Supply owner-approved JSON files matching ai/contracts/playbook.schema.json. Approval fields are declarations from the trusted host, not authentication. Do not expose this directory or its write permissions to API callers.

The API loads this catalog at startup, ignores DRAFT/REVOKED/testOnly/expired books during recommendation, and returns NO_APPROVED_PLAYBOOK when empty. Restart after authorized changes. Tests use temporary synthetic catalogs with explicitly non-operational wording; they are not deployed here.
