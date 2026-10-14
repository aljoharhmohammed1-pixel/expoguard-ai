# Publish the prepared integration release privately

Current remote state: **NOT PUSHED**. GitHub CLI is unavailable on the preparation host. No authentication tokens or credentials were requested or stored.

After installing GitHub CLI through your team's normal process, run these commands from the repository root. Login is interactive in your own account. `expoguard-ai` must be an unused repository name in that account; select the intended organization explicitly if applicable.

```powershell
gh auth login
gh auth status
git lfs install --local
git lfs fsck
gh repo create expoguard-ai --private --source . --remote origin --push
git push origin v0.1.0-integration
gh repo view --json url,visibility
```

Verify visibility is PRIVATE. The Git LFS pre-push hook transfers the weights; do not bypass it. If LFS storage/quota blocks upload, resolve that before advertising a working clone.

If the team has already created a private empty repository, use its actual URL instead of creating another:

```powershell
git remote add origin https://github.com/OWNER/PRIVATE_REPOSITORY.git
git push -u origin main
git push origin v0.1.0-integration
```

OWNER/PRIVATE_REPOSITORY is a placeholder to replace; it is not an existing repository. Verify the repository is private before pushing. Never put tokens in URLs or project files.

After publishing, test a fresh clone with Git LFS, the documented installation and integration verifier. Share both the actual private repository URL and the private deployed Swagger URL. GitHub does not expose your local `127.0.0.1:8000` service to teammates. A shared Swagger address is `https://<your-private-service-host>/docs` only after deployment; the current localhost address is not a shared endpoint.
