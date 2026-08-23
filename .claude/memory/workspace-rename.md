---
name: workspace-rename
description: The project was renamed shipproj -> shipinfer; the old path survives as a symlink
metadata:
  type: project
---

On 2026-08-22 the project was renamed from `shipproj` to **`shipinfer`**, so the working
tree, the Python package and the distribution all share one name.

`/home/dungha15/workspaces/phucnp/shipproj` was left in place as a **symlink** to
`/home/dungha15/workspaces/phucnp/shipinfer`. That is deliberate: renaming the directory a
session is running in would otherwise strand anything holding the old absolute path. The
`.venv` was relocated properly (shebangs, `pyvenv.cfg`, the editable-install `.pth`), so it
does not depend on the symlink — the symlink is only a courtesy for stale references and
can be deleted once nothing needs it.

**How to apply:** prefer the real path in anything new. If a tool reports
`.../shipproj/...`, it is following the symlink, not a bug. See [[git-remotes-ssh]] for the
remote, which is renamed on GitHub separately.
