- A branch create, a merge and a load with `from` now also need `read` on the source branch, at every door (server, CLI and embedded engine); a start at an unreadable commit answers `404`. Grant `read` on the source to actors that copy branches. See [policy][branch-copy-read].

[branch-copy-read]: ../docs/user/operations/policy.md
