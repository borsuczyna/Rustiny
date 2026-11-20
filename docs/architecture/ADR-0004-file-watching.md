# ADR-0004: File watching with `notify`, rescans reuse unchanged `.meta` results

**Status:** accepted.

## Context

The Project panel must follow edits made outside Rustiny (a Unity editor, git, a script) without blocking the UI and without re-reading tens of thousands of `.meta` files each time. On a real project (26 670 files, 27 066 `.meta`), a cold scan takes about 0.5 s, almost all of it reading `.meta` files.

## Decision

* **Watching:** the `notify` crate (8.x, `ReadDirectoryChangesW` on Windows, inotify on Linux, FSEvents on macOS), watching `Assets/` recursively. Raw events are collapsed by our own debounce: after the first relevant event we wait for 300 ms of quiet and then report once. Access events and anything below a name Unity ignores (`.hidden`, `Cache~`, `*.tmp`, `cvs`) are dropped.
* **Rescan:** instead of patching the index in place, a rescan builds a new index and reuses the GUID/importer of every `.meta` whose size and modification time are unchanged (`AssetDatabase::rescan`). The tree walk is repeated in full (about 100 ms on the real project) but no unchanged `.meta` is opened. The result is checked in tests to be identical to a from-scratch scan.
* **Concurrency:** the watcher is started before the first scan, so a change made while the scan walks the tree is not lost (it queues a follow-up scan); a first version started it afterwards and an end-to-end test caught the gap. At most one scan runs at a time. Changes that arrive during a scan are folded into one follow-up scan. A failed refresh keeps showing the last good index. The index is immutable and shared as `Arc<AssetDatabase>`, so the UI never sees a half-updated state.
* **Read-only:** watching opens directory handles and reads; nothing is written to the project.

## Consequences

* A `.meta` rewritten with the same size within the file system's timestamp resolution would not be noticed until its next change. Unity itself hashes contents; Rustiny does not yet.
* Rescan cost is proportional to the number of files and folders, not to the number of changes. Patching only the affected folders would be faster for huge projects and is future work if measurements justify it.
* `notify` is licensed CC0-1.0 (public-domain dedication), compatible with this project's MIT OR Apache-2.0. It adds `walkdir`, `windows-sys`, `log` and `notify-types` to the dependency tree.
* Network drives and some virtual file systems do not deliver change events; the panel shows "watching for changes" only if the watcher started, and there is no manual refresh yet.
