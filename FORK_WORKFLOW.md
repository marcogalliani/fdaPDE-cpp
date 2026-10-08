# Fork workflow: syncing with upstream and managing submodules

This fork of fdaPDE-cpp (`github.com/marcogalliani/fdaPDE-cpp`) and its core fork
(`github.com/marcogalliani/fdaPDE-core`, submodule `fdaPDE/core`) carry development work on top of the
upstream libraries (`github.com/fdaPDE/*`). This document describes how the branches are organised, how to
bring upstream updates into them, how to develop features, and how to handle the submodules.

## 1. Branch layout

```
fdaPDE-core   upstream/stable ─▶ stable ─▶ dev/main ─▶ dev/ode/main

fdaPDE-cpp    upstream/stable ─▶ stable ─▶ dev/main ─┬─▶ dev/fpca/main ─┬─▶ dev/fpca/feat/missing-data
                                                     │                  ├─▶ dev/fpca/feat/lumped-direct
                                                     │                  └─▶ dev/fpca/feat/kcv
                                                     └─▶ dev/ode/main
```

| branch | content | core pin |
|---|---|---|
| `stable` | exactly upstream `stable`, never committed to | upstream's own pin |
| `dev/main` | the core fork in `.gitmodules`, C++20 conformance fixes, this document | core `dev/main` |
| `dev/<module>/main` | a development module (`fpca`, `ode`) | core `dev/main` (fpca), core `dev/ode/main` (ode) |
| `dev/<module>/feat/<name>` | a feature of a module, not yet in its `main` | as its parent |

In fdaPDE-core, `dev/main` adds the Apple Clang workarounds (see `APPLE_CLANG_PATCH.md` there) and
`dev/ode/main` the ODE module.

Downstream projects pin this repository as a submodule: `fpca-rsi` tracks `dev/fpca/main`, `fpca-na` tracks
`dev/fpca/feat/missing-data`.

## 2. Rules

1. **Each branch merges only from its direct parent**, one level at a time, top-down. Never merge `stable`
   straight into a module or feature branch: a conflict is resolved once, where it first appears, and every
   branch sees only its parent's history.
2. **Merge, never rebase** published branches. Downstream projects pin their commits; a rebase orphans the
   pins, and a fresh clone then fails with `upload-pack: not our ref`.
3. **`stable` only fast-forwards** (`git merge --ff-only`). If that fails, something was committed to
   `stable` by mistake: stop and find out what.
4. **Core before cpp.** A cpp commit may only pin a core commit that is already reachable from a branch on
   the core fork (see section 5.4). Push core first.
5. **Dry-run before merging**: `git merge-tree --write-tree <branch> <parent>` lists the conflicts without
   touching the working tree.
6. **Build and run the tests at each level** before merging further down (section 6).
7. Feature branches may lag: update them from their parent when you go back to work on them.

## 3. Syncing with upstream

Fetch only upstream's `stable`, and without submodule recursion: upstream's other branches pin core commits
that the fork does not have, and a plain `git fetch upstream` tries to fetch them and fails with
`not our ref` (harmless, but the command exits with an error).

### 3.1 Core (in `fdaPDE/core`)

```sh
cd fdaPDE/core
git fetch origin
git fetch upstream stable
git log --oneline stable..upstream/stable          # what is new upstream

git switch stable
git merge --ff-only upstream/stable

git merge-tree --write-tree dev/main stable        # dry run
git switch dev/main
git merge stable                                   # resolve conflicts, build, test
git switch dev/ode/main
git merge dev/main                                 # resolve conflicts, build, test

git push origin stable dev/main dev/ode/main
```

Conflicts in core `dev/main` usually come from the Apple Clang workarounds: keep the workaround and port
upstream's change into it, and update `APPLE_CLANG_PATCH.md` if a rewritten spot changes.

### 3.2 fdaPDE-cpp (in the repository root)

```sh
git fetch origin
git fetch --no-recurse-submodules upstream stable
git log --oneline stable..upstream/stable

git switch stable
git merge --ff-only upstream/stable

git switch dev/main
git merge stable
```

Expected conflicts in `dev/main`:

- **`fdaPDE/core`** (submodule conflict), whenever upstream moved its core pin. Resolve by pinning the
  updated core `dev/main` from section 3.1:
  ```sh
  git -C fdaPDE/core switch dev/main
  git add fdaPDE/core
  ```
- **`.gitmodules`**, if upstream edited it: keep the fork URL (`https://github.com/marcogalliani/fdaPDE-core.git`)
  and `branch = dev/main`.
- **the C++20 fixes** (`distributions.h`, `sr.h`, `gsr.h`, `solvers/utility.h`): keep the fix, port
  upstream's change into it; drop the fix if upstream fixed it the same way.

Then continue down the tree:

```sh
git commit                                         # only if the merge stopped on conflicts; build and test first
git switch dev/fpca/main && git merge dev/main     # fpca.h is the usual conflict: upstream also edits it
git switch dev/ode/main  && git merge dev/main
```

`dev/ode/main` pins core `dev/ode/main`, not `dev/main`: if its merge conflicts on `fdaPDE/core`, resolve it
by pinning the updated core `dev/ode/main`:

```sh
git -C fdaPDE/core switch dev/ode/main
git add fdaPDE/core
git commit                                         # after build and tests
git push origin stable dev/main dev/fpca/main dev/ode/main
```

Feature branches, when you work on them:

```sh
git switch dev/fpca/feat/<name> && git merge dev/fpca/main
```

### 3.3 Downstream projects

After pushing, move the pin of the projects that track a branch you updated (see section 5.3):

```sh
cd fpca-rsi                                        # tracks dev/fpca/main
git submodule update --remote fdaPDE-cpp           # moves fdaPDE-cpp to the tip of the tracked branch
git -C fdaPDE-cpp submodule update --init          # moves the nested core to fdaPDE-cpp's pin
# build the project, then
git add fdaPDE-cpp && git commit -m "chore: bump fdaPDE-cpp"
```

## 4. Developing features

```sh
git switch -c dev/fpca/feat/<name> dev/fpca/main   # a local start point sets no upstream tracking
# ... commit ...
git push origin dev/fpca/feat/<name>
```

When the feature is ready, merge it into its module and keep the other features up to date:

```sh
git switch dev/fpca/main
git merge --no-ff dev/fpca/feat/<name>             # --no-ff keeps the feature visible in the history
# build, test, push
git push origin dev/fpca/main
git push origin --delete dev/fpca/feat/<name>      # once nothing pins it (section 5.4)
git branch -d dev/fpca/feat/<name>
```

Before deleting any branch, check that no downstream project pins a commit that only that branch reaches
(section 5.4).

## 5. Submodules

### 5.1 What a submodule is

A superproject does not store the submodule's files: it stores **one commit hash** for it (the *gitlink* or
*pin*), plus its URL and an optional branch name in `.gitmodules`. The pin is what `git submodule update`
checks out; the branch is used only by `git submodule update --remote`.

There are two levels here: downstream projects pin fdaPDE-cpp, and fdaPDE-cpp pins fdaPDE-core.

The URL actually used for fetching is the copy in the superproject's `.git/config`, written when the
submodule is initialised. `git submodule sync` overwrites it with the URL from `.gitmodules`.

### 5.2 Cloning and switching branches

```sh
git clone --recurse-submodules <url>               # or, in an existing clone:
git submodule update --init --recursive            # checks out every pin, nested ones included
```

Switching branches in the superproject does **not** move the submodule. After `git switch`, `git status`
may show `fdaPDE/core` as modified: the submodule is still at the previous branch's pin. Either

```sh
git switch --recurse-submodules <branch>           # moves the submodule to the new pin as well
# or
git switch <branch> && git submodule update        # same, in two steps
```

`git submodule update` leaves the submodule on a *detached HEAD* at the pin. To work in the submodule,
switch it to a branch first (`git -C fdaPDE/core switch dev/main`); commits made on a detached HEAD are
easily lost.

### 5.3 Moving a pin

```sh
# to the tip of the branch recorded in .gitmodules
git submodule update --remote fdaPDE/core
# or to a specific branch or commit
git -C fdaPDE/core switch dev/main                 # or: git -C fdaPDE/core checkout <hash>

git diff --submodule=log fdaPDE/core               # review which commits the pin moves over
git add fdaPDE/core
git commit -m "chore: bump core"
```

Do **not** use `git submodule update --remote --recursive` in a downstream project: it would also move the
nested core to the tip of its `.gitmodules` branch, instead of the commit fdaPDE-cpp pins. Update the outer
submodule with `--remote`, then the nested one with a plain `git submodule update`, as in section 3.3.

### 5.4 Reachability: push the submodule first

A pin is useful only if the pinned commit can be fetched from the submodule's URL. Before pushing a
superproject commit, the pinned commit must be on a branch already pushed to the submodule's remote:

```sh
git -C fdaPDE/core branch -r --contains "$(git ls-tree HEAD fdaPDE/core | awk '{print $3}')"
# must print at least one origin/... branch
```

The same holds when deleting a branch: if a superproject commit pins a commit that only that branch
reaches, a fresh checkout of that superproject commit can no longer fetch its submodule. To list every
fdaPDE-cpp commit a downstream project has ever pinned, with the branches that contain it:

```sh
cd fpca-rsi
git log --all --raw --no-abbrev --format= -- fdaPDE-cpp | awk '{print $4}' | sort -u | while read c; do
  echo "${c:0:7}: $(git -C fdaPDE-cpp branch -r --contains $c 2>/dev/null | tr -d ' ' | tr '\n' ' ')"; done
```

### 5.5 Reading `git status`

| short status | meaning |
|---|---|
| ` M fdaPDE/core` | the submodule's HEAD differs from the pin (other commit checked out) |
| ` m fdaPDE/core` | the submodule has modified tracked files |
| ` ? fdaPDE/core` | the submodule has untracked files |

`git submodule status` prefixes each pin with `-` (not initialised), `+` (checked out commit differs from the
pin) or `U` (merge conflict).

### 5.6 Recovering a broken submodule

If a submodule clone fails halfway (for example `fatal: premature end of pack file`), remove the partial
clone and clone again. The partial clone has two parts: the gitdir under `.git/modules/` and the
working-tree folder.

```sh
rm -rf .git/modules/fdaPDE-cpp/modules/fdaPDE/core fdaPDE-cpp/fdaPDE/core     # paths for the nested core
git -C fdaPDE-cpp submodule update --init
```

If the download keeps failing, clone from a local copy that already has the commits, and copy its objects
so the new clone does not depend on it:

```sh
git -C fdaPDE-cpp submodule update --init --dissociate --reference <path to an existing fdaPDE-core repo>
```

### 5.7 Network-mounted working copies

On a network mount (NFS, FUSE) git can misbehave:

- a cherry-pick, rebase or merge stops with "Your local changes … would be overwritten" although the tree
  is clean: run `git update-index --refresh` and continue. A `git cherry-pick --continue` after such a stop
  may silently skip the commit that failed to start: check `git log`, or use `git cherry-pick --quit` and
  pick the remaining commits again;
- checked-out files can lose their executable bit (`mode change 100755 => 100644`): `chmod +x` them;
- a stale `.git/index.lock` remains after an interrupted command: remove it once no git process runs
  (`pgrep -fl git`);
- an editor's git integration (VS Code) polls the repository and can keep a directory busy: a folder that
  cannot be removed can usually be renamed out of the way.

For read-only checks that must not take the index lock, use `git --no-optional-locks status`.

## 6. Building and testing

```sh
cmake -S test -B build -DCMAKE_CXX_COMPILER=g++-15
cmake --build build -j1                            # -j1: the test suite is a single, memory-hungry unit
mkdir -p test/run && cd test/run && ../../build/fdapde_test
```

The tests read their data from `../data`, so they must run from a subfolder of `test/`. On macOS, if
Homebrew GCC fails inside the SDK headers (`'FILE' has not been declared`), point CMake to an older SDK, for
example `-DCMAKE_OSX_SYSROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX14.sdk`.
