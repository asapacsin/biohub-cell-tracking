# Agent handoff protocol

Before changing this repository, read `.agents/state.json` and `.agents/memory.md`.

After a meaningful work session:

1. update `.agents/state.json` with the current milestone, verified commands, blockers, and next action;
2. append only durable facts and decisions to `.agents/memory.md`;
3. never record secrets, credentials, private dataset contents, or speculative facts;
4. keep official data findings traceable to an inspection command and source path;
5. do not mark a milestone complete unless its listed verification has passed;
6. every time a Kaggle pack is VALIDATE_OK and mirrored to login, append a **Kaggle submission ready**
   record to `.agents/memory.md`. Do not gitignore `outputs/kaggle_submission/` CSVs (size is not a reason).

The competition files and `sample_submission.csv` remain the source of truth. Agent memory is a
handoff aid, not an authoritative competition specification.

