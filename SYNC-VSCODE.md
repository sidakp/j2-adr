1) In VSCode click File -> 'Open a Folder'
2) Select the destination where all files will be saved
3) In VSCode have any file present in your folder to be uploaded to GitHub
4) Open the terminal in VSCode
5) Type and enter `git init` and this should appear "Initialized empty Git repository in [FILE PATH]
6) Type and enter `git add .`
7) Type and enter `git commit -am "initial commit"` and this should appear
[master (root-commit) 08ba31b] initial commit
1 file changed, 0 insertions(+), 0 deletions(-)
create mode 100644 [FILE NAMES]
8) Type and enter `git remote add origin https://github.com/[REPO FILE PATH]`
10) Type and enter `git remote -v` should verify a link for fetch and a link for push

To save and commit changes to GitHub:

OPTIONAL STEP  0: See status changes with `git status`
1) Type `git add .`
2) Type `git commit -m "NAME TO SHOW IN COMMIT HISTORY"` (The name is public)
3) Type `git push -u origin main`