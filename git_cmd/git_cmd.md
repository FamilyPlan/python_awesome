## Installation

### 创建版本库

- Linux (Windows is not officially supported)
- ROS
- Python 3.6
- PyTorch 1.2
- CUDA 10.0


### 创建版本库

通过`git init` 命令把创建的目录变为Git可以管理的版本库，创建后，在当前目录下，多可一个`.git`的目录，这个目录是Git来跟踪和管理版本库的，不要手动修改该目录下的文件。

```shell
$ mkdir git_cmd
$ cd git_cmd
$ git init 
```

### 版本管理

![](工作区与版本库.jpeg)

a. 工作区(Working Directory)
工作区，及自己本地的git项目文件，比如自己本地电脑的git_cmd文件

b. 版本库(Repository)
工作区有一个隐藏目录`.git`，这个不算工作区，而是Git的版本库。Git版本库中，最重要的是称为stage（或者index）的暂存区，还有Git自动创建的第一个分支`master`，以及指向`master`的一个指针`HEAD`。

c. 提交文件
往Git版本库提交文件，是分两步执行的：

- 第一步用 `git add <file>` 把文件添加进去，实际上是把文件修改添加到暂存区（stage or index）。

- 第二步用 `git commit < -m message>` 提交更改，实际上就是把暂存区的所有内容提交到当前分支。

- 第三步可以用 `git status` 查看修改或者提交结果，及当前工作区的状态。

- 第四步可以用 `git diff <file_name>` 来查看对文件的内容修改

- 第五步可以用 `git log` 来查看提交的历史记录。 `git log` 命令可以显示从最近到最远的提交日志。类似`1094adb7b...`是`commit id`（版本号）。

```shell
$ git log

commit 1094adb7b9b3807259d8cb349e7df1d4d6477073 (HEAD -> master)
Author: Michael Liao <askxuefeng@gmail.com>
Date:   Fri May 18 21:06:15 2018 +0800

    append GPL

commit e475afc93c209a690c39c13a46716e8fa000c366
Author: Michael Liao <askxuefeng@gmail.com>
Date:   Fri May 18 21:03:36 2018 +0800

    add distributed

commit eaadf4e385e865d25c48e7ca9c8395c3f7dfaef0
Author: Michael Liao <askxuefeng@gmail.com>
Date:   Fri May 18 20:59:18 2018 +0800

    wrote a readme file
```
- 第六步可以用 `git reset` 来进行版本的回退。Git可以根据版本号来回退。在Git中，用`HEAD`表示当前版本，也就是最新的提交，上一个版本就是`HEAD^`，上上一个版本是`HEAD^^`，当然往上100个版本可以直接写作`HEAD~100`。也可以使用 `git log` 来查看版本号直接回退。版本号没必要写全，前几位就可以了，Git会自动去找。当然也不能只写前一两位，因为Git可能会找到多个版本号，就无法确定是哪一个了。如果后悔回退的版本，想恢复之前比较新的提交，可以用 `git reflog` 查看命令历史，以便确定要回到未来的哪个版本。


```shell
$ git reset --hard HEAD^
$ git reset --hard 1094a
```  
 
 - 第七步可以用 `git checkout -- <file>` 来撤销file在工作区的修改，包括两种情况：1）一种是file自修改后还没有被放到暂存区，现在撤销修改就回到和版本库一模一样的状态；2）一种是file已经被添加到暂存区，又做了修改，现在撤销修改，就回到添加到暂存区后的状态。总之，就是让这个文件回到最近一次 `git commit` 或 `git add` 时的状态。
 
 - 第八步，当你在使用`git add`后，想把刚刚的提交到暂存区的内容撤回（取消修改），可以使用 `git reset HEAD <file>` 命令，来把暂存区的修改撤销，即重新放回工作区；那么，丢弃工作区的修改，可以参考第七步。
 
 综上，对于版本管理的总结如下：
 - 场景1：当你改乱了工作区某个文件的内容，想直接丢弃工作区的修改时，用命令 `git checkout -- <file>`
 - 场景2：当你不但改乱了工作区的某个文件，还把它添加到了暂存区，想丢弃修改，分两步，第一步用命令 `git reset HEAD <file>` ，就回到了场景1，按照场景1操作；
 - 场景3：已经提交了不合适的修改到版本库时，想要撤销本次提交，参考版本回退（第六步），不过前提是没有推送到远程库。
 

### 分支管理
 
#### 1.创建与合并分支 

每次提交，Git都把它们串成一条时间线，这条时间线就是一个分支，及`master` 分支。`HEAD`严格来说不是指向提交，而是指向`master`，`master`才是指向提交的，所以，`HEAD`指向的就是当前分支。

1）一开始，`master`分支是一条线，Git用`master`指向最新的提交，再用`HEAD`指向`master`，就能确定当前分支，以及当前分支的提交点：

![](0.png)

每次提交，`master`分支都会向前移动一步，这样，随着不断提交，`master` 分支的线也会越来越长。

2）当创建新的分支，例如`dev`时，Git新建了一个指针叫做`dev`，指向`master`相同的提交，再把`HEAD`指向`dev`，就表示当前分支在`dev`上。

![](1.png)

```shell
#创建并且切换到分支dev上
$ git checkout -b dev  

# 上一句相当于下面的两句命令
$ git branch dev
$ git checkout dev

# 查看当前分支
git branch
#* dev
#  master
#*表示所在分支
``` 
然后在当前dev分支上提交修改:

```shell
$ git add git_cmd/
$ git commit -m 'add git cmd related' 

```
那么，从现在开始，对工作区的修改就是针对 `dev` 分支了，比如刚刚的提交后，`dev` 指针往前移动一步，而master指针不变，如下图所示：

![](2.png)

3）当在`dev`分支的开发结束后，可以把`dev`合并到`master`。最简单的方法就是直接把`master`指向`dev`的当前提交，就完成了合并：

```shell
$ git checkout master
$ git merge dev

```

![](3.png)

完成合并后，即可删除`dev`分支

```shell
$ git branch -d dev
```

4)目前，切换分支可以使用`switch`来实现

```shell
# 创建并且切换到新的dev分支
$ git switch -c dev

#直接切换到已有的`master`分支
$ git switch master
```
#### 2.解决冲突 

当我们在新分支，如名字为`feature1`的分支上add包括commit修改后；然后切换回master分支，然后在`master`分支没有进行merge，直接进行add/commit操作，这时候，如果合并`feature1`将会爆出'Automatic merge failed...'之类的错误：
 
 ![](4.png)

此时，可以手动修改冲突的文件内容，然后提交

```shell
$ git add readme.txt 
$ git commit -m "conflict fixed"
[master cf810e4] conflict fixed
```
现在，`master`分支和`feature1`分支变成了下图所示的关系：
![](5.png)

可以用带参数的` git log` 看到分支的合并情况。

```shell
$ git log --graph --pretty=oneline --abbrev-commit
*   cf810e4 (HEAD -> master) conflict fixed
|\  
| * 14096d0 (feature1) AND simple
* | 5dc6824 & simple
|/  
* b17d20e branch test
* d46f35e (origin/master) remove test.txt
* b84166e add test.txt
* 519219b git tracks changes
* e43a48b understand how stage works
* 1094adb append GPL
* e475afc add distributed
* eaadf4e wrote a readme file
```
最后删除`feature1`分支

```shell
$ git branch -d feature1
```
完成工作。

#### 3.分支管理策略

通常，通过 `git merge`合并分支时，如果可能，Git会用`Fast forward` 模式，但是这种模式下，删除分支后，会丢掉分支信息。如果要强制禁用`Fast forward` 模式，Git就会在merge时生成一个新的commit，这样，从分支历史上就可以看出分支信息。

在实际开发中，应该按照下面的原则来进行分支管理：`master`分支是非常稳定的，也就是仅用来发布新版本，平时不能在上面干活，应该在`dev`分支上工作。到版本发布的时候，把`dev`分支合并到`master`上。团队合作的分支应该看起来像下面这样：

![](6.png)

```shell
$ git switch -c dev
$ git add readme.txt
$ git commit -m 'add merge'
[dev f52c633] add merge
 1 file changed, 1 insertion(+)
$ git switch master
Switched to branch 'master'

#准备合并dev分支，使用--no-ff参数，以此禁用 Fast forward
$ git merge --no-ff -m 'merge with no-ff' dev
Merge made by the 'recursive' strategy.
 readme.txt | 1 +
 1 file changed, 1 insertion(+)

# 合并后，可以使用git log查看分支历史
$ git log --graph --pretty=oneline --abbrev-commit
*   e1e9c68 (HEAD -> master) merge with no-ff
|\  
| * f52c633 (dev) add merge
|/  
*   cf810e4 conflict fixed
...

```

那么，不使用`Fast forward`模式，merge后就像下面这样：

![](7.png)


#### 4.Bug分支

每个bug都可以通过一个新的临时分支来修复，修复后，合并分支，然后将临时分支删除。

1）储藏现在的工作空间

当接到修复一个代号101的bug任务时，需要创建一个分支issue-101来修复它，但是，目前在dev分支上进行的工作还没有提交，而且工作进行到一半，一时半会不能提交，此时，可以将当前的工作现场储藏起来。

（1）储藏现有未完成的工作空间

```shell
$ git stash
Saved working directory and index state WIP on dev: f52c633 add merge
```

(2)用`git status`查看工作区，此时，工作区间应该是干净的，除非有没被Git管理的文件。

（3）确定要在哪个分支上修复bug，假定需要在`master`分支上修复，就从`master`分支上创建临时分支：

```shell
$ git checkout master
Switched to branch 'master'
Your branch is ahead of 'origin/master' by 6 commits.
  (use "git push" to publish your local commits)

$ git checkout -b issue-101
Switched to a new branch 'issue-101'

#修复bug内容
$ git add xx.txt
$ git commit -m 'fix bug 101'
[issue-101 4c805e2] fix bug 101
 1 file changed, 1 insertion(+), 1 deletion(-) 
```

(4)修复完成后，切换到`master`分支上，完成合并，最后删除bug分支：

```shell
$ git checkout master
Switched to branch 'master'
Your branch is ahead of 'origin/master' by 6 commits.
  (use "git push" to publish your local commits)

$ git merge --no-ff -m 'merge bug fix 101 to master'
Merge made by the 'recursive' strategy.
 xxx.txt | 2 +-
 1 file changed, 1 insertion(+), 1 deletion(-)
$ git branch -d issue-101
 
```


(5)恢复之前储藏的工作空间

```shell
$ git checkout dev

$ git stash list
stash@{0}: WIP on dev: f52c633 add merge

```

工作现场还在，恢复的方式有两种：一是用 `git stash apply` 恢复，但是恢复后，stash内容并不删除，你需要使用 `git stash drop` 来删除；另一种方式是用 `git stash pop` ，恢复的同时把stash内容也删除了；

```shell
$ git stash pop

On branch dev
Changes to be committed:
  (use "git reset HEAD <file>..." to unstage)

	new file:   hello.py

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)

	modified:   readme.txt

Dropped refs/stash@{0} (5d677e2ee266f39ea296182fb2354265b91b3b2a)

```
 
```shell
$ git stash apply stash@{0}
$ git stash drop
```

(6)在dev分支同步更新bug信息

在master分支上修复了bug后，我们要想一想，dev分支是早期从master分支分出来的，所以，这个bug其实在当前dev分支上也存在。

那怎么在dev分支上修复同样的bug？

同样的bug，要在dev上修复，我们只需要把 `4c805e2 fix bug 101` 这个提交所做的修改复制到dev分支，而不是把整个master分支merge过来，使用 cherry-pick命令。

```shell
$ git branch
* dev
  master

$ git cherry-pick 4c805e2
[master 1d4b803] fix bug 101
 1 file changed, 1 insertion(+), 1 deletion(-)
```

以上，完成bug分支的处理。

#### 5.Feature分支

软件开发中，总有很多新的功能需要不断添加进来。添加一个新功能时，你肯定不希望因为一些实验性质的代码，把主分支搞乱了，所以，每添加一个新功能，最好新建一个feature分支，在上面开发，完成后，合并，最后，删除该feature分支。

例如，现在开发一个demo的新功能。

```shell
$ git branch
* dev 
  master

$ git switch -c feature-demo
Switched to a new branch 'feature-demo'
$ git add demo.py
$ git commit -m 'add feature demo'
[feature-demo 287773e] add feature demo
 1 file changed, 2 insertions(+)
 create mode 100644 demo.py
 
# 切回dev，准备合并
$ git checkout dev

```

如果没问题，则可以使用 `git merge feature-demo` 来合并分支，然后把feature-demo分支删除即可。

但是，如果不需要这个特征分支时：

```shell
$ git branch -d feature-demo
error: The branch 'feature-demo' is not fully merged.
If you are sure you want to delete it, run 'git branch -D feature-demo'.
```

销毁失败！

此时，只能通过 `-D` 参数将分支强行删除。
```shell

$ git branch -D feature-demo
Deleted branch feature-demo (was 287773e).
```

#### 6.多人协作

当你从远处仓库克隆时，实际上Git自动把本地的`master`分支和远程的`master`分支对应起来了，并且，远程仓库的默认名称是`origin`。

要查看远程库的信息，用 `git remote`:


```shell
$ git remote
origin
```

或者，用 `git remote -v` 显示更详细的信息：

```shell
$ git remote -v
origin	git@github.com:FamilyPlan/python_awesome.git (fetch)
origin	git@github.com:FamilyPlan/python_awesome.git (push)
```

上面显示了可以抓取和推送`origin`的地址。如果没有推送权限，就看不到push地址。

1）推送分支，把该分支上的所有本地提交推送到远程库。


```shell
# 推送到master分支
$ git push origin master

# 推送到dev分支
$ git push origin dev

```

2）抓取分支

现在，模拟一个你的小伙伴，可以在另一台电脑（注意要把SSH Key添加到GitHub）或者同一台电脑的另一个目录下克隆：

```shell
$ git clone git@github.com:michaelliao/learngit.git
Cloning into 'learngit'...
remote: Counting objects: 40, done.
remote: Compressing objects: 100% (21/21), done.
remote: Total 40 (delta 14), reused 40 (delta 14), pack-reused 0
Receiving objects: 100% (40/40), done.
Resolving deltas: 100% (14/14), done.

```

当你的小伙伴从远程库clone时，默认情况下，你的小伙伴只能看到本地的`master`分支。不信可以用`git branch`命令看看：

```shell
$ git branch
* master

```

现在，你的小伙伴要在dev分支上开发，就必须创建远程origin 的dev分支到本地，于是，他采用了下面命令：

```shell
$ git checkout -b dev origin/dev

```

现在，他就可以在dev上继续修改，然后，时不时地把dev分支push到远程：

```shell
$ git add env.txt

$ git commit -m "add env"
[dev 7a5e5dd] add env
 1 file changed, 1 insertion(+)
 create mode 100644 env.txt

$ git push origin dev
Counting objects: 3, done.
Delta compression using up to 4 threads.
Compressing objects: 100% (2/2), done.
Writing objects: 100% (3/3), 308 bytes | 308.00 KiB/s, done.
Total 3 (delta 0), reused 0 (delta 0)
To github.com:michaelliao/learngit.git
   f52c633..7a5e5dd  dev -> dev


```


但是，当你的小伙伴已经向origin/dev分支推送了他的提交，而碰巧你也对同样的文件作了修改，并试图推送：

```shell
$ cat env.txt
env

$ git add env.txt

$ git commit -m "add new env"
[dev 7bd91f1] add new env
 1 file changed, 1 insertion(+)
 create mode 100644 env.txt

$ git push origin dev
To github.com:michaelliao/learngit.git
 ! [rejected]        dev -> dev (non-fast-forward)
error: failed to push some refs to 'git@github.com:michaelliao/learngit.git'
hint: Updates were rejected because the tip of your current branch is behind
hint: its remote counterpart. Integrate the remote changes (e.g.
hint: 'git pull ...') before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for details.


```

推送失败，因为你的小伙伴的最新提交和你试图推送的提交有冲突，解决办法也很简单，Git已经提示我们，先用`git pull`把最新的提交从`origin/dev`抓下来，然后，在本地合并，解决冲突，再推送：


```shell
$ git pull
There is no tracking information for the current branch.
Please specify which branch you want to merge with.
See git-pull(1) for details.

    git pull <remote> <branch>

If you wish to set tracking information for this branch you can do so with:

    git branch --set-upstream-to=origin/<branch> dev

```

`git pull` 也失败了，原因是没有指定本地`dev`分支与远程`origin/dev` 分支的链接，根据提示，设置`dev`分支与远程`origin/dev` 分支的链接：

```shell
$ git branch --set-upstream-to=origin/dev dev

Branch 'dev' set up to track remote branch 'dev' from 'origin'.

#然后pull
$ git pull
Auto-merging env.txt
CONFLICT (add/add): Merge conflict in env.txt
Automatic merge failed; fix conflicts and then commit the result.


```

这回git pull成功，但是合并有冲突，需要手动解决，解决的方法和分支管理中的解决冲突完全一样。解决后，提交，再push：


```shell
$ git commit -m "fix env conflict"
[dev 57c53ab] fix env conflict

$ git push origin dev
Counting objects: 6, done.
Delta compression using up to 4 threads.
Compressing objects: 100% (4/4), done.
Writing objects: 100% (6/6), 621 bytes | 621.00 KiB/s, done.
Total 6 (delta 0), reused 0 (delta 0)
To github.com:michaelliao/learngit.git
   7a5e5dd..57c53ab  dev -> dev


```

因此，多人协作的工作模式通常如下：

1）首先，可以试图用git push origin <branch-name>推送自己的修改；

2）如果推送失败，则因为远程分支比你的本地更新，需要先用git pull试图合并；

3）如果合并有冲突，则解决冲突，并在本地提交；

4）没有冲突或者解决掉冲突后，再用git push origin <branch-name>推送就能成功！

如果git pull提示no tracking information，则说明本地分支和远程分支的链接关系没有创建，用命令git branch --set-upstream-to <branch-name> origin/<branch-name>。

#### 7.Rebase

多人在同一个分支上协作时，很容易出现冲突。即使没有冲突，后push的童鞋不得不先pull，在本地合并，然后才能push成功。每次合并后再push后，分支变得很乱，而rebase则是解决这一问题。

rebase操作可以把本地未push的分叉提交历史整理成直线，使得我们在查看历史提交的变化时更容易，因为分叉的提交需要三方对比。说明如下：

在和远程分支同步后，我们对hello.py这个文件做了两次提交。用git log命令看看：

```shell
$ git log --graph --pretty=oneline --abbrev-commit
* 582d922 (HEAD -> master) add author
* 8875536 add comment
* d1be385 (origin/master) init hello
*   e5e69f1 Merge branch 'dev'
|\  
| *   57c53ab (origin/dev, dev) fix env conflict
| |\  
| | * 7a5e5dd add env
| * | 7bd91f1 add new env
...

```
注意到Git用(`HEAD -> master`)和(`origin/master`)标识出当前分支的HEAD和远程origin的位置分别是`582d922 add author`和`d1be385 init hello`，本地分支比远程分支快两个提交。

现在，我们尝试推送本地分支：

```shell
$ git push origin master
To github.com:michaelliao/learngit.git
 ! [rejected]        master -> master (fetch first)
error: failed to push some refs to 'git@github.com:michaelliao/learngit.git'
hint: Updates were rejected because the remote contains work that you do
hint: not have locally. This is usually caused by another repository pushing
hint: to the same ref. You may want to first integrate the remote changes
hint: (e.g., 'git pull ...') before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for details.

```

失败了，这说明有人先我们推送了远程分支。按照经验，先pull下：

```shell
$ git pull
remote: Counting objects: 3, done.
remote: Compressing objects: 100% (1/1), done.
remote: Total 3 (delta 1), reused 3 (delta 1), pack-reused 0
Unpacking objects: 100% (3/3), done.
From github.com:michaelliao/learngit
   d1be385..f005ed4  master     -> origin/master
 * [new tag]         v1.0       -> v1.0
Auto-merging hello.py
Merge made by the 'recursive' strategy.
 hello.py | 1 +
 1 file changed, 1 insertion(+)

```

再用 `git status` 查看状态：

```shell
$ git status
On branch master
Your branch is ahead of 'origin/master' by 3 commits.
  (use "git push" to publish your local commits)

nothing to commit, working tree clean
```

加上刚才的合并，现在我们本地分支比远程分支超前3个提交。用 `git log`查看

```shell
$ git log --graph --pretty=oneline --abbrev-commit
*   e0ea545 (HEAD -> master) Merge branch 'master' of github.com:michaelliao/learngit
|\  
| * f005ed4 (origin/master) set exit=1
* | 582d922 add author
* | 8875536 add comment
|/  
* d1be385 init hello
...
```
这个时候，rebase就派上了用场。我们输入命令git rebase试试：

```shell
$ git rebase
First, rewinding head to replay your work on top of it...
Applying: add comment
Using index info to reconstruct a base tree...
M	hello.py
Falling back to patching base and 3-way merge...
Auto-merging hello.py
Applying: add author
Using index info to reconstruct a base tree...
M	hello.py
Falling back to patching base and 3-way merge...
Auto-merging hello.py
```

然后再通过`git log`看下：

```shell
$ git log --graph --pretty=oneline --abbrev-commit
* 7e61ed4 (HEAD -> master) add author
* 3611cfe add comment
* f005ed4 (origin/master) set exit=1
* d1be385 init hello
...

```

原本分叉的提交现在变成一条直线了！这种神奇的操作是怎么实现的？其实原理非常简单。我们注意观察，发现Git把我们本地的提交“挪动”了位置，放到了`f005ed4 (origin/master) set exit=1`之后，这样，整个提交历史就成了一条直线。rebase操作前后，最终的提交内容是一致的，但是，我们本地的commit修改内容已经变化了，它们的修改不再基于`d1be385 init hello`，而是基于`f005ed4 (origin/master) set exit=1`，但最后的提交`7e61ed4`内容是一致的。

这就是rebase操作的特点：把分叉的提交历史“整理”成一条直线，看上去更直观。缺点是本地的分叉提交已经被修改过了。

最后，通过push操作把本地分支推送到远程：

```shell
Mac:~/learngit michael$ git push origin master
Counting objects: 6, done.
Delta compression using up to 4 threads.
Compressing objects: 100% (5/5), done.
Writing objects: 100% (6/6), 576 bytes | 576.00 KiB/s, done.
Total 6 (delta 2), reused 0 (delta 0)
remote: Resolving deltas: 100% (2/2), completed with 1 local object.
To github.com:michaelliao/learngit.git
   f005ed4..7e61ed4  master -> master

```


然后再看下提交记录：

```shell
$ git log --graph --pretty=oneline --abbrev-commit
* 7e61ed4 (HEAD -> master, origin/master) add author
* 3611cfe add comment
* f005ed4 set exit=1
* d1be385 init hello
...

```
这样，远程分支非提交历史也是一条直线。

### 标签管理

1）创建标签

在Git中打标签非常简单，首先，切换到需要打标签的分支上：


```shell
$ git branch
* dev
  master

$ git checkout master
Switched to branch 'master'

```

然后输入命令 `git tag <name>` 就可以打一个新标签：

```shell
$ git tag v1.0
```

可以用命令 `git tag` 查看所有标签：

```shell
$ git tag
v1.0
```

默认标签是打在最新提交的commit上的。有时候，如果忘了打标签，比如，现在已经是周五了，但应该在周一打的标签没有打，怎么办？

方法是找到历史提交的commit id，然后打上就可以了：

```shell

$ git log --pretty=oneline --abbrev-commit
12a631b (HEAD -> master, tag: v1.0, origin/master) merged bug fix 101
4c805e2 fix bug 101
e1e9c68 merge with no-ff
f52c633 add merge
cf810e4 conflict fixed
5dc6824 & simple
14096d0 AND simple
b17d20e branch test
d46f35e remove test.txt
b84166e add test.txt
519219b git tracks changes
e43a48b understand how stage works
1094adb append GPL
e475afc add distributed
eaadf4e wrote a readme file

```

比方说要对add merge这次提交打标签，它对应的commit id是f52c633，敲入命令：

```shell
$ git tag v0.9 f52c633

```

可以用 `git show <tagname>` 查看标签信息：

```shell
$ git show v0.9
commit f52c63349bc3c1593499807e5c8e972b82c8f286 (tag: v0.9)
Author: Michael Liao <askxuefeng@gmail.com>
Date:   Fri May 18 21:56:54 2018 +0800

    add merge

diff --git a/readme.txt b/readme.txt
...

```

还可以创建带有说明的标签，用-a指定标签名，-m指定说明文字：


```shell
$ git tag -a v0.1 -m "version 0.1 released" 1094adb


```


2)操作标签

如果标签打错了，也可以删除：


```shell
$ git tag -d v0.1
Deleted tag 'v0.1' (was f15b0dd)

```

因为创建的标签都只存储在本地，不会自动推送到远程。所以，打错的标签可以在本地安全删除。

如果要推送某个标签到远程，使用命令git push origin <tagname>：

```shell
$ git push origin v1.0
Total 0 (delta 0), reused 0 (delta 0)
To github.com:michaelliao/learngit.git
 * [new tag]         v1.0 -> v1.0


```

或者，一次性推送全部尚未推送到远程的本地标签：

```shell
$ git push origin --tags
Total 0 (delta 0), reused 0 (delta 0)
To github.com:michaelliao/learngit.git
 * [new tag]         v0.9 -> v0.9

```

如果标签已经推送到远程，要删除远程标签麻烦一点：
（1）先从本地删除

```shell
$ git tag -d v0.9
Deleted tag 'v0.9' (was f52c633)

```

(2)从远程删除

```shell
$ git push origin :refs/tags/v0.9
To github.com:michaelliao/learngit.git
 - [deleted]         v0.9
```


以上，为Git版本管理常见命令，参考[廖雪峰老师的网站](https://www.liaoxuefeng.com/wiki/896043488029600)。非常感谢！


