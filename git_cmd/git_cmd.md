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
然后在当前dev分支上提交修改

```shell
#
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








 
