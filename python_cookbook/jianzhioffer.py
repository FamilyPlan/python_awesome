"""
74. 搜索二维矩阵
编写一个高效的算法来判断 m x n 矩阵中，是否存在一个目标值。该矩阵具有如下特性：
每行中的整数从左到右按升序排列。
每行的第一个整数大于前一行的最后一个整数。

"""
# 16ms
class Solution1:
  def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
    if len(matrix) == 0: return False
    for i in range(len(matrix)-1):
      if target >= matrix[i][0] and target < matrix[i+1][0]:
        if target in matrix[i]:return True
    if target in matrix[len(matrix)-1]:return True
    return False
# 20ms-二分法
class Solution2:
  def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
    if len(matrix) == 0: return False
    lo, hi = 0, len(matrix)-1
    while lo<=hi:
      mid = (lo+hi)//2
      if matrix[mid][0]<target:
        lo = mid+1
      elif matrix[mid][0]>target:
        hi = mid-1
      else:return True
    row = hi
    lo,hi=0,len(matrix[0])-1
    while lo<=hi:
      mid =  (lo+hi)//2
      if matrix[row][mid]<target:
        lo = mid+1
      elif matrix[row][mid]>target:
        hi = mid-1
      else:return True
    return False    
# 76ms
class Solution3:
  def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
    if not matrix or len(matrix)<1:return False
    rows,cols = len(matrix),len(matrix[0])
    row,col = 0,cols-1
    while row<rows and col>=0:
      if target > matrix[row][col]:row += 1
      elif target < matrix[row][col]:col -= 1
      else:return True
    return False


"""
102. 二叉树的层次遍历
"""
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
# 40ms
class Solution1:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root: return []
        stack = [root]
        res = []
        while stack:
            tmp_root = stack
            stack =[]
            tmp_val = []
            for i in range(len(tmp_root)):
                tmp_val.append(tmp_root[i].val)
                if tmp_root[i].left: stack.append(tmp_root[i].left)
                if tmp_root[i].right: stack.append(tmp_root[i].right)
            res.append(tmp_val)
        return res

# 8ms
# class Solution2:
#   levels = [] #用于存储每一层的节点值
#   def levelOrder(self, root):
#     if not root: return levels
#   def helper(node,level):
#     if len(levels) == level:
#       levels.append([])
#     levels[level].append(node.val)
#     if node.left: helper(node.left,level+1)
#     if node.right: helper(node.right,level+1)
#   helper(root,0)
#   return levels

"""
107. 二叉树的层次遍历 II
给定一个二叉树，返回其节点值自底向上的层次遍历。 
（即按从叶子节点所在层到根节点所在的层，逐层从左向右遍历）
"""
class Solution(object):
    def levelOrderBottom(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        def helper(node, level):
            if not node:
                return
            else:
                sol[level-1].append(node.val)
                if len(sol) == level:  # 遍历到新层时，只有最左边的结点使得等式成立
                    sol.append([])
                helper(node.left, level+1)
                helper(node.right, level+1)
        sol = [[]]
        helper(root, 1)
        return reversed(sol[:-1])




"""
144. 二叉树的前序遍历
"""
# 28ms
class Solution1:
  def preorderTraversal(self, root: TreeNode) -> List[int]:
    if not root: return []
    left = self.preorderTraversal(root.left)
    right = self.preorderTraversal(root.right)
    return [root.val]+left+right

# 12ms
class Solution2:
  def preorderTraversal(self, root: TreeNode) -> List[int]:
    stack = [root]
    rel = []
    while stack:
      node = stack.pop()
      while node:
        rel.append(node.val)
        stack.append(node.right)
        node = node.left
    return rel


"""
145. 二叉树的后序遍历
"""
#40ms
class Solution1:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        if not root: return []
        left = self.postorderTraversal(root.left)
        right = self.postorderTraversal(root.right)
        return left+right+[root.val]

# 28ms
class Solution2:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
      res, tmp = [],root
      while tmp:
        res.append(tmp.val)
        if tmp.right:
          pre = tmp.right
          while pre.left or pre.right:
            while pre.left:
              pre = pre.left
            if pre.right:
              pre=pre.right
          if tmp.left:
            pre.right = tmp.left
          tmp = tmp.right
        else:tmp =tmp.left
      return res[::-1]

"""
94. 二叉树的中序遍历
"""
# 32ms
class Solution1:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        if not root: return []
        left = self.inorderTraversal(root.left)
        right = self.inorderTraversal(root.right)
        return left+[root.val]+right 

#4ms
class Solution2(object):
    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        stack = list()
        result = list()
        curr = root
        while len(stack) or curr:
            while curr:
                stack.append(curr)
                curr = curr.left
            curr = stack.pop()
            result.append(curr.val)
            curr = curr.right
        return result

"""
105. 从前序与中序遍历序列构造二叉树
"""

# 32ms
class Solution1:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        def helper(left, right):
            nonlocal pre_idx
            if left == right:
                return None
            val = preorder[pre_idx]
            head = TreeNode(val)
            in_idx = idx_map[val]
            pre_idx+=1
            head.left = helper(left, in_idx)
            head.right = helper(in_idx+1, right)
            return head
        
        idx_map = {val:idx for idx,val in enumerate(inorder)}
        pre_idx = 0
        return helper(0, len(preorder))

#208ms
class Solution2:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        if len(preorder)!=len(inorder) or len(preorder)<1:return None
        root = TreeNode(preorder[0])
        root_index=inorder.index(preorder[0])
        left_inorder = inorder[0:root_index]
        right_inorder = inorder[root_index+1:]
        left_preorder = preorder[1:len(left_inorder)+1]
        right_preorder = preorder[len(left_inorder)+1:]
        root.left = self.buildTree(left_preorder,left_inorder)
        root.right = self.buildTree(right_preorder,right_inorder)
        return root  

"""
插入排序
"""
"""
直接插入算法：将一条待排序的记录按照其关键字值的大小插入到已排序的记录序列中的正确位置，一次重复，直到全部记录都插入完成。
"""

arr = [2,45,36,72,34]

def insertSort(arr):
  for i in range(1,len(arr)):
    p = arr[i]
    for j in range(i-1,-1,-1):
      if arr[j]>=p:
        arr[j+1]=arr[j]
      else:break
    arr[j+1]=p
  return arr

"""
希尔排序，又称缩小增量排序，是对直接插入排序的改进算法，其基本思想是分组的直接插入排序。
1）将一个数据元素序列分组，每组由若干个相隔一段距离的元素组成，在一个组内采用直接插入算法进行排序；
2）增量的初值通常为数据元素序列的一半，以后每趟增量减半，最后值为1。随着增量逐渐减小，组数也减少，组内元素的个数增加，数据元素接近有序。
"""
def shellSort(d,arr):
  # d为分段增量的数组集合
  for k in d:
    # 在增量内进行直接插入排序
    j = k
    while j < len(arr):
      p = arr[j]
      m = j
      while m >= k:
        if arr[m-k]>p:
          arr[m] = arr[m-k]
          m = m-k
        else: break
      arr[m]=p
      j+=1
    
"""
交换排序：冒泡排序、快速排序
"""

"""
冒泡排序是两两比较待排序记录的关键字，如果次序相反则交换两个记录的位置，直到序列中的所有记录都有序
"""
def bubbleSort(arr):
  flag = True
  i = 1
  while i <len(arr) and flag:
    flag = False
    for j in range(len(arr)-i):
      if arr[j+1]<arr[j]:
        arr[j+1],arr[j] = arr[j], arr[j+1]
        flag=True
    i+=1

"""
归并排序
"""
class MergeSort(object):
  def __init__(self,arr):
    self.from_lst = arr
    self.to_lst = [None for i in arr]
    self.sort()
  def merge(self,from_lst,to_lst,leftptr,mid,rightbound):
    i,j,k=leftptr,mid+1,leftptr
    while i<=mid and j<=rightbound:
      if from_lst[i]<from_lst[j]:
        to_lst[k]=from_lst[i]
        i+=1
      else:
        to_lst[k]=from_lst[j]
        j+=1
      k+=1
    while i<=mid:
      to_lst[k]=from_lst[i]
      i+=1
      k+=1
    while j<=rightbound:
      to_lst[k]=from_lst[j]
      j+=1
      k+=1
  def merge_pass(self,from_lst, to_lst, llen, slen):
    i = 0
    while i+2*slen-1<llen:
      self.merge(from_lst,to_lst,i,i+slen-1,i+2*slen-1)
      i += 2*slen
    if i+slen-1<llen:
      # 剩余两段，但最后一段长度不足slen
      self.merge(from_lst,to_lst,i,i+slen-1,llen-1)
    else:
      # 剩余最后一段
      for j in range(i,llen):
        to_lst[j]=from_lst[j]
  def sort(self):
    slen,llen=1,len(self.from_lst)
    while slen<llen:
      self.merge_pass(self.from_lst,self.to_lst,llen,slen)
      slen *= 2
      self.merge_pass(self.to_lst,self.from_lst,llen,slen)
      slen *= 2
  def printRes(self):
    print (self.from_lst, self.to_lst)

# ms = MergeSort([25,57,48,37,12,82,75,29,16])   
# ms.printRes()

"""
快速排序
"""

def quick_sort(lst):
  def qsort_rec(lst,left,right):
    if left>=right:
      return
    i=left;j=right;pivot=lst[i]
    while i<j:
      while i<j and lst[j]>=pivot:
        j-=1
      if i<j:
        lst[i] = lst[j];i+=1 #小记录放到左边
      while i<j and lst[i]<=pivot:
        i+=1
      if i<j:lst[j]=lst[i];j-=1
    lst[i]=pivot
    qsort_rec(lst,l,i-1)
    qsort_rec(lst,i+1,r)
  qsort_rec(lst,0,len(lst)-1)

"""
二分查找
如果面试题是要求在排序的数组（或者部分排序的数组）中查找一个数字或者统计某个数字出现的次数，都可以尝试二分查找方法。
二分查找也称折半查找（Binary Search），它是一种效率较高的查找方法，前提是数据结构必须先排好序，可以在数据规模的对数时间
复杂度内完成查找。但是，二分查找要求线性表具有有随机访问的特点（例如数组），也要求线性表能够根据中间元素的特点推测它两侧元素
的性质，以达到缩减问题规模的效果。
"""
"""
35. 搜索插入位置
给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。

你可以假设数组中无重复元素。
示例 1:
输入: [1,3,5,6], 5
输出: 2

示例 2:
输入: [1,3,5,6], 2
输出: 1

示例 3:
输入: [1,3,5,6], 7
输出: 4
"""
class Solution(object):
    def searchInsert(self, nums, target):
      """
      :type nums: List[int]
      :type target: int
      :rtype: int
      """
      left, right = 0, len(nums)
      while left<right:
        mid = (left+right)//2
        if target<nums[mid]:
          right = mid
        elif target>nums[mid]:
          left = mid+1
        else:return mid
      return left

"""
69. x 的平方根
实现 int sqrt(int x) 函数。计算并返回 x 的平方根，其中 x 是非负整数。
由于返回类型是整数，结果只保留整数的部分，小数部分将被舍去。
示例 1:
输入: 4
输出: 2

示例 2:
输入: 8
输出: 2
说明: 8 的平方根是 2.82842..., 
     由于返回类型是整数，小数部分将被舍去。
"""
# 28ms
class Soolution1(object):
  def mySqrt(self,x):
    left = 0
    right = x//2+1
    while left < right:
      mid = (left+right+1)>>1
      square = mid*mid
      if square < x: left = mid
      elif square > x: right = mid-1
      else:return mid
    return left

class Soolution2(object):
  def mySqrt(self,x):
    if x == 0 or x == 1:return x
    left = 1
    right = x>>1
    while left <= right:
      mid = (left+right) >> 1
      square = mid*mid
      if square>x:
        right = mid-1
      elif square<x and (mid+1)*(mid+1)<=x:
        return mid+1
      else:
        return mid
    return 1

"""
167. 两数之和 II - 输入有序数组
给定一个已按照升序排列 的有序数组，找到两个数使得它们相加之和等于目标数。函数应该返回这两个下标值 index1 和 index2，其中 index1 必须小于 index2。
返回的下标值（index1 和 index2）不是从零开始的。你可以假设每个输入只对应唯一的答案，而且你不可以重复使用相同的元素。
示例:
输入: numbers = [2, 7, 11, 15], target = 9
输出: [1,2]
解释: 2 与 7 之和等于目标数 9 。因此 index1 = 1, index2 = 2 。
"""

class Solution(object):
    def twoSum(self, numbers, target):
        """
        :type numbers: List[int]
        :type target: int
        :rtype: List[int]
        """
        left = 0
        right = len(numbers)-1
        while left<right:
            if numbers[left]+numbers[right]==target:return [left+1,right+1]
            elif numbers[left]+numbers[right]<target:
                left+=1
            else:
                right -=1
        return []

"""
33. 搜索旋转排序数组
假设按照升序排序的数组在预先未知的某个点上进行了旋转。
( 例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] )。
搜索一个给定的目标值，如果数组中存在这个目标值，则返回它的索引，否则返回 -1 。
你可以假设数组中不存在重复的元素。
你的算法时间复杂度必须是 O(log n) 级别。
示例 1:
输入: nums = [4,5,6,7,0,1,2], target = 0
输出: 4
"""
class Solution(object):
  def search(self,nums,target):
    if not nums: return -1
    i, j = 0, len(nums)-1
    while i < j-1:
      mid = (i+j)//2
      mv = nums[mid]
      if mv == target:return mid
      if nums[i] < mv:
        if target <= mv and target>=nums[i]:
          j=mid
        else:
          i=mid
      else:
        if target<=nums[j] and target>=mv:
          i=mid
        else:
          j=mid
    if nums[i] == target: return i
    if nums[j] == target:return j
    return -1

"""
34. 在排序数组中查找元素的第一个和最后一个位置
给定一个按照升序排列的整数数组 nums，和一个目标值 target。找出给定目标值在数组中的开始位置和结束位置。
你的算法时间复杂度必须是 O(log n) 级别。如果数组中不存在目标值，返回 [-1, -1]。
示例 1:
输入: nums = [5,7,7,8,8,10], target = 8
输出: [3,4]

示例 2:
输入: nums = [5,7,7,8,8,10], target = 6
输出: [-1,-1]
"""
# 76ms
class Solution1(object):
    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        if not nums: return [-1,-1]
        i,j = 0, len(nums)-1
        res = []
        while i<j:
            mi = (i+j)//2
            mv = nums[mi]
            if mv == target:
                ii,jj=mi,mi
                while nums[ii]==target and ii>=0:
                    ii-=1
                while jj<len(nums) and nums[jj]==target:
                    jj+=1
                if ii == jj :res.append(mi,mi)
                else:res.append(ii+1);res.append(jj-1)
                return sorted(res)
            elif mv < target:
                i = mi+1
            else:
                j = mi-1
        if nums[i] == target: res.append(i);res.append(i)
        return sorted(res) if res!=[] else [-1,-1]


class Solution(object):
    def searchRange(self, nums, target):
        left = self.findLeft(nums, target)
        right = self.findRight(nums, target)
        if left <= right:
            return [left, right]
        else:
            return [-1, -1]

    def findLeft(self, A, x):
        low = 0
        high = len(A) - 1
        while low <= high:
            mid = (low + high) // 2
            if A[mid] < x:
                low = mid + 1
            else:
                high = mid - 1
        return low
    def findRight(self, A, x):
        low = 0
        high = len(A) - 1
        while low <= high:
            mid = (low + high) // 2
            if A[mid] <= x:
                low = mid + 1
            else:
                high = mid - 1
        return high
  

"""
面试题50. 第一个只出现一次的字符
在字符串 s 中找出第一个只出现一次的字符。如果没有，返回一个单空格。

示例:

s = "abaccdeff"
返回 "b"

s = "" 
返回 " "

"""      

# 500ms
# from collections import OrderedDict
# class Solution(object):
#     def firstUniqChar(self, s):
#         """
#         :type s: str
#         :rtype: str
#         """
#         if not s:return ' '
#         temp = OrderedDict()
#         for i in s:
#             if i in temp:
#                 temp[i]+=1
#             else:
#                 temp[i]=1
#         for key,val in temp.items():
#             if val == 1:return key
#         return ' '


# 48ms
class Solution(object):
    def firstUniqChar(self, s):
        """
        :type s: str
        :rtype: str
        """
        if not s: return ' '
        if len(s) == 1:return s
        d ={}
        for i in range(len(s)):
            if s[i] not in d:
                if s[i] not in s[i+1:]:return s[i]
                else:d[s[i]]=1
        return ' '



def dfs(root, node, stack):
            if not root:
                return False
            stack.append(root)
            if root.val == node.val:
                return True          
            if (dfs(root.left, node, stack) or dfs(root.right, node, stack)):
                return True
            stack.pop()


















