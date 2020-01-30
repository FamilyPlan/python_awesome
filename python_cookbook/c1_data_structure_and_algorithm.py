"""
第一章 数据结构和算法
"""
"""
1.1 解压序列赋值给多个变量
说明：任何的序列（或者是可迭代对象）可以通过一个简单的赋值语句解压并赋值给多个变量。 
唯一的前提就是变量的数量必须跟序列元素的数量是一样的。如果变量个数和序列元素的个数不匹配，会产生一个异常。
有时候，你可能只想解压一部分，丢弃其他的值。对于这种情况 Python 并没有提供特殊的语法。
但是你可以使用任意变量名去占位，到时候丢掉这些变量就行了。例如：
>>> data = [ 'ACME', 50, 91.1, (2012, 12, 21) ]
>>> _, shares, price, _ = data
"""

"""
1.2 解压可迭代对象赋值给多个变量
如果一个可迭代对象的元素个数超过变量个数时，会抛出一个 ValueError 。 
那么怎样才能从这个可迭代对象中解压出 N 个元素出来？
Python 的星号表达式可以用来解决这个问题。

"""
# import numpy as np
# def drop_first_last(grades):
#   first, *middle, last = grades
#   return np.average(middle)
# grades = [12,45,65,85]
# print (drop_first_last(grades)) #55

"""
1.3  保留最后 N 个元素
deque类可以被用在任何你只需要一个简单队列的场合，如果不设置maxlen，则得到一个无限大小的队列。
可以在队列两端执行添加和弹出元素的操作,在队列两端添加和删除元素的复杂度都是O(1)
"""
# from collections import deque
# q = deque(maxlen=3) #保持最新的3个历史记录
# q.append(1)
# q.append(2)
# q.append(3)
# # print (q) # deque([1, 2, 3], maxlen=3)
# q.append(4)
# print (q) #deque([2, 3, 4], maxlen=3)

# q.appendleft(4)
# q.appendleft(3)
# q.appendleft(2)
# print (q) #deque([2, 3, 4], maxlen=3)
# print (q.pop()) #4
# print(q.popleft()) #2

"""
1.4 查找最大或最小的N个元素
heapq模块有两个函数：nlargest()和nsmallest()可以解决
"""
# import heapq

# nums = [1, 8, 2, 23, 7, -4, 18, 23, 42, 37, 2]
# print(heapq.nlargest(3, nums)) # Prints [42, 37, 23]
# print(heapq.nsmallest(3, nums)) # Prints [-4, 1, 2]


# portfolio = [
#     {'name': 'IBM', 'shares': 100, 'price': 91.1},
#     {'name': 'AAPL', 'shares': 50, 'price': 543.22},
#     {'name': 'FB', 'shares': 200, 'price': 21.09},
#     {'name': 'HPQ', 'shares': 35, 'price': 31.75},
#     {'name': 'YHOO', 'shares': 45, 'price': 16.35},
#     {'name': 'ACME', 'shares': 75, 'price': 115.65}
# ]
# cheap = heapq.nsmallest(3, portfolio, key=lambda s: s['price'])
# expensive = heapq.nlargest(3, portfolio, key=lambda s: s['price'])
# print(cheap)
# [{'name': 'YHOO', 'shares': 45, 'price': 16.35}, 
# {'name': 'FB', 'shares': 200, 'price': 21.09}, 
# {'name': 'HPQ', 'shares': 35, 'price': 31.75}]

# import heapq
# # 堆数据结构最重要的特征是 heap[0] 永远是最小的元素。并且剩余的元素可以很容易的通过调用 heapq.heappop() 方法得到， 
# # 该方法会先将第一个元素弹出来，然后用下一个最小的元素来取代被弹出元素（这种操作时间复杂度仅仅是 O(log N)，N 是堆大小）
# nums = [1, 8, 2, 23, 7, -4, 18, 23, 42, 37, 2]
# heap = list(nums)
# heapq.heapify(heap)
# print (heap) #[-4, 2, 1, 23, 7, 2, 18, 23, 42, 37, 8]
# print (heapq.heappop(heap)) #-4
# print (heapq.heappop(heap)) #1
# print (heapq.heappop(heap)) #2
# print (heapq.heappop(heap)) #2
# print (heapq.heappop(heap)) #7
# print (heapq.heappop(heap)) #8


"""
1.5 实现一个优先级队列
实现一个优先级队列，在这个队列上面，每次pop操作总是返回优先级最高的那个元素
"""
# import heapq
# class PriorityQueue:
#   def __init__(self):
#     self._queue=[]
#     self._index = 0
#   def push(self,item,priority):
#     heapq.heappush(self._queue,(-priority,self._index,item))
#     self._index += 1
#   def pop(self):
#     return heapq.heappop(self._queue)[-1]

# class Item:
#   def __init__(self,name):
#     self.name = name
#   def __repr__(self):
#     return 'Item({!r})'.format(self.name)

# q=PriorityQueue()
# q.push(Item('foo'), 1)
# q.push(Item('bar'), 5)
# q.push(Item('spam'), 4)
# q.push(Item('grok'), 1)
# # print(q.pop()) #Item('bar')
# # print(q.pop()) #Item('spam')
# # print(q.pop()) #Item('foo')
# # print(q.pop()) #Item('grok')

"""
1.6 字典中的键映射多个值
使用collections中的defaultdict来构造字典(一个键对应多个值)，defaultdict的特征是它会自动初始化
每个key刚开始对应的值，所以只需要关注添加元素操作。
"""
# from collections import defaultdict

# d = defaultdict(list)
# d['a'].append(1)
# d['a'].append(2)
# d['b'].append(3)
# print (d,d['a']) #{'a': [1, 2], 'b': [3]}) [1, 2]

# b=defaultdict(set)
# b['a'].add(1)
# b['a'].add(2)
# b['a'].add(1)
# b['e'].add(1)
# print (b,b['a']) #{'a': {1, 2}, 'e': {1}}) {1, 2}

"""
1.7 字典排序
collectios模块中的OrderedDict类。在迭代操作的时候它会保持元素被插入时的顺序。
当需要构建一个将来需要序列化或编码为其它格式的预设的时候，OrderedDict是非常有用的。
OrderedDict 内部维护着一个根据键插入顺序排序的双向链表。每次当一个新的元素插入进来的时候， 它会被放到链表的尾部。
对于一个已经存在的键的重复赋值不会改变键的顺序。
但一个 OrderedDict 的大小是一个普通字典的两倍，因为它内部维护着另外一个链表。所以，当数据集较大时，得注意内存消耗。
"""
# from collections import OrderedDict

# d = OrderedDict()
# d['foo'] = 1
# d['bar'] = 2
# d['spam'] = 3
# d['grok'] = 4
# d['aa']=5
# d['foo']=6

# a = dict()
# a['foo'] = 1
# a['bar'] = 2
# a['spam'] = 3
# a['grok'] = 4
# a['aa']=5
# a['foo'] = 6
# for k in d:
#   print (k,d[k])
# # foo 6
# # bar 2
# # spam 3
# # grok 4
# # 'aa' 5

# for k in a:
#   print (k,a[k])
# # foo 1
# # bar 2
# # spam 3
# # grok 4
# # 'aa' 5

# # import json
# # print (json.dumps(d)) #{"foo": 3, "bar": 2, "spam": 1, "grok": 4, "aa": 5}
# # print (json.dumps(a)) #{"foo": 3, "bar": 2, "spam": 1, "grok": 4, "aa": 5}

"""
1.8 字典的运算
在字典中，进行一些操作，包括求解最大值、最小值、排序等
"""

# prices = {
#   'ACME': 45.23,
#   'AAPL': 612.78,
#   'IBM': 205.55,
#   'HPQ': 37.20,
#   'FB': 10.75
# }
# prices_sorted = sorted(zip(prices.values(),prices.keys()))
# print(prices_sorted) #[(10.75, 'FB'), (37.2, 'HPQ'), (45.23, 'ACME'), (205.55, 'IBM'), (612.78, 'AAPL')]
# prices_st=sorted(prices, key=lambda k: prices[k])
# print (prices_st) #['FB', 'HPQ', 'ACME', 'IBM', 'AAPL']

"""
1.9 查找两字典的相同点
字典的 keys() 方法返回一个展现键集合的键视图对象。 
键视图的一个很少被了解的特性就是它们也支持集合操作，比如集合并、交、差运算。 
所以，如果你想对集合的键执行一些普通的集合操作，可以直接使用键视图对象而不用先将它们转换成一个 set。

字典的 items() 方法返回一个包含 (键，值) 对的元素视图对象。 
这个对象同样也支持集合操作，并且可以被用来查找两个字典有哪些相同的键值对。

尽管字典的 values() 方法也是类似，但是它并不支持这里介绍的集合操作。 
某种程度上是因为值视图不能保证所有的值互不相同，这样会导致某些集合操作会出现问题。
不过，如果你硬要在值上面执行这些集合操作的话，你可以先将值集合转换成 set，然后再执行集合运算就行了。

"""

# a = {
#     'x' : 1,
#     'y' : 2,
#     'z' : 3
# }

# b = {
#     'w' : 10,
#     'x' : 11,
#     'y' : 2
# }

# print (a.keys() & b.keys()) #{'x', 'y'}
# print (a.keys() - b.keys()) #{'z'}
# print (a.items() & b.items()) #{('y', 2)}

"""
1.10 删除序列相同元素并保持顺序

"""
# 怎样在一个序列上面保持元素顺序的同时消除重复的值？
# 1)当元素是hashable
# def dedupe(items):
#   seen = set()
#   for item in items:
#     if item not in seen:
#       yield item
#       seen.add(item)
# a = [1, 5, 2, 1, 9, 1, 5, 10]
# print(list(dedupe(a))) #[1, 5, 2, 9, 10]

# 2)当元素是not hashable

# def dedupe(items,key=None):
#   seen = set()
#   for item in items:
#     print ('k: ',key(item))
#     val = item if key is None else key(item)
#     if val not in seen:
#       yield item
#       seen.add(val)
# a = [ {'x':1, 'y':2}, {'x':1, 'y':3}, {'x':1, 'y':2}, {'x':2, 'y':4}]
# print (list(dedupe(a, key=lambda d: (d['x'],d['y'])))) #[{'x': 1, 'y': 2}, {'x': 1, 'y': 3}, {'x': 2, 'y': 4}]
# print (list(dedupe(a, key=lambda d: d['x']))) #[{'x': 1, 'y': 2}, {'x': 2, 'y': 4}]

"""
1.11 命名切片
"""

# items = [0, 1, 2, 3, 4, 5, 6]
# a = slice(items)
# print(a) #slice(None, [0, 1, 2, 3, 4, 5, 6], None)
# print(a.start) #None
# print(a.stop) #0, 1, 2, 3, 4, 5, 6]
# print(a.step) #None

# 通过调用切片的 indices(size) 方法将它映射到一个已知大小的序列上。 
# 这个方法返回一个三元组 (start, stop, step) ，所有的值都会被缩小，直到适合这个已知序列的边界为止。
# a = slice(5,50,2)
# print(a) #slice(5, 50, 2)
# print(a.start) #5
# print(a.stop) #50
# print(a.step) #2
# s = 'helloworld'
# a.indices(len(s))
# for i in range(*a.indices(len(s))):
#   print(s[i])
# # w
# # r
# # d

"""
1.12 序列中出现次数最多的元素
collections.Counter 类就是专门为这类问题而设计的
它甚至有一个有用的 most_common() 方法直接给了你答案。
作为输入， Counter 对象可以接受任意的由可哈希（hashable）元素构成的序列对象。 
在底层实现上，一个 Counter 对象就是一个字典，将元素映射到它出现的次数上。比如：word_counts['eyes']
"""
# words = [
#     'look', 'into', 'my', 'eyes', 'look', 'into', 'my', 'eyes',
#     'the', 'eyes', 'the', 'eyes', 'the', 'eyes', 'not', 'around', 'the',
#     'eyes', "don't", 'look', 'around', 'the', 'eyes', 'look', 'into',
#     'my', 'eyes', "you're", 'under'
# ]
# from collections import Counter
# word_counts = Counter(words)
# print (word_counts) 
# #Counter({'eyes': 8, 'the': 5, 'look': 4, 'into': 3, 'my': 3, 'around': 2, 
# # 'not': 1, "don't": 1, "you're": 1, 'under': 1})
# print (word_counts['eyes']) #8
# # 出现频率最高的3个单词
# top_three = word_counts.most_common(3)
# print(top_three)
# # Outputs [('eyes', 8), ('the', 5), ('look', 4)]

"""
1.13 通过某个关键字排序一个字典列表
"""
# from operator import itemgetter
# rows = [
#   {'fname': 'Brian', 'lname': 'Jones', 'uid': 1003},
#   {'fname': 'David', 'lname': 'Beazley', 'uid': 1002},
#   {'fname': 'John', 'lname': 'Cleese', 'uid': 1001},
#   {'fname': 'Big', 'lname': 'Jones', 'uid': 1004}
# ]
# rows_by_fname = sorted(rows, key=itemgetter('fname')) #rows_by_fname = sorted(rows, key=lambda r: r['fname'])
# print(rows_by_fname)

# rows_by_lfname = sorted(rows, key=itemgetter('lname','fname')) #rows_by_lfname = sorted(rows, key=lambda r: (r['lname'],r['fname']))
# print(rows_by_lfname)

"""
1.14 排序不支持原生比较的对象
你想排序类型相同的对象，但是他们不支持原生的比较操作。
内置的 sorted() 函数有一个关键字参数 key ，可以传入一个 callable 对象给它， 这个 callable 对象对每个传入的对象
返回一个值，这个值会被 sorted 用来排序这些对象。 比如，如果你在应用程序里面有一个 User 实例序列，并且你希望通过他们的 
user_id 属性进行排序， 你可以提供一个以 User 实例作为输入并输出对应 user_id 值的 callable 对象。

选择使用 lambda 函数或者是 attrgetter() 可能取决于个人喜好。 但是， attrgetter() 函数通常会运行的快点，并且还能同时允许多个字段进行比较。
"""

# class User:
#   def __init__(self,user_id):
#     self.user_id = user_id
#   def __repr__(self):
#     return 'User({})'.format(self.user_id)

# def sort_notcompare():
#   print (users)
#   print (sorted(users, key=lambda u: u.user_id))
# sort_notcompare()
# # [User(23), User(3), User(99)]
# # [User(3), User(23), User(99)]

# from operator import attrgetter
# users = [User(23),User(3),User(99)]
# sorted(users,key=attrgetter('user_id'))
# by_name = sorted(users, key=attrgetter('last_name', 'first_name'))


"""
1.15 通过某个字段将记录分组
"""

# rows = [
#     {'address': '5412 N CLARK', 'date': '07/01/2012'},
#     {'address': '5148 N CLARK', 'date': '07/04/2012'},
#     {'address': '5800 E 58TH', 'date': '07/02/2012'},
#     {'address': '2122 N CLARK', 'date': '07/03/2012'},
#     {'address': '5645 N RAVENSWOOD', 'date': '07/02/2012'},
#     {'address': '1060 W ADDISON', 'date': '07/02/2012'},
#     {'address': '4801 N BROADWAY', 'date': '07/01/2012'},
#     {'address': '1039 W GRANVILLE', 'date': '07/04/2012'},
# ]

# from operator import itemgetter
# from itertools import groupby

# rows.sort(key=itemgetter('date'))
# for date, item in groupby(rows, key = itemgetter('date')):
#   print (date)
#   for i in item:
#     print (' ',i)

# # 07/01/2012
# #   {'address': '5412 N CLARK', 'date': '07/01/2012'}
# #   {'address': '4801 N BROADWAY', 'date': '07/01/2012'}
# # 07/02/2012
# #   {'address': '5800 E 58TH', 'date': '07/02/2012'}
# #   {'address': '5645 N RAVENSWOOD', 'date': '07/02/2012'}
# #   {'address': '1060 W ADDISON', 'date': '07/02/2012'}
# # 07/03/2012
# #   {'address': '2122 N CLARK', 'date': '07/03/2012'}
# # 07/04/2012
# #   {'address': '5148 N CLARK', 'date': '07/04/2012'}
# #   {'address': '1039 W GRANVILLE', 'date': '07/04/2012'}

"""
1.16 过滤序列元素
"""
# values = ['1', '2', '-3', '-', '4', 'N/A', '5']
# def is_int(val):
#   try:
#     x = int(val)
#     return x
#   except ValueError:
#     return False
# int_vals = list(filter(is_int,values))
# print (int_vals) #['1', '2', '-3', '4', '5']

# mylist = [1, 4, -5, 10, -7, 2, 3, -1]
# clip_neg = [n if n>0 else 0 for n in mylist]
# print(clip_neg) #[1, 4, 0, 10, 0, 2, 3, 0]

addresses = [
    '5412 N CLARK',
    '5148 N CLARK',
    '5800 E 58TH',
    '2122 N CLARK',
    '5645 N RAVENSWOOD',
    '1060 W ADDISON',
    '4801 N BROADWAY',
    '1039 W GRANVILLE',
]
counts = [ 0, 3, 10, 4, 1, 7, 6, 1]

# res = [m for m,n in zip(addresses,counts) if n>5]
# print (res) #['5800 E 58TH', '1060 W ADDISON', '4801 N BROADWAY']

# from itertools import compress
# more5 = [n>5 for n in counts]
# print(more5) #[False, False, True, False, False, True, True, False]
# print(list(compress(addresses,more5)))

"""
1.17 从字典中提取子集
"""

prices = {
    'ACME': 45.23,
    'AAPL': 612.78,
    'IBM': 205.55,
    'HPQ': 37.20,
    'FB': 10.75
}
# Make a dictionary of all prices over 200
# 字典推导式
# res = {k:v for k,v in prices.items() if v>200}
# print (res) #{'AAPL': 612.78, 'IBM': 205.55}

"""
1.18 映射名称到序列元素
你有一段通过下标访问列表或者元组中元素的代码，但是这样有时候会使得你的代码难以阅读， 于是你想通过名称来访问元素。
collections.namedtuple() 函数通过使用一个普通的元组对象来帮你解决这个问题。
 这个函数实际上是一个返回 Python 中标准元组类型子类的一个工厂方法。 
 你需要传递一个类型名和你需要的字段给它，然后它就会返回一个类，你可以初始化这个类，为你定义的字段传递值等
"""

# from collections import namedtuple
# Subscriber = namedtuple('Subscriber',['addr', 'joined'])
# sub = Subscriber('jonesy@example.com', '2012-10-19')
# print (sub) #Subscriber(addr='jonesy@example.com', joined='2012-10-19')
# print (sub.addr) #jonesy@example.com
# print (len(sub)) #2

# 使用普通元组的代码
# def compute_cost(records):
#     total = 0.0
#     for rec in records:
#         total += rec[1] * rec[2]
#     return total

# 使用namedtuple
# from collections import namedtuple
# Stock = namedtuple('Stock',['name','shares','price'])
# def compute_cost(records):
#   total =0.0
#   for rec in records:
#     s = Stock(rec)
#     total += s.shares*s.price
#   return total

# s=Stock('ACME',100,123.45)
# s = s._replace(shares=75)

# _replace() 方法还有一个很有用的特性就是当你的命名元组拥有可选或者缺失字段时候， 
# 它是一个非常方便的填充数据的方法。
# from collections import namedtuple
# Stock = namedtuple('Stock', ['name', 'shares', 'price', 'date', 'time'])
# stock_prototype = Stock('', 0, 0.0, None, None)
# def dict_to_stock(s):
#   return stock_prototype._replace(**s)

# a = {'name': 'ACME', 'shares': 100, 'price': 123.45}
# print (dict_to_stock(a)) #Stock(name='ACME', shares=100, price=123.45, date=None, time=None)

"""
1.19 转换并同时计算数据
你需要在数据序列上执行聚集函数（比如 sum() , min() , max() ）， 但是首先你需要先转换或者过滤数据。
一个非常优雅的方式去结合数据计算与转换就是使用一个生成器表达式参数。
"""
# 生成器
# nums = [1, 2, 3, 4, 5]
# s = sum(x*x for x in nums)
# print ((x*x for x in nums)) #<generator object <genexpr> at 0x10ee7ab88>
# print (s) #55

# # Determine if any .py files exist in a directory
# import os
# files = os.listdir('dirname')
# if any(name.endswith('.py') for name in files):
#     print('There be python!')
# else:
#     print('Sorry, no python.')

# Data reduction across fields of a data structure
# portfolio = [
#     {'name':'GOOG', 'shares': 50},
#     {'name':'YHOO', 'shares': 75},
#     {'name':'AOL', 'shares': 20},
#     {'name':'SCOX', 'shares': 65}
# ]
# min_shares = min(s['shares'] for s in portfolio)

# s = sum((x * x for x in nums)) # 显式的传递一个生成器表达式对象
# s = sum(x * x for x in nums) # 更加优雅的实现方式，省略了括号

"""
1.20 合并多个字典或映射
现在有多个字典或者映射，你想将它们从逻辑上合并为一个单一的映射后执行某些操作， 比如查找值或者检查某些键是否存在。
现在假设你必须在两个字典中执行查找操作（比如先从 a 中找，如果找不到再在 b 中找）。 
一个非常简单的解决方案就是使用 collections 模块中的 ChainMap 类。
一个 ChainMap 接受多个字典并将它们在逻辑上变为一个字典。 然后，这些字典并不是真的合并在一起了， 
ChainMap 类只是在内部创建了一个容纳这些字典的列表 并重新定义了一些常见的字典操作来遍历这个列表。
"""
# from collections import ChainMap

# a = {'x': 1, 'z': 3 }
# b = {'y': 2, 'z': 4 }
# c = ChainMap(a,b) #先在a中查找，没有才去b
# d = ChainMap(b,a) #先在b中查找，没有才去a
# print (c['x'],c['y'],c['z']) #1 2 3
# print (d['x'],d['y'],d['z']) #1 2 4

# ChainMap 对于编程语言中的作用范围变量（比如 globals , locals 等）是非常有用的。 
# 事实上，有一些方法可以使它变得简单：
# from collections import ChainMap
# values = ChainMap()
# values['x'] = 1
# # add a new mapping
# values = values.new_child()
# values['x'] = 2
# # add a new mapping
# values = values.new_child()
# values['y'] = 3
# print (values) #ChainMap({'y': 3}, {'x': 2}, {'x': 1})

# # discard last mapping
# values = values.parents
# print (values['x']) #2

# # discard last mapping
# values = values.parents
# print (values['x']) #1

# print (values) #ChainMap({'x': 1})






