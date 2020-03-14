
"""
2.1 使用多个界定符分割字符串
你需要将一个字符串分割为多个字段，但是分隔符(还有周围的空格)并不是固定的。
使用re.split
"""
# line = 'asdf fjdk; afed, fjek,asdf, foo'
# import re
# line_split = re.split(r'[;,\s]\s*',line) #分隔符可以是分号、逗号、空格、任意个空格
# print (line_split) #['asdf', 'fjdk', 'afed', 'fjek', 'asdf', 'foo']

# 如果需要捕获分组元素
# line = 'asdf fjdk; afed, fjek,asdf, foo'
# import re
# line_split = re.split(r'(;|,|\s)\s*',line)
# print (line_split) #['asdf', ' ', 'fjdk', ';', 'afed', ',', 'fjek', ',', 'asdf', ',', 'foo']
# values = line_split[::2]
# print (values) #['asdf', 'fjdk', 'afed', 'fjek', 'asdf', 'foo']
# delimiters = line_split[1::2]+['']
# print (delimiters) #delimiters
# # reform the line using the same delimiters
# reform_line = ''.join(v+d for v,d in zip(values,delimiters))
# print(reform_line)
# # 不捕获分组元素
# line_split_2 = re.split(r'(?:;|,|\s)\s*',line)
# print (line_split_2) #['asdf', 'fjdk', 'afed', 'fjek', 'asdf', 'foo']


"""
2.2 正则表达式
一、基础
元字符：. ^ $ * + ? \ | { } [ ] ( )
生成正则表达式对象：re.compile(pattern, flag=0)：r1 = re.compile("abc")
检索：
re.search(pattern, string, flag=0)
分割：
re.split(pattern,string,maxsplit=0,flags=0)
maxsplit指明分割数，0表示做完整个string
找到所有的匹配串：
re.findall(pattern,string,flag=0)

二：检索
1.字符组表达式[...]匹配括号中列出的任一个字符
[abc]可以匹配字符a或b或c
[0-9]匹配所有十进制数字字符
[0-9a-zA-Z]匹配所有英文字母和数字
[^...]中的^表示求补，这种模式匹配所有未在括号里列出的字符
[^0-9]匹配所有非十进制数字的字符
2.圆点字符.匹配任意一个字符
a..b匹配所有以a开头以b结束的四字字符串
a[1-9][0-9]a10, a11, ..., a99
3.re用换意串形式定义了几个常用字符组，包括：
\d:等价于[0-9]
\D:等价于[^0-9]
\s:等价于[ \t\v\n\f\r]，与所有空白字符匹配
\S:与所有非空白字符串匹配
\w:与所有字母数字字符匹配，等价于[0-9a-zA-Z]
\W:与所有非字母数字字符匹配，等价于[^0-9a-zA-Z]

三、重复
1.基本重复运算符是*，a*与a的0次或任意多次出现匹配
2.+表示重复1次或多次
3.可选（片段）用?运算符表示，？表示0次或1次重复
4.确定次数的重复用{n}表示，a{n}与a匹配的串的n次重复匹配，描述北京常用的固定号码：'(010-)?[2-9][0-9]{7}'
5.重复范围用{m,n}表示，a{m,n}与a匹配的串的m到n次重复匹配。m和n均可以省略，a{,n}表示出现0至n次；a{m,}表示出现m次以上。

四、选择
1.行首匹配：
以^符号开头的模式，只能与一行的前缀子串匹配：re.search('^for','books for children') 得到None
以$符号结束的模式，只能与一行的后缀子串匹配：re.search('fish$','cats like to eat fishes') 得到None
注意，“一行的”前缀/后缀包括整个被匹配串的前缀和后缀。如串里有 换行符，还包括换行符前的子串(一行的后缀)和其后的子串(前缀)

五、模式里的组(group)
被匹配的组可用 \n 形式在模式里“引用”，要求匹配同样字符段。
这里的 n 表示一个整数序号，组从 1 开始编号
r'(.{2}) \1' 可匹配 'ok ok' 或 'no no'，不匹配 'no oh'

六、其余
re.fullmatch(pattern,string,flags=0) #如果整个 string 与 pattern 匹配则成功并返回相应的 match 对象，
否则返回 None
re.finditer(pattern,string,flags=0) #功能与findall类似，但返回的是迭代器，使用该迭代器可以顺序取得表示各非重叠匹配的match对象
re.sub(pattern,repl,string,count=0,flags=0)
"""


import re

# r1 = re.compile('abc')
# res = re.search(r1, 'aaabcbcbabcb')
# print (res) #<re.Match object; span=(2, 5), match='abc'>
# res = re.search(r1, 'aaa')
# print (res) #None

# print (re.split(' ',"abc abb are not the same")) #['abc', 'abb', 'are', 'not', 'the', 'same']
# print (re.split(" ", "1 2   3    4")) #['1', '2', '', '', '3', '', '', '', '4']


# print(re.split('[ ,]', '1 2, 3 4, , 5')) #['1', '2', '', '3', '4', '', '', '', '5']
# print(re.split('[ ,]*', '1 2, 3 4, , 5')) #['', '1', '', '2', '', '3', '', '4', '', '5', '']s
# print (re.split('a*', 'abbaaabbdbbabbababbabb')) #['', '', 'b', 'b', '', 'b', 'b', 'd', 'b', 'b', '', 'b', 'b', '', 'b', '', 'b', 'b', '', 'b', 'b', '']

# print (re.match('ab*','abbbbbbc')) #<re.Match object; span=(0, 7), match='abbbbbb'>
# print (re.match('ab*','a')) #<re.Match object; span=(0, 1), match='a'>
# print (re.match('ab*','')) #None

# print (re.match('a?','bbhdvaaad')) #<re.Match object; span=(0, 0), match=''>
# print (re.match('a?','bbd')) #<re.Match object; span=(0, 0), match=''>

# print (re.match('a?','qbbhdvaaad')) #<re.Match object; span=(0, 0), match=''>
# print (re.match('a?','bbda')) #<re.Match object; span=(0, 0), match=''>

# print (re.search('\d+','12sa4')) #<re.Match object; span=(0, 2), match='12'>

# print (re.search('-?\d+','12sa4')) #<re.Match object; span=(0, 2), match='12'>


# print(re.fullmatch('(010-?)8{2}[0-9]{6}','01088034567')) #<re.Match object; span=(0, 11), match='01088034567'>

# print(re.fullmatch('(010-?)[2-9]{3}','01088034567')) None

# print(re.finditer('(010-?)8{2}[0-9]{6}','01088034567')) #<callable_iterator object at 0x10fd45c88>


# print (re.sub('[abc]', '*', 'aabhskduasdd')) #***hskdu*sdd

# mat = re.fullmatch('(010-?)8{2}[0-9]{6}','01088034567')
# print (mat.start(),mat.end(),mat.group()) #0 11 01088034567
# print (mat.span()) (0,11)

# mat = re.finditer('(010-?)8{2}[0-9]{6}','01088034567,01088879202')
# for mi in mat:
# 	print (mi.group())















































